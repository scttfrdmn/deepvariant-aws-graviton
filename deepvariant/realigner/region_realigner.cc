#include "deepvariant/realigner/region_realigner.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "deepvariant/allelecounter.h"
#include "deepvariant/protos/deepvariant.pb.h"
#include "deepvariant/realigner/debruijn_graph.h"
#include "deepvariant/realigner/fast_pass_aligner.h"
#include "deepvariant/realigner/window_selector.h"
#include "third_party/nucleus/util/proto_ptr.h"
#include "third_party/nucleus/util/utils.h"

namespace learning {
namespace genomics {
namespace deepvariant {
namespace {

using nucleus::genomics::v1::Range;
using nucleus::genomics::v1::Read;

constexpr int64_t kRefAlignMargin = 20;  // realigner.py _REF_ALIGN_MARGIN

Range MakeRange(const std::string& name, int64_t start, int64_t end) {
  Range r;
  r.set_reference_name(name);
  r.set_start(start);
  r.set_end(end);
  return r;
}

// utils.read_range: alignment reference_name, start, ReadEnd.
Range ReadRangeOf(const Read& read) {
  return MakeRange(read.alignment().position().reference_name(),
                   read.alignment().position().position(),
                   nucleus::ReadEnd(read));
}

// ranges.overlap_len.
int64_t OverlapLen(const Range& a, const Range& b) {
  if (a.reference_name() != b.reference_name()) return 0;
  return std::max<int64_t>(0, std::min(a.end(), b.end()) - std::max(a.start(), b.start()));
}

// GenomeReference bases; false where Python's query would raise.
bool Bases(const nucleus::GenomeReference& ref, const Range& r, std::string* out) {
  auto s = ref.GetBases(r);
  if (!s.ok()) return false;
  *out = s.ValueOrDie();
  return true;
}

// window_selector._candidates_from_reads (VARIANT_READS) + _candidates_to_windows.
bool SelectWindows(const std::vector<const Read*>& reads, const Range& region,
                   const nucleus::GenomeReference& ref,
                   const WindowSelectorOptions& ws, bool light,
                   std::vector<Range>* windows) {
  windows->clear();
  if (reads.empty()) return true;
  if (ws.realign_all()) {
    windows->push_back(region);
    return true;
  }
  if (ws.window_selector_model().model_type() != WindowSelectorModel::VARIANT_READS)
    return false;
  auto contig = ref.Contig(region.reference_name());
  if (!contig.ok()) return false;
  const int64_t n_bp = ws.region_expansion_in_bp();
  if (n_bp < 0) return false;
  const Range expanded = MakeRange(
      region.reference_name(), std::max<int64_t>(region.start() - n_bp, 0),
      std::min<int64_t>(region.end() + n_bp, contig.ValueOrDie()->n_bases()));
  AlleleCounterOptions options;
  options.mutable_read_requirements()->set_min_mapping_quality(ws.min_mapq());
  options.mutable_read_requirements()->set_min_base_quality(ws.min_base_quality());
  options.set_keep_legacy_behavior(ws.keep_legacy_behavior());
  AlleleCounter counter(&ref, expanded, std::vector<int>(), options);
  if (light) counter.SetLightMode(true);
  for (const Read* r : reads) counter.Add(*r, "placeholder_sample_id");
  const std::vector<int> counts = VariantReadsWindowSelectorCandidates(counter, ws);
  const auto& m = ws.window_selector_model().variant_reads_model();
  const int64_t d = ws.min_windows_distance();
  bool have = false;
  int64_t start_pos = 0, end_pos = 0;
  for (size_t i = 0; i < counts.size(); ++i) {
    if (counts[i] < m.min_num_supporting_reads() || counts[i] > m.max_num_supporting_reads())
      continue;
    const int64_t pos = expanded.start() + static_cast<int64_t>(i);
    if (!have) {
      have = true;
      start_pos = end_pos = pos;
    } else if (pos > end_pos + 2 * d) {
      windows->push_back(MakeRange(region.reference_name(), start_pos - d, end_pos + d));
      start_pos = end_pos = pos;
    } else {
      end_pos = pos;
    }
  }
  if (have) windows->push_back(MakeRange(region.reference_name(), start_pos - d, end_pos + d));
  // Already sorted by (start, end): positions ascend and windows don't overlap.
  return true;
}

}  // namespace

bool RealignRegion(const std::vector<const Read*>& reads, const Range& region,
                   const nucleus::GenomeReference& ref,
                   const RealignerOptions& config, bool ws_light,
                   RegionRealignment* out) {
  out->order.clear();
  out->replaced_pos.clear();
  out->replaced.clear();
  out->last_read_size = -1;
  if (reads.empty()) return true;  // realign_reads returns [], [] -> caller keeps []
  if (config.split_skip_reads() || config.diagnostics().enabled()) return false;

  // 1. Windows.
  std::vector<Range> windows;
  if (!SelectWindows(reads, region, ref, config.ws_config(), ws_light, &windows)) return false;

  // 2. Assembly per window (call_debruijn_graph).
  struct Assembled {
    Range span;
    std::vector<std::string> haplotypes;
    std::vector<int> reads;  // input indices, in assignment order
  };
  std::vector<Assembled> regions;
  for (const Range& w : windows) {
    if (w.end() - w.start() > config.ws_config().max_window_size()) continue;
    if (!ref.IsValidInterval(w)) continue;
    std::string ref_bases;
    if (!Bases(ref, w, &ref_bases)) return false;
    std::vector<nucleus::ConstProtoPtr<const Read>> window_reads;
    for (const Read* r : reads)
      if (nucleus::ReadOverlapsRegion(*r, w)) window_reads.emplace_back(r);
    std::unique_ptr<DeBruijnGraph> graph =
        DeBruijnGraph::Build(ref_bases, window_reads, config.dbg_config());
    std::vector<std::string> haps;
    if (!graph) haps.push_back(ref_bases);
    else haps = graph->CandidateHaplotypes();
    if (!haps.empty() && !(haps.size() == 1 && haps[0] == ref_bases)) {
      Assembled a;
      a.span = w;
      a.haplotypes = std::move(haps);
      regions.push_back(std::move(a));
    }
  }

  // 3. assign_reads_to_assembled_regions: maximal overlap, lowest index on ties.
  std::vector<int> unassigned;
  for (int i = 0; i < static_cast<int>(reads.size()); ++i) {
    if (regions.empty()) { unassigned.push_back(i); continue; }
    const Range rr = ReadRangeOf(*reads[i]);
    int best = 0;
    int64_t best_len = OverlapLen(rr, regions[0].span);
    for (int k = 1; k < static_cast<int>(regions.size()); ++k) {
      const int64_t len = OverlapLen(rr, regions[k].span);
      if (len > best_len) { best = k; best_len = len; }
    }
    if (best_len == 0) unassigned.push_back(i);
    else regions[best].reads.push_back(i);
  }
  out->order = unassigned;

  // 4. call_fast_pass_aligner per assembled region, in order.
  for (const Assembled& a : regions) {
    if (a.reads.empty()) continue;
    const std::string& contig = a.span.reference_name();
    int64_t span_start = 0, span_end = 0;
    for (size_t j = 0; j < a.reads.size(); ++j) {
      const Range rr = ReadRangeOf(*reads[a.reads[j]]);
      if (j == 0 || rr.start() < span_start) span_start = rr.start();
      if (j == 0 || rr.end() > span_end) span_end = rr.end();
    }
    auto ci = ref.Contig(contig);
    if (!ci.ok()) return false;
    const int64_t ref_start =
        std::max<int64_t>(0, std::min(span_start, a.span.start()) - kRefAlignMargin);
    const int64_t ref_end = std::min<int64_t>(
        ci.ValueOrDie()->n_bases(), std::max(span_end, a.span.end()) + kRefAlignMargin);
    std::string prefix, center, suffix;
    if (!Bases(ref, MakeRange(contig, ref_start, a.span.start()), &prefix)) return false;
    if (!Bases(ref, a.span, &center)) return false;
    const size_t base = out->order.size();
    if (ref_end <= a.span.end()) {
      for (int idx : a.reads) out->order.push_back(idx);
      continue;
    }
    if (!Bases(ref, MakeRange(contig, a.span.end(), ref_end), &suffix)) return false;
    AlignerOptions aln = config.aln_config();
    aln.set_read_size(reads[a.reads[0]]->aligned_sequence().size());
    aln.set_force_alignment(false);
    out->last_read_size = aln.read_size();
    FastPassAligner fpa;
    fpa.set_normalize_reads(config.normalize_reads());
    fpa.set_options(aln);
    fpa.set_reference(prefix + center + suffix);
    fpa.set_ref_start(contig, ref_start);
    fpa.set_ref_prefix_len(prefix.size());
    fpa.set_ref_suffix_len(suffix.size());
    std::vector<std::string> extended;
    extended.reserve(a.haplotypes.size());
    for (const std::string& h : a.haplotypes) extended.push_back(prefix + h + suffix);
    fpa.set_haplotypes(extended);
    std::vector<const Read*> ptrs;
    ptrs.reserve(a.reads.size());
    for (int idx : a.reads) ptrs.push_back(reads[idx]);
    std::vector<int> changed_idx;
    std::vector<Read> changed;
    fpa.AlignReadsViews(ptrs, &changed_idx, &changed);
    for (int idx : a.reads) out->order.push_back(idx);
    for (size_t c = 0; c < changed_idx.size(); ++c) {
      out->replaced_pos.push_back(static_cast<int>(base) + changed_idx[c]);
      out->replaced.push_back(std::move(changed[c]));
    }
  }
  return true;
}

}  // namespace deepvariant
}  // namespace genomics
}  // namespace learning
