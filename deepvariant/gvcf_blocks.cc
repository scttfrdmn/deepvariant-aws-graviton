#include "deepvariant/gvcf_blocks.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "deepvariant/protos/deepvariant.pb.h"
#include "third_party/nucleus/protos/struct.pb.h"
#include "third_party/nucleus/protos/variants.pb.h"

namespace learning {
namespace genomics {
namespace deepvariant {
namespace {

using nucleus::genomics::v1::Variant;
using nucleus::genomics::v1::VariantCall;

// variant_caller.py CANONICAL_DNA_BASES and EXTENDED_IUPAC_CODES.
bool IsCanonical(const std::string& b) {
  return b == "A" || b == "C" || b == "G" || b == "T";
}
bool IsExtendedIupac(const std::string& b) {
  static const char* kCodes = "RYSWKMBDHVN";
  return b.size() == 1 && std::string(kCodes).find(b[0]) != std::string::npos;
}

// _quantize_gq.
int QuantizeGq(int raw_gq, int binsize) {
  if (raw_gq < 1) return 0;
  // Python // on non-negative ints.
  return ((raw_gq - 1) / binsize) * binsize + 1;
}

// _rescale_read_counts_if_necessary, same float operations.
void Rescale(int64_t* n_ref, int64_t* n_total, int64_t max_allowed) {
  if (*n_total > max_allowed) {
    double ratio = static_cast<double>(*n_ref) / (1.0 * static_cast<double>(*n_total));
    *n_ref = static_cast<int64_t>(std::ceil(ratio * static_cast<double>(max_allowed)));
    *n_total = max_allowed;
  }
}

struct Site {
  const AlleleCountSummary* s;
  bool none_gq;  // ambiguous reference base: quantized_gq is None
  int quantized_gq;
  int raw_gq;
  const std::array<double, 3>* likelihoods;
  int64_t read_depth;
  bool has_valid_gl;
};

void SetIntField(VariantCall* call, const std::string& key, int64_t v) {
  auto& lv = (*call->mutable_info())[key];
  lv.clear_values();
  lv.add_values()->set_int_value(v);
}

Variant MakeBlock(const Site& first, const Site& last, const std::string& sample,
                  const std::array<double, 3>* gl, std::vector<int> genotype,
                  int64_t gq, int64_t min_dp, bool med, int64_t med_dp) {
  Variant v;
  v.set_reference_name(first.s->reference_name());
  v.set_reference_bases(first.s->ref_base());
  v.add_alternate_bases("<*>");
  v.set_start(first.s->position());
  v.set_end(last.s->position() + 1);
  VariantCall* call = v.add_calls();
  call->set_call_set_name(sample);
  for (int g : genotype) call->add_genotype(g);
  if (gl) for (double x : *gl) call->add_genotype_likelihood(x);
  SetIntField(call, "GQ", gq);
  SetIntField(call, "MIN_DP", min_dp);
  if (med) SetIntField(call, "MED_DP", med_dp);
  return v;
}

}  // namespace

std::vector<std::string> MakeGvcfRecords(const AlleleCounter& counter,
                                         int left_padding, int right_padding,
                                         const GvcfTable& table,
                                         const GvcfOptions& o) {
  const std::vector<AlleleCountSummary> summaries =
      counter.SummaryCounts(left_padding, right_padding);
  std::vector<Site> sites;
  sites.reserve(summaries.size());
  for (const AlleleCountSummary& s : summaries) {
    Site e{&s, false, 0, 0, nullptr, s.total_read_count(), true};
    if (!IsCanonical(s.ref_base())) {
      if (!IsExtendedIupac(s.ref_base()))
        throw std::invalid_argument("Invalid reference base=" + s.ref_base() +
                                    " found during gvcf calculation");
      e.none_gq = true;
    } else {
      const int hap = o.haploid_contigs.count(s.reference_name()) ? 1 : 0;
      int64_t n_ref = s.ref_supporting_read_count(), n_total = s.total_read_count();
      Rescale(&n_ref, &n_total, table.max_cache_coverage);
      const size_t idx = n_total * (n_total + 1) / 2 + n_ref;
      e.raw_gq = table.gq[hap][idx];
      e.likelihoods = &table.likelihoods[hap][idx];
      e.quantized_gq = QuantizeGq(e.raw_gq, o.gq_resolution);
      const auto& l = *e.likelihoods;
      // np.amax(likelihoods) == likelihoods[0]
      e.has_valid_gl = std::max({l[0], l[1], l[2]}) == l[0];
    }
    sites.push_back(e);
  }
  std::vector<std::string> out;
  size_t i = 0;
  while (i < sites.size()) {
    // itertools.groupby key: (quantized_gq, has_valid_gl); None compares equal to None.
    size_t j = i + 1;
    while (j < sites.size() && sites[j].none_gq == sites[i].none_gq &&
           (sites[i].none_gq || sites[j].quantized_gq == sites[i].quantized_gq) &&
           sites[j].has_valid_gl == sites[i].has_valid_gl)
      ++j;
    if (!sites[i].none_gq) {
      if (sites[i].has_valid_gl) {
        size_t min_idx = i;
        int64_t min_dp = sites[i].read_depth;
        std::vector<int64_t> dps;
        dps.reserve(j - i);
        for (size_t k = i; k < j; ++k) {
          if (sites[k].raw_gq < sites[min_idx].raw_gq) min_idx = k;  // first minimum
          min_dp = std::min(min_dp, sites[k].read_depth);
          dps.push_back(sites[k].read_depth);
        }
        int64_t med_dp = 0;
        if (o.include_med_dp) {
          // int(statistics.median(ints)): mean of the two middle values when even.
          std::sort(dps.begin(), dps.end());
          const size_t n = dps.size();
          med_dp = (n % 2) ? dps[n / 2]
                           : static_cast<int64_t>((dps[n / 2 - 1] + dps[n / 2]) / 2.0);
        }
        out.push_back(MakeBlock(sites[i], sites[j - 1], o.sample_name, sites[min_idx].likelihoods,
                                {0, 0}, sites[min_idx].raw_gq, min_dp, o.include_med_dp, med_dp)
                          .SerializeAsString());
      } else {
        for (size_t k = i; k < j; ++k)
          out.push_back(MakeBlock(sites[k], sites[k], o.sample_name, sites[k].likelihoods,
                                  {-1, -1}, sites[k].raw_gq, sites[k].read_depth,
                                  o.include_med_dp, sites[k].read_depth)
                            .SerializeAsString());
      }
    }
    i = j;
  }
  return out;
}

}  // namespace deepvariant
}  // namespace genomics
}  // namespace learning
