/*
 * gVCF reference blocks from an AlleleCounter, in C++ (deepvariant-aws-graviton).
 *
 * A line-for-line port of VariantCaller.make_gvcfs (variant_caller.py) for the
 * cached-table path: per-site GQ and genotype likelihoods come from the same
 * table Python builds (passed in, not recomputed), so the floating-point values
 * are identical by construction. Blocks are grouped exactly as
 * itertools.groupby on (quantized_gq, has_valid_gl).
 */
#ifndef LEARNING_GENOMICS_DEEPVARIANT_GVCF_BLOCKS_H_
#define LEARNING_GENOMICS_DEEPVARIANT_GVCF_BLOCKS_H_

#include <array>
#include <set>
#include <string>
#include <vector>

#include "deepvariant/allelecounter.h"

namespace learning {
namespace genomics {
namespace deepvariant {

struct GvcfTable {
  int max_cache_coverage = 0;
  // Indexed [is_haploid][n_total * (n_total + 1) / 2 + n_ref].
  std::vector<int> gq[2];
  std::vector<std::array<double, 3>> likelihoods[2];
};

struct GvcfOptions {
  std::string sample_name;
  int gq_resolution = 1;
  bool include_med_dp = false;
  std::set<std::string> haploid_contigs;
};

// Serialized Variant protos, as make_gvcfs(counter.summary_counts(lp, rp)).
std::vector<std::string> MakeGvcfRecords(const AlleleCounter& counter,
                                         int left_padding, int right_padding,
                                         const GvcfTable& table,
                                         const GvcfOptions& options);

}  // namespace deepvariant
}  // namespace genomics
}  // namespace learning

#endif  // LEARNING_GENOMICS_DEEPVARIANT_GVCF_BLOCKS_H_
