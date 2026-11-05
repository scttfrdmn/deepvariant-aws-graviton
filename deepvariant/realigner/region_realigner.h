/*
 * One region's realignment in C++ (deepvariant-aws-graviton#10, design step 5).
 *
 * A port of Realigner.realign_reads (realigner.py) for the production
 * configuration: VARIANT_READS window selection with the counts-only counter,
 * De Bruijn assembly per window, read assignment to the maximally overlapping
 * assembled region, and the fast-pass aligner per assembled region. Reads are
 * taken as pointers to the caller's protos and the output order is returned as
 * input indices, so unchanged reads are never copied. Any condition the port
 * does not cover makes it return false, and the caller runs the Python path.
 */
#ifndef LEARNING_GENOMICS_DEEPVARIANT_REALIGNER_REGION_REALIGNER_H_
#define LEARNING_GENOMICS_DEEPVARIANT_REALIGNER_REGION_REALIGNER_H_

#include <vector>

#include "deepvariant/protos/realigner.pb.h"
#include "third_party/nucleus/io/reference.h"
#include "third_party/nucleus/protos/range.pb.h"
#include "third_party/nucleus/protos/reads.pb.h"

namespace learning {
namespace genomics {
namespace deepvariant {

struct RegionRealignment {
  // Output reads in Python's order: input indices.
  std::vector<int> order;
  // Positions in `order` whose read was replaced, and the replacements.
  std::vector<int> replaced_pos;
  std::vector<nucleus::genomics::v1::Read> replaced;
  // AlignerOptions.read_size as Python leaves it in config (-1: untouched).
  int last_read_size = -1;
};

// Realigner.realign_reads(reads, region) for `config`; false means "not covered,
// use Python" and leaves *out unspecified.
bool RealignRegion(const std::vector<const nucleus::genomics::v1::Read*>& reads,
                   const nucleus::genomics::v1::Range& region,
                   const nucleus::GenomeReference& ref,
                   const RealignerOptions& config, bool ws_light,
                   RegionRealignment* out);

}  // namespace deepvariant
}  // namespace genomics
}  // namespace learning

#endif  // LEARNING_GENOMICS_DEEPVARIANT_REALIGNER_REGION_REALIGNER_H_
