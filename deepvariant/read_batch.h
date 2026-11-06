/*
 * A region's reads kept in C++ (deepvariant-aws-graviton#10).
 *
 * make_examples used to turn every read into a Python Read proto (one per
 * read per region fetch), keep them in a Python list, and pass that list to
 * each C++ stage, which converted it back to pointers. A ReadBatch holds the
 * region's reads as C++ protos allocated in one protobuf Arena, so the region
 * engine, the allele counters and the pileup encoder take pointers directly,
 * and Python only sees the batch. Reads are filled through the same
 * SamReader::Query iterator (same filters, same order) as the Python path.
 */
#ifndef LEARNING_GENOMICS_DEEPVARIANT_READ_BATCH_H_
#define LEARNING_GENOMICS_DEEPVARIANT_READ_BATCH_H_

#include <memory>
#include <vector>

#include "google/protobuf/arena.h"
#include "third_party/nucleus/io/sam_reader.h"
#include "third_party/nucleus/protos/range.pb.h"
#include "third_party/nucleus/protos/reads.pb.h"
#include "third_party/nucleus/util/proto_ptr.h"

namespace learning {
namespace genomics {
namespace deepvariant {

class ReadBatch {
 public:
  using Read = nucleus::genomics::v1::Read;

  ReadBatch();
  ReadBatch(const ReadBatch&) = delete;
  ReadBatch& operator=(const ReadBatch&) = delete;

  // Appends reader.Query(region)'s reads, in iterator order. False with
  // `error` set when the iterator fails (as the Python path would raise).
  bool AppendQuery(nucleus::SamReader& reader,
                   const nucleus::genomics::v1::Range& region,
                   std::string* error);
  // Appends a copy of `read`.
  void Append(const Read& read);

  size_t size() const { return reads_.size(); }
  const Read& at(size_t i) const { return *reads_[i]; }
  const std::vector<const Read*>& reads() const { return reads_; }

  // Keeps reads_[idx[0]], reads_[idx[1]], ... in that order (reservoir sample).
  void Select(const std::vector<int>& idx);
  // New order of the batch: reads_[order[0]], ..., with reads at positions
  // `replaced_pos` of the new order replaced by `replaced` (moved in).
  void Reorder(const std::vector<int>& order,
               const std::vector<int>& replaced_pos,
               std::vector<Read>* replaced);

  // The reads as ConstProtoPtrs, for the existing consumer signatures.
  std::vector<nucleus::ConstProtoPtr<const Read>> ConstPtrs() const;
  std::vector<nucleus::ConstProtoPtr<Read>> MutablePtrs() const;

 private:
  Read* New();
  std::unique_ptr<google::protobuf::Arena> arena_;
  std::vector<const Read*> reads_;
  // Replacements from realignment, owned here (they were made on the heap).
  std::vector<std::unique_ptr<Read>> extra_;
};

}  // namespace deepvariant
}  // namespace genomics
}  // namespace learning

#endif  // LEARNING_GENOMICS_DEEPVARIANT_READ_BATCH_H_
