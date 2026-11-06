#include "deepvariant/read_batch.h"

#include <string>
#include <utility>

namespace learning {
namespace genomics {
namespace deepvariant {

namespace {
google::protobuf::ArenaOptions BatchArenaOptions() {
  google::protobuf::ArenaOptions o;
  o.start_block_size = 256 << 10;
  o.max_block_size = 4 << 20;
  return o;
}
}  // namespace

ReadBatch::ReadBatch()
    : arena_(std::make_unique<google::protobuf::Arena>(BatchArenaOptions())) {}

ReadBatch::Read* ReadBatch::New() {
  return google::protobuf::Arena::CreateMessage<Read>(arena_.get());
}

bool ReadBatch::AppendQuery(nucleus::SamReader& reader,
                            const nucleus::genomics::v1::Range& region,
                            std::string* error) {
  auto it = reader.Query(region);
  if (!it.ok()) {
    *error = it.error_message();
    return false;
  }
  std::shared_ptr<nucleus::SamIterable> iterable = it.ValueOrDie();
  while (true) {
    Read* r = New();
    nucleus::StatusOr<bool> more = iterable->Next(r);
    if (!more.ok()) {
      *error = more.error_message();
      return false;
    }
    if (!more.ValueOrDie()) break;  // r stays unused in the arena
    reads_.push_back(r);
  }
  return true;
}

void ReadBatch::Append(const Read& read) {
  Read* r = New();
  r->CopyFrom(read);
  reads_.push_back(r);
}

void ReadBatch::Select(const std::vector<int>& idx) {
  std::vector<const Read*> out;
  out.reserve(idx.size());
  for (int i : idx) out.push_back(reads_[i]);
  reads_ = std::move(out);
}

void ReadBatch::Reorder(const std::vector<int>& order,
                        const std::vector<int>& replaced_pos,
                        std::vector<Read>* replaced) {
  std::vector<const Read*> out;
  out.reserve(order.size());
  for (int i : order) out.push_back(reads_[i]);
  for (size_t k = 0; k < replaced_pos.size(); ++k) {
    extra_.push_back(std::make_unique<Read>(std::move((*replaced)[k])));
    out[replaced_pos[k]] = extra_.back().get();
  }
  reads_ = std::move(out);
}

std::vector<nucleus::ConstProtoPtr<const ReadBatch::Read>>
ReadBatch::ConstPtrs() const {
  std::vector<nucleus::ConstProtoPtr<const Read>> v;
  v.reserve(reads_.size());
  for (const Read* r : reads_) v.emplace_back(r);
  return v;
}

std::vector<nucleus::ConstProtoPtr<ReadBatch::Read>> ReadBatch::MutablePtrs()
    const {
  std::vector<nucleus::ConstProtoPtr<Read>> v;
  v.reserve(reads_.size());
  for (const Read* r : reads_) v.emplace_back(const_cast<Read*>(r));
  return v;
}

}  // namespace deepvariant
}  // namespace genomics
}  // namespace learning
