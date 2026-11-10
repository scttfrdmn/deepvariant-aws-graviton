/*
 * Copyright 2017 Google LLC.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef LEARNING_GENOMICS_DEEPVARIANT_PILEUP_IMAGE_NATIVE_H_
#define LEARNING_GENOMICS_DEEPVARIANT_PILEUP_IMAGE_NATIVE_H_

#include <memory>
#include <string>
#include <cstring>
#include <vector>

#include "deepvariant/pileup_channel_lib.h"
#include "deepvariant/protos/deepvariant.pb.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/nucleus/protos/reads.pb.h"
#include "third_party/nucleus/util/proto_ptr.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace learning {
namespace genomics {
namespace deepvariant {

using std::string;

constexpr unsigned char kChannelValue255 = 255;
constexpr unsigned char kChannelValue200 = 200;
// Different ways alt aligned reads can be expressed.
// This enum matches flag values in make_example_options.
// It helps to avoid string comparison in performance critical parts.
enum AltAlignedPileup {
  kNone = 0,
  kBaseChannels = 1,
  kDiffChannels = 2,
  kRows = 3,
  kSingleRow = 4,
};

template <class T>
std::vector<T> ToVector(const google::protobuf::RepeatedPtrField<T> container) {
  return std::vector<T>(std::make_move_iterator(container.begin()),
                        std::make_move_iterator(container.end()));
}

struct ImageRow {
  int width;
  int num_channels;
  std::vector<DeepVariantChannelEnum> channel_enums;
  std::vector<std::vector<unsigned char>> channel_data;

  int Width() const;
  explicit ImageRow(int width, int num_channels);
  bool operator==(const ImageRow& other) const {
    if (channel_enums != other.channel_enums) {
      LOG(INFO) << "ImageRow channel_enums mismatch";
    }
    if (channel_data != other.channel_data) {
      LOG(INFO) << "ImageRow channel_data mismatch";
    }
    return width == other.width && num_channels == other.num_channels &&
           channel_enums == other.channel_enums &&
           channel_data == other.channel_data;
  }
};

class PileupImageEncoderNative {
 public:
  // All channel enums that will be processed by this PileupImageEncoder
  std::vector<DeepVariantChannelEnum> channel_enums_;

  // Essential API methods.
  explicit PileupImageEncoderNative(const PileupImageOptions& options);

  // Return all the channels in this image, as a list of enums.
  std::vector<DeepVariantChannelEnum> AllChannelsEnum(
      absl::string_view alt_aligned_representation);

  // Create read pileup image section for one sample.
  std::vector<std::unique_ptr<ImageRow>> BuildPileupForOneSample(
      const DeepVariantCall& dv_call, const string& ref_bases,
      const std::vector<const ::nucleus::genomics::v1::Read*>& reads,
      int image_start_pos, const std::vector<std::string>& alt_alleles,
      const SampleOptions& sample_options, float mean_coverage,
      // Contains original alignment positions for trimmed reads. This array has
      // the same order as reads.
      const std::vector<int64_t>* alignment_positions = nullptr,
      absl::flat_hash_set<DeepVariantChannelEnum> channels_enum_to_blank = {});

  // Simple wrapper around BuildPileupForOneSample that allows us to efficiently
  // pass large protobufs in from Python. Simply unwraps the ConstProtoPtr
  // objects and calls BuildPileupForOneSample().
  std::vector<std::unique_ptr<ImageRow>> BuildPileupForOneSamplePython(
      const nucleus::ConstProtoPtr<
          const learning::genomics::deepvariant::DeepVariantCall>&
          wrapped_dv_call,
      const string& ref_bases,
      absl::Span<
          const nucleus::ConstProtoPtr<const ::nucleus::genomics::v1::Read>>
          wrapped_reads,
      int image_start_pos, const std::vector<std::string>& alt_alleles,
      const SampleOptions& sample_options) {
    std::vector<const nucleus::genomics::v1::Read *> reads;
    reads.reserve(wrapped_reads.size());
    for (const auto& wrapped_read : wrapped_reads) {
      reads.emplace_back(wrapped_read.p_);
    }
    return BuildPileupForOneSample(*(wrapped_dv_call.p_), ref_bases, reads,
                                   image_start_pos, alt_alleles, sample_options,
                                   /*mean_coverage=*/0.0);
  }

  // Encode one read into a row of pixels for our image.
  std::unique_ptr<ImageRow> EncodeRead(
      const learning::genomics::deepvariant::DeepVariantCall& dv_call,
      absl::string_view ref_bases, const nucleus::genomics::v1::Read& read,
      int image_start_pos, const std::vector<std::string>& alt_alleles,
      const absl::flat_hash_set<DeepVariantChannelEnum>& channels_enum_to_blank =
          {});

  // Simple wrapper around EncodeRead that allows us to efficiently pass large
  // protobufs in from Python. Simply unwraps the ConstProtoPtr objects and
  // calls EncodeRead().
  std::unique_ptr<ImageRow> EncodeReadPython(
      const nucleus::ConstProtoPtr<
          const learning::genomics::deepvariant::DeepVariantCall>&
          wrapped_dv_call,
      absl::string_view ref_bases,
      const nucleus::ConstProtoPtr<const ::nucleus::genomics::v1::Read>&
          wrapped_read,
      int image_start_pos, const std::vector<std::string>& alt_alleles) {
    return EncodeRead(*(wrapped_dv_call.p_), ref_bases, *(wrapped_read.p_),
                      image_start_pos, alt_alleles);
  }

  // Encode the reference bases into a single row of pixels.
  std::unique_ptr<ImageRow> EncodeReference(const string& ref_bases);

 private:
  // aws-realign#130: env-gated capture (DV_DUMP_PILEUP) of one call's inputs and rows.
  void DumpPileupCall(
      const DeepVariantCall& dv_call, const string& ref_bases,
      const std::vector<const ::nucleus::genomics::v1::Read*>& reads,
      int image_start_pos, const std::vector<std::string>& alt_alleles,
      int max_reads, const std::vector<int>& sampled_indices,
      const std::vector<int64_t>* alignment_positions,
      const absl::flat_hash_set<DeepVariantChannelEnum>& channels_enum_to_blank,
      const std::vector<std::unique_ptr<ImageRow>>& rows);

  int GetHapIndex(const nucleus::genomics::v1::Read& read);

  const PileupImageOptions options_;
  // Exact table-driven read encoder, when the channel set allows it (#10).
  std::shared_ptr<const FastReadEncoder> fast_encoder_;
};

// Exact fast pileup path (deepvariant-aws-graviton#10): per-read channels
// without per-base overhead, the reference band encoded once per pileup, and
// rows interleaved into the image array a row at a time. Same bytes as the
// original code; DV_PILEUP_FAST=0 selects the original code.
bool PileupFastPath();

namespace internal {
// In-place transpose of an 8x8 byte matrix held as eight little-endian words
// (row r = x[r], column c = byte c): afterwards byte c of x[r] is the old
// byte r of x[c].
inline void SwapBits(uint64_t& a, uint64_t& b, int shift, uint64_t mask) {
  const uint64_t t = ((a >> shift) ^ b) & mask;
  a ^= t << shift;
  b ^= t;
}
inline void Transpose8x8Bytes(uint64_t x[8]) {
  constexpr uint64_t k32 = 0x00000000FFFFFFFFull, k16 = 0x0000FFFF0000FFFFull,
                     k8 = 0x00FF00FF00FF00FFull;
  SwapBits(x[0], x[4], 32, k32); SwapBits(x[1], x[5], 32, k32);
  SwapBits(x[2], x[6], 32, k32); SwapBits(x[3], x[7], 32, k32);
  SwapBits(x[0], x[2], 16, k16); SwapBits(x[1], x[3], 16, k16);
  SwapBits(x[4], x[6], 16, k16); SwapBits(x[5], x[7], 16, k16);
  SwapBits(x[0], x[1], 8, k8); SwapBits(x[2], x[3], 8, k8);
  SwapBits(x[4], x[5], 8, k8); SwapBits(x[6], x[7], 8, k8);
}
}  // namespace internal

// Converting a vector<ImageRow> into a 3-d vector where channels are stored in
// the lowest dimension. Special processing is required for alt aligned
// channels.
// * --alt_aligned_pileup=diff_channels Add channel 5 (base differs
// from ref) of both alts as channels
// * --alt_aligned_pileup=base_channels Add channel 0 (bases ATCG) of both alts
// as channels.
// * --alt_aligned_pileup=rows alt aligned data is stored as extra rows.
//
// TODO: Consolidate streaming examples and normal examples so
// that we don't need to use the template anymore.
template <class T>
void FillPileupArray(
    absl::Span<const std::unique_ptr<ImageRow>> image,
    absl::Span<const std::vector<std::unique_ptr<ImageRow>>> alt_image,
    AltAlignedPileup alt_aligned_representation, T* pileup_array,
    int buffer_size, int buffer_pos = 0,
    absl::Span<const int> alt_image_row_indices = {}) {
  // TODO Although channels 0 and 5 should not change we need to set
  // channel number programmatically.
  int alt_channel_index = 0;
  switch (alt_aligned_representation) {
    case AltAlignedPileup::kDiffChannels:
      alt_channel_index = 5;
      break;
    case AltAlignedPileup::kBaseChannels:
      alt_channel_index = 0;
      break;
    default:
      alt_channel_index = 0;
  }
  const bool alt_channels =
      alt_aligned_representation == AltAlignedPileup::kBaseChannels ||
      alt_aligned_representation == AltAlignedPileup::kDiffChannels;
  if (!alt_channels && PileupFastPath()) {
    // Same element order as the loop below: row, then column, then channel.
    for (int row = 0; row < image.size(); row++) {
      const auto& channel_data = image[row]->channel_data;
      if (channel_data.empty()) continue;
      const int width = image[row]->Width();
      const int num_channels = channel_data.size();
      CHECK_LE(buffer_pos + width * num_channels, buffer_size);
      constexpr int kMaxFastChannels = 32;
      CHECK_LE(num_channels, kMaxFastChannels);
      const unsigned char* src[kMaxFastChannels];
      for (int channel = 0; channel < num_channels; channel++) {
        CHECK_GE(channel_data[channel].size(), width);
        src[channel] = channel_data[channel].data();
      }
      int column = 0;
      if (num_channels == 7) {
        // The standard short-read image: as below, unrolled, with the words
        // in registers; each column's 7 bytes are stored as an 8-byte write
        // whose spare byte the next column's write overwrites, and the
        // block's last column as an exact 7-byte write.
        unsigned char* const out = &(*pileup_array)[0];
        for (; column + 8 <= width; column += 8) {
          uint64_t x[8];
          for (int channel = 0; channel < 7; channel++) {
            std::memcpy(&x[channel], src[channel] + column, 8);
          }
          x[7] = 0;
          internal::Transpose8x8Bytes(x);
          unsigned char* o = out + buffer_pos;
          std::memcpy(o + 0, &x[0], 8);
          std::memcpy(o + 7, &x[1], 8);
          std::memcpy(o + 14, &x[2], 8);
          std::memcpy(o + 21, &x[3], 8);
          std::memcpy(o + 28, &x[4], 8);
          std::memcpy(o + 35, &x[5], 8);
          std::memcpy(o + 42, &x[6], 8);
          std::memcpy(o + 49, &x[7], 7);
          buffer_pos += 56;
        }
      } else if (num_channels <= 8) {
        // Eight columns at a time: load 8 bytes of each channel, transpose the
        // 8x8 byte block so each word holds one column's channels, and store
        // num_channels bytes per column (little-endian byte order).
        unsigned char* const out = &(*pileup_array)[0];
        for (; column + 8 <= width; column += 8) {
          uint64_t x[8] = {0, 0, 0, 0, 0, 0, 0, 0};
          for (int channel = 0; channel < num_channels; channel++) {
            std::memcpy(&x[channel], src[channel] + column, 8);
          }
          internal::Transpose8x8Bytes(x);
          for (int k = 0; k < 8; k++) {
            std::memcpy(out + buffer_pos, &x[k], num_channels);
            buffer_pos += num_channels;
          }
        }
      }
      for (; column < width; column++) {
        for (int channel = 0; channel < num_channels; channel++) {
          (*pileup_array)[buffer_pos + channel] = src[channel][column];
        }
        buffer_pos += num_channels;
      }
    }
  } else
  for (int row = 0; row < image.size(); row++) {
    for (int column = 0; column < image[row]->Width(); column++) {
      if (!image[row]->channel_data.empty()) {
        // Lower dimension is a channel data. Here we iterate all channels to
        // fill one position of the pileup image.
        for (int channel = 0; channel < image[row]->channel_data.size();
             channel++) {
          CHECK_LT(buffer_pos, buffer_size);
          (*pileup_array)[buffer_pos] =
              image[row]->channel_data[channel][column];
          buffer_pos++;
        }
        // Fill alt aligned channels if needed.
        if (alt_aligned_representation == AltAlignedPileup::kBaseChannels ||
            alt_aligned_representation == AltAlignedPileup::kDiffChannels) {
          CHECK_EQ(alt_image.size(), 2);
          unsigned char alt_aligned_channel_1_value = 0;
          if (!alt_image[0].empty() &&
              !alt_image[0][row]->channel_data.empty()) {
            alt_aligned_channel_1_value =
                alt_image[0][row]->channel_data[alt_channel_index][column];
          }
          CHECK_LT(buffer_pos, buffer_size);
          // Fill alt aligned channel 1.
          (*pileup_array)[buffer_pos] = alt_aligned_channel_1_value;
          buffer_pos++;
          // Fill alt aligned channel 2.
          if (alt_image[1].empty() || alt_image[1][row]->channel_data.empty()) {
            // Fill with alt aligned channel 1 if alt2 is empty.
            CHECK_LT(buffer_pos, buffer_size);
            (*pileup_array)[buffer_pos] = alt_aligned_channel_1_value;
            buffer_pos++;
          } else {
            CHECK_LT(buffer_pos, buffer_size);
            (*pileup_array)[buffer_pos] =
                alt_image[1][row]->channel_data[alt_channel_index][column];
            buffer_pos++;
          }
        }  // if need_alt_alignment
      }  // for row->Width
    }  // if !channel_data.empty()
  }  // for row

  // Fill alt aligned channels as rows given by alt_image_row_indices.
  for (const int active_alt_image_index : alt_image_row_indices) {
    auto& one_alt_image = alt_image[active_alt_image_index];
    if (one_alt_image.empty()) {
      CHECK_LT(buffer_pos, buffer_size);
      auto pos_offset_to_end =
          image.size() * image[0]->Width() * image[0]->channel_data.size();
      CHECK_LE(buffer_pos + pos_offset_to_end, buffer_size);
      auto* const fill_from = &(*pileup_array)[buffer_pos];
      std::fill(fill_from, fill_from + pos_offset_to_end,
                static_cast<unsigned char>(0));
      buffer_pos +=
          image.size() * image[0]->Width() * image[0]->channel_data.size();
      continue;
    }
    for (int row = 0; row < one_alt_image.size(); row++) {
      for (int column = 0; column < one_alt_image[row]->Width(); column++) {
        if (!one_alt_image[row]->channel_data.empty()) {
          // Lower dimension is a channel data. Here we iterate all channels
          // to fill one position of the pileup image.
          for (int channel = 0;
               channel < one_alt_image[row]->channel_data.size(); channel++) {
            CHECK_LT(buffer_pos, buffer_size);
            (*pileup_array)[buffer_pos] =
                one_alt_image[row]->channel_data[channel][column];
            buffer_pos++;
          }
        }
      }
    }
  }
}

}  // namespace deepvariant
}  // namespace genomics
}  // namespace learning

#endif  // LEARNING_GENOMICS_DEEPVARIANT_PILEUP_IMAGE_NATIVE_H_
