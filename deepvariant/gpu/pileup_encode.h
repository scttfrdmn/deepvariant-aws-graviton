// Pileup image encoding for the GPU, written to reproduce
// PileupImageEncoderNative::BuildPileupForOneSample + FillPileupArray
// (kNone alt-aligned representation) bit-for-bit (aws-realign#130).
//
// The same functions compile for the host (the harness's CPU reference) and
// the device. Each piece mirrors the C++ it replaces:
//   EncodeReadRow   Channels::CalculateChannels / CalculateBaseLevelData and
//                   the seven channel FillReadBase methods (one read, one row,
//                   written in HWC order: [col][channel]).
//   EncodeRefRow    Channels::CalculateRefRows (FillRefBase per channel).
//   StdSort         libstdc++ std::sort (introsort, threshold 16, median of
//                   three, final insertion sort), ported exactly so that reads
//                   with equal (hap, position, fragment_name, read_number) keys
//                   land in the same order as the CPU's std::sort. std::sort is
//                   not stable; a stable sort would not be identical.
//
// Floating point: every float expression here is a single IEEE division or
// multiplication followed by truncation to int (no add, so no contraction),
// and builds use -ffp-contract=off and correctly rounded division (HIP's
// default; nvcc -prec-div=true, its default).
#ifndef DEEPVARIANT_GPU_PILEUP_ENCODE_H_
#define DEEPVARIANT_GPU_PILEUP_ENCODE_H_

#include <cstdint>

#include "deepvariant/gpu/gpu_compat.h"

namespace dvgpu {

// nucleus CigarUnit::Operation.
enum : uint8_t {
  kOpUnspecified = 0, kOpMatch = 1, kOpInsert = 2, kOpDelete = 3, kOpSkip = 4,
  kOpSoftClip = 5, kOpHardClip = 6, kOpPad = 7, kOpSeqMatch = 8,
  kOpSeqMismatch = 9
};
// DeepVariantChannelEnum values this encoder implements.
enum : int32_t {
  kChReadBase = 1, kChBaseQuality = 2, kChMappingQuality = 3, kChStrand = 4,
  kChReadSupportsVariant = 5, kChBaseDiffersFromRef = 6, kChInsertSize = 19
};

constexpr int kMaxChannels = 16;
constexpr float kMaxPixelValueAsFloat = 254;  // Channel::kMaxPixelValueAsFloat
constexpr float kMaxFragmentLength = 1000;    // Channel::kMaxFragmentLength

struct Options {
  int32_t width, height, ref_band, nch;
  int32_t ch[kMaxChannels];
  int32_t off_ag, off_tc, stride, ref_bq;
  float bq_cap, mq_cap;
  int32_t pos_color, neg_color;
  float a_supp, a_unsupp, a_other, a_match, a_mismatch;
  uint8_t anchor;
  int32_t min_mq, min_bq;
};

struct CallDesc {
  int64_t vstart;
  int32_t img_start, max_reads, n_sampled, read0;
};

struct ReadDesc {
  int32_t sort_pos, hap, align_pos, mq, rev, frag_len, support, rank;
  int32_t cig0, ncig, seq0, seq_len;
};

DV_HD inline int BaseColor(const Options& o, char base) {
  switch (base) {
    case 'A': return o.off_ag + o.stride * 3;
    case 'G': return o.off_ag + o.stride * 2;
    case 'T': return o.off_tc + o.stride * 1;
    case 'C': return o.off_tc + o.stride * 0;
    default: return 0;
  }
}

// BaseQualityChannel / MappingQualityChannel::ScaleColor.
DV_HD inline uint8_t ScaleColor(int value, float max_val) {
  if (static_cast<float>(value) > max_val) value = max_val;
  return static_cast<int>(kMaxPixelValueAsFloat *
                          (static_cast<float>(value) / max_val));
}

DV_HD inline int AlphaColor(float alpha) {
  return static_cast<int>(kMaxPixelValueAsFloat * alpha);
}

DV_HD inline int SupportsAltColor(const Options& o, int support) {
  float alpha = support == 0 ? o.a_unsupp : (support == 1 ? o.a_supp : o.a_other);
  return AlphaColor(alpha);
}

DV_HD inline int InsertSizeColor(int fragment_length) {
  int fl = fragment_length < 0 ? -fragment_length : fragment_length;
  if (static_cast<float>(fl) > kMaxFragmentLength) fl = static_cast<int>(kMaxFragmentLength);
  return static_cast<int>(kMaxPixelValueAsFloat *
                          (static_cast<float>(fl) / kMaxFragmentLength));
}

// One read's row, HWC within the row (row[col * nch + c]); row must be zeroed.
// Returns false exactly when Channels::CalculateChannels does (a low-quality
// base at the call position); the caller then drops the row. The CPU walks
// CIGAR elements in order and returns at the first failing element, so a
// sequential walk per read is the faithful form; parallelism is across reads.
DV_HD inline bool EncodeReadRow(const Options& o, const CallDesc& c,
                                const ReadDesc& r, const uint8_t* cig_op,
                                const int32_t* cig_len, const char* seq,
                                const uint8_t* qual, const char* ref,
                                uint8_t* row) {
  const int mq_color = ScaleColor(r.mq, o.mq_cap);
  const int strand_color = r.rev ? o.neg_color : o.pos_color;
  const int support_color = SupportsAltColor(o, r.support);
  const int insert_color = InsertSizeColor(r.frag_len);
  // The action applied per (ref_i, read_i, op) segment.
  auto action = [&](int ref_i, int read_i, uint8_t op) -> bool {
    char read_base = 0;
    if (op == kOpInsert) {
      read_base = static_cast<char>(o.anchor);
    } else if (op == kOpDelete) {
      ref_i -= 1;
      read_base = static_cast<char>(o.anchor);
    } else if (op == kOpMatch || op == kOpSeqMatch || op == kOpSeqMismatch) {
      read_base = seq[read_i];
    }
    // size_t in the CPU code: a negative column wraps and fails the bound.
    const unsigned long long col =
        static_cast<unsigned long long>(static_cast<long long>(ref_i) - c.img_start);
    if (read_base && col < static_cast<unsigned long long>(o.width)) {
      const uint8_t bq = qual[read_i];
      if (ref_i == c.vstart && bq < o.min_bq) return false;
      const char ref_base = ref[col];
      uint8_t* px = row + col * o.nch;
      for (int k = 0; k < o.nch; ++k) {
        int v = 0;
        switch (o.ch[k]) {
          case kChReadBase: v = BaseColor(o, read_base); break;
          case kChBaseQuality: v = ScaleColor(bq, o.bq_cap); break;
          case kChMappingQuality: v = mq_color; break;
          case kChStrand: v = strand_color; break;
          case kChReadSupportsVariant: v = support_color; break;
          case kChBaseDiffersFromRef:
            v = AlphaColor(read_base == ref_base ? o.a_match : o.a_mismatch);
            break;
          case kChInsertSize: v = insert_color; break;
        }
        px[k] = static_cast<uint8_t>(v);
      }
    }
    return true;
  };
  int ref_i = r.align_pos;
  int read_i = 0;
  for (int e = 0; e < r.ncig; ++e) {
    const uint8_t op = cig_op[r.cig0 + e];
    const int len = cig_len[r.cig0 + e];
    bool ok = true;
    switch (op) {
      case kOpMatch: case kOpSeqMatch: case kOpSeqMismatch:
        for (int i = 0; i < len; ++i) {
          ok = ok && action(ref_i, read_i, op);
          ref_i++; read_i++;
        }
        break;
      case kOpInsert: case kOpSoftClip:
        if (ref_i > 0) ok = action(ref_i - 1, read_i, op);
        read_i += len;
        break;
      case kOpDelete: case kOpSkip:
        if (read_i > 0) ok = action(ref_i, read_i - 1, op);
        ref_i += len;
        break;
      default:  // hard clip, pad: ignored. (Unknown ops are fatal on the CPU.)
        break;
    }
    if (!ok) return false;
  }
  return true;
}

// Reference band row (Channels::CalculateRefRows), HWC within the row.
DV_HD inline void EncodeRefRow(const Options& o, const char* ref, int col,
                               uint8_t* px) {
  for (int k = 0; k < o.nch; ++k) {
    int v = 0;
    switch (o.ch[k]) {
      case kChReadBase: v = BaseColor(o, ref[col]); break;
      case kChBaseQuality: v = ScaleColor(o.ref_bq, o.bq_cap); break;
      // MappingQualityChannel::FillRefBase scales by base_quality_cap.
      case kChMappingQuality: v = ScaleColor(o.ref_bq, o.bq_cap); break;
      case kChStrand: v = o.pos_color; break;
      case kChReadSupportsVariant: v = SupportsAltColor(o, 0); break;
      case kChBaseDiffersFromRef: v = AlphaColor(o.a_match); break;
      case kChInsertSize: v = static_cast<int>(kMaxPixelValueAsFloat); break;
    }
    px[k] = static_cast<uint8_t>(v);
  }
}

// ---- libstdc++ std::sort, ported (bits/stl_algo.h, bits/stl_heap.h) ----
// Sorts an array of read indices with SortByAlignment's order:
// (hap, position) then (fragment_name, read_number), the latter as a rank.
struct ByAlignment {
  const ReadDesc* reads;
  DV_HD bool operator()(int a, int b) const {
    const ReadDesc& x = reads[a];
    const ReadDesc& y = reads[b];
    if (x.hap == y.hap && x.sort_pos == y.sort_pos) return x.rank < y.rank;
    if (x.hap != y.hap) return x.hap < y.hap;
    return x.sort_pos < y.sort_pos;
  }
};

DV_HD inline void IterSwap(int* a, int* b) { int t = *a; *a = *b; *b = t; }

DV_HD inline void PushHeap(int* first, long hole, long top, int value, const ByAlignment& comp) {
  long parent = (hole - 1) / 2;
  while (hole > top && comp(first[parent], value)) {
    first[hole] = first[parent];
    hole = parent;
    parent = (hole - 1) / 2;
  }
  first[hole] = value;
}

DV_HD inline void AdjustHeap(int* first, long hole, long len, int value, const ByAlignment& comp) {
  const long top = hole;
  long second = hole;
  while (second < (len - 1) / 2) {
    second = 2 * (second + 1);
    if (comp(first[second], first[second - 1])) second--;
    first[hole] = first[second];
    hole = second;
  }
  if ((len & 1) == 0 && second == (len - 2) / 2) {
    second = 2 * (second + 1);
    first[hole] = first[second - 1];
    hole = second - 1;
  }
  PushHeap(first, hole, top, value, comp);
}

DV_HD inline void HeapSortAll(int* first, int* last, const ByAlignment& comp) {
  // std::__partial_sort(first, last, last): __heap_select (make_heap only,
  // as middle == last) then __sort_heap.
  const long len = last - first;
  if (len >= 2) {
    long parent = (len - 2) / 2;
    while (true) {
      AdjustHeap(first, parent, len, first[parent], comp);
      if (parent == 0) break;
      parent--;
    }
  }
  while (last - first > 1) {
    --last;
    int value = *last;
    *last = *first;
    AdjustHeap(first, 0, last - first, value, comp);
  }
}

DV_HD inline void MoveMedianToFirst(int* result, int* a, int* b, int* c, const ByAlignment& comp) {
  if (comp(*a, *b)) {
    if (comp(*b, *c)) IterSwap(result, b);
    else if (comp(*a, *c)) IterSwap(result, c);
    else IterSwap(result, a);
  } else if (comp(*a, *c)) {
    IterSwap(result, a);
  } else if (comp(*b, *c)) {
    IterSwap(result, c);
  } else {
    IterSwap(result, b);
  }
}

DV_HD inline int* UnguardedPartition(int* first, int* last, int* pivot, const ByAlignment& comp) {
  while (true) {
    while (comp(*first, *pivot)) ++first;
    --last;
    while (comp(*pivot, *last)) --last;
    if (!(first < last)) return first;
    IterSwap(first, last);
    ++first;
  }
}

// Iterative form of __introsort_loop (the recursion is on [cut, last), the
// loop continues on [first, cut)); an explicit stack keeps the same order of
// operations. Returns the number of heap-sort fallbacks taken.
DV_HD inline int IntrosortLoop(int* first, int* last, int depth_limit, const ByAlignment& comp) {
  struct Frame { int* first; int* last; int depth; };
  Frame stack[64];
  int sp = 0, heaps = 0;
  stack[sp++] = {first, last, depth_limit};
  while (sp > 0) {
    Frame f = stack[--sp];
    while (f.last - f.first > 16) {
      if (f.depth == 0) {
        HeapSortAll(f.first, f.last, comp);
        heaps++;
        break;
      }
      --f.depth;
      int* mid = f.first + (f.last - f.first) / 2;
      MoveMedianToFirst(f.first, f.first + 1, mid, f.last - 1, comp);
      int* cut = UnguardedPartition(f.first + 1, f.last, f.first, comp);
      // Recursive call on [cut, last) runs to completion before the loop
      // continues on [first, cut): push the continuation, then the callee.
      stack[sp++] = {f.first, cut, f.depth};
      stack[sp++] = {cut, f.last, f.depth};
      break;
    }
  }
  return heaps;
}

DV_HD inline void UnguardedLinearInsert(int* last, const ByAlignment& comp) {
  int val = *last;
  int* next = last - 1;
  while (comp(val, *next)) {
    *last = *next;
    last = next;
    --next;
  }
  *last = val;
}

DV_HD inline void InsertionSort(int* first, int* last, const ByAlignment& comp) {
  if (first == last) return;
  for (int* i = first + 1; i != last; ++i) {
    if (comp(*i, *first)) {
      int val = *i;
      for (int* p = i; p != first; --p) *p = *(p - 1);
      *first = val;
    } else {
      UnguardedLinearInsert(i, comp);
    }
  }
}

DV_HD inline int Lg(long n) {
  int k = 0;
  while (n > 1) { n >>= 1; ++k; }
  return k;
}

// std::sort(first, last, comp). Returns heap-sort fallbacks (normally 0).
DV_HD inline int StdSort(int* first, int* last, const ByAlignment& comp) {
  if (first == last) return 0;
  int heaps = IntrosortLoop(first, last, Lg(last - first) * 2, comp);
  if (last - first > 16) {
    InsertionSort(first, first + 16, comp);
    for (int* i = first + 16; i != last; ++i) UnguardedLinearInsert(i, comp);
  } else {
    InsertionSort(first, last, comp);
  }
  return heaps;
}

}  // namespace dvgpu

#endif  // DEEPVARIANT_GPU_PILEUP_ENCODE_H_
