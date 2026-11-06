/*
 * Copyright 2024 Google LLC.
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

#include "third_party/nucleus/protos/reads.pb.h"
#if true  // Trick to stop tooling from moving the #include around.
// MUST appear before any standard headers are included.
#include <pybind11/pybind11.h>
#endif

#include <pybind11/stl.h>

#include <utility>
#include <vector>

#include "deepvariant/realigner/fast_pass_aligner.h"
#include "deepvariant/realigner/region_realigner.h"
#include "deepvariant/read_batch.h"
#include "third_party/nucleus/io/reference.h"
#include "third_party/nucleus/core/python/type_caster_nucleus_status.h"
#include "third_party/nucleus/core/python/type_caster_nucleus_statusor.h"
#include "third_party/nucleus/util/python/type_caster_nucleus_proto_ptr.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace py = pybind11;
PYBIND11_MODULE(fast_pass_aligner, m) {
  pybind11_protobuf::ImportNativeProtoCasters();
  using namespace ::learning::genomics::deepvariant;  // NOLINT

  py::classh<FastPassAligner>(m, "FastPassAligner")
      .def(py::init<>())
      .def("set_reference", &FastPassAligner::set_reference, py::arg("ref"))
      .def("set_ref_start", &FastPassAligner::set_ref_start, py::arg("chr"),
           py::arg("pos"))
      .def("set_haplotypes", &FastPassAligner::set_haplotypes,
           py::arg("hyplotypes"))
      .def("set_options", &FastPassAligner::set_options, py::arg("options"))
      .def("set_is_debug", &FastPassAligner::set_is_debug, py::arg("isDebug"))
      .def("set_normalize_reads", &FastPassAligner::set_normalize_reads,
           py::arg("normalizeReads"))
      .def("set_debug_read_id", &FastPassAligner::set_debug_read_id,
           py::arg("readId"))
      .def("set_ref_prefix_len", &FastPassAligner::set_ref_prefix_len,
           py::arg("ref_prefix_len"))
      .def("set_ref_suffix_len", &FastPassAligner::set_ref_suffix_len,
           py::arg("set_ref_suffix_len"))
      // Copy-free variant (deepvariant-aws-graviton#10): the reads are
      // passed as pointers to the Python protos, and only reads whose
      // alignment is replaced come back, with their input indices.
      .def("realign_reads_views",
           [](FastPassAligner* self,
              const std::vector<
                  nucleus::ConstProtoPtr<const nucleus::genomics::v1::Read>>&
                  reads) {
             std::vector<const nucleus::genomics::v1::Read*> ptrs;
             ptrs.reserve(reads.size());
             for (const auto& r : reads) ptrs.push_back(r.p_);
             std::vector<int> idx;
             std::vector<nucleus::genomics::v1::Read> changed;
             self->AlignReadsViews(ptrs, &idx, &changed);
             return std::make_pair(std::move(idx), std::move(changed));
           })
      .def("realign_reads",
           [](FastPassAligner* self,
              const std::vector<nucleus::genomics::v1::Read>& reads_param) {
             auto cpp_result = self->AlignReads(reads_param);
             return std::move(*cpp_result);
             // TODO): possible that `return *cpp_result` could work.
             // bug doing a move to be safe.
             // https://sigcpp.github.io/2020/06/08/return-value-optimization
           });

  // Realigner.realign_reads for one region in C++ (deepvariant-aws-graviton#10):
  // returns (order, replaced_pos, replaced, last_read_size), or None when the
  // configuration isn't covered and Python should run.
  m.def(
      "realign_region",
      [](const std::vector<
             nucleus::ConstProtoPtr<const nucleus::genomics::v1::Read>>& reads,
         const nucleus::genomics::v1::Range& region,
         const nucleus::GenomeReference* ref, const RealignerOptions& config,
         bool ws_light) -> py::object {
        std::vector<const nucleus::genomics::v1::Read*> ptrs;
        ptrs.reserve(reads.size());
        for (const auto& r : reads) ptrs.push_back(r.p_);
        RegionRealignment out;
        if (!RealignRegion(ptrs, region, *ref, config, ws_light, &out))
          return py::none();
        return py::make_tuple(std::move(out.order), std::move(out.replaced_pos),
                              std::move(out.replaced), out.last_read_size);
      },
      py::arg("reads"), py::arg("region"), py::arg("ref"), py::arg("config"),
      py::arg("ws_light"));

  // make_examples' realign_reads on a ReadBatch, in place: reads longer than
  // max_read_length_to_realign (when > 0) first, in input order, then the
  // realigned short reads in realign_reads' output order. Returns
  // last_read_size (or -1), or None when the configuration isn't covered and
  // Python should run (batch unchanged).
  m.def(
      "realign_batch",
      [](ReadBatch* batch, const nucleus::genomics::v1::Range& region,
         const nucleus::GenomeReference* ref, const RealignerOptions& config,
         bool ws_light, int max_read_length) -> py::object {
        const auto& reads = batch->reads();
        std::vector<int> long_idx, short_idx;
        for (int i = 0; i < static_cast<int>(reads.size()); ++i) {
          if (max_read_length > 0 &&
              static_cast<int>(reads[i]->aligned_sequence().size()) >
                  max_read_length)
            long_idx.push_back(i);
          else
            short_idx.push_back(i);
        }
        RegionRealignment out;
        if (!short_idx.empty()) {
          std::vector<const nucleus::genomics::v1::Read*> ptrs;
          ptrs.reserve(short_idx.size());
          for (int i : short_idx) ptrs.push_back(reads[i]);
          if (!RealignRegion(ptrs, region, *ref, config, ws_light, &out))
            return py::none();
        }
        std::vector<int> order(long_idx);
        for (int j : out.order) order.push_back(short_idx[j]);
        std::vector<int> replaced_pos;
        for (int p : out.replaced_pos)
          replaced_pos.push_back(p + static_cast<int>(long_idx.size()));
        batch->Reorder(order, replaced_pos, &out.replaced);
        return py::int_(out.last_read_size);
      },
      py::arg("batch"), py::arg("region"), py::arg("ref"), py::arg("config"),
      py::arg("ws_light"), py::arg("max_read_length"));
}
