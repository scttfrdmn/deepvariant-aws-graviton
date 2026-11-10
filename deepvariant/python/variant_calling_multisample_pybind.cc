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

#include "deepvariant/allelecounter.h"
#if true  // Trick to stop tooling from moving the #include around.
// MUST appear before any standard headers are included.
#include <pybind11/pybind11.h>
#endif

#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "deepvariant/protos/deepvariant.pb.h"
#include "deepvariant/variant_calling_multisample.h"
#include "third_party/nucleus/core/python/type_caster_nucleus_status.h"
#include "third_party/nucleus/core/python/type_caster_nucleus_statusor.h"
#include "third_party/nucleus/util/python/type_caster_nucleus_proto_ptr.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace py = pybind11;

PYBIND11_MODULE(variant_calling_multisample, m) {
  pybind11_protobuf::ImportNativeProtoCasters();
  using namespace ::learning::genomics::deepvariant;  // NOLINT

  py::classh<multi_sample::VariantCaller>(m, "VariantCaller")
      .def(py::init<const VariantCallerOptions&>(), py::arg("options"))
      .def("calls_from_allele_counts",
           [](multi_sample::VariantCaller& self,
              const std::unordered_map<std::string, AlleleCounter*>&
                  allele_counts_wrapper,
              const std::string& target_sample) {
             auto cpp_result = self.CallsFromAlleleCounts(allele_counts_wrapper,
                                                          target_sample);
             return cpp_result;
           })
      // calls_from_allele_counts plus, per candidate, the read sums the small
      // model's one-sample features are computed from (deepvariant-aws-graviton#10):
      //   [n, sum mapping_quality, sum average_base_quality, n reverse] for the
      //   ref reads, then the same 4 for each alternate_bases[i] (all -1 if
      //   allele_support_ext has no entry for it), then total depth (ref reads
      //   plus every allele_support_ext entry's reads), then
      //   allele_frequency_at_position.get(start + o, 0) for o in
      //   [-vaf_window / 2, vaf_window / 2].
      // Integer sums of int32/bool fields, so the same values Python's sum()
      // gives.
      .def("calls_from_allele_counts_with_small_model_sums",
           [](multi_sample::VariantCaller& self,
              const std::unordered_map<std::string, AlleleCounter*>&
                  allele_counts_wrapper,
              const std::string& target_sample, int vaf_window) {
             auto calls = self.CallsFromAlleleCounts(allele_counts_wrapper,
                                                     target_sample);
             std::vector<std::vector<int64_t>> sums;
             sums.reserve(calls.size());
             const int half = vaf_window > 0 ? vaf_window / 2 : 0;
             for (const DeepVariantCall& c : calls) {
               std::vector<int64_t> v;
               v.reserve(4 * (1 + c.variant().alternate_bases_size()) + 1 +
                         (vaf_window > 0 ? 2 * half + 1 : 0));
               auto add = [&v](const DeepVariantCall::SupportingReadsExt& e) {
                 int64_t mq = 0, bq = 0, rs = 0;
                 for (const auto& r : e.read_infos()) {
                   mq += r.mapping_quality();
                   bq += r.average_base_quality();
                   rs += r.is_reverse_strand() ? 1 : 0;
                 }
                 v.push_back(e.read_infos_size());
                 v.push_back(mq);
                 v.push_back(bq);
                 v.push_back(rs);
               };
               add(c.ref_support_ext());
               for (const std::string& alt : c.variant().alternate_bases()) {
                 auto it = c.allele_support_ext().find(alt);
                 if (it == c.allele_support_ext().end()) {
                   v.insert(v.end(), {-1, -1, -1, -1});
                 } else {
                   add(it->second);
                 }
               }
               int64_t total = c.ref_support_ext().read_infos_size();
               for (const auto& kv : c.allele_support_ext())
                 total += kv.second.read_infos_size();
               v.push_back(total);
               if (vaf_window > 0) {
                 const auto& afp = c.allele_frequency_at_position();
                 for (int o = -half; o <= half; ++o) {
                   auto it = afp.find(static_cast<int32_t>(c.variant().start() + o));
                   v.push_back(it == afp.end() ? 0 : it->second);
                 }
               }
               sums.push_back(std::move(v));
             }
             return py::make_tuple(py::cast(std::move(calls)),
                                   py::cast(std::move(sums)));
           })
      .def("call_positions_from_allele_counts",
           [](multi_sample::VariantCaller& self,
              const std::unordered_map<std::string, AlleleCounter*>&
                  allele_counters,
              const std::string& target_sample) {
             auto cpp_result = self.CallPositionsFromAlleleCounts(
                 allele_counters, target_sample);
             return cpp_result;
           });
}
