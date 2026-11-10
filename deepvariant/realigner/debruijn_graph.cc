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

#include "deepvariant/realigner/debruijn_graph.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "deepvariant/protos/realigner.pb.h"
#include <utility>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/depth_first_search.hpp"
#include "boost/graph/graphviz.hpp"
#include "boost/graph/reverse_graph.hpp"
#include "third_party/nucleus/protos/reads.pb.h"
#include "third_party/nucleus/util/proto_ptr.h"
#include "third_party/nucleus/util/utils.h"

namespace learning {
namespace genomics {
namespace deepvariant {

using Vertex = DeBruijnGraph::Vertex;
using VertexIndexMap = DeBruijnGraph::VertexIndexMap;
using Edge = DeBruijnGraph::Edge;
using Path = DeBruijnGraph::Path;

using Read = nucleus::genomics::v1::Read;

using absl::string_view;

namespace {

// Visitor classes we will use to run boost algorithms.  N.B.: these classes
// operate by side effect, modifying pointers that are passed in.  This is not
// an optimal design, rather it is to work around the copying done by
// boost::visitor which makes it difficult for our code to retain a pointer to
// the visitor object that is actually used.

class CycleDetector : public boost::dfs_visitor<> {
 public:
  explicit CycleDetector(bool* has_cycle) : has_cycle(has_cycle) {}

  template <class Edge, class Graph>
  void back_edge(Edge, const Graph&) {
    *has_cycle = true;
  }

 private:
  bool* has_cycle;
};

template <class BoostGraph>
class EdgeLabelWriter {
 public:
  explicit EdgeLabelWriter(const BoostGraph& g) : g_(g) {}

  void operator()(std::ostream& out, const Edge e) const {
    EdgeInfo ei = g_[e];
    out << "[label=" << std::to_string(ei.weight)
        << (ei.is_ref ? " color=red" : "") << "]";
  }

 private:
  const BoostGraph& g_;
};

class ReachableVertexVisitor : public boost::dfs_visitor<> {
 public:
  explicit ReachableVertexVisitor(std::set<Vertex>* reachable_vertices)
      : reachable_vertices(reachable_vertices) {}

  template <class Edge, class Graph>
  void tree_edge(Edge e, const Graph& g) {
    Vertex from = boost::source(e, g);
    if (reachable_vertices->find(from) != reachable_vertices->end()) {
      Vertex to = boost::target(e, g);
      reachable_vertices->insert(to);
    }
  }

 private:
  std::set<Vertex>* reachable_vertices;
};

template <class BoostGraphT, class VertexIndexMapT>
std::set<Vertex> VerticesReachableFrom(
    Vertex v, const BoostGraphT& g, const VertexIndexMapT& vertex_index_map) {
  std::set<Vertex> reachable_vertices{v};
  ReachableVertexVisitor vis(&reachable_vertices);
  boost::depth_first_search(
      g, boost::visitor(vis).root_vertex(v).vertex_index_map(vertex_index_map));
  return reachable_vertices;
}

}  // namespace

Vertex DeBruijnGraph::EnsureVertex(string_view kmer) {
  Vertex v;
  auto vertex_find = kmer_to_vertex_.find(kmer);
  if (vertex_find != kmer_to_vertex_.end()) {
    v = (*vertex_find).second;
  } else {
    string kmer_copy(kmer);
    v = boost::add_vertex(VertexInfo{kmer_copy}, g_);
    // N.B.: must use the long-lived string in the map key as the referent of
    // the string_view key.
    kmer_to_vertex_[g_[v].kmer] = v;
  }

  return v;
}

Vertex DeBruijnGraph::VertexForKmer(string_view kmer) const {
  return kmer_to_vertex_.at(kmer);
}

// The same indices as before (vertices() order), stored in the vertex itself
// instead of a std::map<Vertex, int>, so index lookups in DFS are O(1).
void DeBruijnGraph::RebuildIndexMap() {
  VertexIterator vi, vend;
  std::tie(vi, vend) = boost::vertices(g_);
  int index = 0;
  for (; vi != vend; ++vi) {
    g_[*vi].index = index;
    ++index;
  }
}

VertexIndexMap DeBruijnGraph::IndexMap() const {
  return boost::get(&VertexInfo::index, g_);
}

// Whether the directed graph has a cycle: iterative three-color DFS over all
// vertices. Cycle existence doesn't depend on traversal order, so this is the
// same answer as boost::depth_first_search with a back-edge visitor.
bool DeBruijnGraph::HasCycle() const {
  const int n = boost::num_vertices(g_);
  std::vector<uint8_t> color(n, 0);  // 0 white, 1 on stack, 2 done
  std::vector<std::pair<Vertex, AdjacencyIterator>> stack;
  VertexIterator vi, vend;
  for (std::tie(vi, vend) = boost::vertices(g_); vi != vend; ++vi) {
    if (color[g_[*vi].index] != 0) continue;
    color[g_[*vi].index] = 1;
    stack.emplace_back(*vi, boost::adjacent_vertices(*vi, g_).first);
    while (!stack.empty()) {
      auto& top = stack.back();
      if (top.second == boost::adjacent_vertices(top.first, g_).second) {
        color[g_[top.first].index] = 2;
        stack.pop_back();
        continue;
      }
      const Vertex w = *top.second;
      ++top.second;
      const uint8_t c = color[g_[w].index];
      if (c == 1) return true;  // back edge
      if (c == 0) {
        color[g_[w].index] = 1;
        stack.emplace_back(w, boost::adjacent_vertices(w, g_).first);
      }
    }
  }
  return false;
}

DeBruijnGraph::DeBruijnGraph(
    absl::string_view ref,
    absl::Span<const nucleus::ConstProtoPtr<const Read>> reads,
    const Options& options, int k)
    : options_(options), k_(k) {
  CHECK_GT(k, 0);  // k should always be a positive integer.
  CHECK(static_cast<uint32_t>(k) < ref.size());
  AddEdgesForReference(ref);
  source_ = VertexForKmer(ref.substr(0, k_));
  sink_ = VertexForKmer(ref.substr(ref.size() - k_, k_));
  for (const nucleus::ConstProtoPtr<const Read>& read_ptr : reads) {
    const Read& read = *read_ptr.p_;
    if (read.alignment().mapping_quality() >= options.min_mapq()) {
      AddEdgesForRead(read);
    }
  }
  RebuildIndexMap();
}

// Indicates that we couldn't find a minimum k that can be used.
constexpr int kBoundsNoWorkingK = -1;
struct KBounds {
  int min_k;  // Minimum k to consider (inclusive).
  int max_k;  // Maximum k to consider (inclusive).
};


KBounds KMinMaxFromReference(const string_view ref,
                             const DeBruijnGraph::Options& options) {
  KBounds bounds;
  bounds.min_k = kBoundsNoWorkingK;
  bounds.max_k  = std::min(options.max_k(), static_cast<int>(ref.size()) - 1);

  for (int k = options.min_k(); k <= bounds.max_k; k += options.step_k()) {
    bool has_cycle = false;
    absl::btree_set<string_view> kmers;

    for (int i = 0; i < ref.size() - k + 1; i++) {
      string_view kmer = ref.substr(i, k);
      if (kmers.insert(kmer).second == false) {
        // No insertion took place because the kmer already exists. This implies
        // that there's a cycle in the graph.
        has_cycle = true;
        break;
      }
    }

    if (!has_cycle) {
      bounds.min_k = k;
      break;
    }
  }

  return bounds;
}

namespace {

// Flat De Bruijn graph for one k (deepvariant-aws-graviton#10). Same graph as
// the boost version: vertices are distinct k-mers, edges are distinct
// (from, to) pairs with weight = multiplicity and is_ref; k-mers are views
// into the reference and into one buffer of upper-cased read bases (built once
// per Build, not per k). Buffers keep their capacity across k and windows.
//
// Order. The boost graph keeps out-edges in a std::set ordered by the target
// vertex descriptor, which for listS vertices is a heap address. Every output
// is independent of that order (HasCycle, reachability, the sorted haplotype
// list) except CandidatePaths' max_num_paths cut-off, which checks the running
// path count before each BFS pop; the last count checked depends on which
// path the BFS pops last. Here out-edges are in insertion order.
// DV_DBG_SELFCHECK=1 compares both engines window by window.
// 2-bit codes for upper-case ACGT; every other character (N, IUPAC, lower
// case) is -1, and k-mers containing one are keyed by their string, so the
// vertex identity is the k-mer string exactly as in the boost version.
struct BaseCodes {
  int8_t c[256];
  BaseCodes() {
    for (int i = 0; i < 256; ++i) c[i] = -1;
    c['A'] = 0; c['C'] = 1; c['G'] = 2; c['T'] = 3;
  }
};
const BaseCodes kCodes;

struct FlatGraph {
  struct E { int to, next, weight; bool is_ref, alive; };
  int k = 0;
  uint64_t mask = 0;
  std::vector<absl::string_view> kmer;
  std::vector<int> head, tail, out_deg;
  std::vector<E> edges;
  absl::flat_hash_map<absl::string_view, int> index;      // k > 32 or non-ACGT
  absl::flat_hash_map<uint64_t, int> index64;             // k <= 32, ACGT
  // 32 < k <= 64, ACGT (#10): two 64-bit words of the 2-bit code, so these
  // k-mers are hashed as integers too; the vertex identity is unchanged.
  using Key128 = std::pair<uint64_t, uint64_t>;
  absl::flat_hash_map<Key128, int> index128;
  unsigned __int128 mask128 = 0;
  static Key128 Split(unsigned __int128 v) {
    return {static_cast<uint64_t>(v >> 64), static_cast<uint64_t>(v)};
  }
  std::vector<uint8_t> color;
  std::vector<std::pair<int, int>> stack;
  // Closed-walk check (deepvariant-aws-graviton#10): seen[v] is the id of the
  // last read segment that visited v. Consecutive k-mers of one segment are
  // joined by edges, so a segment that revisits a vertex has walked a closed
  // path, and a graph with a closed walk has a cycle: HasCycle() is then true
  // for this k whatever the remaining reads add (edges are only ever added
  // before HasCycle). DV_DBG_NOEARLY=1 builds every k to the end.
  std::vector<int> seen;
  int segment = 0;

  void Reset(int k_in, size_t hint) {
    k = k_in;
    mask = k >= 32 ? ~uint64_t{0} : ((uint64_t{1} << (2 * k)) - 1);
    kmer.clear(); head.clear(); tail.clear(); out_deg.clear(); edges.clear();
    seen.clear(); segment = 0;
    index.clear();
    index64.clear();
    index128.clear();
    mask128 = k >= 64 ? ~static_cast<unsigned __int128>(0)
                      : ((static_cast<unsigned __int128>(1) << (2 * k)) - 1);
    if (k <= 32) index64.reserve(hint);
    else if (k <= 64) index128.reserve(hint);
    else index.reserve(hint);
  }
  int NewVertex(absl::string_view km) {
    const int v = static_cast<int>(kmer.size());
    kmer.push_back(km);
    head.push_back(-1); tail.push_back(-1); out_deg.push_back(0);
    seen.push_back(0);
    return v;
  }
  int EnsureStr(absl::string_view km) {
    auto it = index.find(km);
    if (it != index.end()) return it->second;
    const int v = NewVertex(km);
    index.emplace(km, v);
    return v;
  }
  int Ensure64(uint64_t key, absl::string_view km) {
    auto [it, inserted] = index64.try_emplace(key, 0);
    if (inserted) it->second = NewVertex(km);
    return it->second;
  }
  int Ensure128(unsigned __int128 key, absl::string_view km) {
    auto [it, inserted] = index128.try_emplace(Split(key), 0);
    if (inserted) it->second = NewVertex(km);
    return it->second;
  }
  // Vertex for one k-mer (source/sink lookup); -1 if absent.
  int Find(absl::string_view km) const {
    if (k <= 32) {
      uint64_t key = 0;
      bool acgt = true;
      for (char ch : km) {
        const int8_t c = kCodes.c[static_cast<uint8_t>(ch)];
        if (c < 0) { acgt = false; break; }
        key = (key << 2) | c;
      }
      if (acgt) { auto it = index64.find(key); return it == index64.end() ? -1 : it->second; }
    } else if (k <= 64) {
      unsigned __int128 key = 0;
      bool acgt = true;
      for (char ch : km) {
        const int8_t c = kCodes.c[static_cast<uint8_t>(ch)];
        if (c < 0) { acgt = false; break; }
        key = (key << 2) | static_cast<unsigned __int128>(c);
      }
      if (acgt) { auto it = index128.find(Split(key)); return it == index128.end() ? -1 : it->second; }
    }
    auto it = index.find(km);
    return it == index.end() ? -1 : it->second;
  }
  void AddEdge(int from, int to, bool is_ref) {
    for (int e = head[from]; e != -1; e = edges[e].next) {
      if (edges[e].to == to) {
        edges[e].weight++;
        edges[e].is_ref |= is_ref;
        return;
      }
    }
    const int id = static_cast<int>(edges.size());
    edges.push_back(E{to, -1, 1, is_ref, true});
    if (tail[from] == -1) head[from] = id; else edges[tail[from]].next = id;
    tail[from] = id;
    out_deg[from]++;
  }
  // Returns true iff check_walk and this segment revisited one of its own
  // vertices (a closed walk, so the graph has a cycle at this k).
  bool AddKmersAndEdges(absl::string_view bases, int start, int end, bool is_ref,
                        bool check_walk = false) {
    CHECK_GE(start, 0);
    CHECK_LE(start + k, bases.size());
    CHECK_LE(end + k, bases.size());
    if (end <= 0) return false;
    const int seg = check_walk ? ++segment : 0;
    auto visit = [&](int v) {
      if (!check_walk) return false;
      if (seen[v] == seg) return true;
      seen[v] = seg;
      return false;
    };
    if (k > 32 && k <= 64) {
      unsigned __int128 key = 0;
      int bad = start - 1;
      for (int j = start; j < start + k; ++j) {
        const int8_t c = kCodes.c[static_cast<uint8_t>(bases[j])];
        if (c < 0) { bad = j; key = key << 2; } else { key = (key << 2) | static_cast<unsigned __int128>(c); }
      }
      key &= mask128;
      auto vertex_at = [&](int i) {
        return bad < i ? Ensure128(key, bases.substr(i, k)) : EnsureStr(bases.substr(i, k));
      };
      int prev = vertex_at(start);
      bool closed = visit(prev);
      for (int i = start + 1; i <= end; ++i) {
        const int j = i + k - 1;
        const int8_t c = kCodes.c[static_cast<uint8_t>(bases[j])];
        if (c < 0) { bad = j; key = (key << 2) & mask128; } else { key = ((key << 2) | static_cast<unsigned __int128>(c)) & mask128; }
        const int cur = vertex_at(i);
        AddEdge(prev, cur, is_ref);
        closed |= visit(cur);
        prev = cur;
      }
      return closed;
    }
    if (k > 32) {
      int prev = EnsureStr(bases.substr(start, k));
      bool closed = visit(prev);
      for (int i = start + 1; i <= end; ++i) {
        const int cur = EnsureStr(bases.substr(i, k));
        AddEdge(prev, cur, is_ref);
        closed |= visit(cur);
        prev = cur;
      }
      return closed;
    }
    // Rolling 2-bit key over k-mers bases[i, i + k); `bad` is the last index
    // with a non-ACGT character, so k-mer i is all-ACGT iff bad < i.
    uint64_t key = 0;
    int bad = start - 1;
    for (int j = start; j < start + k; ++j) {
      const int8_t c = kCodes.c[static_cast<uint8_t>(bases[j])];
      if (c < 0) { bad = j; key = key << 2; } else { key = (key << 2) | c; }
    }
    key &= mask;
    auto vertex_at = [&](int i) {
      return bad < i ? Ensure64(key, bases.substr(i, k)) : EnsureStr(bases.substr(i, k));
    };
    int prev = vertex_at(start);
    bool closed = visit(prev);
    for (int i = start + 1; i <= end; ++i) {
      const int j = i + k - 1;
      const int8_t c = kCodes.c[static_cast<uint8_t>(bases[j])];
      if (c < 0) { bad = j; key = (key << 2) & mask; } else { key = ((key << 2) | c) & mask; }
      const int cur = vertex_at(i);
      AddEdge(prev, cur, is_ref);
      closed |= visit(cur);
      prev = cur;
    }
    return closed;
  }
  bool HasCycle() {
    const int n = static_cast<int>(kmer.size());
    color.assign(n, 0);
    stack.clear();
    for (int s = 0; s < n; ++s) {
      if (color[s]) continue;
      color[s] = 1;
      stack.emplace_back(s, head[s]);
      while (!stack.empty()) {
        auto& top = stack.back();
        if (top.second == -1) { color[top.first] = 2; stack.pop_back(); continue; }
        const int w = edges[top.second].to;
        top.second = edges[top.second].next;
        if (color[w] == 1) return true;
        if (color[w] == 0) { color[w] = 1; stack.emplace_back(w, head[w]); }
      }
    }
    return false;
  }
};

struct FlatScratch {
  FlatGraph g;
  std::string read_bases;                        // upper-cased, all reads
  std::vector<std::pair<int, int>> read_spans;   // offset, length
  std::vector<const Read*> read_ptrs;
  absl::flat_hash_set<absl::string_view> ref_kmers;
  absl::flat_hash_set<uint64_t> ref_kmers64;
  absl::flat_hash_set<std::pair<uint64_t, uint64_t>> ref_kmers128;
  std::vector<int> in_head, in_next, in_from, rstack;
  std::vector<uint8_t> fwd, rev;
  std::vector<int> pnode_v, pnode_parent, queue, terminated;
  std::vector<int> bad_pos, bad_off;             // per-read bad positions (FlatBuild)
};

FlatScratch& Scratch() {
  static thread_local FlatScratch s;
  return s;
}

// The edges of the pruned graph, iterated as CandidatePaths does.
bool FlatCandidateHaplotypes(FlatScratch& sc, int source, int sink,
                             const DeBruijnGraph::Options& options,
                             std::vector<std::string>* out) {
  FlatGraph& g = sc.g;
  const int n = static_cast<int>(g.kmer.size());
  // Prune: drop non-reference edges below min_edge_weight.
  for (auto& e : g.edges)
    if (!e.is_ref && e.weight < options.min_edge_weight()) e.alive = false;
  // Forward reachability from source, reverse reachability from sink.
  sc.fwd.assign(n, 0); sc.rev.assign(n, 0);
  sc.rstack.clear();
  sc.fwd[source] = 1; sc.rstack.push_back(source);
  while (!sc.rstack.empty()) {
    const int v = sc.rstack.back(); sc.rstack.pop_back();
    for (int e = g.head[v]; e != -1; e = g.edges[e].next)
      if (g.edges[e].alive && !sc.fwd[g.edges[e].to]) { sc.fwd[g.edges[e].to] = 1; sc.rstack.push_back(g.edges[e].to); }
  }
  sc.in_head.assign(n, -1); sc.in_next.assign(g.edges.size(), -1); sc.in_from.assign(g.edges.size(), -1);
  for (int v = 0; v < n; ++v)
    for (int e = g.head[v]; e != -1; e = g.edges[e].next)
      if (g.edges[e].alive) { const int t = g.edges[e].to; sc.in_from[e] = v; sc.in_next[e] = sc.in_head[t]; sc.in_head[t] = e; }
  sc.rev[sink] = 1; sc.rstack.push_back(sink);
  while (!sc.rstack.empty()) {
    const int v = sc.rstack.back(); sc.rstack.pop_back();
    for (int e = sc.in_head[v]; e != -1; e = sc.in_next[e])
      if (!sc.rev[sc.in_from[e]]) { sc.rev[sc.in_from[e]] = 1; sc.rstack.push_back(sc.in_from[e]); }
  }
  // Remove vertices not on a source-to-sink path, with all their edges.
  for (int v = 0; v < n; ++v) {
    const bool keep = sc.fwd[v] && sc.rev[v];
    for (int e = g.head[v]; e != -1; e = g.edges[e].next)
      if (!keep || !(sc.fwd[g.edges[e].to] && sc.rev[g.edges[e].to])) g.edges[e].alive = false;
  }
  for (int v = 0; v < n; ++v) {
    int d = 0;
    for (int e = g.head[v]; e != -1; e = g.edges[e].next) d += g.edges[e].alive;
    g.out_deg[v] = d;
  }
  // CandidatePaths: BFS over paths, as a tree of path nodes (no path copies).
  CHECK_GT(g.out_deg[source], 0);
  sc.pnode_v.clear(); sc.pnode_parent.clear(); sc.queue.clear(); sc.terminated.clear();
  sc.pnode_v.push_back(source); sc.pnode_parent.push_back(-1);
  sc.queue.push_back(0);
  size_t qhead = 0;
  while (qhead < sc.queue.size()) {
    const int n_total = static_cast<int>(sc.terminated.size() + (sc.queue.size() - qhead));
    if (n_total > options.max_num_paths()) { out->clear(); return true; }
    const int node = sc.queue[qhead++];
    const int last = sc.pnode_v[node];
    for (int e = g.head[last]; e != -1; e = g.edges[e].next) {
      if (!g.edges[e].alive) continue;
      const int w = g.edges[e].to;
      const int child = static_cast<int>(sc.pnode_v.size());
      sc.pnode_v.push_back(w); sc.pnode_parent.push_back(node);
      if (w == sink || g.out_deg[w] == 0) sc.terminated.push_back(child);
      else sc.queue.push_back(child);
    }
  }
  out->clear();
  out->reserve(sc.terminated.size());
  std::vector<int> chain;
  for (int t : sc.terminated) {
    chain.clear();
    for (int x = t; x != -1; x = sc.pnode_parent[x]) chain.push_back(sc.pnode_v[x]);
    std::string h;
    h.reserve(chain.size() + g.k);
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) h.push_back(g.kmer[*it][0]);
    if (!chain.empty()) h.append(g.kmer[chain.front()].substr(1, g.k - 1));
    out->push_back(std::move(h));
  }
  std::sort(out->begin(), out->end());
  return true;
}

bool DbgEnv(const char* name);

// Build() with the flat engine. Returns false if no k works (Build -> nullptr);
// otherwise *k_used and *haplotypes.
bool FlatBuild(absl::string_view ref,
               const std::vector<nucleus::ConstProtoPtr<const Read>>& reads,
               const DeBruijnGraph::Options& options, int* k_used,
               std::vector<std::string>* haplotypes) {
  FlatScratch& sc = Scratch();
  // KMinMaxFromReference: the smallest k whose reference k-mers are distinct.
  int min_k = -1;
  const int max_k = std::min(options.max_k(), static_cast<int>(ref.size()) - 1);
  for (int k = options.min_k(); k <= max_k; k += options.step_k()) {
    bool has_cycle = false;
    sc.ref_kmers.clear();
    sc.ref_kmers64.clear();
    const int n_kmers = static_cast<int>(ref.size()) - k + 1;
    if (k <= 32) {
      sc.ref_kmers64.reserve(n_kmers);
      const uint64_t mask = k >= 32 ? ~uint64_t{0} : ((uint64_t{1} << (2 * k)) - 1);
      uint64_t key = 0;
      int bad = -1;
      for (int j = 0; j < static_cast<int>(ref.size()) && !has_cycle; ++j) {
        const int8_t c = kCodes.c[static_cast<uint8_t>(ref[j])];
        if (c < 0) { bad = j; key = (key << 2) & mask; } else { key = ((key << 2) | c) & mask; }
        const int i = j - k + 1;
        if (i < 0) continue;
        if (bad < i) { if (!sc.ref_kmers64.insert(key).second) has_cycle = true; }
        else if (!sc.ref_kmers.insert(ref.substr(i, k)).second) has_cycle = true;
      }
    } else if (k <= 64) {
      sc.ref_kmers128.clear();
      sc.ref_kmers128.reserve(n_kmers);
      const unsigned __int128 mask = k >= 64 ? ~static_cast<unsigned __int128>(0)
                                             : ((static_cast<unsigned __int128>(1) << (2 * k)) - 1);
      unsigned __int128 key = 0;
      int bad = -1;
      for (int j = 0; j < static_cast<int>(ref.size()) && !has_cycle; ++j) {
        const int8_t c = kCodes.c[static_cast<uint8_t>(ref[j])];
        if (c < 0) { bad = j; key = (key << 2) & mask; } else { key = ((key << 2) | static_cast<unsigned __int128>(c)) & mask; }
        const int i = j - k + 1;
        if (i < 0) continue;
        if (bad < i) { if (!sc.ref_kmers128.insert(FlatGraph::Split(key)).second) has_cycle = true; }
        else if (!sc.ref_kmers.insert(ref.substr(i, k)).second) has_cycle = true;
      }
    } else {
      sc.ref_kmers.reserve(n_kmers);
      for (int i = 0; i < n_kmers; i++) {
        if (!sc.ref_kmers.insert(ref.substr(i, k)).second) { has_cycle = true; break; }
      }
    }
    if (!has_cycle) { min_k = k; break; }
  }
  if (min_k == -1) return false;
  // Upper-cased read bases once (the boost version does it per k).
  size_t total = 0;
  sc.read_ptrs.clear();
  for (const auto& rp : reads) {
    const Read& r = *rp.p_;
    if (r.alignment().mapping_quality() >= options.min_mapq()) {
      sc.read_ptrs.push_back(&r);
      total += r.aligned_sequence().size();
    }
  }
  sc.read_bases.clear();
  sc.read_bases.reserve(total);
  sc.read_spans.clear();
  for (const Read* r : sc.read_ptrs) {
    const int off = static_cast<int>(sc.read_bases.size());
    for (char c : r->aligned_sequence()) sc.read_bases.push_back(absl::ascii_toupper(c));
    sc.read_spans.emplace_back(off, static_cast<int>(r->aligned_sequence().size()));
  }
  const absl::string_view all_bases(sc.read_bases);
  // Bad positions (non-ACGT base or base quality below min_base_quality) of
  // every read, found once here instead of once per k (#10); read r's are
  // bad_pos[bad_off[r] .. bad_off[r + 1]), ascending.
  sc.bad_pos.clear();
  sc.bad_off.assign(1, 0);
  for (size_t r = 0; r < sc.read_ptrs.size(); ++r) {
    const absl::string_view bases = all_bases.substr(sc.read_spans[r].first, sc.read_spans[r].second);
    const auto& qual = sc.read_ptrs[r]->aligned_quality();
    for (int i = 0; i < static_cast<int>(bases.size()); ++i)
      if (!IsCanonicalBase(bases[i], nucleus::CanonicalBases::ACGT) ||
          qual[i] < options.min_base_quality())
        sc.bad_pos.push_back(i);
    sc.bad_off.push_back(static_cast<int>(sc.bad_pos.size()));
  }
  static const bool early = !DbgEnv("DV_DBG_NOEARLY");
  for (int k = min_k; k <= max_k; k += options.step_k()) {
    FlatGraph& g = sc.g;
    CHECK_GT(k, 0);
    CHECK(static_cast<uint32_t>(k) < ref.size());
    g.Reset(k, ref.size() + total / 8);
    g.AddKmersAndEdges(ref, 0, ref.size() - k, true);
    const int source = g.Find(ref.substr(0, k));
    const int sink = g.Find(ref.substr(ref.size() - k, k));
    CHECK(source >= 0 && sink >= 0);
    bool cycle = false;
    for (size_t r = 0; r < sc.read_ptrs.size(); ++r) {
      const absl::string_view bases = all_bases.substr(sc.read_spans[r].first, sc.read_spans[r].second);
      const int size = static_cast<int>(bases.size());
      // next_bad(i): the first bad position >= i, or size. Successive calls
      // ask for ascending i, so a cursor over the read's bad positions serves.
      const int* bp = sc.bad_pos.data() + sc.bad_off[r];
      const int* bp_end = sc.bad_pos.data() + sc.bad_off[r + 1];
      auto next_bad = [&](int start) -> int {
        while (bp != bp_end && *bp < start) ++bp;
        return bp != bp_end ? *bp : size;
      };
      const int stop = size - k;
      int i = 0;
      while (i < stop) {
        const int nb = next_bad(i);
        cycle |= g.AddKmersAndEdges(bases, i, nb - k, false, early);
        i = nb + 1;
      }
      if (cycle) break;  // HasCycle() would be true: next k
    }
    if (cycle || g.HasCycle()) continue;
    *k_used = k;
    return FlatCandidateHaplotypes(sc, source, sink, options, haplotypes);
  }
  return false;
}

bool DbgEnv(const char* name) {
  const char* v = getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

std::atomic<long> g_dbg_checked{0}, g_dbg_mismatch{0};
void DbgReport() {
  if (g_dbg_checked.load())
    fprintf(stderr, "DV_DBG_SELFCHECK windows %ld mismatched %ld\n",
            g_dbg_checked.load(), g_dbg_mismatch.load());
}

}  // namespace

std::unique_ptr<DeBruijnGraph> DeBruijnGraph::Build(
    absl::string_view ref,
    const std::vector<nucleus::ConstProtoPtr<const Read>>& reads,
    const DeBruijnGraph::Options& options) {
  static const bool use_boost = DbgEnv("DV_DBG_BOOST");
  static const bool selfcheck = DbgEnv("DV_DBG_SELFCHECK");
  if (use_boost) return BuildBoost(ref, reads, options);
  int k = 0;
  std::vector<std::string> haps;
  const bool ok = FlatBuild(ref, reads, options, &k, &haps);
  if (selfcheck) {
    static bool reg = (atexit(DbgReport), true);
    (void)reg;
    std::unique_ptr<DeBruijnGraph> b = BuildBoost(ref, reads, options);
    ++g_dbg_checked;
    const bool same = (ok == (b != nullptr)) &&
                      (!ok || (k == b->KmerSize() && haps == b->CandidateHaplotypes()));
    if (!same && g_dbg_mismatch++ < 5)
      fprintf(stderr, "DV_DBG_SELFCHECK mismatch: flat ok=%d k=%d n=%zu; boost ok=%d k=%d n=%zu\n",
              ok, k, haps.size(), b != nullptr, b ? b->KmerSize() : 0,
              b ? b->CandidateHaplotypes().size() : 0);
  }
  if (!ok) return nullptr;
  std::unique_ptr<DeBruijnGraph> graph(new DeBruijnGraph());
  graph->flat_ = true;
  graph->k_ = k;
  graph->options_ = options;
  graph->flat_haplotypes_ = std::move(haps);
  return graph;
}

std::unique_ptr<DeBruijnGraph> DeBruijnGraph::BuildBoost(
    absl::string_view ref,
    const std::vector<nucleus::ConstProtoPtr<const Read>>& reads,
    const DeBruijnGraph::Options& options) {
  KBounds bounds = KMinMaxFromReference(ref, options);
  if (bounds.min_k == kBoundsNoWorkingK) return nullptr;

  for (int k = bounds.min_k; k <= bounds.max_k; k += options.step_k()) {
    std::unique_ptr<DeBruijnGraph> graph = std::unique_ptr<DeBruijnGraph>(
        new DeBruijnGraph(ref, reads, options, k));
    if (graph->HasCycle()) {
      continue;
    } else {
      graph->Prune();
      return graph;
    }
  }
  return nullptr;
}

Edge DeBruijnGraph::AddEdge(Vertex from_vertex, Vertex to_vertex, bool is_ref) {
  bool was_present;
  Edge edge;
  std::tie(edge, was_present) = boost::edge(from_vertex, to_vertex, g_);
  if (!was_present) {
    std::tie(edge, std::ignore) = boost::add_edge(from_vertex, to_vertex,
                                                  EdgeInfo{0, false}, g_);
  }
  EdgeInfo& ei = g_[edge];
  ei.weight++;
  ei.is_ref |= is_ref;
  return edge;
}

void DeBruijnGraph::AddKmersAndEdges(string_view bases, int start, int end,
                                     bool is_ref) {
  CHECK_GE(start, 0);
  CHECK_LE(start + k_, bases.size());
  CHECK_LE(end + k_, bases.size());

  // End can be less than 0, in which case we return without doing any work.
  if (end > 0) {
    Vertex vertex_prev = EnsureVertex(bases.substr(start, k_));
    for (int i = start + 1; i <= end; ++i) {
      Vertex vertex_cur = EnsureVertex(bases.substr(i, k_));
      AddEdge(vertex_prev, vertex_cur, is_ref);
      vertex_prev = vertex_cur;
    }
  }
}

void DeBruijnGraph::AddEdgesForReference(string_view ref) {
  AddKmersAndEdges(ref, 0, ref.size() - k_, true /* is_ref */);
}


void DeBruijnGraph::AddEdgesForRead(const nucleus::genomics::v1::Read& read) {
  const string bases = absl::AsciiStrToUpper(read.aligned_sequence());

  // Lambda function to find the next bad position in the read, if one exists,
  // starting from offset `start` in the read. If all remains bases/quals are
  // good, returns bases.size().
  auto NextBadPosition = [&read, &bases, this](int start) -> int {
    for (int i = start; i < bases.size(); ++i) {
      if (!IsCanonicalBase(bases[i], nucleus::CanonicalBases::ACGT) ||
          read.aligned_quality()[i] < options_.min_base_quality()) {
        return i;
      }
    }
    return bases.size();
  };

  // This algorithm is simple and fast, but it isn't the most straightforward
  // implementation so it merits a few comments.
  //
  // Suppose I have the following data:
  //
  // offset: 01234567
  // bases:  ACGTAACC
  // bad? :  00010000
  // k_   :  2 <= using a kmer size of 2
  //
  // The algorithm below loops over positions (variable `i`), pulling kmers of
  // length k from positions `i` and `i + 1` to add as edges. The key
  // calculation is NextBadPosition that searches from the current `i` position
  // for the next position that is bad. In the above example, this would be the
  // 3 position. We then loop from i until `next_bad_position - k`, to create
  // our edges, since we know that everything from i to next_bad_position is
  // good but we cannot construct a valid kmer that overlaps next_bad_position
  // so it invalidates all kmer starts from `next_bad_position - k`. Finally, we
  // set i to `next_bad_position + 1`, which is the very next starting position
  // after the last bad position, and the algorithm repeats.
  //
  // This algorithm has many important properties for performance:
  //
  //   * It doesn't allocate any data structures to support the calculation.
  //   * It only examines whether a given position is good/bad once.
  //   * The loop to add edges is streamlined, without any unnecessary checks.
  //
  const string_view bases_view(bases);
  // Note that this SIGNED int type declaration is key to avoid
  // bases.size() - k_ underflowing.
  const int stop = bases.size() - k_;
  int i = 0;
  while (i < stop) {
    int next_bad_position = NextBadPosition(i);
    AddKmersAndEdges(bases_view, i, next_bad_position - k_, false /* is_ref */);
    i = next_bad_position + 1;
  }
}

std::vector<Path> DeBruijnGraph::CandidatePaths() const {
  std::vector<Path> terminated_paths;
  std::queue<Path> extendable_paths;

  CHECK_GT(boost::out_degree(source_, g_), 0);
  extendable_paths.push({source_});

  // Inefficient.
  while (!extendable_paths.empty()) {
    // Some windows can have an extremely branchy graph.  Ideally windows would
    // be chosen to avoid this.  We give up if we encounter too many paths.
    int n_total_paths = terminated_paths.size() + extendable_paths.size();
    if (n_total_paths > options_.max_num_paths()) {
      return {};
    }

    Path path = extendable_paths.front();
    extendable_paths.pop();
    Vertex last_v = path.back();
    // For each successor of last_v, add path::successor to the
    // appropriate queue.
    AdjacencyIterator vi, vend;
    std::tie(vi, vend) = boost::adjacent_vertices(last_v, g_);
    for (; vi != vend; ++vi) {
      Path extended_path(path);
      extended_path.push_back(*vi);
      if (*vi == sink_ || boost::out_degree(*vi, g_) == 0) {
        terminated_paths.push_back(extended_path);
      } else {
        extendable_paths.push(extended_path);
      }
    }
  }
  return terminated_paths;
}

string DeBruijnGraph::HaplotypeForPath(const Path& path) const {
  std::stringstream haplotype;
  for (Vertex v : path) {
    haplotype << g_[v].kmer[0];
  }
  if (!path.empty()) {
    haplotype << g_[path.back()].kmer.substr(1, k_ - 1);
  }
  return haplotype.str();
}

std::vector<std::string> DeBruijnGraph::CandidateHaplotypes() const {
  if (flat_) return flat_haplotypes_;
  std::vector<std::string> haplotypes;
  for (const Path& path : CandidatePaths()) {
    haplotypes.push_back(HaplotypeForPath(path));
  }
  std::sort(haplotypes.begin(), haplotypes.end());
  return haplotypes;
}

string DeBruijnGraph::GraphViz() const {
  if (flat_) {
    LOG(WARNING) << "GraphViz needs the boost graph; set DV_DBG_BOOST=1";
    return "";
  }
  std::stringstream graphviz;
  auto vertex_label_writer = boost::make_label_writer(
      boost::get(&VertexInfo::kmer, g_));
  boost::write_graphviz(
      graphviz,
      g_,
      vertex_label_writer,
      EdgeLabelWriter<BoostGraph>(g_),
      boost::default_writer(),
      IndexMap());
  return graphviz.str();
}

void DeBruijnGraph::Prune() {
  // Remove low-weight edges not in the reference.
  boost::remove_edge_if(
      [this](const Edge& e) {
        return !g_[e].is_ref && g_[e].weight < options_.min_edge_weight();
      },
      g_);

  // Remove vertices not reachable forward from src or backward from sink.
  VertexIterator vbegin, vend;
  std::tie(vbegin, vend) = boost::vertices(g_);
  absl::flat_hash_set<Vertex> all_vertices(vbegin, vend);

  std::set<Vertex> fwd_reachable_vertices, rev_reachable_vertices;
  fwd_reachable_vertices = VerticesReachableFrom(
      source_, g_, IndexMap());
  rev_reachable_vertices = VerticesReachableFrom(
      sink_, boost::make_reverse_graph(g_), IndexMap());

  absl::flat_hash_set<Vertex> reachable_vertices;
  std::set_intersection(
      fwd_reachable_vertices.begin(), fwd_reachable_vertices.end(),
      rev_reachable_vertices.begin(), rev_reachable_vertices.end(),
      std::inserter(reachable_vertices, reachable_vertices.end()));
  for (Vertex v : all_vertices) {
    if (reachable_vertices.find(v) == reachable_vertices.end()) {
      kmer_to_vertex_.erase(g_[v].kmer);
      boost::clear_vertex(v, g_);
      boost::remove_vertex(v, g_);
    }
  }
  RebuildIndexMap();
}

}  // namespace deepvariant
}  // namespace genomics
}  // namespace learning
