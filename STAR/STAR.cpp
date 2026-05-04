#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kd_tree.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

struct Coord {
    double x;
    double y;
};

struct InstanceData {
    std::vector<int> nodes;
    std::vector<Coord> coords;
    std::vector<int> demands;
    std::unordered_map<int, int> node_to_index;
    int depot;
    int capacity;
    int edge_weight_mode;  // 0=EUC_2D, 1=CEIL_2D, 2=raw int
};

InstanceData make_instance(
    const std::vector<int>& nodes,
    const std::vector<std::vector<double>>& coords,
    const std::vector<int>& demands,
    int depot,
    int capacity,
    const std::string& edge_weight_type) {
    if (nodes.size() != coords.size()) {
        throw std::invalid_argument("nodes and coords must have the same length");
    }
    if (!demands.empty() && demands.size() != nodes.size()) {
        throw std::invalid_argument("demands must be empty or match nodes length");
    }

    InstanceData data;
    data.nodes = nodes;
    data.coords.reserve(coords.size());
    data.demands = demands.empty() ? std::vector<int>(nodes.size(), 0) : demands;
    data.depot = depot;
    data.capacity = capacity;
    data.edge_weight_mode = 2;
    if (edge_weight_type == "CEIL_2D") {
        data.edge_weight_mode = 1;
    } else if (edge_weight_type == "EUC_2D") {
        data.edge_weight_mode = 0;
    }

    for (size_t i = 0; i < coords.size(); ++i) {
        if (coords[i].size() < 2) {
            throw std::invalid_argument("each coord row must contain x and y");
        }
        data.coords.push_back({coords[i][0], coords[i][1]});
        data.node_to_index[nodes[i]] = static_cast<int>(i);
    }
    return data;
}

int demand_of(const InstanceData& data, int node) {
    auto it = data.node_to_index.find(node);
    if (it == data.node_to_index.end()) {
        return 0;
    }
    return data.demands[it->second];
}

int dist(const InstanceData& data, int a, int b) {
    auto ia = data.node_to_index.find(a);
    auto ib = data.node_to_index.find(b);
    if (ia == data.node_to_index.end() || ib == data.node_to_index.end()) {
        throw std::invalid_argument("distance requested for unknown node");
    }
    const Coord& ca = data.coords[ia->second];
    const Coord& cb = data.coords[ib->second];
    double value = std::hypot(ca.x - cb.x, ca.y - cb.y);
    if (data.edge_weight_mode == 1) {
        return static_cast<int>(std::ceil(value));
    }
    if (data.edge_weight_mode == 0) {
        return static_cast<int>(std::floor(value + 0.5));
    }
    return static_cast<int>(value);
}

double raw_dist(const InstanceData& data, int a, int b) {
    auto ia = data.node_to_index.find(a);
    auto ib = data.node_to_index.find(b);
    if (ia == data.node_to_index.end() || ib == data.node_to_index.end()) {
        throw std::invalid_argument("distance requested for unknown node");
    }
    const Coord& ca = data.coords[ia->second];
    const Coord& cb = data.coords[ib->second];
    return std::hypot(ca.x - cb.x, ca.y - cb.y);
}

std::vector<int> nearest_nodes_excluding(const InstanceData& data, int node, int excluded, int limit) {
    std::vector<std::pair<int, int>> keyed;
    keyed.reserve(data.nodes.size());
    for (int other : data.nodes) {
        if (other == excluded) {
            continue;
        }
        keyed.push_back({dist(data, node, other), other});
    }
    if (limit < static_cast<int>(keyed.size())) {
        std::partial_sort(
            keyed.begin(),
            keyed.begin() + limit,
            keyed.end(),
            [](const auto& left, const auto& right) {
                return left.first < right.first || (left.first == right.first && left.second < right.second);
            });
        keyed.resize(limit);
    } else {
        std::sort(keyed.begin(), keyed.end(), [](const auto& left, const auto& right) {
            return left.first < right.first || (left.first == right.first && left.second < right.second);
        });
    }
    std::vector<int> result;
    result.reserve(keyed.size());
    for (const auto& item : keyed) {
        result.push_back(item.second);
    }
    return result;
}

std::vector<int> nearest_order_excluding(const InstanceData& data, int node, int excluded, int limit) {
    std::vector<std::pair<double, int>> keyed;
    keyed.reserve(data.nodes.size());
    for (int other : data.nodes) {
        if (other == excluded) {
            continue;
        }
        keyed.push_back({raw_dist(data, node, other), other});
    }
    if (limit < static_cast<int>(keyed.size())) {
        std::partial_sort(
            keyed.begin(),
            keyed.begin() + limit,
            keyed.end(),
            [](const auto& left, const auto& right) {
                return left.first < right.first || (left.first == right.first && left.second < right.second);
            });
        keyed.resize(limit);
    } else {
        std::sort(keyed.begin(), keyed.end(), [](const auto& left, const auto& right) {
            return left.first < right.first || (left.first == right.first && left.second < right.second);
        });
    }
    std::vector<int> result;
    result.reserve(keyed.size());
    for (const auto& item : keyed) {
        result.push_back(item.second);
    }
    return result;
}

int select_tsp_knn_action(
    const std::vector<int>& nodes,
    const std::vector<std::vector<double>>& coords,
    const std::string& edge_weight_type,
    int current,
    const std::vector<int>& candidates,
    const std::vector<double>& probabilities,
    const std::vector<double>& memory_weights,
    int k,
    int backup_k,
    double random01) {
    if (candidates.empty()) {
        throw std::invalid_argument("select_tsp_knn_action received no candidates");
    }
    if (nodes.size() != probabilities.size()) {
        throw std::invalid_argument("probabilities must align with nodes");
    }
    if (!memory_weights.empty() && memory_weights.size() != nodes.size()) {
        throw std::invalid_argument("memory_weights must be empty or align with nodes");
    }
    InstanceData data = make_instance(nodes, coords, {}, nodes.empty() ? 0 : nodes.front(), 0, edge_weight_type);
    auto current_it = data.node_to_index.find(current);
    if (current_it == data.node_to_index.end()) {
        throw std::invalid_argument("current node is not in nodes");
    }

    std::unordered_map<int, bool> legal;
    legal.reserve(candidates.size() * 2);
    for (int node : candidates) {
        legal[node] = true;
    }

    const int total = std::max(0, k) + std::max(0, backup_k);
    std::vector<int> order = nearest_order_excluding(data, current, current, total > 0 ? total : static_cast<int>(nodes.size()) - 1);

    std::vector<std::pair<int, double>> primary;
    primary.reserve(std::max(0, k));
    double sum = 0.0;
    int primary_limit = std::min(std::max(0, k), static_cast<int>(order.size()));
    for (int i = 0; i < primary_limit; ++i) {
        int node = order[i];
        if (legal.find(node) == legal.end()) {
            continue;
        }
        int index = data.node_to_index.at(node);
        double weight = probabilities[index];
        if (!memory_weights.empty()) {
            weight *= memory_weights[index];
        }
        if (weight <= 0.0 || !std::isfinite(weight)) {
            continue;
        }
        primary.push_back({node, weight});
        sum += weight;
    }

    if (!primary.empty() && sum > 0.0) {
        if (primary.size() == 1) {
            return primary[0].first;
        }
        double threshold = std::min(std::max(random01, 0.0), std::nextafter(1.0, 0.0)) * sum;
        double running = 0.0;
        for (const auto& item : primary) {
            running += item.second;
            if (running >= threshold) {
                return item.first;
            }
        }
        return primary.back().first;
    }

    int backup_start = primary_limit;
    int backup_limit = std::min(primary_limit + std::max(0, backup_k), static_cast<int>(order.size()));
    for (int i = backup_start; i < backup_limit; ++i) {
        int node = order[i];
        if (legal.find(node) != legal.end()) {
            return node;
        }
    }

    int best = candidates[0];
    double best_dist = raw_dist(data, current, best);
    for (int node : candidates) {
        double d = raw_dist(data, current, node);
        if (d < best_dist || (d == best_dist && node < best)) {
            best = node;
            best_dist = d;
        }
    }
    return best;
}

std::vector<int> select_tsp_candidate_actions_batch(
    const std::vector<int>& nodes,
    const std::vector<int>& currents,
    const std::vector<std::vector<int>>& candidates_batch,
    const std::vector<std::vector<double>>& probabilities_batch,
    const std::vector<std::vector<double>>& memory_weights_batch,
    const std::vector<double>& random01_batch) {
    const size_t batch = candidates_batch.size();
    if (currents.size() != batch || probabilities_batch.size() != batch || random01_batch.size() != batch) {
        throw std::invalid_argument("batched selector input sizes do not match");
    }
    if (!memory_weights_batch.empty() && memory_weights_batch.size() != batch) {
        throw std::invalid_argument("memory_weights_batch must be empty or match batch size");
    }
    std::unordered_map<int, int> node_to_index;
    node_to_index.reserve(nodes.size() * 2);
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        node_to_index[nodes[i]] = i;
    }

    std::vector<int> picked(batch, -1);
    for (size_t row = 0; row < batch; ++row) {
        const auto& candidates = candidates_batch[row];
        const auto& probabilities = probabilities_batch[row];
        if (candidates.empty()) {
            throw std::invalid_argument("batched selector received an empty candidate row");
        }
        if (probabilities.size() != nodes.size()) {
            throw std::invalid_argument("probability row must align with nodes");
        }
        const bool use_memory = !memory_weights_batch.empty() && !memory_weights_batch[row].empty();
        if (use_memory && memory_weights_batch[row].size() != nodes.size()) {
            throw std::invalid_argument("memory weight row must align with nodes");
        }

        double sum = 0.0;
        std::vector<double> weights(candidates.size(), 0.0);
        for (size_t j = 0; j < candidates.size(); ++j) {
            auto index_it = node_to_index.find(candidates[j]);
            if (index_it == node_to_index.end()) {
                throw std::invalid_argument("candidate node is not in nodes");
            }
            double weight = probabilities[index_it->second];
            if (use_memory) {
                weight *= memory_weights_batch[row][index_it->second];
            }
            if (weight > 0.0 && std::isfinite(weight)) {
                weights[j] = weight;
                sum += weight;
            }
        }
        if (sum <= 0.0) {
            picked[row] = candidates.front();
            continue;
        }
        double threshold = std::min(std::max(random01_batch[row], 0.0), std::nextafter(1.0, 0.0)) * sum;
        double running = 0.0;
        picked[row] = candidates.back();
        for (size_t j = 0; j < candidates.size(); ++j) {
            running += weights[j];
            if (running >= threshold) {
                picked[row] = candidates[j];
                break;
            }
        }
    }
    return picked;
}

std::vector<int> select_tsp_candidate_weight_actions_batch(
    const std::vector<std::vector<int>>& candidates_batch,
    const std::vector<std::vector<double>>& weights_batch,
    const std::vector<double>& random01_batch) {
    const size_t batch = candidates_batch.size();
    if (weights_batch.size() != batch || random01_batch.size() != batch) {
        throw std::invalid_argument("batched candidate-weight selector input sizes do not match");
    }
    std::vector<int> picked(batch, -1);
    for (size_t row = 0; row < batch; ++row) {
        const auto& candidates = candidates_batch[row];
        const auto& weights_in = weights_batch[row];
        if (candidates.empty()) {
            throw std::invalid_argument("candidate-weight selector received an empty candidate row");
        }
        if (weights_in.size() != candidates.size()) {
            throw std::invalid_argument("candidate weights must align with candidates");
        }
        double sum = 0.0;
        std::vector<double> weights(candidates.size(), 0.0);
        for (size_t j = 0; j < candidates.size(); ++j) {
            double weight = weights_in[j];
            if (weight > 0.0 && std::isfinite(weight)) {
                weights[j] = weight;
                sum += weight;
            }
        }
        if (sum <= 0.0) {
            picked[row] = candidates.front();
            continue;
        }
        double threshold = std::min(std::max(random01_batch[row], 0.0), std::nextafter(1.0, 0.0)) * sum;
        double running = 0.0;
        picked[row] = candidates.back();
        for (size_t j = 0; j < candidates.size(); ++j) {
            running += weights[j];
            if (running >= threshold) {
                picked[row] = candidates[j];
                break;
            }
        }
    }
    return picked;
}

std::vector<int> nearest_nodes_in_route(
    const InstanceData& data,
    int node,
    const std::vector<int>& route,
    int limit) {
    std::vector<std::pair<int, int>> keyed;
    keyed.reserve(route.size());
    for (int other : route) {
        if (other == node) {
            continue;
        }
        keyed.push_back({dist(data, node, other), other});
    }
    if (limit < static_cast<int>(keyed.size())) {
        std::partial_sort(
            keyed.begin(),
            keyed.begin() + limit,
            keyed.end(),
            [](const auto& left, const auto& right) {
                return left.first < right.first || (left.first == right.first && left.second < right.second);
            });
        keyed.resize(limit);
    } else {
        std::sort(keyed.begin(), keyed.end(), [](const auto& left, const auto& right) {
            return left.first < right.first || (left.first == right.first && left.second < right.second);
        });
    }
    std::vector<int> result;
    result.reserve(keyed.size());
    for (const auto& item : keyed) {
        result.push_back(item.second);
    }
    return result;
}

std::unordered_map<int, int> tsp_positions(const std::vector<int>& route) {
    std::unordered_map<int, int> positions;
    positions.reserve(route.size() * 2);
    for (int i = 0; i < static_cast<int>(route.size()); ++i) {
        positions[route[i]] = i;
    }
    return positions;
}

void update_positions(const std::vector<int>& route, std::unordered_map<int, int>& positions, int first, int last) {
    for (int i = first; i < last; ++i) {
        positions[route[i]] = i;
    }
}

void flip_tsp_section(std::vector<int>& route, int start_node, int end_node, std::unordered_map<int, int>& positions) {
    if (start_node == end_node) {
        return;
    }
    const int n = static_cast<int>(route.size());
    int first = positions.at(start_node);
    int last = positions.at(end_node);
    if (first > last) {
        std::swap(first, last);
    }

    const int segment_length = last - first;
    const int remaining_length = n - segment_length;
    if (segment_length <= remaining_length) {
        std::reverse(route.begin() + first, route.begin() + last);
        update_positions(route, positions, first, last);
        return;
    }

    std::vector<int> indices;
    indices.reserve((n - last) + first);
    for (int index = last; index < n; ++index) {
        indices.push_back(index);
    }
    for (int index = 0; index < first; ++index) {
        indices.push_back(index);
    }
    std::vector<int> values;
    values.reserve(indices.size());
    for (int index : indices) {
        values.push_back(route[index]);
    }
    std::reverse(values.begin(), values.end());
    for (size_t i = 0; i < indices.size(); ++i) {
        route[indices[i]] = values[i];
        positions[values[i]] = indices[i];
    }
}

int successor(const std::vector<int>& route, const std::unordered_map<int, int>& positions, int node) {
    int index = positions.at(node);
    return route[(index + 1) % static_cast<int>(route.size())];
}

int predecessor(const std::vector<int>& route, const std::unordered_map<int, int>& positions, int node) {
    int index = positions.at(node);
    return route[(index - 1 + static_cast<int>(route.size())) % static_cast<int>(route.size())];
}

std::vector<std::vector<int>> build_neighbor_order(const InstanceData& data, int total);

std::vector<int> refine_tsp_srr_with_data(
    const InstanceData& data,
    std::vector<int> route,
    const std::vector<int>& changed,
    int refine_k,
    const std::vector<std::vector<int>>* neighbor_order = nullptr);

std::vector<int> refine_tsp_srr(
    const std::vector<int>& nodes,
    const std::vector<std::vector<double>>& coords,
    const std::string& edge_weight_type,
    std::vector<int> route,
    const std::vector<int>& changed,
    int refine_k) {
    InstanceData data = make_instance(nodes, coords, {}, 1, 0, edge_weight_type);
    auto neighbor_order = build_neighbor_order(data, refine_k);
    return refine_tsp_srr_with_data(data, std::move(route), changed, refine_k, &neighbor_order);
}

std::vector<int> refine_tsp_srr_with_order(
    const std::vector<int>& nodes,
    const std::vector<std::vector<double>>& coords,
    const std::string& edge_weight_type,
    std::vector<int> route,
    const std::vector<int>& changed,
    int refine_k,
    const std::vector<std::vector<int>>& neighbor_order) {
    InstanceData data = make_instance(nodes, coords, {}, 1, 0, edge_weight_type);
    return refine_tsp_srr_with_data(data, std::move(route), changed, refine_k, &neighbor_order);
}

std::vector<int> refine_tsp_srr_with_data(
    const InstanceData& data,
    std::vector<int> route,
    const std::vector<int>& changed,
    int refine_k,
    const std::vector<std::vector<int>>* neighbor_order) {
    if (route.size() < 4 || changed.empty()) {
        return route;
    }

    auto positions = tsp_positions(route);
    std::vector<int> checklist = changed;
    size_t cursor = 0;
    while (cursor < checklist.size()) {
        int a = checklist[cursor++];
        if (positions.find(a) == positions.end()) {
            continue;
        }

        int a_next = successor(route, positions, a);
        int a_prev = predecessor(route, positions, a);
        int dist_a_to_next = dist(data, a, a_next);
        int dist_a_to_prev = dist(data, a_prev, a);

        double max_diff = 0.0;
        int best_first = -1;
        int best_last = -1;
        int best_move[4] = {-1, -1, -1, -1};
        std::vector<int> fallback_neighbors;
        const std::vector<int>* neighbors = nullptr;
        if (neighbor_order != nullptr) {
            auto node_it = data.node_to_index.find(a);
            if (node_it != data.node_to_index.end() && node_it->second < static_cast<int>(neighbor_order->size())) {
                neighbors = &(*neighbor_order)[node_it->second];
            }
        }
        if (neighbors == nullptr) {
            fallback_neighbors = nearest_nodes_excluding(data, a, a, refine_k);
            neighbors = &fallback_neighbors;
        }

        int seen = 0;
        for (int b : *neighbors) {
            if (b == a) {
                continue;
            }
            if (seen++ >= refine_k) {
                break;
            }
            int dist_ab = dist(data, a, b);
            if (dist_a_to_next > dist_ab) {
                int b_next = successor(route, positions, b);
                double diff = static_cast<double>(dist_a_to_next + dist(data, b, b_next) - dist_ab - dist(data, a_next, b_next));
                if (diff > max_diff) {
                    best_first = a_next;
                    best_last = b_next;
                    best_move[0] = a_next;
                    best_move[1] = b_next;
                    best_move[2] = a;
                    best_move[3] = b;
                    max_diff = diff;
                }
            } else {
                break;
            }
        }

        seen = 0;
        for (int b : *neighbors) {
            if (b == a) {
                continue;
            }
            if (seen++ >= refine_k) {
                break;
            }
            int dist_ab = dist(data, a, b);
            if (dist_a_to_prev > dist_ab) {
                int b_prev = predecessor(route, positions, b);
                double diff = static_cast<double>(dist_a_to_prev + dist(data, b_prev, b) - dist_ab - dist(data, a_prev, b_prev));
                if (diff > max_diff) {
                    best_first = a;
                    best_last = b;
                    best_move[0] = a;
                    best_move[1] = b;
                    best_move[2] = a_prev;
                    best_move[3] = b_prev;
                    max_diff = diff;
                }
            } else {
                break;
            }
        }

        if (best_first >= 0 && max_diff > 0.0) {
            flip_tsp_section(route, best_first, best_last, positions);
            for (int node : best_move) {
                if (node >= 0 && std::find(checklist.begin(), checklist.end(), node) == checklist.end()) {
                    checklist.push_back(node);
                }
            }
        }
    }
    return route;
}

double tsp_route_cost_cpp(const InstanceData& data, const std::vector<int>& route) {
    double cost = 0.0;
    for (size_t i = 0; i < route.size(); ++i) {
        cost += dist(data, route[i], route[(i + 1) % route.size()]);
    }
    return cost;
}

double cvrp_routes_cost_cpp(const InstanceData& data, const std::vector<std::vector<int>>& routes) {
    double cost = 0.0;
    for (const auto& route : routes) {
        int current = data.depot;
        for (int node : route) {
            cost += dist(data, current, node);
            current = node;
        }
        cost += dist(data, current, data.depot);
    }
    return cost;
}

py::tuple refine_tsp_srr_batch(
    const std::vector<int>& nodes,
    const std::vector<std::vector<double>>& coords,
    const std::string& edge_weight_type,
    const std::vector<std::vector<int>>& routes,
    const std::vector<std::vector<int>>& changeds,
    int refine_k) {
    if (routes.size() != changeds.size()) {
        throw std::invalid_argument("routes and changeds must have the same length");
    }

    InstanceData data = make_instance(nodes, coords, {}, 1, 0, edge_weight_type);
    auto neighbor_order = build_neighbor_order(data, refine_k);
    std::vector<std::vector<int>> refined(routes.size());
    std::vector<double> costs(routes.size(), 0.0);

    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < static_cast<int>(routes.size()); ++i) {
            refined[i] = refine_tsp_srr_with_data(data, routes[i], changeds[i], refine_k, &neighbor_order);
            costs[i] = tsp_route_cost_cpp(data, refined[i]);
        }
    }

    return py::make_tuple(refined, costs);
}

py::tuple refine_tsp_srr_batch_with_order(
    const std::vector<int>& nodes,
    const std::vector<std::vector<double>>& coords,
    const std::string& edge_weight_type,
    const std::vector<std::vector<int>>& routes,
    const std::vector<std::vector<int>>& changeds,
    int refine_k,
    const std::vector<std::vector<int>>& neighbor_order) {
    if (routes.size() != changeds.size()) {
        throw std::invalid_argument("routes and changeds must have the same length");
    }

    InstanceData data = make_instance(nodes, coords, {}, 1, 0, edge_weight_type);
    std::vector<std::vector<int>> refined(routes.size());
    std::vector<double> costs(routes.size(), 0.0);

    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < static_cast<int>(routes.size()); ++i) {
            refined[i] = refine_tsp_srr_with_data(data, routes[i], changeds[i], refine_k, &neighbor_order);
            costs[i] = tsp_route_cost_cpp(data, refined[i]);
        }
    }

    return py::make_tuple(refined, costs);
}

void rebuild_route_positions(const std::vector<int>& route, std::unordered_map<int, int>& positions) {
    positions.clear();
    positions.reserve(route.size() * 2);
    for (int i = 0; i < static_cast<int>(route.size()); ++i) {
        positions[route[i]] = i;
    }
}

void relocate_after_cpp(std::vector<int>& route, std::unordered_map<int, int>& positions, int target, int node) {
    if (target == node) {
        return;
    }
    auto target_it = positions.find(target);
    auto node_it = positions.find(node);
    if (target_it == positions.end() || node_it == positions.end()) {
        throw std::invalid_argument("relocate requested unknown node");
    }
    int target_pos = target_it->second;
    int node_pos = node_it->second;
    int value = route[node_pos];
    route.erase(route.begin() + node_pos);
    if (node_pos < target_pos) {
        --target_pos;
    }
    route.insert(route.begin() + target_pos + 1, value);
    rebuild_route_positions(route, positions);
}

std::vector<Vec2d> instance_points(const InstanceData& data) {
    std::vector<Vec2d> points(data.coords.size());
    for (size_t i = 0; i < data.coords.size(); ++i) {
        points[i] = Vec2d{data.coords[i].x, data.coords[i].y};
    }
    return points;
}

std::vector<std::vector<int>> build_neighbor_order_from_tree(
    const InstanceData& data,
    const KDTree& shared_kdtree,
    int total) {
    const int n = static_cast<int>(data.nodes.size());
    const int row_limit = std::min(std::max(0, total), std::max(0, n - 1));
    std::vector<std::vector<int>> order(static_cast<size_t>(n));
    if (n == 0 || row_limit <= 0) {
        return order;
    }

#ifdef _OPENMP
#pragma omp parallel default(none) shared(shared_kdtree, order, data) firstprivate(n, row_limit)
    {
        KDTree kdtree = shared_kdtree;
        std::vector<uint32_t> deleted;
        deleted.reserve(static_cast<size_t>(row_limit));

#pragma omp for schedule(static)
        for (int u = 0; u < n; ++u) {
            auto& row = order[static_cast<size_t>(u)];
            row.reserve(static_cast<size_t>(row_limit));
            deleted.clear();
            for (int j = 0; j < row_limit; ++j) {
                uint32_t pt_idx = kdtree.nn_bottom_up(static_cast<uint32_t>(u));
                row.push_back(data.nodes[static_cast<size_t>(pt_idx)]);
                kdtree.delete_point(pt_idx);
                deleted.push_back(pt_idx);
            }
            for (uint32_t pt_idx : deleted) {
                kdtree.undelete_point(pt_idx);
            }
        }
    }
#else
    KDTree kdtree = shared_kdtree;
    std::vector<uint32_t> deleted;
    deleted.reserve(static_cast<size_t>(row_limit));
    for (int u = 0; u < n; ++u) {
        auto& row = order[static_cast<size_t>(u)];
        row.reserve(static_cast<size_t>(row_limit));
        deleted.clear();
        for (int j = 0; j < row_limit; ++j) {
            uint32_t pt_idx = kdtree.nn_bottom_up(static_cast<uint32_t>(u));
            row.push_back(data.nodes[static_cast<size_t>(pt_idx)]);
            kdtree.delete_point(pt_idx);
            deleted.push_back(pt_idx);
        }
        for (uint32_t pt_idx : deleted) {
            kdtree.undelete_point(pt_idx);
        }
    }
#endif

    return order;
}

std::vector<std::vector<int>> build_neighbor_order(const InstanceData& data, int total) {
    KDTree shared_kdtree(instance_points(data), /*round_distances=*/false);
    return build_neighbor_order_from_tree(data, shared_kdtree, total);
}

std::vector<std::vector<int>> build_tsp_neighbor_order(
    const std::vector<int>& nodes,
    const std::vector<std::vector<double>>& coords,
    const std::string& edge_weight_type,
    int total) {
    InstanceData data = make_instance(nodes, coords, {}, nodes.empty() ? 0 : nodes.front(), 0, edge_weight_type);
    return build_neighbor_order(data, total);
}

std::vector<int> greedy_tsp_multi_start_with_data(
    const InstanceData& data,
    const KDTree& shared_kdtree,
    int starts,
    int k) {
    const int n = static_cast<int>(data.nodes.size());
    if (n == 0) {
        return {};
    }
    const int num_starts = std::max(1, std::min(std::max(1, starts), n));
    (void)k;

    std::vector<std::vector<int>> routes(static_cast<size_t>(num_starts));
    std::vector<double> costs(static_cast<size_t>(num_starts), std::numeric_limits<double>::infinity());

    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int start_index = 0; start_index < num_starts; ++start_index) {
            KDTree kdtree = shared_kdtree;
            std::vector<int> route_indices;
            route_indices.reserve(static_cast<size_t>(n));
            int current_index = start_index;
            route_indices.push_back(current_index);

            while (static_cast<int>(route_indices.size()) < n) {
                uint32_t next_index = kdtree.nn_bottom_up(static_cast<uint32_t>(current_index));
                kdtree.delete_point(static_cast<uint32_t>(current_index));
                current_index = static_cast<int>(next_index);
                route_indices.push_back(current_index);
            }

            std::vector<int> route;
            route.reserve(static_cast<size_t>(n));
            for (int node_index : route_indices) {
                route.push_back(data.nodes[static_cast<size_t>(node_index)]);
            }
            costs[static_cast<size_t>(start_index)] = tsp_route_cost_cpp(data, route);
            routes[static_cast<size_t>(start_index)] = std::move(route);
        }
    }

    int best_index = 0;
    for (int i = 1; i < num_starts; ++i) {
        if (costs[static_cast<size_t>(i)] < costs[static_cast<size_t>(best_index)]) {
            best_index = i;
        }
    }
    return routes[static_cast<size_t>(best_index)];
}

std::vector<int> greedy_tsp_multi_start(
    const std::vector<int>& nodes,
    const std::vector<std::vector<double>>& coords,
    const std::string& edge_weight_type,
    int starts,
    int k) {
    InstanceData data = make_instance(nodes, coords, {}, nodes.empty() ? 0 : nodes.front(), 0, edge_weight_type);
    KDTree shared_kdtree(instance_points(data), /*round_distances=*/false);
    return greedy_tsp_multi_start_with_data(data, shared_kdtree, starts, k);
}

int select_nearest_STAR_node_cpp(
    const InstanceData& data,
    const std::vector<std::vector<int>>& neighbor_order,
    int current,
    const std::unordered_map<int, bool>& visited,
    int k,
    int backup_k,
    const std::vector<int>& route) {
    int current_index = data.node_to_index.at(current);
    const auto& order = neighbor_order[current_index];
    int primary_limit = std::min(std::max(0, k), static_cast<int>(order.size()));
    for (int i = 0; i < primary_limit; ++i) {
        int node = order[i];
        auto it = visited.find(node);
        if (it == visited.end() || !it->second) {
            return node;
        }
    }

    int backup_limit = std::min(primary_limit + std::max(0, backup_k), static_cast<int>(order.size()));
    for (int i = primary_limit; i < backup_limit; ++i) {
        int node = order[i];
        auto it = visited.find(node);
        if (it == visited.end() || !it->second) {
            return node;
        }
    }

    int best = -1;
    double best_dist = std::numeric_limits<double>::infinity();
    for (int node : route) {
        auto it = visited.find(node);
        if (it != visited.end() && it->second) {
            continue;
        }
        double d = raw_dist(data, current, node);
        if (d < best_dist || (d == best_dist && node < best)) {
            best = node;
            best_dist = d;
        }
    }
    if (best < 0) {
        throw std::runtime_error("no unvisited TSP node left during nearest STAR perturbation");
    }
    return best;
}

int select_nearest_heuristic_STAR_node_cpp(
    const InstanceData& data,
    const std::vector<std::vector<int>>& neighbor_order,
    int current,
    const std::vector<unsigned char>& visited,
    int k,
    int backup_k,
    const std::vector<int>& route,
    std::mt19937_64& rng) {
    int current_index = data.node_to_index.at(current);
    const auto& order = neighbor_order[current_index];
    int primary_limit = std::min(std::max(0, k), static_cast<int>(order.size()));
    std::vector<std::pair<int, double>> primary;
    primary.reserve(primary_limit);
    double sum = 0.0;
    for (int i = 0; i < primary_limit; ++i) {
        int node = order[i];
        int node_index = data.node_to_index.at(node);
        if (visited[node_index]) {
            continue;
        }
        double h = 1.0 / std::max(raw_dist(data, current, node), 1e-12);
        if (h > 0.0 && std::isfinite(h)) {
            primary.push_back({node, h});
            sum += h;
        }
    }
    if (!primary.empty() && sum > 0.0) {
        std::uniform_real_distribution<double> dist01(0.0, std::nextafter(1.0, 0.0));
        double threshold = dist01(rng) * sum;
        double running = 0.0;
        for (const auto& item : primary) {
            running += item.second;
            if (running >= threshold) {
                return item.first;
            }
        }
        return primary.back().first;
    }

    int backup_limit = std::min(primary_limit + std::max(0, backup_k), static_cast<int>(order.size()));
    for (int i = primary_limit; i < backup_limit; ++i) {
        int node = order[i];
        if (!visited[data.node_to_index.at(node)]) {
            return node;
        }
    }

    int best = -1;
    double best_dist = std::numeric_limits<double>::infinity();
    for (int node : route) {
        if (visited[data.node_to_index.at(node)]) {
            continue;
        }
        double d = raw_dist(data, current, node);
        if (d < best_dist || (d == best_dist && node < best)) {
            best = node;
            best_dist = d;
        }
    }
    if (best < 0) {
        throw std::runtime_error("no unvisited TSP node left during nearest heuristic STAR perturbation");
    }
    return best;
}

std::pair<std::vector<int>, std::vector<int>> perturb_nearest_heuristic_tsp_sample(
    const InstanceData& data,
    const std::vector<std::vector<int>>& neighbor_order,
    const std::vector<int>& base_route,
    int min_new_edges,
    int k,
    int backup_k,
    uint64_t seed) {
    std::vector<int> candidate = base_route;
    std::unordered_map<int, int> positions;
    rebuild_route_positions(candidate, positions);
    std::vector<unsigned char> visited(data.nodes.size(), 0);
    std::mt19937_64 rng(seed);

    std::uniform_int_distribution<int> start_dist(0, static_cast<int>(candidate.size()) - 1);
    int current = candidate[start_dist(rng)];
    visited[data.node_to_index.at(current)] = 1;
    int visited_count = 1;
    int new_edges = 0;
    std::vector<int> changed;
    std::unordered_map<int, bool> in_changed;
    in_changed.reserve(std::max(16, min_new_edges * 4));

    auto add_changed = [&](int node) {
        if (in_changed.find(node) == in_changed.end()) {
            in_changed[node] = true;
            changed.push_back(node);
        }
    };

    while (new_edges < min_new_edges && visited_count < static_cast<int>(candidate.size())) {
        int current_pos = positions.at(current);
        int successor_node = candidate[(current_pos + 1) % static_cast<int>(candidate.size())];
        int picked = select_nearest_heuristic_STAR_node_cpp(data, neighbor_order, current, visited, k, backup_k, candidate, rng);
        int picked_index = data.node_to_index.at(picked);
        if (visited[picked_index]) {
            throw std::runtime_error("nearest heuristic STAR selected invalid visited node");
        }

        if (picked != successor_node) {
            int picked_pos = positions.at(picked);
            int picked_pred = candidate[(picked_pos - 1 + static_cast<int>(candidate.size())) % static_cast<int>(candidate.size())];
            relocate_after_cpp(candidate, positions, current, picked);
            add_changed(current);
            add_changed(picked);
            add_changed(picked_pred);
            add_changed(successor_node);
            ++new_edges;
        }

        visited[picked_index] = 1;
        ++visited_count;
        current = picked;
    }

    return {candidate, changed};
}

std::vector<int> run_STAR_nearest_tsp(
    const std::vector<int>& nodes,
    const std::vector<std::vector<double>>& coords,
    const std::string& edge_weight_type,
    std::vector<int> initial_route,
    int iterations,
    int min_new_edges,
    int refine_k,
    int k,
    int backup_k,
    uint64_t seed,
    bool refine) {
    if (iterations < 0) {
        throw std::invalid_argument("iterations must be non-negative");
    }
    if (initial_route.size() <= 2 || iterations == 0 || min_new_edges == 0) {
        return initial_route;
    }
    InstanceData data = make_instance(nodes, coords, {}, nodes.empty() ? 0 : nodes.front(), 0, edge_weight_type);
    auto neighbor_order = build_neighbor_order(data, std::max({0, k + backup_k, refine ? refine_k : 0}));
    std::mt19937_64 rng(seed);
    std::vector<int> best = initial_route;
    double best_cost = tsp_route_cost_cpp(data, best);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<int> candidate = best;
        std::unordered_map<int, int> positions;
        rebuild_route_positions(candidate, positions);
        std::unordered_map<int, bool> visited;
        visited.reserve(candidate.size() * 2);
        for (int node : candidate) {
            visited[node] = false;
        }

        std::uniform_int_distribution<int> start_dist(0, static_cast<int>(candidate.size()) - 1);
        int current = candidate[start_dist(rng)];
        visited[current] = true;
        int visited_count = 1;
        int new_edges = 0;
        std::vector<int> changed;
        std::unordered_map<int, bool> in_changed;

        auto add_changed = [&](int node) {
            if (in_changed.find(node) == in_changed.end()) {
                in_changed[node] = true;
                changed.push_back(node);
            }
        };

        while (new_edges < min_new_edges && visited_count < static_cast<int>(candidate.size())) {
            int current_pos = positions.at(current);
            int successor_node = candidate[(current_pos + 1) % static_cast<int>(candidate.size())];
            int picked = select_nearest_STAR_node_cpp(data, neighbor_order, current, visited, k, backup_k, candidate);
            auto picked_visited = visited.find(picked);
            if (picked_visited == visited.end() || picked_visited->second) {
                throw std::runtime_error("nearest STAR selected invalid visited node");
            }

            if (picked != successor_node) {
                int picked_pos = positions.at(picked);
                int picked_pred = candidate[(picked_pos - 1 + static_cast<int>(candidate.size())) % static_cast<int>(candidate.size())];
                relocate_after_cpp(candidate, positions, current, picked);
                add_changed(current);
                add_changed(picked);
                add_changed(picked_pred);
                add_changed(successor_node);
                ++new_edges;
            }

            visited[picked] = true;
            ++visited_count;
            current = picked;
        }

        if (refine && !changed.empty()) {
            candidate = refine_tsp_srr_with_data(data, std::move(candidate), changed, refine_k, &neighbor_order);
        }
        double candidate_cost = tsp_route_cost_cpp(data, candidate);
        if (candidate_cost <= best_cost + 1e-9) {
            best = candidate;
            best_cost = candidate_cost;
        }
    }
    return best;
}

std::vector<int> run_STAR_nearest_tsp_samples(
    const std::vector<int>& nodes,
    const std::vector<std::vector<double>>& coords,
    const std::string& edge_weight_type,
    std::vector<int> initial_route,
    int iterations,
    int min_new_edges,
    int samples,
    int refine_k,
    int k,
    int backup_k,
    uint64_t seed,
    bool refine) {
    if (iterations < 0) {
        throw std::invalid_argument("iterations must be non-negative");
    }
    if (samples <= 0) {
        throw std::invalid_argument("samples must be positive");
    }
    if (initial_route.size() <= 2 || iterations == 0 || min_new_edges == 0) {
        return initial_route;
    }

    InstanceData data = make_instance(nodes, coords, {}, nodes.empty() ? 0 : nodes.front(), 0, edge_weight_type);
    auto neighbor_order = build_neighbor_order(data, std::max({0, k + backup_k, refine ? refine_k : 0}));
    std::vector<int> best = initial_route;
    double best_cost = tsp_route_cost_cpp(data, best);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<std::vector<int>> candidates(samples);
        std::vector<double> costs(samples, 0.0);
        {
            py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (int sample = 0; sample < samples; ++sample) {
                uint64_t sample_seed = seed
                    ^ (static_cast<uint64_t>(iteration + 1) * 0x9E3779B97F4A7C15ULL)
                    ^ (static_cast<uint64_t>(sample + 1) * 0xBF58476D1CE4E5B9ULL);
                auto perturbed = perturb_nearest_heuristic_tsp_sample(
                    data, neighbor_order, best, min_new_edges, k, backup_k, sample_seed);
                if (refine && !perturbed.second.empty()) {
                    candidates[sample] = refine_tsp_srr_with_data(data, std::move(perturbed.first), perturbed.second, refine_k, &neighbor_order);
                } else {
                    candidates[sample] = std::move(perturbed.first);
                }
                costs[sample] = tsp_route_cost_cpp(data, candidates[sample]);
            }
        }

        int best_sample = -1;
        double iteration_best_cost = best_cost;
        for (int sample = 0; sample < samples; ++sample) {
            if (costs[sample] <= iteration_best_cost + 1e-9) {
                iteration_best_cost = costs[sample];
                best_sample = sample;
            }
        }
        if (best_sample >= 0) {
            best = std::move(candidates[best_sample]);
            best_cost = iteration_best_cost;
        }
    }
    return best;
}

std::pair<int, int> find_route_pos(const std::vector<std::vector<int>>& routes, int node) {
    for (int route_index = 0; route_index < static_cast<int>(routes.size()); ++route_index) {
        const auto& route = routes[route_index];
        for (int pos = 0; pos < static_cast<int>(route.size()); ++pos) {
            if (route[pos] == node) {
                return {route_index, pos};
            }
        }
    }
    return {-1, -1};
}

void cvrp_intra_route_ls_srr(
    const InstanceData& data,
    std::vector<std::vector<int>>& routes,
    const std::vector<int>& checklist,
    int refine_k) {
    size_t cursor = 0;
    while (cursor < checklist.size()) {
        int a = checklist[cursor++];
        auto [route_index, a_pos] = find_route_pos(routes, a);
        if (route_index < 0) {
            continue;
        }

        auto& seq = routes[route_index];
        if (seq.size() < 2) {
            continue;
        }

        int a_prev = a_pos > 0 ? seq[a_pos - 1] : data.depot;
        int a_next = a_pos < static_cast<int>(seq.size()) - 1 ? seq[a_pos + 1] : data.depot;
        int dist_a_prev = dist(data, a_prev, a);
        int dist_a_next = dist(data, a, a_next);

        double max_diff = 0.0;
        int best_i = -1;
        int best_j = -1;

        for (int b : nearest_nodes_in_route(data, a, seq, refine_k)) {
            int dist_ab = dist(data, a, b);
            if (dist_a_prev <= dist_ab) {
                break;
            }
            auto b_it = std::find(seq.begin(), seq.end(), b);
            if (b_it == seq.end()) {
                continue;
            }
            int b_pos = static_cast<int>(std::distance(seq.begin(), b_it));
            if (b_pos == a_pos) {
                continue;
            }
            int b_prev = b_pos > 0 ? seq[b_pos - 1] : data.depot;
            double diff;
            if (b_pos > a_pos) {
                diff = static_cast<double>(dist_a_prev + dist(data, b_prev, b) - dist(data, a_prev, b_prev) - dist_ab);
                if (diff > max_diff) {
                    max_diff = diff;
                    best_i = a_pos;
                    best_j = b_pos;
                }
            } else {
                diff = static_cast<double>(dist(data, b_prev, b) + dist_a_prev - dist(data, b_prev, a) - dist(data, b, a_prev));
                if (diff > max_diff) {
                    max_diff = diff;
                    best_i = b_pos;
                    best_j = a_pos;
                }
            }
        }

        for (int b : nearest_nodes_in_route(data, a, seq, refine_k)) {
            int dist_ab = dist(data, a, b);
            if (dist_a_next <= dist_ab) {
                break;
            }
            auto b_it = std::find(seq.begin(), seq.end(), b);
            if (b_it == seq.end()) {
                continue;
            }
            int b_pos = static_cast<int>(std::distance(seq.begin(), b_it));
            if (b_pos == a_pos) {
                continue;
            }
            int b_next = b_pos < static_cast<int>(seq.size()) - 1 ? seq[b_pos + 1] : data.depot;
            if (b_pos > a_pos) {
                double diff = static_cast<double>(dist_a_next + dist(data, b, b_next) - dist_ab - dist(data, a_next, b_next));
                if (diff > max_diff) {
                    max_diff = diff;
                    best_i = a_pos + 1;
                    best_j = b_pos + 1;
                }
            }
        }

        if (max_diff > 1e-6 && best_i >= 0 && best_j > best_i) {
            std::reverse(seq.begin() + best_i, seq.begin() + best_j);
        }
    }
}

void touch_node(int node, int depot, std::vector<int>& checklist, std::unordered_map<int, bool>& in_checklist, std::unordered_map<int, bool>& dlb) {
    if (node == depot || node < 0) {
        return;
    }
    dlb.erase(node);
    if (in_checklist.find(node) == in_checklist.end()) {
        checklist.push_back(node);
        in_checklist[node] = true;
    }
}

void update_route_state(
    const InstanceData& data,
    int route_index,
    const std::vector<int>& sentinels,
    const std::unordered_map<int, int>& next_node,
    std::unordered_map<int, int>& node_route,
    std::unordered_map<int, int>& cum_load,
    std::vector<int>& route_loads) {
    int sentinel = sentinels[route_index];
    int current = next_node.at(sentinel);
    int load = 0;
    node_route[sentinel] = route_index;
    cum_load[sentinel] = 0;
    while (current >= 0) {
        load += demand_of(data, current);
        node_route[current] = route_index;
        cum_load[current] = load;
        current = next_node.at(current);
    }
    route_loads[route_index] = load;
}

void cvrp_inter_route_ls_srr(
    const InstanceData& data,
    std::vector<std::vector<int>>& routes,
    std::vector<int>& checklist,
    std::unordered_map<int, bool>& in_checklist,
    int refine_k) {
    if (checklist.empty()) {
        return;
    }

    std::unordered_map<int, int> next_node;
    std::unordered_map<int, int> prev_node;
    std::unordered_map<int, int> node_route;
    std::unordered_map<int, int> cum_load;
    std::vector<int> route_loads(routes.size(), 0);
    std::vector<int> sentinels(routes.size());
    for (int route_index = 0; route_index < static_cast<int>(routes.size()); ++route_index) {
        sentinels[route_index] = -(route_index + 1);
    }

    for (int route_index = 0; route_index < static_cast<int>(routes.size()); ++route_index) {
        int sentinel = sentinels[route_index];
        int prev = sentinel;
        int load = 0;
        node_route[sentinel] = route_index;
        cum_load[sentinel] = 0;
        for (int node : routes[route_index]) {
            next_node[prev] = node;
            prev_node[node] = prev;
            node_route[node] = route_index;
            load += demand_of(data, node);
            cum_load[node] = load;
            prev = node;
        }
        next_node[prev] = sentinel;
        prev_node[sentinel] = prev;
        route_loads[route_index] = load;
    }

    auto as_depot = [&data](int node) { return node < 0 ? data.depot : node; };
    std::unordered_map<int, bool> dlb;

    size_t head = 0;
    while (head < checklist.size()) {
        int u = checklist[head++];
        in_checklist.erase(u);
        if (dlb.find(u) != dlb.end() || node_route.find(u) == node_route.end()) {
            continue;
        }

        bool improved = false;
        for (int v : nearest_nodes_excluding(data, u, u, refine_k)) {
            if (v == data.depot || node_route.find(v) == node_route.end()) {
                continue;
            }
            int r_u = node_route[u];
            int r_v = node_route[v];
            if (r_u == r_v) {
                continue;
            }

            int next_u = next_node[u];
            int prev_u = prev_node[u];
            int next_v = next_node[v];
            int prev_v = prev_node[v];

            if (route_loads[r_v] + demand_of(data, u) <= data.capacity) {
                double delta = static_cast<double>(
                    dist(data, as_depot(prev_u), as_depot(next_u))
                    + dist(data, v, u)
                    + dist(data, u, as_depot(next_v))
                    - dist(data, as_depot(prev_u), u)
                    - dist(data, u, as_depot(next_u))
                    - dist(data, v, as_depot(next_v)));
                if (delta < -1e-5) {
                    next_node[prev_u] = next_u;
                    prev_node[next_u] = prev_u;
                    next_node[v] = u;
                    prev_node[u] = v;
                    next_node[u] = next_v;
                    prev_node[next_v] = u;
                    update_route_state(data, r_u, sentinels, next_node, node_route, cum_load, route_loads);
                    update_route_state(data, r_v, sentinels, next_node, node_route, cum_load, route_loads);
                    for (int node : {u, v, prev_u, next_u, next_v}) {
                        touch_node(node, data.depot, checklist, in_checklist, dlb);
                    }
                    improved = true;
                    break;
                }

                next_u = next_node[u];
                prev_u = prev_node[u];
                prev_v = prev_node[v];
                double delta2 = static_cast<double>(
                    dist(data, as_depot(prev_u), as_depot(next_u))
                    + dist(data, as_depot(prev_v), u)
                    + dist(data, u, v)
                    - dist(data, as_depot(prev_u), u)
                    - dist(data, u, as_depot(next_u))
                    - dist(data, as_depot(prev_v), v));
                if (delta2 < -1e-5) {
                    next_node[prev_u] = next_u;
                    prev_node[next_u] = prev_u;
                    next_node[prev_v] = u;
                    prev_node[u] = prev_v;
                    next_node[u] = v;
                    prev_node[v] = u;
                    update_route_state(data, r_u, sentinels, next_node, node_route, cum_load, route_loads);
                    update_route_state(data, r_v, sentinels, next_node, node_route, cum_load, route_loads);
                    for (int node : {u, prev_v, v, prev_u, next_u}) {
                        touch_node(node, data.depot, checklist, in_checklist, dlb);
                    }
                    improved = true;
                    break;
                }
            }

            r_u = node_route[u];
            r_v = node_route[v];
            if (r_u == r_v) {
                continue;
            }
            next_u = next_node[u];
            prev_u = prev_node[u];
            next_v = next_node[v];
            prev_v = prev_node[v];
            int load_u_new = route_loads[r_u] - demand_of(data, u) + demand_of(data, v);
            int load_v_new = route_loads[r_v] - demand_of(data, v) + demand_of(data, u);
            if (load_u_new <= data.capacity && load_v_new <= data.capacity) {
                double delta = static_cast<double>(
                    dist(data, as_depot(prev_u), v)
                    + dist(data, v, as_depot(next_u))
                    + dist(data, as_depot(prev_v), u)
                    + dist(data, u, as_depot(next_v))
                    - dist(data, as_depot(prev_u), u)
                    - dist(data, u, as_depot(next_u))
                    - dist(data, as_depot(prev_v), v)
                    - dist(data, v, as_depot(next_v)));
                if (delta < -1e-5) {
                    next_node[prev_u] = v;
                    prev_node[next_u] = v;
                    next_node[prev_v] = u;
                    prev_node[next_v] = u;
                    next_node[u] = next_v;
                    prev_node[u] = prev_v;
                    next_node[v] = next_u;
                    prev_node[v] = prev_u;
                    update_route_state(data, r_u, sentinels, next_node, node_route, cum_load, route_loads);
                    update_route_state(data, r_v, sentinels, next_node, node_route, cum_load, route_loads);
                    for (int node : {u, v, prev_u, next_u, prev_v, next_v}) {
                        touch_node(node, data.depot, checklist, in_checklist, dlb);
                    }
                    improved = true;
                    break;
                }
            }

            r_u = node_route[u];
            r_v = node_route[v];
            if (r_u == r_v) {
                continue;
            }
            next_u = next_node[u];
            next_v = next_node[v];
            int head_u = cum_load[u];
            int tail_u = route_loads[r_u] - head_u;
            int head_v = cum_load[v];
            int tail_v = route_loads[r_v] - head_v;
            if (head_u + tail_v <= data.capacity && head_v + tail_u <= data.capacity) {
                double delta = static_cast<double>(
                    dist(data, u, as_depot(next_v))
                    + dist(data, v, as_depot(next_u))
                    - dist(data, u, as_depot(next_u))
                    - dist(data, v, as_depot(next_v)));
                if (delta < -1e-5) {
                    next_node[u] = next_v;
                    prev_node[next_v] = u;
                    next_node[v] = next_u;
                    prev_node[next_u] = v;
                    update_route_state(data, r_u, sentinels, next_node, node_route, cum_load, route_loads);
                    update_route_state(data, r_v, sentinels, next_node, node_route, cum_load, route_loads);
                    for (int node : {u, v, next_u, next_v}) {
                        touch_node(node, data.depot, checklist, in_checklist, dlb);
                    }
                    improved = true;
                    break;
                }
            }
        }
        if (!improved) {
            dlb[u] = true;
        }
    }

    std::vector<std::vector<int>> rebuilt;
    rebuilt.reserve(routes.size());
    for (int sentinel : sentinels) {
        std::vector<int> route;
        int current = next_node[sentinel];
        while (current >= 0) {
            route.push_back(current);
            current = next_node[current];
        }
        if (!route.empty()) {
            rebuilt.push_back(route);
        }
    }
    routes = rebuilt;
}

std::vector<std::vector<int>> refine_cvrp_srr(
    const std::vector<int>& nodes,
    const std::vector<std::vector<double>>& coords,
    const std::vector<int>& demands,
    int depot,
    int capacity,
    const std::string& edge_weight_type,
    std::vector<std::vector<int>> routes,
    const std::vector<int>& changed,
    int refine_k) {
    std::vector<std::vector<int>> candidate;
    candidate.reserve(routes.size());
    for (auto& route : routes) {
        if (!route.empty()) {
            candidate.push_back(route);
        }
    }
    std::vector<int> checklist;
    std::unordered_map<int, bool> in_checklist;
    for (int node : changed) {
        if (node == depot) {
            continue;
        }
        checklist.push_back(node);
        in_checklist[node] = true;
    }
    if (checklist.empty()) {
        return candidate;
    }

    InstanceData data = make_instance(nodes, coords, demands, depot, capacity, edge_weight_type);
    cvrp_intra_route_ls_srr(data, candidate, checklist, refine_k);
    cvrp_inter_route_ls_srr(data, candidate, checklist, in_checklist, refine_k);
    cvrp_intra_route_ls_srr(data, candidate, checklist, refine_k);

    std::vector<std::vector<int>> rebuilt;
    rebuilt.reserve(candidate.size());
    for (const auto& route : candidate) {
        if (!route.empty()) {
            rebuilt.push_back(route);
        }
    }
    return rebuilt;
}

py::tuple refine_cvrp_srr_batch(
    const std::vector<int>& nodes,
    const std::vector<std::vector<double>>& coords,
    const std::vector<int>& demands,
    int depot,
    int capacity,
    const std::string& edge_weight_type,
    const std::vector<std::vector<std::vector<int>>>& routes_batch,
    const std::vector<std::vector<int>>& changeds,
    int refine_k) {
    if (routes_batch.size() != changeds.size()) {
        throw std::invalid_argument("routes_batch and changeds must have the same length");
    }

    InstanceData data = make_instance(nodes, coords, demands, depot, capacity, edge_weight_type);
    std::vector<std::vector<std::vector<int>>> refined(routes_batch.size());
    std::vector<double> costs(routes_batch.size(), 0.0);

    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < static_cast<int>(routes_batch.size()); ++i) {
            refined[static_cast<size_t>(i)] = refine_cvrp_srr(
                nodes,
                coords,
                demands,
                depot,
                capacity,
                edge_weight_type,
                routes_batch[static_cast<size_t>(i)],
                changeds[static_cast<size_t>(i)],
                refine_k);
            costs[static_cast<size_t>(i)] = cvrp_routes_cost_cpp(data, refined[static_cast<size_t>(i)]);
        }
    }

    return py::make_tuple(refined, costs);
}

int cvrp_find_route_index(const std::vector<std::vector<int>>& routes, int node, int* pos_out = nullptr) {
    for (int route_index = 0; route_index < static_cast<int>(routes.size()); ++route_index) {
        const auto& route = routes[static_cast<size_t>(route_index)];
        for (int pos = 0; pos < static_cast<int>(route.size()); ++pos) {
            if (route[static_cast<size_t>(pos)] == node) {
                if (pos_out != nullptr) {
                    *pos_out = pos;
                }
                return route_index;
            }
        }
    }
    throw std::invalid_argument("CVRP route mutation requested unknown customer");
}

int cvrp_route_prefix_load_cpp(const InstanceData& data, const std::vector<int>& route, int through_node) {
    int load = 0;
    for (int node : route) {
        load += demand_of(data, node);
        if (node == through_node) {
            return load;
        }
    }
    return load;
}

int cvrp_route_load_cpp(const InstanceData& data, const std::vector<int>& route) {
    int load = 0;
    for (int node : route) {
        load += demand_of(data, node);
    }
    return load;
}

uint64_t cvrp_edge_key(int a, int b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}

uint64_t cvrp_undirected_customer_edge_key(const InstanceData& data, int a, int b) {
    if (a != data.depot && b != data.depot && a > b) {
        std::swap(a, b);
    }
    return cvrp_edge_key(a, b);
}

std::unordered_set<uint64_t> cvrp_source_edges_cpp(
    const InstanceData& data,
    const std::vector<std::vector<int>>& routes) {
    std::unordered_set<uint64_t> edges;
    for (const auto& route : routes) {
        int previous = data.depot;
        for (int node : route) {
            edges.insert(cvrp_undirected_customer_edge_key(data, previous, node));
            previous = node;
        }
        edges.insert(cvrp_undirected_customer_edge_key(data, previous, data.depot));
    }
    return edges;
}

std::unordered_set<uint64_t> cvrp_directed_memory_edges_cpp(
    const InstanceData& data,
    const std::vector<std::vector<int>>& routes) {
    std::unordered_set<uint64_t> edges;
    for (const auto& route : routes) {
        int previous = data.depot;
        for (int node : route) {
            edges.insert(cvrp_edge_key(previous, node));
            edges.insert(cvrp_edge_key(node, previous));
            previous = node;
        }
        edges.insert(cvrp_edge_key(previous, data.depot));
        edges.insert(cvrp_edge_key(data.depot, previous));
    }
    return edges;
}

std::unordered_map<int, int> cvrp_source_route_ids_cpp(const std::vector<std::vector<int>>& routes) {
    std::unordered_map<int, int> ids;
    for (int route_index = 0; route_index < static_cast<int>(routes.size()); ++route_index) {
        for (int node : routes[static_cast<size_t>(route_index)]) {
            ids[node] = route_index;
        }
    }
    return ids;
}

void cvrp_split_after_current_cpp(std::vector<std::vector<int>>& routes, int current) {
    int pos = -1;
    int route_index = cvrp_find_route_index(routes, current, &pos);
    auto& route = routes[static_cast<size_t>(route_index)];
    if (pos + 1 >= static_cast<int>(route.size())) {
        return;
    }
    std::vector<int> tail(route.begin() + pos + 1, route.end());
    route.erase(route.begin() + pos + 1, route.end());
    routes.insert(routes.begin() + route_index + 1, std::move(tail));
}

void cvrp_remove_customer_cpp(std::vector<std::vector<int>>& routes, int node) {
    int pos = -1;
    int route_index = cvrp_find_route_index(routes, node, &pos);
    auto& route = routes[static_cast<size_t>(route_index)];
    route.erase(route.begin() + pos);
    if (route.empty()) {
        routes.erase(routes.begin() + route_index);
    }
}

void cvrp_relocate_after_current_route_cpp(std::vector<std::vector<int>>& routes, int current, int picked) {
    cvrp_remove_customer_cpp(routes, picked);
    int pos = -1;
    int current_route = cvrp_find_route_index(routes, current, &pos);
    int insert_route = std::min(current_route + 1, static_cast<int>(routes.size()));
    if (insert_route == static_cast<int>(routes.size())) {
        routes.push_back(std::vector<int>{picked});
        return;
    }
    routes[static_cast<size_t>(insert_route)].insert(routes[static_cast<size_t>(insert_route)].begin(), picked);
}

void cvrp_relocate_customer_cpp(const InstanceData& data, std::vector<std::vector<int>>& routes, int current, int picked) {
    cvrp_remove_customer_cpp(routes, picked);
    int pos = -1;
    int route_index = cvrp_find_route_index(routes, current, &pos);
    auto& route = routes[static_cast<size_t>(route_index)];
    if (cvrp_route_load_cpp(data, route) + demand_of(data, picked) > data.capacity) {
        routes.insert(routes.begin() + route_index + 1, std::vector<int>{picked});
        return;
    }
    route.insert(route.begin() + pos + 1, picked);
}

struct CvrpPerturbBatchStateCpp {
    std::vector<std::vector<int>> routes;
    std::unordered_map<int, int> source_route_id;
    std::unordered_set<uint64_t> source_edges;
    std::unordered_set<uint64_t> source_memory_edges;
    std::unordered_set<int> visited;
    std::unordered_set<int> changed;
    std::vector<int> prefix;
    int current = -1;
    int remaining_capacity = 0;
    int new_edges_cross = 0;
    int steps = 0;
    bool done = false;
};

uint64_t tsp_directed_edge_key(int a, int b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}

void tsp_add_symmetric_edge(std::unordered_set<uint64_t>& edges, int a, int b) {
    edges.insert(tsp_directed_edge_key(a, b));
    edges.insert(tsp_directed_edge_key(b, a));
}

struct TspWeightedActionCpp {
    int node;
    double weight;
};

struct TspPerturbBatchStateCpp {
    std::vector<int> route;
    std::unordered_map<int, int> positions;
    std::unordered_set<int> visited;
    std::unordered_set<int> changed;
    std::unordered_set<uint64_t> introduced_edges;
    std::unordered_set<uint64_t> removed_edges;
    std::vector<int> prefix;
    int current = -1;
    int new_edges = 0;
    bool active = true;
};

class TspPerturbBatchCpp {
public:
    TspPerturbBatchCpp(
        const std::vector<int>& nodes,
        const std::vector<std::vector<double>>& coords,
        const std::string& edge_weight_type,
        const std::vector<int>& route,
        int samples,
        int min_new_edges,
        int k,
        int backup_k,
        uint64_t seed,
        const std::vector<std::vector<int>>& neighbor_order = {})
        : data_(make_instance(nodes, coords, {}, nodes.empty() ? 0 : nodes.front(), 0, edge_weight_type)),
          min_new_edges_(min_new_edges),
          k_(std::max(0, k)),
          backup_k_(std::max(0, backup_k)),
          neighbor_order_(
              neighbor_order.empty()
                  ? build_neighbor_order(data_, std::max(1, k_ + backup_k_))
                  : neighbor_order) {
        if (samples <= 0) {
            throw std::invalid_argument("TSP perturb batch requires positive samples");
        }
        if (route.size() <= 2 || min_new_edges_ == 0) {
            for (int sample = 0; sample < samples; ++sample) {
                TspPerturbBatchStateCpp state;
                state.route = route;
                state.active = false;
                states_.push_back(std::move(state));
            }
            return;
        }
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> start_dist(0, static_cast<int>(route.size()) - 1);
        states_.reserve(static_cast<size_t>(samples));
        for (int sample = 0; sample < samples; ++sample) {
            int start = route[static_cast<size_t>(start_dist(rng))];
            TspPerturbBatchStateCpp state;
            state.route = route;
            rebuild_route_positions(state.route, state.positions);
            if (state.positions.find(start) == state.positions.end()) {
                throw std::invalid_argument("TSP perturb start node is not in route");
            }
            state.current = start;
            state.visited.insert(start);
            state.prefix.push_back(start);
            state.changed.insert(start);
            state.active = true;
            states_.push_back(std::move(state));
        }
    }

    py::tuple requests() {
        std::vector<int> indices;
        std::vector<std::vector<int>> prefixes;
        std::vector<std::vector<int>> candidate_rows;
        for (int i = 0; i < static_cast<int>(states_.size()); ++i) {
            auto& state = states_[static_cast<size_t>(i)];
            if (!state.active || state.new_edges >= min_new_edges_ || state.visited.size() >= state.route.size()) {
                state.active = false;
                continue;
            }
            auto candidates = collect_legal_candidates(state, 0, k_, false);
            if (candidates.empty()) {
                candidates = collect_legal_candidates(state, k_, backup_k_, false);
            }
            if (candidates.empty()) {
                candidates = collect_legal_candidates(state, 0, 0, true);
            }
            if (candidates.empty()) {
                state.active = false;
                continue;
            }
            indices.push_back(i);
            prefixes.push_back(state.prefix);
            candidate_rows.push_back(std::move(candidates));
        }
        return py::make_tuple(indices, prefixes, candidate_rows);
    }

    void step(
        const std::vector<int>& indices,
        const std::vector<std::vector<double>>& probability_rows,
        const std::vector<std::tuple<int, int, double>>& memory_values,
        const std::vector<double>& random01) {
        if (indices.size() != probability_rows.size() || indices.size() != random01.size()) {
            throw std::invalid_argument("TSP perturb step batch sizes do not match");
        }
        std::unordered_map<uint64_t, double> memory;
        memory.reserve(memory_values.size() * 2);
        for (const auto& entry : memory_values) {
            int u = 0;
            int v = 0;
            double value = 1.0;
            std::tie(u, v, value) = entry;
            if (value > 0.0 && std::isfinite(value)) {
                memory[tsp_directed_edge_key(u, v)] = value;
            }
        }
        std::vector<std::string> errors(indices.size());
        if (indices.size() < 256) {
            for (size_t row = 0; row < indices.size(); ++row) {
                int state_index = indices[row];
                if (state_index < 0 || state_index >= static_cast<int>(states_.size())) {
                    throw std::invalid_argument("TSP perturb step received invalid state index");
                }
                apply_probability_row(states_[static_cast<size_t>(state_index)], probability_rows[row], memory, random01[row]);
            }
        } else {
            py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (int row = 0; row < static_cast<int>(indices.size()); ++row) {
                try {
                    int state_index = indices[static_cast<size_t>(row)];
                    if (state_index < 0 || state_index >= static_cast<int>(states_.size())) {
                        throw std::invalid_argument("TSP perturb step received invalid state index");
                    }
                    apply_probability_row(
                        states_[static_cast<size_t>(state_index)],
                        probability_rows[static_cast<size_t>(row)],
                        memory,
                        random01[static_cast<size_t>(row)]);
                } catch (const std::exception& exc) {
                    errors[static_cast<size_t>(row)] = exc.what();
                } catch (...) {
                    errors[static_cast<size_t>(row)] = "unknown TSP perturb step error";
                }
            }
        }
        for (const auto& error : errors) {
            if (!error.empty()) {
                throw std::runtime_error(error);
            }
        }
    }

    void step_candidates(
        const std::vector<int>& indices,
        const std::vector<std::vector<int>>& candidates_batch,
        const std::vector<std::vector<double>>& probability_rows,
        const std::vector<std::tuple<int, int, double>>& memory_values,
        const std::vector<double>& random01) {
        if (indices.size() != candidates_batch.size() || indices.size() != probability_rows.size() || indices.size() != random01.size()) {
            throw std::invalid_argument("TSP perturb candidate step batch sizes do not match");
        }
        std::unordered_map<uint64_t, double> memory;
        memory.reserve(memory_values.size() * 2);
        for (const auto& entry : memory_values) {
            int u = 0;
            int v = 0;
            double value = 1.0;
            std::tie(u, v, value) = entry;
            if (value > 0.0 && std::isfinite(value)) {
                memory[tsp_directed_edge_key(u, v)] = value;
            }
        }
        std::vector<std::string> errors(indices.size());
        if (indices.size() < 256) {
            for (size_t row = 0; row < indices.size(); ++row) {
                int state_index = indices[row];
                if (state_index < 0 || state_index >= static_cast<int>(states_.size())) {
                    throw std::invalid_argument("TSP perturb candidate step received invalid state index");
                }
                auto actions = weighted_candidate_actions(
                    states_[static_cast<size_t>(state_index)],
                    candidates_batch[row],
                    probability_rows[row],
                    memory);
                auto action = sample_action(std::move(actions), random01[row]);
                apply_pick(states_[static_cast<size_t>(state_index)], action.node);
            }
        } else {
            py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (int row = 0; row < static_cast<int>(indices.size()); ++row) {
                try {
                    int state_index = indices[static_cast<size_t>(row)];
                    if (state_index < 0 || state_index >= static_cast<int>(states_.size())) {
                        throw std::invalid_argument("TSP perturb candidate step received invalid state index");
                    }
                    auto actions = weighted_candidate_actions(
                        states_[static_cast<size_t>(state_index)],
                        candidates_batch[static_cast<size_t>(row)],
                        probability_rows[static_cast<size_t>(row)],
                        memory);
                    auto action = sample_action(std::move(actions), random01[static_cast<size_t>(row)]);
                    apply_pick(states_[static_cast<size_t>(state_index)], action.node);
                } catch (const std::exception& exc) {
                    errors[static_cast<size_t>(row)] = exc.what();
                } catch (...) {
                    errors[static_cast<size_t>(row)] = "unknown TSP perturb candidate step error";
                }
            }
        }
        for (const auto& error : errors) {
            if (!error.empty()) {
                throw std::runtime_error(error);
            }
        }
    }

    py::tuple results() const {
        std::vector<std::vector<int>> routes;
        std::vector<std::vector<int>> changeds;
        std::vector<std::vector<std::pair<int, int>>> introduced_batch;
        std::vector<std::vector<std::pair<int, int>>> removed_batch;
        routes.reserve(states_.size());
        changeds.reserve(states_.size());
        introduced_batch.reserve(states_.size());
        removed_batch.reserve(states_.size());
        for (const auto& state : states_) {
            routes.push_back(state.route);
            changeds.emplace_back(state.changed.begin(), state.changed.end());
            introduced_batch.push_back(edge_set_to_pairs(state.introduced_edges));
            removed_batch.push_back(edge_set_to_pairs(state.removed_edges));
        }
        return py::make_tuple(routes, changeds, introduced_batch, removed_batch);
    }

private:
    std::vector<std::pair<int, int>> edge_set_to_pairs(const std::unordered_set<uint64_t>& edges) const {
        std::vector<std::pair<int, int>> pairs;
        pairs.reserve(edges.size());
        for (uint64_t key : edges) {
            pairs.push_back({static_cast<int>(key >> 32), static_cast<int>(key & 0xffffffffULL)});
        }
        return pairs;
    }

    double memory_weight(const std::unordered_map<uint64_t, double>& memory, int u, int v) const {
        auto it = memory.find(tsp_directed_edge_key(u, v));
        return it == memory.end() ? 1.0 : it->second;
    }

    std::vector<int> collect_legal_candidates(
        const TspPerturbBatchStateCpp& state,
        int start,
        int limit,
        bool global) const {
        std::vector<int> candidates;
        if (global) {
            for (int node : state.route) {
                if (state.visited.find(node) == state.visited.end()) {
                    candidates.push_back(node);
                }
            }
            return candidates;
        }
        if (limit <= 0) {
            return candidates;
        }
        auto current_it = data_.node_to_index.find(state.current);
        if (current_it == data_.node_to_index.end()) {
            throw std::invalid_argument("TSP perturb current node is not in node set");
        }
        const auto& order = neighbor_order_[static_cast<size_t>(current_it->second)];
        const int end = std::min(start + limit, static_cast<int>(order.size()));
        for (int i = start; i < end; ++i) {
            int node = order[static_cast<size_t>(i)];
            if (state.visited.find(node) == state.visited.end()) {
                candidates.push_back(node);
            }
        }
        return candidates;
    }

    std::vector<TspWeightedActionCpp> weighted_actions(
        const TspPerturbBatchStateCpp& state,
        const std::vector<int>& candidates,
        const std::vector<double>& row,
        const std::unordered_map<uint64_t, double>& memory) const {
        std::vector<TspWeightedActionCpp> actions;
        actions.reserve(candidates.size());
        for (int node : candidates) {
            auto it = data_.node_to_index.find(node);
            if (it == data_.node_to_index.end() || it->second >= static_cast<int>(row.size())) {
                throw std::invalid_argument("TSP probability row does not align with node set");
            }
            double d = raw_dist(data_, state.current, node);
            double weight = row[static_cast<size_t>(it->second)] / std::max(d, 1e-12);
            weight *= memory_weight(memory, state.current, node);
            if (weight > 0.0 && std::isfinite(weight)) {
                actions.push_back({node, weight});
            }
        }
        return actions;
    }

    std::vector<TspWeightedActionCpp> weighted_candidate_actions(
        const TspPerturbBatchStateCpp& state,
        const std::vector<int>& candidates,
        const std::vector<double>& candidate_probs,
        const std::unordered_map<uint64_t, double>& memory) const {
        if (candidates.size() != candidate_probs.size()) {
            throw std::invalid_argument("TSP candidate probabilities must align with candidates");
        }
        std::vector<TspWeightedActionCpp> actions;
        actions.reserve(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            int node = candidates[i];
            if (state.visited.find(node) != state.visited.end()) {
                continue;
            }
            double d = raw_dist(data_, state.current, node);
            double weight = candidate_probs[i] / std::max(d, 1e-12);
            weight *= memory_weight(memory, state.current, node);
            if (weight > 0.0 && std::isfinite(weight)) {
                actions.push_back({node, weight});
            }
        }
        return actions;
    }

    TspWeightedActionCpp sample_action(std::vector<TspWeightedActionCpp> actions, double random01) const {
        double sum = 0.0;
        for (const auto& action : actions) {
            sum += action.weight;
        }
        if (actions.empty() || sum <= 0.0 || !std::isfinite(sum)) {
            throw std::invalid_argument("neural TSP STAR produced no positive probability for legal k-NN actions");
        }
        double threshold = std::min(std::max(random01, 0.0), std::nextafter(1.0, 0.0)) * sum;
        double running = 0.0;
        for (const auto& action : actions) {
            running += action.weight;
            if (running >= threshold) {
                return action;
            }
        }
        return actions.back();
    }

    void apply_probability_row(
        TspPerturbBatchStateCpp& state,
        const std::vector<double>& row,
        const std::unordered_map<uint64_t, double>& memory,
        double random01) {
        auto primary = collect_legal_candidates(state, 0, k_, false);
        auto actions = weighted_actions(state, primary, row, memory);
        if (actions.empty()) {
            auto backup = collect_legal_candidates(state, k_, backup_k_, false);
            actions = weighted_actions(state, backup, row, memory);
        }
        if (actions.empty()) {
            auto global = collect_legal_candidates(state, 0, 0, true);
            actions = weighted_actions(state, global, row, memory);
        }
        auto action = sample_action(std::move(actions), random01);
        apply_pick(state, action.node);
    }

    void apply_pick(TspPerturbBatchStateCpp& state, int picked) {
        if (state.visited.find(picked) != state.visited.end()) {
            throw std::invalid_argument("TSP perturb picked an already visited node");
        }
        int current_pos = state.positions.at(state.current);
        int n = static_cast<int>(state.route.size());
        int successor_node = state.route[static_cast<size_t>((current_pos + 1) % n)];
        if (picked != successor_node) {
            int picked_pos = state.positions.at(picked);
            int picked_pred = state.route[static_cast<size_t>((picked_pos - 1 + n) % n)];
            int picked_succ = state.route[static_cast<size_t>((picked_pos + 1) % n)];
            tsp_add_symmetric_edge(state.introduced_edges, state.current, picked);
            tsp_add_symmetric_edge(state.introduced_edges, picked, successor_node);
            tsp_add_symmetric_edge(state.introduced_edges, picked_pred, picked_succ);
            tsp_add_symmetric_edge(state.removed_edges, state.current, successor_node);
            tsp_add_symmetric_edge(state.removed_edges, picked_pred, picked);
            tsp_add_symmetric_edge(state.removed_edges, picked, picked_succ);
            relocate_after_cpp(state.route, state.positions, state.current, picked);
            state.changed.insert(state.current);
            state.changed.insert(picked);
            state.changed.insert(picked_pred);
            state.new_edges += 1;
        }
        state.visited.insert(picked);
        state.prefix.push_back(picked);
        state.current = picked;
        state.active = state.new_edges < min_new_edges_ && state.visited.size() < state.route.size();
    }

    InstanceData data_;
    int min_new_edges_;
    int k_;
    int backup_k_;
    std::vector<std::vector<int>> neighbor_order_;
    std::vector<TspPerturbBatchStateCpp> states_;
};

struct CvrpWeightedActionCpp {
    int node;
    bool route_break;
    double weight;
};

class CvrpPerturbBatchCpp {
public:
    CvrpPerturbBatchCpp(
        const std::vector<int>& nodes,
        const std::vector<std::vector<double>>& coords,
        const std::vector<int>& demands,
        int depot,
        int capacity,
        const std::string& edge_weight_type,
        const std::vector<std::vector<int>>& routes,
        int samples,
        int min_new_edges,
        int k,
        int backup_k,
        uint64_t seed,
        const std::vector<std::vector<int>>& neighbor_order = {})
        : data_(make_instance(nodes, coords, demands, depot, capacity, edge_weight_type)),
          customers_(),
          min_new_edges_(min_new_edges),
          k_(std::max(0, k)),
          backup_k_(std::max(0, backup_k)),
          max_steps_(1),
          neighbor_order_(
              neighbor_order.empty()
                  ? build_neighbor_order(data_, std::max(1, k_ + backup_k_))
                  : neighbor_order) {
        if (samples <= 0) {
            throw std::invalid_argument("samples must be positive");
        }
        if (data_.nodes.empty() || data_.nodes.front() != data_.depot) {
            throw std::invalid_argument("CVRP neural probability rows expect the depot to be the first sorted node");
        }
        for (const auto& route : routes) {
            for (int node : route) {
                customers_.push_back(node);
            }
        }
        if (customers_.empty()) {
            return;
        }
        max_steps_ = std::max(1, static_cast<int>(customers_.size()) * 4);
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> start_dist(0, static_cast<int>(customers_.size()) - 1);
        states_.reserve(static_cast<size_t>(samples));
        for (int sample = 0; sample < samples; ++sample) {
            CvrpPerturbBatchStateCpp state;
            for (const auto& route : routes) {
                if (!route.empty()) {
                    state.routes.push_back(route);
                }
            }
            state.source_route_id = cvrp_source_route_ids_cpp(state.routes);
            state.source_edges = cvrp_source_edges_cpp(data_, state.routes);
            state.source_memory_edges = cvrp_directed_memory_edges_cpp(data_, state.routes);
            state.current = customers_[static_cast<size_t>(start_dist(rng))];
            int pos = -1;
            int route_index = cvrp_find_route_index(state.routes, state.current, &pos);
            state.remaining_capacity = std::max(
                0,
                data_.capacity - cvrp_route_prefix_load_cpp(data_, state.routes[static_cast<size_t>(route_index)], state.current));
            state.prefix.push_back(state.current);
            state.visited.insert(state.current);
            states_.push_back(std::move(state));
        }
    }

    py::tuple requests() {
        std::vector<int> indices;
        std::vector<std::vector<int>> prefixes;
        std::vector<int> remaining_capacities;
        for (int i = 0; i < static_cast<int>(states_.size()); ++i) {
            auto& state = states_[static_cast<size_t>(i)];
            if (state.done || state.new_edges_cross >= min_new_edges_
                || state.visited.size() >= customers_.size() || state.steps > max_steps_) {
                state.done = true;
                continue;
            }
            indices.push_back(i);
            prefixes.push_back(state.prefix);
            remaining_capacities.push_back(state.remaining_capacity);
        }
        return py::make_tuple(indices, prefixes, remaining_capacities);
    }

    void step(
        const std::vector<int>& indices,
        const std::vector<std::vector<double>>& probability_rows,
        const std::vector<std::tuple<int, int, double>>& memory_values,
        const std::vector<double>& random01) {
        if (indices.size() != probability_rows.size() || indices.size() != random01.size()) {
            throw std::invalid_argument("CVRP perturb step batch sizes do not match");
        }
        std::unordered_map<uint64_t, double> memory;
        memory.reserve(memory_values.size() * 2);
        for (const auto& entry : memory_values) {
            int u = 0;
            int v = 0;
            double value = 1.0;
            std::tie(u, v, value) = entry;
            if (value > 0.0 && std::isfinite(value)) {
                memory[cvrp_edge_key(u, v)] = value;
            }
        }
        std::vector<std::string> errors(indices.size());
        if (indices.size() < 256) {
            for (size_t row = 0; row < indices.size(); ++row) {
                int state_index = indices[row];
                if (state_index < 0 || state_index >= static_cast<int>(states_.size())) {
                    throw std::invalid_argument("CVRP perturb step received invalid state index");
                }
                apply_probability_row(states_[static_cast<size_t>(state_index)], probability_rows[row], memory, random01[row]);
            }
        } else {
            py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (int row = 0; row < static_cast<int>(indices.size()); ++row) {
                try {
                    int state_index = indices[static_cast<size_t>(row)];
                    if (state_index < 0 || state_index >= static_cast<int>(states_.size())) {
                        throw std::invalid_argument("CVRP perturb step received invalid state index");
                    }
                    apply_probability_row(
                        states_[static_cast<size_t>(state_index)],
                        probability_rows[static_cast<size_t>(row)],
                        memory,
                        random01[static_cast<size_t>(row)]);
                } catch (const std::exception& exc) {
                    errors[static_cast<size_t>(row)] = exc.what();
                } catch (...) {
                    errors[static_cast<size_t>(row)] = "unknown CVRP perturb step error";
                }
            }
        }
        for (const auto& error : errors) {
            if (!error.empty()) {
                throw std::runtime_error(error);
            }
        }
    }

    py::tuple results() const {
        std::vector<std::vector<std::vector<int>>> routes_batch;
        std::vector<std::vector<int>> changeds_batch;
        std::vector<std::vector<std::pair<int, int>>> introduced_batch;
        std::vector<std::vector<std::pair<int, int>>> removed_batch;
        routes_batch.reserve(states_.size());
        changeds_batch.reserve(states_.size());
        introduced_batch.reserve(states_.size());
        removed_batch.reserve(states_.size());
        for (const auto& state : states_) {
            std::vector<std::vector<int>> routes;
            for (const auto& route : state.routes) {
                if (!route.empty()) {
                    routes.push_back(route);
                }
            }
            routes_batch.push_back(routes);
            changeds_batch.emplace_back(state.changed.begin(), state.changed.end());

            auto final_edges = cvrp_directed_memory_edges_cpp(data_, routes);
            std::vector<std::pair<int, int>> introduced;
            std::vector<std::pair<int, int>> removed;
            for (uint64_t key : final_edges) {
                if (state.source_memory_edges.find(key) == state.source_memory_edges.end()) {
                    introduced.push_back({static_cast<int>(key >> 32), static_cast<int>(key & 0xffffffffULL)});
                }
            }
            for (uint64_t key : state.source_memory_edges) {
                if (final_edges.find(key) == final_edges.end()) {
                    removed.push_back({static_cast<int>(key >> 32), static_cast<int>(key & 0xffffffffULL)});
                }
            }
            introduced_batch.push_back(std::move(introduced));
            removed_batch.push_back(std::move(removed));
        }
        return py::make_tuple(routes_batch, changeds_batch, introduced_batch, removed_batch);
    }

private:
    double memory_weight(const std::unordered_map<uint64_t, double>& memory, int u, int v) const {
        auto it = memory.find(cvrp_edge_key(u, v));
        return it == memory.end() ? 1.0 : it->second;
    }

    double probability_for(const std::vector<double>& row, int node, bool route_break) const {
        auto it = data_.node_to_index.find(node);
        if (it == data_.node_to_index.end() || it->second == 0) {
            throw std::invalid_argument("CVRP probability requested for invalid customer");
        }
        const int split = static_cast<int>(data_.nodes.size()) - 1;
        const int offset = it->second - 1;
        const int index = (route_break ? split : 0) + offset;
        if (index < 0 || index >= static_cast<int>(row.size())) {
            throw std::invalid_argument("CVRP probability row does not align with node set");
        }
        double value = row[static_cast<size_t>(index)];
        return (value > 0.0 && std::isfinite(value)) ? value : 0.0;
    }

    void push_candidate(
        std::vector<CvrpWeightedActionCpp>& actions,
        const std::vector<double>& row,
        const std::unordered_map<uint64_t, double>& memory,
        int current,
        int node,
        bool route_break) const {
        double weight = probability_for(row, node, route_break);
        if (route_break) {
            weight *= memory_weight(memory, current, data_.depot) * memory_weight(memory, data_.depot, node);
        } else {
            weight *= memory_weight(memory, current, node);
        }
        if (weight > 0.0 && std::isfinite(weight)) {
            actions.push_back({node, route_break, weight});
        }
    }

    std::vector<CvrpWeightedActionCpp> collect_actions(
        const CvrpPerturbBatchStateCpp& state,
        const std::vector<double>& row,
        const std::unordered_map<uint64_t, double>& memory,
        int start,
        int limit) const {
        std::vector<CvrpWeightedActionCpp> actions;
        if (limit <= 0) {
            return actions;
        }
        auto current_it = data_.node_to_index.find(state.current);
        auto depot_it = data_.node_to_index.find(data_.depot);
        if (current_it == data_.node_to_index.end() || depot_it == data_.node_to_index.end()) {
            throw std::invalid_argument("CVRP perturb state current/depot is not in node set");
        }
        const auto& direct_order = neighbor_order_[static_cast<size_t>(current_it->second)];
        const auto& via_order = neighbor_order_[static_cast<size_t>(depot_it->second)];
        const int direct_end = std::min(start + limit, static_cast<int>(direct_order.size()));
        for (int i = start; i < direct_end; ++i) {
            int node = direct_order[static_cast<size_t>(i)];
            if (node == data_.depot || state.visited.find(node) != state.visited.end()) {
                continue;
            }
            if (demand_of(data_, node) <= state.remaining_capacity) {
                push_candidate(actions, row, memory, state.current, node, false);
            }
        }
        const int via_end = std::min(start + limit, static_cast<int>(via_order.size()));
        for (int i = start; i < via_end; ++i) {
            int node = via_order[static_cast<size_t>(i)];
            if (node == data_.depot || state.visited.find(node) != state.visited.end()) {
                continue;
            }
            if (demand_of(data_, node) <= data_.capacity) {
                push_candidate(actions, row, memory, state.current, node, true);
            }
        }
        return actions;
    }

    std::vector<CvrpWeightedActionCpp> collect_global_actions(
        const CvrpPerturbBatchStateCpp& state,
        const std::vector<double>& row,
        const std::unordered_map<uint64_t, double>& memory) const {
        std::vector<CvrpWeightedActionCpp> actions;
        for (int node : customers_) {
            if (state.visited.find(node) != state.visited.end()) {
                continue;
            }
            int demand = demand_of(data_, node);
            if (demand <= state.remaining_capacity) {
                push_candidate(actions, row, memory, state.current, node, false);
            }
            if (demand <= data_.capacity) {
                push_candidate(actions, row, memory, state.current, node, true);
            }
        }
        return actions;
    }

    CvrpWeightedActionCpp sample_action(std::vector<CvrpWeightedActionCpp> actions, double random01) const {
        double sum = 0.0;
        for (const auto& action : actions) {
            sum += action.weight;
        }
        if (actions.empty() || sum <= 0.0 || !std::isfinite(sum)) {
            throw std::invalid_argument("neural CVRP STAR produced no positive probability for legal k-NN actions");
        }
        double threshold = std::min(std::max(random01, 0.0), std::nextafter(1.0, 0.0)) * sum;
        double running = 0.0;
        for (const auto& action : actions) {
            running += action.weight;
            if (running >= threshold) {
                return action;
            }
        }
        return actions.back();
    }

    void apply_probability_row(
        CvrpPerturbBatchStateCpp& state,
        const std::vector<double>& row,
        const std::unordered_map<uint64_t, double>& memory,
        double random01) {
        std::vector<CvrpWeightedActionCpp> actions = collect_actions(state, row, memory, 0, k_);
        if (actions.empty()) {
            actions = collect_actions(state, row, memory, k_, backup_k_);
        }
        if (actions.empty()) {
            actions = collect_global_actions(state, row, memory);
        }
        CvrpWeightedActionCpp action = sample_action(std::move(actions), random01);
        apply_decision(state, action.node, action.route_break);
    }

    void apply_decision(CvrpPerturbBatchStateCpp& state, int picked, bool route_break) {
        if (state.visited.find(picked) != state.visited.end()) {
            throw std::invalid_argument("CVRP perturb picked an already visited customer");
        }
        int pos = -1;
        int route_index = cvrp_find_route_index(state.routes, state.current, &pos);
        int successor = (pos + 1 < static_cast<int>(state.routes[static_cast<size_t>(route_index)].size()))
            ? state.routes[static_cast<size_t>(route_index)][static_cast<size_t>(pos + 1)]
            : data_.depot;
        int demand = demand_of(data_, picked);
        bool force_break = route_break || demand > state.remaining_capacity;
        int transition_to = force_break ? data_.depot : picked;
        bool is_new = state.source_edges.find(cvrp_undirected_customer_edge_key(data_, state.current, transition_to)) == state.source_edges.end();

        if (force_break) {
            cvrp_split_after_current_cpp(state.routes, state.current);
            state.changed.insert(state.current);
            if (is_new) {
                state.new_edges_cross += 1;
            }
            int target_pos = -1;
            int target_route_index = cvrp_find_route_index(state.routes, state.current, &target_pos);
            int insert_route_index = std::min(target_route_index + 1, static_cast<int>(state.routes.size()) - 1);
            int route_successor = state.routes[static_cast<size_t>(insert_route_index)].empty()
                ? -1
                : state.routes[static_cast<size_t>(insert_route_index)].front();
            if (picked != route_successor) {
                cvrp_relocate_after_current_route_cpp(state.routes, state.current, picked);
                state.changed.insert(picked);
                if (route_successor >= 0) {
                    state.changed.insert(route_successor);
                }
            }
            state.remaining_capacity = std::max(0, data_.capacity - demand);
        } else if (picked == successor) {
            state.remaining_capacity -= demand;
        } else {
            state.changed.insert(state.current);
            state.changed.insert(picked);
            if (successor != data_.depot) {
                state.changed.insert(successor);
            }
            cvrp_relocate_customer_cpp(data_, state.routes, state.current, picked);
            int picked_pos = -1;
            int picked_route = cvrp_find_route_index(state.routes, picked, &picked_pos);
            state.remaining_capacity = std::max(
                0,
                data_.capacity - cvrp_route_prefix_load_cpp(data_, state.routes[static_cast<size_t>(picked_route)], picked));
            if (is_new) {
                int u_route = state.source_route_id.count(state.current) ? state.source_route_id.at(state.current) : -1;
                int v_route = state.source_route_id.count(picked) ? state.source_route_id.at(picked) : -2;
                if (state.current == data_.depot || u_route != v_route) {
                    state.new_edges_cross += 1;
                }
            }
        }

        state.visited.insert(picked);
        state.prefix.push_back(picked);
        state.current = picked;
        state.steps += 1;
    }

    InstanceData data_;
    std::vector<int> customers_;
    int min_new_edges_;
    int k_;
    int backup_k_;
    int max_steps_;
    std::vector<std::vector<int>> neighbor_order_;
    std::vector<CvrpPerturbBatchStateCpp> states_;
};

class STARCpp {
public:
    STARCpp(
        const std::vector<int>& nodes,
        const std::vector<std::vector<double>>& coords,
        const std::vector<int>& demands,
        int depot,
        int capacity,
        const std::string& edge_weight_type)
        : data_(make_instance(nodes, coords, demands, depot, capacity, edge_weight_type)),
          kdtree_(instance_points(data_), /*round_distances=*/false),
          cached_total_(0),
          build_count_(1),
          edge_weight_type_(edge_weight_type) {}

    std::vector<std::vector<int>> neighbor_order(int total) {
        ensure_neighbor_total(total);
        return slice_neighbor_order(total);
    }

    std::vector<int> neighbor_row(int node, int total) {
        ensure_neighbor_total(total);
        auto it = data_.node_to_index.find(node);
        if (it == data_.node_to_index.end()) {
            throw std::invalid_argument("STAR neighbor_row requested unknown node");
        }
        const int row_limit = std::min(std::max(0, total), cached_total_);
        const auto& row = neighbor_order_[static_cast<size_t>(it->second)];
        return std::vector<int>(row.begin(), row.begin() + std::min(row_limit, static_cast<int>(row.size())));
    }

    int cached_total() const {
        return cached_total_;
    }

    int build_count() const {
        return build_count_;
    }

    std::vector<int> greedy_tsp_multi_start(int starts, int k) const {
        return greedy_tsp_multi_start_with_data(data_, kdtree_, starts, k);
    }

    std::vector<int> run_nearest_tsp(
        std::vector<int> initial_route,
        int iterations,
        int min_new_edges,
        int refine_k,
        int k,
        int backup_k,
        uint64_t seed,
        bool refine) {
        if (iterations < 0) {
            throw std::invalid_argument("iterations must be non-negative");
        }
        if (initial_route.size() <= 2 || iterations == 0 || min_new_edges == 0) {
            return initial_route;
        }
        ensure_neighbor_total(std::max({0, k + backup_k, refine ? refine_k : 0}));
        std::mt19937_64 rng(seed);
        std::vector<int> best = initial_route;
        double best_cost = tsp_route_cost_cpp(data_, best);

        for (int iteration = 0; iteration < iterations; ++iteration) {
            std::vector<int> candidate = best;
            std::unordered_map<int, int> positions;
            rebuild_route_positions(candidate, positions);
            std::unordered_map<int, bool> visited;
            visited.reserve(candidate.size() * 2);
            for (int node : candidate) {
                visited[node] = false;
            }

            std::uniform_int_distribution<int> start_dist(0, static_cast<int>(candidate.size()) - 1);
            int current = candidate[start_dist(rng)];
            visited[current] = true;
            int visited_count = 1;
            int new_edges = 0;
            std::vector<int> changed;
            std::unordered_map<int, bool> in_changed;

            auto add_changed = [&](int node) {
                if (in_changed.find(node) == in_changed.end()) {
                    in_changed[node] = true;
                    changed.push_back(node);
                }
            };

            while (new_edges < min_new_edges && visited_count < static_cast<int>(candidate.size())) {
                int current_pos = positions.at(current);
                int successor_node = candidate[(current_pos + 1) % static_cast<int>(candidate.size())];
                int picked = select_nearest_STAR_node_cpp(data_, neighbor_order_, current, visited, k, backup_k, candidate);
                auto picked_visited = visited.find(picked);
                if (picked_visited == visited.end() || picked_visited->second) {
                    throw std::runtime_error("nearest STAR selected invalid visited node");
                }

                if (picked != successor_node) {
                    int picked_pos = positions.at(picked);
                    int picked_pred = candidate[(picked_pos - 1 + static_cast<int>(candidate.size())) % static_cast<int>(candidate.size())];
                    relocate_after_cpp(candidate, positions, current, picked);
                    add_changed(current);
                    add_changed(picked);
                    add_changed(picked_pred);
                    add_changed(successor_node);
                    ++new_edges;
                }

                visited[picked] = true;
                ++visited_count;
                current = picked;
            }

            if (refine && !changed.empty()) {
                candidate = refine_tsp_srr_with_data(data_, std::move(candidate), changed, refine_k, &neighbor_order_);
            }
            double candidate_cost = tsp_route_cost_cpp(data_, candidate);
            if (candidate_cost <= best_cost + 1e-9) {
                best = candidate;
                best_cost = candidate_cost;
            }
        }
        return best;
    }

    std::vector<int> run_nearest_tsp_samples(
        std::vector<int> initial_route,
        int iterations,
        int min_new_edges,
        int samples,
        int refine_k,
        int k,
        int backup_k,
        uint64_t seed,
        bool refine) {
        if (iterations < 0) {
            throw std::invalid_argument("iterations must be non-negative");
        }
        if (samples <= 0) {
            throw std::invalid_argument("samples must be positive");
        }
        if (initial_route.size() <= 2 || iterations == 0 || min_new_edges == 0) {
            return initial_route;
        }

        ensure_neighbor_total(std::max({0, k + backup_k, refine ? refine_k : 0}));
        std::vector<int> best = initial_route;
        double best_cost = tsp_route_cost_cpp(data_, best);

        for (int iteration = 0; iteration < iterations; ++iteration) {
            std::vector<std::vector<int>> candidates(samples);
            std::vector<double> costs(samples, 0.0);
            {
                py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                for (int sample = 0; sample < samples; ++sample) {
                    uint64_t sample_seed = seed
                        ^ (static_cast<uint64_t>(iteration + 1) * 0x9E3779B97F4A7C15ULL)
                        ^ (static_cast<uint64_t>(sample + 1) * 0xBF58476D1CE4E5B9ULL);
                    auto perturbed = perturb_nearest_heuristic_tsp_sample(
                        data_, neighbor_order_, best, min_new_edges, k, backup_k, sample_seed);
                    if (refine && !perturbed.second.empty()) {
                        candidates[sample] = refine_tsp_srr_with_data(data_, std::move(perturbed.first), perturbed.second, refine_k, &neighbor_order_);
                    } else {
                        candidates[sample] = std::move(perturbed.first);
                    }
                    costs[sample] = tsp_route_cost_cpp(data_, candidates[sample]);
                }
            }

            int best_sample = -1;
            double iteration_best_cost = best_cost;
            for (int sample = 0; sample < samples; ++sample) {
                if (costs[sample] <= iteration_best_cost + 1e-9) {
                    iteration_best_cost = costs[sample];
                    best_sample = sample;
                }
            }
            if (best_sample >= 0) {
                best = std::move(candidates[best_sample]);
                best_cost = iteration_best_cost;
            }
        }
        return best;
    }

    std::vector<int> refine_tsp(std::vector<int> route, const std::vector<int>& changed, int refine_k) {
        ensure_neighbor_total(refine_k);
        return refine_tsp_srr_with_data(data_, std::move(route), changed, refine_k, &neighbor_order_);
    }

    py::tuple refine_tsp_batch(
        const std::vector<std::vector<int>>& routes,
        const std::vector<std::vector<int>>& changeds,
        int refine_k) {
        if (routes.size() != changeds.size()) {
            throw std::invalid_argument("routes and changeds must have the same length");
        }
        ensure_neighbor_total(refine_k);
        std::vector<std::vector<int>> refined(routes.size());
        std::vector<double> costs(routes.size(), 0.0);

        {
            py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (int i = 0; i < static_cast<int>(routes.size()); ++i) {
                refined[static_cast<size_t>(i)] = refine_tsp_srr_with_data(data_, routes[static_cast<size_t>(i)], changeds[static_cast<size_t>(i)], refine_k, &neighbor_order_);
                costs[static_cast<size_t>(i)] = tsp_route_cost_cpp(data_, refined[static_cast<size_t>(i)]);
            }
        }

        return py::make_tuple(refined, costs);
    }

    TspPerturbBatchCpp tsp_perturb_batch(
        const std::vector<int>& route,
        int samples,
        int min_new_edges,
        int k,
        int backup_k,
        uint64_t seed) {
        ensure_neighbor_total(std::max(1, std::max(0, k) + std::max(0, backup_k)));
        return TspPerturbBatchCpp(
            data_.nodes,
            coords_payload(),
            edge_weight_type_,
            route,
            samples,
            min_new_edges,
            k,
            backup_k,
            seed,
            neighbor_order_);
    }

    CvrpPerturbBatchCpp cvrp_perturb_batch(
        const std::vector<std::vector<int>>& routes,
        int samples,
        int min_new_edges,
        int k,
        int backup_k,
        uint64_t seed) {
        ensure_neighbor_total(std::max(1, std::max(0, k) + std::max(0, backup_k)));
        return CvrpPerturbBatchCpp(
            data_.nodes,
            coords_payload(),
            data_.demands,
            data_.depot,
            data_.capacity,
            edge_weight_type_,
            routes,
            samples,
            min_new_edges,
            k,
            backup_k,
            seed,
            neighbor_order_);
    }

private:
    void ensure_neighbor_total(int total) {
        const int row_limit = std::min(std::max(0, total), std::max(0, static_cast<int>(data_.nodes.size()) - 1));
        if (row_limit <= cached_total_) {
            return;
        }
        neighbor_order_ = build_neighbor_order_from_tree(data_, kdtree_, row_limit);
        cached_total_ = row_limit;
    }

    std::vector<std::vector<int>> slice_neighbor_order(int total) const {
        const int row_limit = std::min(std::max(0, total), cached_total_);
        std::vector<std::vector<int>> result;
        result.reserve(neighbor_order_.size());
        for (const auto& row : neighbor_order_) {
            result.emplace_back(row.begin(), row.begin() + std::min(row_limit, static_cast<int>(row.size())));
        }
        return result;
    }

    std::vector<std::vector<double>> coords_payload() const {
        std::vector<std::vector<double>> coords;
        coords.reserve(data_.coords.size());
        for (const auto& coord : data_.coords) {
            coords.push_back({coord.x, coord.y});
        }
        return coords;
    }

    InstanceData data_;
    KDTree kdtree_;
    int cached_total_;
    int build_count_;
    std::vector<std::vector<int>> neighbor_order_;
    std::string edge_weight_type_;
};

}  // namespace

PYBIND11_MODULE(_STAR, m) {
    m.doc() = "C++ kernels for STAR: Scoped Test-time Adapt and Refine";
    m.def("refine_tsp_srr", &refine_tsp_srr, py::arg("nodes"), py::arg("coords"), py::arg("edge_weight_type"),
          py::arg("route"), py::arg("changed"), py::arg("refine_k"));
    m.def("refine_tsp_srr_with_order", &refine_tsp_srr_with_order, py::arg("nodes"), py::arg("coords"),
          py::arg("edge_weight_type"), py::arg("route"), py::arg("changed"), py::arg("refine_k"),
          py::arg("neighbor_order"));
    m.def("refine_tsp_srr_batch", &refine_tsp_srr_batch, py::arg("nodes"), py::arg("coords"), py::arg("edge_weight_type"),
          py::arg("routes"), py::arg("changeds"), py::arg("refine_k"));
    m.def("refine_tsp_srr_batch_with_order", &refine_tsp_srr_batch_with_order, py::arg("nodes"), py::arg("coords"),
          py::arg("edge_weight_type"), py::arg("routes"), py::arg("changeds"), py::arg("refine_k"),
          py::arg("neighbor_order"));
    m.def("refine_cvrp_srr", &refine_cvrp_srr, py::arg("nodes"), py::arg("coords"), py::arg("demands"),
          py::arg("depot"), py::arg("capacity"), py::arg("edge_weight_type"), py::arg("routes"),
          py::arg("changed"), py::arg("refine_k"));
    m.def("refine_cvrp_srr_batch", &refine_cvrp_srr_batch, py::arg("nodes"), py::arg("coords"), py::arg("demands"),
          py::arg("depot"), py::arg("capacity"), py::arg("edge_weight_type"), py::arg("routes_batch"),
          py::arg("changeds"), py::arg("refine_k"));
    m.def("build_tsp_neighbor_order", &build_tsp_neighbor_order, py::arg("nodes"), py::arg("coords"),
          py::arg("edge_weight_type"), py::arg("total"));
    py::class_<STARCpp>(m, "STAR")
        .def(py::init<const std::vector<int>&, const std::vector<std::vector<double>>&,
                      const std::vector<int>&, int, int, const std::string&>(),
             py::arg("nodes"), py::arg("coords"), py::arg("demands"), py::arg("depot"),
             py::arg("capacity"), py::arg("edge_weight_type"))
        .def("neighbor_order", &STARCpp::neighbor_order, py::arg("total"))
        .def("neighbor_row", &STARCpp::neighbor_row, py::arg("node"), py::arg("total"))
        .def("cached_total", &STARCpp::cached_total)
        .def("build_count", &STARCpp::build_count)
        .def("greedy_tsp_multi_start", &STARCpp::greedy_tsp_multi_start, py::arg("starts"), py::arg("k"))
        .def("run_nearest_tsp", &STARCpp::run_nearest_tsp, py::arg("initial_route"), py::arg("iterations"),
             py::arg("min_new_edges"), py::arg("refine_k"), py::arg("k"), py::arg("backup_k"), py::arg("seed"),
             py::arg("refine") = true)
        .def("run_nearest_tsp_samples", &STARCpp::run_nearest_tsp_samples, py::arg("initial_route"),
             py::arg("iterations"), py::arg("min_new_edges"), py::arg("samples"), py::arg("refine_k"),
             py::arg("k"), py::arg("backup_k"), py::arg("seed"), py::arg("refine") = true)
        .def("refine_tsp", &STARCpp::refine_tsp, py::arg("route"), py::arg("changed"), py::arg("refine_k"))
        .def("refine_tsp_batch", &STARCpp::refine_tsp_batch, py::arg("routes"), py::arg("changeds"), py::arg("refine_k"))
        .def("tsp_perturb_batch", &STARCpp::tsp_perturb_batch, py::arg("route"), py::arg("samples"),
             py::arg("min_new_edges"), py::arg("k"), py::arg("backup_k"), py::arg("seed"))
        .def("cvrp_perturb_batch", &STARCpp::cvrp_perturb_batch, py::arg("routes"), py::arg("samples"),
             py::arg("min_new_edges"), py::arg("k"), py::arg("backup_k"), py::arg("seed"));
    m.def("greedy_tsp_multi_start", &greedy_tsp_multi_start, py::arg("nodes"), py::arg("coords"),
          py::arg("edge_weight_type"), py::arg("starts"), py::arg("k"));
    m.def("select_tsp_knn_action", &select_tsp_knn_action, py::arg("nodes"), py::arg("coords"),
          py::arg("edge_weight_type"), py::arg("current"), py::arg("candidates"), py::arg("probabilities"),
          py::arg("memory_weights"), py::arg("k"), py::arg("backup_k"), py::arg("random01"));
    m.def("select_tsp_candidate_actions_batch", &select_tsp_candidate_actions_batch, py::arg("nodes"),
          py::arg("currents"), py::arg("candidates_batch"), py::arg("probabilities_batch"),
          py::arg("memory_weights_batch"), py::arg("random01_batch"));
    m.def("select_tsp_candidate_weight_actions_batch", &select_tsp_candidate_weight_actions_batch,
          py::arg("candidates_batch"), py::arg("weights_batch"), py::arg("random01_batch"));
    m.def("run_STAR_nearest_tsp", &run_STAR_nearest_tsp, py::arg("nodes"), py::arg("coords"),
          py::arg("edge_weight_type"), py::arg("initial_route"), py::arg("iterations"), py::arg("min_new_edges"),
          py::arg("refine_k"), py::arg("k"), py::arg("backup_k"), py::arg("seed"), py::arg("refine") = true);
    m.def("run_STAR_nearest_tsp_samples", &run_STAR_nearest_tsp_samples, py::arg("nodes"), py::arg("coords"),
          py::arg("edge_weight_type"), py::arg("initial_route"), py::arg("iterations"), py::arg("min_new_edges"),
          py::arg("samples"), py::arg("refine_k"), py::arg("k"), py::arg("backup_k"), py::arg("seed"),
          py::arg("refine") = true);
    py::class_<TspPerturbBatchCpp>(m, "TspPerturbBatch")
        .def(py::init<const std::vector<int>&, const std::vector<std::vector<double>>&,
                      const std::string&, const std::vector<int>&, int, int, int, int, uint64_t,
                      const std::vector<std::vector<int>>&>(),
             py::arg("nodes"), py::arg("coords"), py::arg("edge_weight_type"), py::arg("route"),
             py::arg("samples"), py::arg("min_new_edges"), py::arg("k"), py::arg("backup_k"), py::arg("seed"),
             py::arg("neighbor_order") = std::vector<std::vector<int>>())
        .def("requests", &TspPerturbBatchCpp::requests)
        .def("step", &TspPerturbBatchCpp::step, py::arg("indices"), py::arg("probability_rows"),
             py::arg("memory_values"), py::arg("random01"))
        .def("step_candidates", &TspPerturbBatchCpp::step_candidates, py::arg("indices"), py::arg("candidates_batch"),
             py::arg("probability_rows"), py::arg("memory_values"), py::arg("random01"))
        .def("results", &TspPerturbBatchCpp::results);
    py::class_<CvrpPerturbBatchCpp>(m, "CvrpPerturbBatch")
        .def(py::init<const std::vector<int>&, const std::vector<std::vector<double>>&,
                      const std::vector<int>&, int, int, const std::string&,
                      const std::vector<std::vector<int>>&, int, int, int, int, uint64_t,
                      const std::vector<std::vector<int>>&>(),
             py::arg("nodes"), py::arg("coords"), py::arg("demands"), py::arg("depot"), py::arg("capacity"),
             py::arg("edge_weight_type"), py::arg("routes"), py::arg("samples"), py::arg("min_new_edges"),
             py::arg("k"), py::arg("backup_k"), py::arg("seed"),
             py::arg("neighbor_order") = std::vector<std::vector<int>>())
        .def("requests", &CvrpPerturbBatchCpp::requests)
        .def("step", &CvrpPerturbBatchCpp::step, py::arg("indices"), py::arg("probability_rows"),
             py::arg("memory_values"), py::arg("random01"))
        .def("results", &CvrpPerturbBatchCpp::results);
}
