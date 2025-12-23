#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <functional>

const int EMPTY = 0;
const int HOUSE = 1;
const int ROAD = 2;
const int TOWN_CENTER = 3;
const int TRADING_POST = 4;

const int HOUSE_H = 3;
const int HOUSE_W = 3;
const int TOWN_CENTER_H = 7;
const int TOWN_CENTER_W = 5;
const int TRADING_POST_H = 1;
const int TRADING_POST_W = 5;
const int TOWN_CENTER_RADIUS_SQ = 26 * 26;
const bool ENABLE_REWARD_SHAPING = true;
const double REWARD_PLACEMENT_SUCCESS = 10.0;
const double PENALTY_INVALID_PLACEMENT = -1.0;
const double PENALTY_DISCONNECTED_HOUSE = -50.0;
const double REWARD_CONSECUTIVE_SUCCESS_BONUS = 5.0;
const double REWARD_HIGH_HOUSE_COUNT_BONUS = 10.0;
const double REWARD_GREEDY_ADDITIONAL_HOUSE = 5.0;
const double REWARD_BLOCK_ALIGNED = 15.0;
const double REWARD_EXTENSION_BONUS_PER_POTENTIAL = 1.0;
const double REWARD_EPISODE_LENGTH_BONUS = 0.1;
const double PENALTY_ROAD_EXPOSURE_PER_SIDE = -1.0;
const int ROAD_EXPOSURE_PENALTY_START_THRESHOLD = 10;
const double REWARD_MULTIPLE_HOUSE_ALIGNMENT = 10.0;

struct Dimensions { int h, w; };

struct Pos {
    int r, c;

    inline bool operator<(const Pos& other) const noexcept {
        if (r != other.r) return r < other.r;
        return c < other.c;
    }

    inline bool operator==(const Pos& other) const noexcept {
        return r == other.r && c == other.c;
    }

    inline bool operator!=(const Pos& other) const noexcept {
        return !(*this == other);
    }
};

struct PosHash {
    inline size_t operator()(const Pos& p) const noexcept {
        return std::hash<int>()(p.r) ^ (std::hash<int>()(p.c) << 1);
    }
};

struct PlacedBuilding { int r, c, h, w, type; };

struct State {
    int num_houses_placed;
    int block_aligned_positions;
    int consecutive_successes;
    int episode_length;

    inline bool operator<(const State& other) const noexcept {
        if (num_houses_placed != other.num_houses_placed) return num_houses_placed < other.num_houses_placed;
        if (block_aligned_positions != other.block_aligned_positions) return block_aligned_positions < other.block_aligned_positions;
        if (consecutive_successes != other.consecutive_successes) return consecutive_successes < other.consecutive_successes;
        return episode_length < other.episode_length;
    }

    inline bool operator==(const State& other) const noexcept {
        return num_houses_placed == other.num_houses_placed &&
            block_aligned_positions == other.block_aligned_positions &&
            consecutive_successes == other.consecutive_successes &&
            episode_length == other.episode_length;
    }
};

// Hash function for State (for unordered_map)
struct StateHash {
    inline size_t operator()(const State& s) const noexcept {
        size_t h1 = std::hash<int>()(s.num_houses_placed);
        size_t h2 = std::hash<int>()(s.block_aligned_positions);
        size_t h3 = std::hash<int>()(s.consecutive_successes);
        size_t h4 = std::hash<int>()(s.episode_length);
        return h1 ^ (h2 << 8) ^ (h3 << 16) ^ (h4 << 24);
    }
};

// Union-Find data structure for fast connectivity checks
class UnionFind {
public:
    UnionFind(int size) : parent_(size), rank_(size, 0), component_size_(size, 1) {
        for (int i = 0; i < size; ++i) {
            parent_[i] = i;
        }
    }

    inline int find(int x) const noexcept {
        if (parent_[x] != x) {
            parent_[x] = find(parent_[x]); // Path compression
        }
        return parent_[x];
    }

    inline bool unite(int x, int y) noexcept {
        int root_x = find(x);
        int root_y = find(y);

        if (root_x == root_y) return false;

        // Union by rank
        if (rank_[root_x] < rank_[root_y]) {
            parent_[root_x] = root_y;
            component_size_[root_y] += component_size_[root_x];
        } else if (rank_[root_x] > rank_[root_y]) {
            parent_[root_y] = root_x;
            component_size_[root_x] += component_size_[root_y];
        } else {
            parent_[root_y] = root_x;
            component_size_[root_x] += component_size_[root_y];
            rank_[root_x]++;
        }
        return true;
    }

    inline bool connected(int x, int y) noexcept {
        return find(x) == find(y);
    }

    inline int get_component_size(int x) const noexcept {
        return component_size_[find(x)];
    }

    void reset(int size) {
        parent_.resize(size);
        rank_.assign(size, 0);
        component_size_.assign(size, 1);
        for (int i = 0; i < size; ++i) {
            parent_[i] = i;
        }
    }

private:
    mutable std::vector<int> parent_;
    std::vector<int> rank_;
    std::vector<int> component_size_;
};

// Spatial hashing for fast neighborhood queries
class SpatialHash {
public:
    SpatialHash(int cell_size_r, int cell_size_c)
        : cell_size_r_(cell_size_r), cell_size_c_(cell_size_c) {}

    inline int hash_position(int r, int c) const noexcept {
        int grid_r = r / cell_size_r_;
        int grid_c = c / cell_size_c_;
        return grid_r * 10000 + grid_c;
    }

    void insert(const Pos& pos) {
        int h = hash_position(pos.r, pos.c);
        grid_[h].push_back(pos);
    }

    void clear() {
        grid_.clear();
    }

    const std::vector<Pos>* get_nearby(int r, int c) const {
        int h = hash_position(r, c);
        auto it = grid_.find(h);
        return (it != grid_.end()) ? &it->second : nullptr;
    }

    std::vector<Pos> get_all_nearby(int r, int c, int search_radius = 1) const {
        std::vector<Pos> result;
        int grid_r = r / cell_size_r_;
        int grid_c = c / cell_size_c_;

        for (int dr = -search_radius; dr <= search_radius; ++dr) {
            for (int dc = -search_radius; dc <= search_radius; ++dc) {
                int h = (grid_r + dr) * 10000 + (grid_c + dc);
                auto it = grid_.find(h);
                if (it != grid_.end()) {
                    result.insert(result.end(), it->second.begin(), it->second.end());
                }
            }
        }
        return result;
    }

private:
    std::unordered_map<int, std::vector<Pos>> grid_;
    int cell_size_r_;
    int cell_size_c_;
};