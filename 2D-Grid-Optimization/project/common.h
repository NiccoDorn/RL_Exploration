#pragma once

#include <vector>
#include <map>
#include <cmath>

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
    bool operator<(const Pos& other) const {
        if (r != other.r) return r < other.r;
        return c < other.c;
    }
    bool operator==(const Pos& other) const {
        return r == other.r && c == other.c;
    }
    bool operator!=(const Pos& other) const {
        return !(*this == other);
    }
};

struct PosHash {
    size_t operator()(const Pos& p) const {
        return std::hash<int>()(p.r) ^ (std::hash<int>()(p.c) << 1);
    }
};

struct PlacedBuilding { int r, c, h, w, type; };

struct State {
    int num_houses_placed;
    int block_aligned_positions;
    int consecutive_successes;
    int episode_length;

    bool operator<(const State& other) const {
        if (num_houses_placed != other.num_houses_placed) return num_houses_placed < other.num_houses_placed;
        if (block_aligned_positions != other.block_aligned_positions) return block_aligned_positions < other.block_aligned_positions;
        if (consecutive_successes != other.consecutive_successes) return consecutive_successes < other.consecutive_successes;
        return episode_length < other.episode_length;
    }
    
    bool operator==(const State& other) const {
        return num_houses_placed == other.num_houses_placed &&
            block_aligned_positions == other.block_aligned_positions &&
            consecutive_successes == other.consecutive_successes &&
            episode_length == other.episode_length;
    }
};