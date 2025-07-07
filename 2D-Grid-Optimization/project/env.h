#pragma once

#include <vector>
#include <random>
#include <tuple>
#include <unordered_set>

#include "common.h"
#include "utils.h"

class EnhancedCityPlanningEnv {
public:
    EnhancedCityPlanningEnv(int rows, int cols);

    State reset();
    std::tuple<State, double, bool> step(int action_idx);
    std::vector<int> get_valid_actions();
    int get_num_connected_houses() const;
    const std::vector<std::vector<int>>& get_grid() const;

private:
    int rows_, cols_;
    std::vector<std::vector<int>> grid_;
    std::vector<PlacedBuilding> placed_buildings_info_;
    std::vector<Pos> kontor_coords_list_;
    std::vector<Pos> tc_coords_list_;
    Pos tc_center_;
    int num_houses_placed_;
    int consecutive_successes_;
    int episode_length_;
    Pos last_house_pos_;
    std::vector<Pos> house_positions_;
    std::vector<std::vector<bool>> road_reach_from_kontor_cache_;
    std::vector<std::vector<bool>> road_reach_from_tc_cache_;
    bool episode_end_processing_;
    std::mt19937 rng_;

    std::vector<Pos> cached_valid_positions_;
    std::unordered_set<int> cached_valid_actions_;
    bool cache_valid_;

    int max_actions_per_episode_;

    State _get_reduced_state();
    bool _is_block_aligned(int r, int c) const;
    int _count_potential_block_extensions(int r, int c);
    int _fill_gaps_greedy();
    bool _is_valid_position(int r, int c) const;
    std::vector<int> _get_valid_actions();
    std::vector<int> _get_valid_actions_optimized();
    Pos _action_to_position(int action_idx);
    bool _check_connectivity_fast(int new_house_r, int new_house_c);
    bool _check_all_buildings_connectivity(int new_house_r = -1, int new_house_c = -1);
    int _count_adjacent_roads(int r, int c, int h, int w) const;
    int _calculate_road_exposure(int r, int c) const;
    int _count_adjacent_houses(int r, int c) const;
    double _calculate_reward_shaping(int r, int c);
    void _update_connectivity_cache();
    void _invalidate_cache();
};