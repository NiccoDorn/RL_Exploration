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
    int get_num_connected_houses() const noexcept;
    const std::vector<int>& get_grid() const noexcept;
    int get_rows() const noexcept { return rows_; }
    int get_cols() const noexcept { return cols_; }

private:
    int rows_, cols_;
    std::vector<int> grid_;
    std::vector<PlacedBuilding> placed_buildings_info_;
    std::vector<Pos> kontor_coords_list_;
    std::vector<Pos> tc_coords_list_;
    Pos tc_center_;
    int num_houses_placed_;
    int consecutive_successes_;
    int episode_length_;
    Pos last_house_pos_;

    SpatialHash house_spatial_hash_;
    std::vector<Pos> house_positions_;

    std::unordered_set<int> valid_action_set_;
    std::vector<int> valid_actions_cache_;
    bool valid_actions_dirty_;

    // BFS caching for performance
    mutable std::vector<std::vector<bool>> cached_tc_reachable_;
    mutable std::vector<std::vector<bool>> cached_kontor_reachable_;
    mutable bool bfs_cache_valid_;

    bool episode_end_processing_;
    std::mt19937 rng_;
    int max_actions_per_episode_;

    inline int grid_index(int r, int c) const noexcept { return r * cols_ + c; }
    inline int& at(int r, int c) noexcept { return grid_[grid_index(r, c)]; }
    inline const int& at(int r, int c) const noexcept { return grid_[grid_index(r, c)]; }

    State _get_reduced_state() const;
    bool _is_block_aligned(int r, int c) const;
    int _count_potential_block_extensions(int r, int c) const;
    int _fill_gaps_greedy();
    bool _is_valid_position(int r, int c) const;
    std::vector<int> _get_valid_actions();
    void _recompute_valid_actions();
    Pos _action_to_position(int action_idx) const noexcept;
    void _update_bfs_caches() const;
    bool _check_all_buildings_connectivity(int new_house_r = -1, int new_house_c = -1) const;
    int _count_adjacent_roads(int r, int c, int h, int w) const;
    int _calculate_road_exposure(int r, int c) const;
    int _count_adjacent_houses(int r, int c) const;
    double _calculate_reward_shaping(int r, int c) const;
    void _invalidate_valid_actions(int r, int c);
};
