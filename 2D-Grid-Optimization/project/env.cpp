#include <iostream>
#include <algorithm>
#include <unordered_set>

#include "env.h"

EnhancedCityPlanningEnv::EnhancedCityPlanningEnv(int rows, int cols) :
    rows_(rows), cols_(cols), episode_length_(0), cache_valid_(false), max_actions_per_episode_(200) {
    grid_.resize(rows_, std::vector<int>(cols_));
    std::random_device rd;
    rng_ = std::mt19937(rd());
    episode_end_processing_ = true;
    road_reach_from_kontor_cache_.resize(rows_, std::vector<bool>(cols_, false));
    road_reach_from_tc_cache_.resize(rows_, std::vector<bool>(cols_, false));
}

State EnhancedCityPlanningEnv::_get_reduced_state() {
    int block_aligned_positions = 0;
    for (int r = 0; r < rows_ - HOUSE_H + 1; ++r) {
        for (int c = 0; c < cols_ - HOUSE_W + 1; ++c) {
            if (_is_valid_position(r, c)) {
                std::vector<Pos> house_coords = get_building_coords(r, c, HOUSE_H, HOUSE_W);
                if (is_within_influence(house_coords, tc_center_) && _is_block_aligned(r, c)) {
                    block_aligned_positions++;
                }
            }
        }
    }
    return {
        std::min(num_houses_placed_, 50),
        std::min(block_aligned_positions, 20),
        std::min(consecutive_successes_, 10),
        std::min(episode_length_, 100)
    };
}

bool EnhancedCityPlanningEnv::_is_block_aligned(int r, int c) const {
    if (house_positions_.empty()) { return true; }

    for (const auto& house_pos : house_positions_) {
        if (std::abs(r - house_pos.r) < HOUSE_H && (c == house_pos.c + HOUSE_W || c == house_pos.c - HOUSE_W)) {
            return true;
        }
        if (std::abs(c - house_pos.c) < HOUSE_W && (r == house_pos.r + HOUSE_H || r == house_pos.r - HOUSE_H)) {
            return true;
        }
    }
    return false;
}

int EnhancedCityPlanningEnv::_count_potential_block_extensions(int r, int c) {
    int extensions = 0;
    std::vector<Pos> directions = {
        {0, HOUSE_W}, {0, -HOUSE_W}, {HOUSE_H, 0}, {-HOUSE_H, 0}
    };

    for (const auto& dir : directions) {
        int new_r = r + dir.r;
        int new_c = c + dir.c;
        if ((new_r >= 0 && new_r < rows_ - HOUSE_H + 1) &&
            (new_c >= 0 && new_c < cols_ - HOUSE_W + 1)) {
            if (_is_valid_position(new_r, new_c)) {
                std::vector<Pos> house_coords = get_building_coords(new_r, new_c, HOUSE_H, HOUSE_W);
                if (is_within_influence(house_coords, tc_center_)) {
                    extensions++;
                }
            }
        }
    }
    return extensions;
}

bool EnhancedCityPlanningEnv::_check_connectivity_fast(int new_house_r, int new_house_c) {
    std::vector<Pos> house_coords = get_building_coords(new_house_r, new_house_c, HOUSE_H, HOUSE_W);
    for (const auto& coord : house_coords) {
        for (const auto& dir : {Pos{-1,0}, Pos{1,0}, Pos{0,-1}, Pos{0,1}}) {
            int nr = coord.r + dir.r;
            int nc = coord.c + dir.c;
            if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_ &&
                grid_[nr][nc] == ROAD &&
                road_reach_from_kontor_cache_[nr][nc] &&
                road_reach_from_tc_cache_[nr][nc]) {
                return true;
            }
        }
    }
    return false;
}

void EnhancedCityPlanningEnv::_update_connectivity_cache() {
    road_reach_from_kontor_cache_ = bfs_road_reachable(grid_, kontor_coords_list_);
    road_reach_from_tc_cache_ = bfs_road_reachable(grid_, tc_coords_list_);
}

void EnhancedCityPlanningEnv::_invalidate_cache() {
    cache_valid_ = false;
    cached_valid_positions_.clear();
    cached_valid_actions_.clear();
}

std::vector<int> EnhancedCityPlanningEnv::_get_valid_actions_optimized() {
    std::vector<int> valid_actions;
    std::vector<int> block_aligned_actions;
    
    int action_idx = 0;
    for (int r = 0; r < rows_ - HOUSE_H + 1; ++r) {
        for (int c = 0; c < cols_ - HOUSE_W + 1; ++c) {
            if (_is_valid_position(r, c)) {
                std::vector<Pos> house_coords = get_building_coords(r, c, HOUSE_H, HOUSE_W);
                if (is_within_influence(house_coords, tc_center_)) {
                    if (_check_connectivity_fast(r, c)) {
                        if (_is_block_aligned(r, c)) {
                            block_aligned_actions.push_back(action_idx);
                        } else {
                            valid_actions.push_back(action_idx);
                        }
                    }
                }
            }
            action_idx++;
        }
    }
    block_aligned_actions.insert(block_aligned_actions.end(), valid_actions.begin(), valid_actions.end());
    return block_aligned_actions.empty() ? std::vector<int>{0} : block_aligned_actions;
}

std::vector<int> EnhancedCityPlanningEnv::_get_valid_actions() {
    return _get_valid_actions_optimized();
}

int EnhancedCityPlanningEnv::_fill_gaps_greedy() {
    int houses_added = 0;
    int max_overall_attempts = 100;

    for (int overall_attempt = 0; overall_attempt < max_overall_attempts; ++overall_attempt) {
        std::vector<Pos> potential_gaps;
        std::unordered_set<int> tried_and_failed_hashes_in_this_pass;

        for (int r = 0; r < rows_ - HOUSE_H + 1; ++r) {
            for (int c = 0; c < cols_ - HOUSE_W + 1; ++c) {
                if (_is_valid_position(r, c)) {
                    std::vector<Pos> house_coords = get_building_coords(r, c, HOUSE_H, HOUSE_W);
                    if (is_within_influence(house_coords, tc_center_)) {
                        potential_gaps.push_back({r, c});
                    }
                }
            }
        }

        if (potential_gaps.empty()) { break; }

        std::sort(potential_gaps.begin(), potential_gaps.end(),
                [this](const Pos& a, const Pos& b) {
                    bool a_aligned = _is_block_aligned(a.r, a.c);
                    bool b_aligned = _is_block_aligned(b.r, b.c);
                    if (a_aligned != b_aligned) {
                        return a_aligned > b_aligned;
                    }
                    return _count_potential_block_extensions(a.r, a.c) > _count_potential_block_extensions(b.r, b.c);
                });

        bool house_placed_in_this_overall_attempt = false;
        for (const auto& gap : potential_gaps) {
            int gap_hash = gap.r * cols_ + gap.c;
            if (tried_and_failed_hashes_in_this_pass.count(gap_hash)) {
                continue;
            }
            if (_check_all_buildings_connectivity(gap.r, gap.c)) {
                place_building_on_grid(grid_, HOUSE, gap.r, gap.c, HOUSE_H, HOUSE_W);
                placed_buildings_info_.push_back({gap.r, gap.c, HOUSE_H, HOUSE_W, HOUSE});
                house_positions_.push_back(gap);
                num_houses_placed_++;
                houses_added++;
                _update_connectivity_cache();
                _invalidate_cache();
                house_placed_in_this_overall_attempt = true;
                break;
            } else { tried_and_failed_hashes_in_this_pass.insert(gap_hash); }
        }
        if (!house_placed_in_this_overall_attempt) { break; }
    }
    return houses_added;
}

bool EnhancedCityPlanningEnv::_is_valid_position(int r, int c) const {
    int house_h = HOUSE_H;
    int house_w = HOUSE_W;
    if (check_overlap(r, c, house_h, house_w, placed_buildings_info_)) {
        return false;
    }
    if (!is_area_road(grid_, r, c, house_h, house_w)) {
        return false;
    }
    return true;
}

Pos EnhancedCityPlanningEnv::_action_to_position(int action_idx) {
    int max_actions = (rows_ - HOUSE_H + 1) * (cols_ - HOUSE_W + 1);
    if (action_idx < 0 || action_idx >= max_actions) {
        return {0, 0};
    }
    int cols_available = cols_ - HOUSE_W + 1;
    int r = action_idx / cols_available;
    int c = action_idx % cols_available;
    return {r, c};
}

bool EnhancedCityPlanningEnv::_check_all_buildings_connectivity(int new_house_r, int new_house_c) {
    PlacedBuilding hypothetical_building = {new_house_r, new_house_c, HOUSE_H, HOUSE_W, HOUSE};
    if (!is_connected_by_road(grid_, tc_coords_list_, kontor_coords_list_, &hypothetical_building)) {
        return false;
    }

    for (const auto& house_pos : house_positions_) {
        std::vector<Pos> house_coords = get_building_coords(house_pos.r, house_pos.c, HOUSE_H, HOUSE_W);
        if (!is_connected_by_road(grid_, house_coords, tc_coords_list_, &hypothetical_building) ||
            !is_connected_by_road(grid_, house_coords, kontor_coords_list_, &hypothetical_building)) {
            return false;
        }
    }

    if (new_house_r >= 0 && new_house_c >= 0) {
        std::vector<Pos> new_house_coords = get_building_coords(new_house_r, new_house_c, HOUSE_H, HOUSE_W);
        if (!is_connected_by_road(grid_, new_house_coords, tc_coords_list_, &hypothetical_building) ||
            !is_connected_by_road(grid_, new_house_coords, kontor_coords_list_, &hypothetical_building)) {
            return false;
        }
    }
    return true;
}

int EnhancedCityPlanningEnv::_count_adjacent_roads(int r, int c, int h, int w) const {
    int adjacent_roads = 0;
    std::vector<Pos> house_coords = get_building_coords(r, c, h, w);
    
    for (const auto& coord : house_coords) {
        for (const auto& dir : {Pos{-1,0}, Pos{1,0}, Pos{0,-1}, Pos{0,1}}) {
            int nr = coord.r + dir.r;
            int nc = coord.c + dir.c;
            if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_ && grid_[nr][nc] == ROAD) {
                adjacent_roads++;
            }
        }
    }
    return adjacent_roads;
}

int EnhancedCityPlanningEnv::_calculate_road_exposure(int r, int c) const {
    int exposure = 0;
    std::vector<Pos> house_cells = get_building_coords(r, c, HOUSE_H, HOUSE_W);

    for (const auto& cell : house_cells) {
        std::vector<Pos> neighbors = {
            {cell.r - 1, cell.c}, {cell.r + 1, cell.c},
            {cell.r, cell.c - 1}, {cell.r, cell.c + 1}
        };

        for (const auto& neighbor : neighbors) {
            if (neighbor.r >= 0 && neighbor.r < rows_ &&
                neighbor.c >= 0 && neighbor.c < cols_) {

                bool is_current_house_cell = false;
                for(const auto& hc : house_cells) {
                    if (hc.r == neighbor.r && hc.c == neighbor.c) {
                        is_current_house_cell = true;
                        break;
                    }
                }

                if (grid_[neighbor.r][neighbor.c] == ROAD && !is_current_house_cell) {
                    exposure++;
                }
            } else {

                exposure++;
            }
        }
    }
    return exposure;
}


int EnhancedCityPlanningEnv::_count_adjacent_houses(int r, int c) const {
    int adjacent_houses_count = 0;
    std::unordered_set<Pos, PosHash> counted_houses;
    std::vector<Pos> new_house_coords = get_building_coords(r, c, HOUSE_H, HOUSE_W);

    for (const auto& cell : new_house_coords) {
        std::vector<Pos> neighbors = {
            {cell.r - 1, cell.c}, {cell.r + 1, cell.c},
            {cell.r, cell.c - 1}, {cell.r, cell.c + 1}
        };

        for (const auto& neighbor : neighbors) {
            if (neighbor.r >= 0 && neighbor.r < rows_ &&
                neighbor.c >= 0 && neighbor.c < cols_) {

                if (grid_[neighbor.r][neighbor.c] == HOUSE) {
                    for (const auto& existing_house_pos : house_positions_) {
                        std::vector<Pos> existing_house_coords = get_building_coords(existing_house_pos.r, existing_house_pos.c, HOUSE_H, HOUSE_W);
                        
                        bool is_part_of_existing_house = false;
                        for (const auto& existing_house_cell : existing_house_coords) {
                            if (existing_house_cell == neighbor) {
                                is_part_of_existing_house = true;
                                break;
                            }
                        }

                        if (is_part_of_existing_house) {
                            if (counted_houses.find(existing_house_pos) == counted_houses.end()) {
                                adjacent_houses_count++;
                                counted_houses.insert(existing_house_pos);
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
    return adjacent_houses_count;
}


double EnhancedCityPlanningEnv::_calculate_reward_shaping(int r, int c) {
    if (!ENABLE_REWARD_SHAPING) return 0.0;
    
    double reward = 0.0;

    if (_is_block_aligned(r, c)) {
        reward += REWARD_BLOCK_ALIGNED;
    }

    if (consecutive_successes_ > 5) {
        reward += REWARD_CONSECUTIVE_SUCCESS_BONUS * std::min(consecutive_successes_, 3);
    }

    int extensions = _count_potential_block_extensions(r, c);
    reward += REWARD_EXTENSION_BONUS_PER_POTENTIAL * extensions;
    reward += REWARD_EPISODE_LENGTH_BONUS * episode_length_;

    if (num_houses_placed_ >= 80) {
        reward += REWARD_HIGH_HOUSE_COUNT_BONUS;
    }

    if (num_houses_placed_ >= ROAD_EXPOSURE_PENALTY_START_THRESHOLD) {
        int road_exposure = _calculate_road_exposure(r, c);
        reward += road_exposure * PENALTY_ROAD_EXPOSURE_PER_SIDE;
    }

    int adjacent_houses = _count_adjacent_houses(r, c);
    if (adjacent_houses >= 2) {
        reward += REWARD_MULTIPLE_HOUSE_ALIGNMENT * 2.0;
        // reward += (adjacent_houses - 1) * REWARD_PER_ADDITIONAL_ADJACENCY;
    }

    return reward;
}


State EnhancedCityPlanningEnv::reset() {
    for (auto& row : grid_) {
        std::fill(row.begin(), row.end(), ROAD);
    }

    placed_buildings_info_.clear();
    house_positions_.clear();
    num_houses_placed_ = 0;
    consecutive_successes_ = 0;
    episode_length_ = 0;
    last_house_pos_ = {-1, -1};

    _invalidate_cache();

    int tc_r = (rows_ - TOWN_CENTER_H) / 2;
    int tc_c = (cols_ - TOWN_CENTER_W) / 2;
    place_building_on_grid(grid_, TOWN_CENTER, tc_r, tc_c, TOWN_CENTER_H, TOWN_CENTER_W);
    placed_buildings_info_.push_back({tc_r, tc_c, TOWN_CENTER_H, TOWN_CENTER_W, TOWN_CENTER});
    tc_coords_list_ = get_building_coords(tc_r, tc_c, TOWN_CENTER_H, TOWN_CENTER_W);
    tc_center_ = {tc_r + TOWN_CENTER_H / 2, tc_c + TOWN_CENTER_W / 2};
    
    int kontor_r, kontor_c;
    int attempts = 0;
    
    std::vector<Pos> edge_positions;
    for (int c = 0; c < cols_ - TRADING_POST_W + 1; ++c) {
        edge_positions.push_back({0, c});
        edge_positions.push_back({rows_ - TRADING_POST_H, c});
    }

    for (int r = 1; r < rows_ - TRADING_POST_H; ++r) {
        edge_positions.push_back({r, 0});
        edge_positions.push_back({r, cols_ - TRADING_POST_W});
    }

    std::shuffle(edge_positions.begin(), edge_positions.end(), rng_);

    bool placed = false;
    for (const auto& pos : edge_positions) {
        if (!check_overlap(pos.r, pos.c, TRADING_POST_H, TRADING_POST_W, placed_buildings_info_)) {
            kontor_r = pos.r;
            kontor_c = pos.c;
            placed = true;
            break;
        }
    }

    if (!placed) {
        std::uniform_int_distribution<> r_dist(0, rows_ - TRADING_POST_H);
        std::uniform_int_distribution<> c_dist(0, cols_ - TRADING_POST_W);
        do {
            kontor_r = r_dist(rng_);
            kontor_c = c_dist(rng_);
            attempts++;
        } while (attempts < 100 && check_overlap(kontor_r, kontor_c, TRADING_POST_H, TRADING_POST_W, placed_buildings_info_));
    }
    
    place_building_on_grid(grid_, TRADING_POST, kontor_r, kontor_c, TRADING_POST_H, TRADING_POST_W);
    placed_buildings_info_.push_back({kontor_r, kontor_c, TRADING_POST_H, TRADING_POST_W, TRADING_POST});
    kontor_coords_list_ = get_building_coords(kontor_r, kontor_c, TRADING_POST_H, TRADING_POST_W);
    _update_connectivity_cache();
    
    return _get_reduced_state();
}

std::tuple<State, double, bool> EnhancedCityPlanningEnv::step(int action_idx) {
    episode_length_++;

    if (episode_length_ >= max_actions_per_episode_) {
        int additional_houses = _fill_gaps_greedy();
        double final_reward = additional_houses * REWARD_GREEDY_ADDITIONAL_HOUSE;
        return {_get_reduced_state(), final_reward, true};
    }
    
    Pos pos = _action_to_position(action_idx);
    int r = pos.r;
    int c = pos.c;
    if (!_is_valid_position(r, c)) {
        return {_get_reduced_state(), PENALTY_INVALID_PLACEMENT, false};
    }

    std::vector<Pos> house_coords = get_building_coords(r, c, HOUSE_H, HOUSE_W);
    if (!is_within_influence(house_coords, tc_center_)) {
        return {_get_reduced_state(), PENALTY_INVALID_PLACEMENT, false};
    }

    if (!_check_all_buildings_connectivity(r, c)) {
        consecutive_successes_ = 0;
        return {_get_reduced_state(), PENALTY_DISCONNECTED_HOUSE, false};
    }

    place_building_on_grid(grid_, HOUSE, r, c, HOUSE_H, HOUSE_W);
    placed_buildings_info_.push_back({r, c, HOUSE_H, HOUSE_W, HOUSE});
    house_positions_.push_back({r, c});
    num_houses_placed_++;
    consecutive_successes_++;
    last_house_pos_ = {r, c};

    _update_connectivity_cache();
    _invalidate_cache();

    double reward = REWARD_PLACEMENT_SUCCESS;
    reward += _calculate_reward_shaping(r, c);

    std::vector<int> next_valid_actions = _get_valid_actions();
    bool done = next_valid_actions.empty() || next_valid_actions.size() == 1;
    
    if (done && episode_end_processing_) {
        int additional_houses = _fill_gaps_greedy();
        reward += additional_houses * REWARD_GREEDY_ADDITIONAL_HOUSE;
    }
    
    return {_get_reduced_state(), reward, done};
}

std::vector<int> EnhancedCityPlanningEnv::get_valid_actions() {
    return _get_valid_actions();
}

int EnhancedCityPlanningEnv::get_num_connected_houses() const {
    int connected_houses = 0;
    for (const auto& house_pos : house_positions_) {
        std::vector<Pos> house_coords = get_building_coords(house_pos.r, house_pos.c, HOUSE_H, HOUSE_W);
        if (is_connected_by_road(grid_, house_coords, tc_coords_list_) &&
            is_connected_by_road(grid_, house_coords, kontor_coords_list_)) {
            connected_houses++;
        }
    }
    return connected_houses;
}

const std::vector<std::vector<int>>& EnhancedCityPlanningEnv::get_grid() const {
    return grid_;
}