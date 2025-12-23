#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <queue>

#include "env.h"

EnhancedCityPlanningEnv::EnhancedCityPlanningEnv(int rows, int cols) :
    rows_(rows), cols_(cols), episode_length_(0),
    house_spatial_hash_(HOUSE_H, HOUSE_W),
    valid_actions_dirty_(true),
    bfs_cache_valid_(false),
    max_actions_per_episode_(200) {

    grid_.resize(rows_ * cols_, ROAD);
    cached_tc_reachable_.resize(rows_, std::vector<bool>(cols_, false));
    cached_kontor_reachable_.resize(rows_, std::vector<bool>(cols_, false));
    std::random_device rd;
    rng_ = std::mt19937(rd());
    episode_end_processing_ = true;
    valid_action_set_.reserve(1200);
    valid_actions_cache_.reserve(1200);
}

inline Pos EnhancedCityPlanningEnv::_action_to_position(int action_idx) const noexcept {
    int cols_available = cols_ - HOUSE_W + 1;
    int r = action_idx / cols_available;
    int c = action_idx % cols_available;
    return {r, c};
}

State EnhancedCityPlanningEnv::_get_reduced_state() const {
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

    const std::vector<Pos> nearby = house_spatial_hash_.get_all_nearby(r, c, 2);
    for (const auto& house_pos : nearby) {
        if (std::abs(r - house_pos.r) < HOUSE_H && (c == house_pos.c + HOUSE_W || c == house_pos.c - HOUSE_W)) {
            return true;
        }
        if (std::abs(c - house_pos.c) < HOUSE_W && (r == house_pos.r + HOUSE_H || r == house_pos.r - HOUSE_H)) {
            return true;
        }
    }
    return false;
}

int EnhancedCityPlanningEnv::_count_potential_block_extensions(int r, int c) const {
    int extensions = 0;
    constexpr Pos directions[] = {{0, HOUSE_W}, {0, -HOUSE_W}, {HOUSE_H, 0}, {-HOUSE_H, 0}};

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

void EnhancedCityPlanningEnv::_update_bfs_caches() const {
    cached_tc_reachable_ = bfs_road_reachable_flat(grid_, rows_, cols_, tc_coords_list_, nullptr);
    cached_kontor_reachable_ = bfs_road_reachable_flat(grid_, rows_, cols_, kontor_coords_list_, nullptr);
    bfs_cache_valid_ = true;
}

bool EnhancedCityPlanningEnv::_check_all_buildings_connectivity(int new_house_r, int new_house_c) const {
    // Update BFS cache if invalid
    if (!bfs_cache_valid_) {
        _update_bfs_caches();
    }

    auto is_hypothetical_cell = [&](int r, int c) {
        if (new_house_r >= 0 && new_house_c >= 0) {
            return r >= new_house_r && r < new_house_r + HOUSE_H &&
                   c >= new_house_c && c < new_house_c + HOUSE_W;
        }
        return false;
    };

    // Check TC ↔ Kontor connectivity: There must be at least one road reachable from both TC and Kontor
    bool tc_kontor_connected = false;
    for (int r = 0; r < rows_ && !tc_kontor_connected; ++r) {
        for (int c = 0; c < cols_ && !tc_kontor_connected; ++c) {
            if (at(r, c) == ROAD && !is_hypothetical_cell(r, c)) {
                if (cached_tc_reachable_[r][c] && cached_kontor_reachable_[r][c]) {
                    tc_kontor_connected = true;
                }
            }
        }
    }
    if (!tc_kontor_connected) {
        return false;
    }

    // Check ALL existing houses remain connected to TC and Kontor
    for (const auto& house_pos : house_positions_) {
        std::vector<Pos> house_coords = get_building_coords(house_pos.r, house_pos.c, HOUSE_H, HOUSE_W);

        bool connected_to_tc = false;
        bool connected_to_kontor = false;

        for (const auto& coord : house_coords) {
            constexpr Pos dirs[] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
            for (const auto& dir : dirs) {
                int nr = coord.r + dir.r;
                int nc = coord.c + dir.c;
                if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_) {
                    if (at(nr, nc) == ROAD && !is_hypothetical_cell(nr, nc)) {
                        if (cached_tc_reachable_[nr][nc]) connected_to_tc = true;
                        if (cached_kontor_reachable_[nr][nc]) connected_to_kontor = true;
                        if (connected_to_tc && connected_to_kontor) goto house_done;
                    }
                }
            }
        }
        house_done:
        if (!connected_to_tc || !connected_to_kontor) {
            return false;
        }
    }

    // Check new house connectivity (if valid position)
    if (new_house_r >= 0 && new_house_c >= 0) {
        std::vector<Pos> new_house_coords = get_building_coords(new_house_r, new_house_c, HOUSE_H, HOUSE_W);

        bool connected_to_tc = false;
        bool connected_to_kontor = false;

        for (const auto& coord : new_house_coords) {
            constexpr Pos dirs[] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
            for (const auto& dir : dirs) {
                int nr = coord.r + dir.r;
                int nc = coord.c + dir.c;
                if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_) {
                    if (at(nr, nc) == ROAD && !is_hypothetical_cell(nr, nc)) {
                        if (cached_tc_reachable_[nr][nc]) connected_to_tc = true;
                        if (cached_kontor_reachable_[nr][nc]) connected_to_kontor = true;
                        if (connected_to_tc && connected_to_kontor) goto new_house_done;
                    }
                }
            }
        }
        new_house_done:
        if (!connected_to_tc || !connected_to_kontor) {
            return false;
        }
    }

    return true;
}

void EnhancedCityPlanningEnv::_invalidate_valid_actions(int r, int c) {
    int start_r = std::max(0, r - HOUSE_H);
    int end_r = std::min(rows_ - HOUSE_H, r + HOUSE_H);
    int start_c = std::max(0, c - HOUSE_W);
    int end_c = std::min(cols_ - HOUSE_W, c + HOUSE_W);

    int cols_available = cols_ - HOUSE_W + 1;
    for (int rr = start_r; rr <= end_r; ++rr) {
        for (int cc = start_c; cc <= end_c; ++cc) {
            int action_id = rr * cols_available + cc;
            valid_action_set_.erase(action_id);
        }
    }
    valid_actions_dirty_ = true;
}

void EnhancedCityPlanningEnv::_recompute_valid_actions() {
    valid_action_set_.clear();
    int cols_available = cols_ - HOUSE_W + 1;

    for (int r = 0; r < rows_ - HOUSE_H + 1; ++r) {
        for (int c = 0; c < cols_ - HOUSE_W + 1; ++c) {
            if (_is_valid_position(r, c)) {
                std::vector<Pos> house_coords = get_building_coords(r, c, HOUSE_H, HOUSE_W);
                if (is_within_influence(house_coords, tc_center_)) {
                    if (_check_all_buildings_connectivity(r, c)) {
                        int action_id = r * cols_available + c;
                        valid_action_set_.insert(action_id);
                    }
                }
            }
        }
    }
    valid_actions_dirty_ = false;
}

std::vector<int> EnhancedCityPlanningEnv::_get_valid_actions() {
    if (valid_actions_dirty_) {
        _recompute_valid_actions();
    }

    valid_actions_cache_.clear();
    valid_actions_cache_.reserve(valid_action_set_.size());

    std::vector<int> block_aligned;
    std::vector<int> others;

    for (int action_id : valid_action_set_) {
        Pos pos = _action_to_position(action_id);
        if (_is_block_aligned(pos.r, pos.c)) {
            block_aligned.push_back(action_id);
        } else {
            others.push_back(action_id);
        }
    }

    valid_actions_cache_.insert(valid_actions_cache_.end(), block_aligned.begin(), block_aligned.end());
    valid_actions_cache_.insert(valid_actions_cache_.end(), others.begin(), others.end());

    return valid_actions_cache_.empty() ? std::vector<int>{0} : valid_actions_cache_;
}

int EnhancedCityPlanningEnv::_fill_gaps_greedy() {
    int houses_added = 0;
    constexpr int max_overall_attempts = 100;

    for (int overall_attempt = 0; overall_attempt < max_overall_attempts; ++overall_attempt) {
        std::vector<Pos> potential_gaps;

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
            if (_check_all_buildings_connectivity(gap.r, gap.c)) {
                place_building_on_grid_flat(grid_, cols_, HOUSE, gap.r, gap.c, HOUSE_H, HOUSE_W);
                placed_buildings_info_.push_back({gap.r, gap.c, HOUSE_H, HOUSE_W, HOUSE});
                house_positions_.push_back(gap);
                house_spatial_hash_.insert(gap);
                num_houses_placed_++;
                houses_added++;
                _invalidate_valid_actions(gap.r, gap.c);
                bfs_cache_valid_ = false;
                house_placed_in_this_overall_attempt = true;
                break;
            }
        }
        if (!house_placed_in_this_overall_attempt) { break; }
    }
    return houses_added;
}

bool EnhancedCityPlanningEnv::_is_valid_position(int r, int c) const {
    if (check_overlap(r, c, HOUSE_H, HOUSE_W, placed_buildings_info_)) {
        return false;
    }
    if (!is_area_road_flat(grid_, cols_, r, c, HOUSE_H, HOUSE_W)) {
        return false;
    }
    return true;
}

int EnhancedCityPlanningEnv::_count_adjacent_roads(int r, int c, int h, int w) const {
    int adjacent_roads = 0;
    std::vector<Pos> house_coords = get_building_coords(r, c, h, w);

    for (const auto& coord : house_coords) {
        constexpr Pos dirs[] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        for (const auto& dir : dirs) {
            int nr = coord.r + dir.r;
            int nc = coord.c + dir.c;
            if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_ && at(nr, nc) == ROAD) {
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
        constexpr Pos neighbors[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (const auto& neighbor : neighbors) {
            int nr = cell.r + neighbor.r;
            int nc = cell.c + neighbor.c;

            if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_) {
                bool is_current_house_cell = false;
                for(const auto& hc : house_cells) {
                    if (hc.r == nr && hc.c == nc) {
                        is_current_house_cell = true;
                        break;
                    }
                }

                if (at(nr, nc) == ROAD && !is_current_house_cell) {
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

    const std::vector<Pos> nearby_houses = house_spatial_hash_.get_all_nearby(r, c, 2);

    for (const auto& cell : new_house_coords) {
        constexpr Pos neighbors[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (const auto& neighbor : neighbors) {
            int nr = cell.r + neighbor.r;
            int nc = cell.c + neighbor.c;

            if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_ && at(nr, nc) == HOUSE) {
                for (const auto& existing_house_pos : nearby_houses) {
                    std::vector<Pos> existing_house_coords = get_building_coords(existing_house_pos.r, existing_house_pos.c, HOUSE_H, HOUSE_W);

                    for (const auto& existing_house_cell : existing_house_coords) {
                        if (existing_house_cell.r == nr && existing_house_cell.c == nc) {
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

double EnhancedCityPlanningEnv::_calculate_reward_shaping(int r, int c) const {
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
    }

    return reward;
}

State EnhancedCityPlanningEnv::reset() {
    std::fill(grid_.begin(), grid_.end(), ROAD);

    placed_buildings_info_.clear();
    house_positions_.clear();
    house_spatial_hash_.clear();
    num_houses_placed_ = 0;
    consecutive_successes_ = 0;
    episode_length_ = 0;
    last_house_pos_ = {-1, -1};
    valid_actions_dirty_ = true;
    valid_action_set_.clear();
    bfs_cache_valid_ = false;

    int tc_r = (rows_ - TOWN_CENTER_H) / 2;
    int tc_c = (cols_ - TOWN_CENTER_W) / 2;
    place_building_on_grid_flat(grid_, cols_, TOWN_CENTER, tc_r, tc_c, TOWN_CENTER_H, TOWN_CENTER_W);
    placed_buildings_info_.push_back({tc_r, tc_c, TOWN_CENTER_H, TOWN_CENTER_W, TOWN_CENTER});
    tc_coords_list_ = get_building_coords(tc_r, tc_c, TOWN_CENTER_H, TOWN_CENTER_W);
    tc_center_ = {tc_r + TOWN_CENTER_H / 2, tc_c + TOWN_CENTER_W / 2};

    int kontor_r, kontor_c;

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
        int attempts = 0;
        do {
            kontor_r = r_dist(rng_);
            kontor_c = c_dist(rng_);
            attempts++;
        } while (attempts < 100 && check_overlap(kontor_r, kontor_c, TRADING_POST_H, TRADING_POST_W, placed_buildings_info_));
    }

    place_building_on_grid_flat(grid_, cols_, TRADING_POST, kontor_r, kontor_c, TRADING_POST_H, TRADING_POST_W);
    placed_buildings_info_.push_back({kontor_r, kontor_c, TRADING_POST_H, TRADING_POST_W, TRADING_POST});
    kontor_coords_list_ = get_building_coords(kontor_r, kontor_c, TRADING_POST_H, TRADING_POST_W);

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

    place_building_on_grid_flat(grid_, cols_, HOUSE, r, c, HOUSE_H, HOUSE_W);
    placed_buildings_info_.push_back({r, c, HOUSE_H, HOUSE_W, HOUSE});
    house_positions_.push_back({r, c});
    house_spatial_hash_.insert({r, c});
    num_houses_placed_++;
    consecutive_successes_++;
    last_house_pos_ = {r, c};

    _invalidate_valid_actions(r, c);
    bfs_cache_valid_ = false;

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

int EnhancedCityPlanningEnv::get_num_connected_houses() const noexcept {
    return static_cast<int>(house_positions_.size());
}

const std::vector<int>& EnhancedCityPlanningEnv::get_grid() const noexcept {
    return grid_;
}
