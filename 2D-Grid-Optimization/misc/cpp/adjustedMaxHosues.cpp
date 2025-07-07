#include <iostream>
#include <vector>
#include <deque>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <chrono>
#include <limits>
#include <tuple>

// Building Types
const int EMPTY = 0;
const int HOUSE = 1;
const int ROAD = 2;
const int TOWN_CENTER = 3;
const int TRADING_POST = 4;

// Building Dimensions
const int HOUSE_H = 3;
const int HOUSE_W = 3;
const int TOWN_CENTER_H = 7;
const int TOWN_CENTER_W = 5;
const int TRADING_POST_H = 1;
const int TRADING_POST_W = 5;

// Town Center Influence Radius (squared for distance calculation)
const int TOWN_CENTER_RADIUS_SQ = 26 * 26;

// Reward Configuration
const bool ENABLE_REWARD_SHAPING = true; // Set to false to disable all shaping bonuses

const double REWARD_PLACEMENT_SUCCESS = 2.0;
const double PENALTY_INVALID_PLACEMENT = -5.0;
const double PENALTY_DISCONNECTED_HOUSE = -10.0;
const double REWARD_CONSECUTIVE_SUCCESS_BONUS = 0.0; // Bonus if consecutive_successes_ >= 5
const double REWARD_HIGH_HOUSE_COUNT_BONUS = 0.0;   // Bonus if num_houses_placed_ > 25
const double REWARD_GREEDY_ADDITIONAL_HOUSE = 5.0;  // Reward per house added by greedy fill

// Reward Shaping Specific Weights (only applied if ENABLE_REWARD_SHAPING is true)
const double REWARD_BLOCK_ALIGNED = 10.0;
const double REWARD_EXTENSION_BONUS_PER_POTENTIAL = 0.5;
const double REWARD_CLOSE_TO_LAST_HOUSE = 1.5; // If distance <= 5
const double REWARD_TC_INFLUENCE_FACTOR = 2.0; // Multiplier for TC distance bonus
const double PENALTY_TOO_MANY_ROADSIDES = -3.0; // If adjacent_roads > 2

struct Dimensions {
    int h, w;
};

std::map<int, Dimensions> BUILDING_TYPES = {
    {HOUSE, {HOUSE_H, HOUSE_W}},
    {TOWN_CENTER, {TOWN_CENTER_H, TOWN_CENTER_W}},
    {TRADING_POST, {TRADING_POST_H, TRADING_POST_W}}
};

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

struct PlacedBuilding {
    int r, c, h, w, type;
};

// Pre-calculated relative coordinates for a HOUSE (3x3)
static const std::vector<Pos> HOUSE_RELATIVE_COORDS = {
    {0,0}, {0,1}, {0,2},
    {1,0}, {1,1}, {1,2},
    {2,0}, {2,1}, {2,2}
};

struct State {
    int num_houses_placed;
    int available_positions_in_tc_radius;
    int road_cells;
    int dist_to_last;
    int consecutive_successes;
    int block_aligned_positions;

    bool operator<(const State& other) const {
        if (num_houses_placed != other.num_houses_placed) return num_houses_placed < other.num_houses_placed;
        if (available_positions_in_tc_radius != other.available_positions_in_tc_radius) return available_positions_in_tc_radius < other.available_positions_in_tc_radius;
        if (road_cells != other.road_cells) return road_cells < other.road_cells;
        if (dist_to_last != other.dist_to_last) return dist_to_last < other.dist_to_last;
        if (consecutive_successes != other.consecutive_successes) return consecutive_successes < other.consecutive_successes;
        return block_aligned_positions < other.block_aligned_positions;
    }
    bool operator==(const State& other) const {
        return num_houses_placed == other.num_houses_placed &&
            available_positions_in_tc_radius == other.available_positions_in_tc_radius &&
            road_cells == other.road_cells &&
            dist_to_last == other.dist_to_last &&
            consecutive_successes == other.consecutive_successes &&
            block_aligned_positions == other.block_aligned_positions;
    }
};

bool is_area_road(const std::vector<std::vector<int>>& grid, int r, int c, int h, int w);
void place_building_on_grid(std::vector<std::vector<int>>& grid, int building_type_val, int r, int c, int h, int w);
std::vector<Pos> get_building_coords(int r, int c, int h, int w);
double calculate_distance_sq(Pos p1, Pos p2);
bool is_within_influence(const std::vector<Pos>& house_coords, const Pos& tc_center);

// Modified BFS to accept a hypothetical building
std::vector<std::vector<bool>> bfs_road_reachable(
    const std::vector<std::vector<int>>& grid,
    const std::vector<Pos>& start_building_coords,
    const PlacedBuilding* hypothetical_building = nullptr
);

// Modified connectivity check to use the hypothetical building in BFS
bool is_connected_by_road(
    const std::vector<std::vector<int>>& grid,
    const std::vector<Pos>& start_building_coords,
    const std::vector<Pos>& target_building_coords,
    const PlacedBuilding* hypothetical_building = nullptr
);
bool check_overlap(int new_r, int new_c, int new_h, int new_w, const std::vector<PlacedBuilding>& placed_buildings_info);


class ExperienceReplay {
public:
    struct Experience {
        State state;
        int action;
        double reward;
        State next_state;
    };

    ExperienceReplay(size_t capacity = 5000) : capacity_(capacity) {
        std::random_device rd;
        rng_ = std::mt19937(rd());
    }
    
    void add(const State& state, int action, double reward, const State& next_state) {
        if (buffer_.size() == capacity_) {
            buffer_.pop_front();
        }
        buffer_.push_back({state, action, reward, next_state});
    }
    
    std::vector<Experience> sample(size_t batch_size = 32) {
        if (buffer_.empty()) {
            return {};
        }
        batch_size = std::min(batch_size, buffer_.size());
        std::vector<Experience> samples;
        samples.reserve(batch_size);
        std::sample(buffer_.begin(), buffer_.end(), std::back_inserter(samples), batch_size, rng_);
        return samples;
    }
    
    size_t size() const {
        return buffer_.size();
    }

private:
    std::deque<Experience> buffer_;
    size_t capacity_;
    std::mt19937 rng_;
};

class EnhancedCityPlanningEnv {
public:
    EnhancedCityPlanningEnv(int rows, int cols) : rows_(rows), cols_(cols) {
        grid_.resize(rows_, std::vector<int>(cols_));
        std::random_device rd;
        rng_ = std::mt19937(rd());
        episode_end_processing_ = true;
        road_reach_from_kontor_cache_.resize(rows_, std::vector<bool>(cols_, false));
        road_reach_from_tc_cache_.resize(rows_, std::vector<bool>(cols_, false));
    }

    State _get_reduced_state() {
        int available_positions_in_tc_radius = 0;
        int block_aligned_positions = 0;
        
        for (int r = 0; r < rows_ - HOUSE_H + 1; ++r) {
            for (int c = 0; c < cols_ - HOUSE_W + 1; ++c) {
                if (_is_valid_position(r, c)) {
                    std::vector<Pos> house_coords = get_building_coords(r, c, HOUSE_H, HOUSE_W);
                    if (is_within_influence(house_coords, tc_center_)) {
                        available_positions_in_tc_radius++;
                        
                        if (_is_block_aligned(r, c)) {
                            block_aligned_positions++;
                        }
                    }
                }
            }
        }
        
        int road_cells = 0;
        for(int r = 0; r < rows_; ++r) {
            for(int c = 0; c < cols_; ++c) {
                if(grid_[r][c] == ROAD) {
                    road_cells++;
                }
            }
        }
        
        int dist_to_last = 0;
        if (last_house_pos_.r != -1 && tc_center_.r != -1) {
            dist_to_last = static_cast<int>(std::min(10.0, std::sqrt(calculate_distance_sq(last_house_pos_, tc_center_))));
        }
        
        return {
            std::min(num_houses_placed_, 50),
            std::min(available_positions_in_tc_radius, 100),
            std::min(road_cells, 200),
            dist_to_last,
            consecutive_successes_,
            std::min(block_aligned_positions, 20)
        };
    }

    bool _is_block_aligned(int r, int c) const {
        if (house_positions_.empty()) {
            return true;
        }
        
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

    int _count_potential_block_extensions(int r, int c) {
        int extensions = 0;
        std::vector<Pos> directions = {
            {0, HOUSE_W},
            {0, -HOUSE_W},
            {HOUSE_H, 0},
            {-HOUSE_H, 0}
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

    int _fill_gaps_greedy() {
        int houses_added = 0;
        int max_attempts = 100;
        
        for (int _ = 0; _ < max_attempts; ++_) {
            std::vector<Pos> valid_gaps;
            for (int r = 0; r < rows_ - HOUSE_H + 1; ++r) {
                for (int c = 0; c < cols_ - HOUSE_W + 1; ++c) {
                    if (_is_valid_position(r, c)) {
                        std::vector<Pos> house_coords = get_building_coords(r, c, HOUSE_H, HOUSE_W);
                        if (is_within_influence(house_coords, tc_center_)) {
                            valid_gaps.push_back({r, c});
                        }
                    }
                }
            }

            if (valid_gaps.empty()) {
                break;
            }
            
            std::vector<Pos> block_aligned_gaps;
            for (const auto& gap : valid_gaps) {
                if (_is_block_aligned(gap.r, gap.c)) {
                    block_aligned_gaps.push_back(gap);
                }
            }
            
            Pos best_gap = {-1, -1};
            if (!block_aligned_gaps.empty()) {
                best_gap = *std::max_element(block_aligned_gaps.begin(), block_aligned_gaps.end(),
                                            [this](const Pos& a, const Pos& b) {
                                                return _count_potential_block_extensions(a.r, a.c) < _count_potential_block_extensions(b.r, b.c);
                                            });
            } else if (!valid_gaps.empty()) {
                best_gap = *std::max_element(valid_gaps.begin(), valid_gaps.end(),
                                            [this](const Pos& a, const Pos& b) {
                                                return _count_potential_block_extensions(a.r, a.c) < _count_potential_block_extensions(b.r, b.c);
                                            });
            } else {
                break;
            }
            
            if (best_gap.r == -1) {
                break;
            }

            // Call connectivity check with hypothetical building
            if (_check_all_buildings_connectivity(best_gap.r, best_gap.c)) {
                place_building_on_grid(grid_, HOUSE, best_gap.r, best_gap.c, HOUSE_H, HOUSE_W);
                placed_buildings_info_.push_back({best_gap.r, best_gap.c, HOUSE_H, HOUSE_W, HOUSE});
                house_positions_.push_back(best_gap);
                num_houses_placed_++;
                houses_added++;
                
                // Update caches only after actual placement
                road_reach_from_kontor_cache_ = bfs_road_reachable(grid_, kontor_coords_list_);
                road_reach_from_tc_cache_ = bfs_road_reachable(grid_, tc_coords_list_);
            } else {
                break;
            }
        }
        return houses_added;
    }

    bool _is_valid_position(int r, int c) const {
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

    std::vector<int> _get_valid_actions() {
        std::vector<int> valid_actions;
        int action_idx = 0;
        for (int r = 0; r < rows_ - HOUSE_H + 1; ++r) {
            for (int c = 0; c < cols_ - HOUSE_W + 1; ++c) {
                if (_is_valid_position(r, c)) {
                    std::vector<Pos> house_coords = get_building_coords(r, c, HOUSE_H, HOUSE_W);
                    if (is_within_influence(house_coords, tc_center_)) {
                        valid_actions.push_back(action_idx);
                    }
                }
                action_idx++;
            }
        }
        return valid_actions.empty() ? std::vector<int>{0} : valid_actions; 
    }

    Pos _action_to_position(int action_idx) {
        int max_actions = (rows_ - HOUSE_H + 1) * (cols_ - HOUSE_W + 1);
        if (action_idx < 0 || action_idx >= max_actions) {
            return {0, 0};
        }
        int r = action_idx / (cols_ - HOUSE_W + 1);
        int c = action_idx % (cols_ - HOUSE_W + 1);
        return {r, c};
    }

    State reset() {
        
        for (int i = 0; i < rows_; ++i) {
            std::fill(grid_[i].begin(), grid_[i].end(), ROAD);
        }

        placed_buildings_info_.clear();
        kontor_coords_list_.clear();
        tc_coords_list_.clear();
        tc_center_ = { -1, -1 };
        num_houses_placed_ = 0;
        consecutive_successes_ = 0;
        last_house_pos_ = { -1, -1 };
        house_positions_.clear();
        
        int kontor_r = 0;
        int kontor_c = 0;
        int kontor_h = TRADING_POST_H;
        int kontor_w = TRADING_POST_W;
        place_building_on_grid(grid_, TRADING_POST, kontor_r, kontor_c, kontor_h, kontor_w);
        placed_buildings_info_.push_back({kontor_r, kontor_c, kontor_h, kontor_w, TRADING_POST});
        kontor_coords_list_ = get_building_coords(kontor_r, kontor_c, kontor_h, kontor_w);
        
        
        int tc_h = TOWN_CENTER_H;
        int tc_w = TOWN_CENTER_W;
        std::vector<Pos> possible_tc_placements;
        for (int r = 0; r < rows_ - tc_h + 1; ++r) {
            for (int c = 0; c < cols_ - tc_w + 1; ++c) {
                possible_tc_placements.push_back({r, c});
            }
        }
        std::shuffle(possible_tc_placements.begin(), possible_tc_placements.end(), rng_);

        bool tc_placed = false;
        for (const auto& pos : possible_tc_placements) {
            if (!check_overlap(pos.r, pos.c, tc_h, tc_w, placed_buildings_info_) &&
                is_area_road(grid_, pos.r, pos.c, tc_h, tc_w)) {

                // For TC placement, we can use a temporary grid as it's only once per reset
                std::vector<std::vector<int>> temp_grid = grid_;
                place_building_on_grid(temp_grid, TOWN_CENTER, pos.r, pos.c, tc_h, tc_w);
                
                std::vector<Pos> potential_tc_coords = get_building_coords(pos.r, pos.c, tc_h, tc_w);
                if (is_connected_by_road(temp_grid, kontor_coords_list_, potential_tc_coords)) { // No hypothetical building for TC placement
                    place_building_on_grid(grid_, TOWN_CENTER, pos.r, pos.c, tc_h, tc_w);
                    placed_buildings_info_.push_back({pos.r, pos.c, tc_h, tc_w, TOWN_CENTER});
                    tc_coords_list_ = potential_tc_coords;
                    tc_center_ = {pos.r + tc_h / 2, pos.c + tc_w / 2};
                    tc_placed = true;
                    break;
                }
            }
        }

        if (!tc_placed) {
            std::cerr << "Warning: Could not place Town Center. Resetting again." << std::endl;
            return reset();
        }

        road_reach_from_kontor_cache_ = bfs_road_reachable(grid_, kontor_coords_list_);
        road_reach_from_tc_cache_ = bfs_road_reachable(grid_, tc_coords_list_);
        return _get_reduced_state();
    }

    // Modified to not copy the grid, and accept hypothetical building directly
    bool _check_all_buildings_connectivity(int new_house_r = -1, int new_house_c = -1) {
        PlacedBuilding hypothetical_house_obj;
        const PlacedBuilding* hypothetical_house_ptr = nullptr;

        if (new_house_r != -1) {
            hypothetical_house_obj = {new_house_r, new_house_c, HOUSE_H, HOUSE_W, HOUSE};
            hypothetical_house_ptr = &hypothetical_house_obj;
        }

        // Check Kontor-TC connectivity first, with hypothetical house
        if (!is_connected_by_road(grid_, kontor_coords_list_, tc_coords_list_, hypothetical_house_ptr)) {
            return false;
        }

        // Now check all houses (existing + hypothetical)
        std::vector<PlacedBuilding> all_buildings_to_check = placed_buildings_info_;
        if (hypothetical_house_ptr) {
            all_buildings_to_check.push_back(hypothetical_house_obj);
        }

        // Calculate temporary road reachability for checking, with hypothetical house
        std::vector<std::vector<bool>> temp_kontor_reach = bfs_road_reachable(grid_, kontor_coords_list_, hypothetical_house_ptr);
        std::vector<std::vector<bool>> temp_tc_reach = bfs_road_reachable(grid_, tc_coords_list_, hypothetical_house_ptr);

        for (const auto& building : all_buildings_to_check) {
            if (building.type == HOUSE) {
                // Ensure to get coordinates based on the actual type of building (house_h, house_w for HOUSE)
                // If the building in all_buildings_to_check is the hypothetical one, its r,c,h,w are already set
                std::vector<Pos> building_coords = get_building_coords(building.r, building.c, building.h, building.w);
                
                bool connected_to_kontor = false;
                for (const auto& b_coord : building_coords) {
                    for (const auto& dir : {Pos{-1,0}, Pos{1,0}, Pos{0,-1}, Pos{0,1}}) {
                        int nr = b_coord.r + dir.r;
                        int nc = b_coord.c + dir.c;
                        // Check if adjacent cell is within bounds, is a ROAD, and reachable from kontor
                        if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_ && grid_[nr][nc] == ROAD && temp_kontor_reach[nr][nc]) {
                             // Also ensure the road cell is not part of the hypothetical building itself if checking other buildings
                            bool is_hypothetical_cell = false;
                            if (hypothetical_house_ptr) {
                                if (nr >= hypothetical_house_ptr->r && nr < hypothetical_house_ptr->r + hypothetical_house_ptr->h &&
                                    nc >= hypothetical_house_ptr->c && nc < hypothetical_house_ptr->c + hypothetical_house_ptr->w) {
                                    is_hypothetical_cell = true;
                                }
                            }
                            if (!is_hypothetical_cell) {
                                connected_to_kontor = true;
                                break;
                            }
                        }
                    }
                    if (connected_to_kontor) break;
                }
                if (!connected_to_kontor) return false;

                bool connected_to_tc = false;
                for (const auto& b_coord : building_coords) {
                    for (const auto& dir : {Pos{-1,0}, Pos{1,0}, Pos{0,-1}, Pos{0,1}}) {
                        int nr = b_coord.r + dir.r;
                        int nc = b_coord.c + dir.c;
                        // Check if adjacent cell is within bounds, is a ROAD, and reachable from TC
                        if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_ && grid_[nr][nc] == ROAD && temp_tc_reach[nr][nc]) {
                             // Also ensure the road cell is not part of the hypothetical building itself
                            bool is_hypothetical_cell = false;
                            if (hypothetical_house_ptr) {
                                if (nr >= hypothetical_house_ptr->r && nr < hypothetical_house_ptr->r + hypothetical_house_ptr->h &&
                                    nc >= hypothetical_house_ptr->c && nc < hypothetical_house_ptr->c + hypothetical_house_ptr->w) {
                                    is_hypothetical_cell = true;
                                }
                            }
                            if (!is_hypothetical_cell) {
                                connected_to_tc = true;
                                break;
                            }
                        }
                    }
                    if (connected_to_tc) break;
                }
                if (!connected_to_tc) return false;
            }
        }
        return true;
    }

    int _count_adjacent_roads(int r, int c, int h, int w) const {
        int road_sides = 0;

        bool top_has_road = false;
        for (int j = c; j < c + w; ++j) {
            if (r - 1 >= 0 && grid_[r - 1][j] == ROAD) {
                top_has_road = true;
                break;
            }
        }
        if (top_has_road) road_sides++;

        bool bottom_has_road = false;
        for (int j = c; j < c + w; ++j) {
            if (r + h < rows_ && grid_[r + h][j] == ROAD) {
                bottom_has_road = true;
                break;
            }
        }
        if (bottom_has_road) road_sides++;

        bool left_has_road = false;
        for (int i = r; i < r + h; ++i) {
            if (c - 1 >= 0 && grid_[i][c - 1] == ROAD) {
                left_has_road = true;
                break;
            }
        }
        if (left_has_road) road_sides++;

        bool right_has_road = false;
        for (int i = r; i < r + h; ++i) {
            if (c + w < cols_ && grid_[i][c + w] == ROAD) {
                right_has_road = true;
                break;
            }
        }
        if (right_has_road) road_sides++;

        return road_sides;
    }

    double _calculate_reward_shaping(int r, int c) {
        double reward_bonus = 0.0;
        
        if (!ENABLE_REWARD_SHAPING) {
            return reward_bonus;
        }

        if (_is_block_aligned(r, c)) {
            reward_bonus += REWARD_BLOCK_ALIGNED;
        }
        
        int extensions = _count_potential_block_extensions(r, c);
        reward_bonus += extensions * REWARD_EXTENSION_BONUS_PER_POTENTIAL;
        
        if (last_house_pos_.r != -1) {
            double distance = std::sqrt(calculate_distance_sq({r, c}, last_house_pos_));
            if (distance <= 5) {
                reward_bonus += REWARD_CLOSE_TO_LAST_HOUSE;
            }
        }
        
        double tc_distance = std::sqrt(calculate_distance_sq({r, c}, tc_center_));
        double radius = std::sqrt(static_cast<double>(TOWN_CENTER_RADIUS_SQ));
        double tc_bonus = std::max(0.0, REWARD_TC_INFLUENCE_FACTOR * (1.0 - tc_distance / radius));
        reward_bonus += tc_bonus;
        
        int adjacent_roads = _count_adjacent_roads(r, c, HOUSE_H, HOUSE_W);
        if (adjacent_roads > 2) {
            reward_bonus += PENALTY_TOO_MANY_ROADSIDES; // Penalty is a negative reward
        }
        
        return reward_bonus;
    }

    std::tuple<State, double, bool> step(int action_idx) {
        Pos pos = _action_to_position(action_idx);
        int r = pos.r;
        int c = pos.c;
        int house_h = HOUSE_H;
        int house_w = HOUSE_W;
        double reward = 0.0;
        bool done = false;
        
        if (check_overlap(r, c, house_h, house_w, placed_buildings_info_)) {
            reward = PENALTY_INVALID_PLACEMENT;
            consecutive_successes_ = 0;
            return std::make_tuple(_get_reduced_state(), reward, done);
        }

        if (!is_area_road(grid_, r, c, house_h, house_w)) {
            reward = PENALTY_INVALID_PLACEMENT;
            consecutive_successes_ = 0;
            return std::make_tuple(_get_reduced_state(), reward, done);
        }

        std::vector<Pos> house_coords = get_building_coords(r, c, house_h, house_w);
        if (!is_within_influence(house_coords, tc_center_)) {
            reward = PENALTY_INVALID_PLACEMENT;
            consecutive_successes_ = 0;
            return std::make_tuple(_get_reduced_state(), reward, done);
        }

        // Call connectivity check with hypothetical building, no grid copy here
        if (_check_all_buildings_connectivity(r, c)) {
            place_building_on_grid(grid_, HOUSE, r, c, house_h, house_w);
            placed_buildings_info_.push_back({r, c, house_h, house_w, HOUSE});
            house_positions_.push_back({r, c});
            num_houses_placed_++;
            last_house_pos_ = {r + house_h / 2, c + house_w / 2};
            reward = REWARD_PLACEMENT_SUCCESS;
            consecutive_successes_++;
            
            reward += _calculate_reward_shaping(r, c);
            
            if (consecutive_successes_ >= 5) {
                reward += REWARD_CONSECUTIVE_SUCCESS_BONUS;
            }
            
            if (num_houses_placed_ > 25) {
                reward += REWARD_HIGH_HOUSE_COUNT_BONUS;
            }
            
            // Update caches only after actual placement
            road_reach_from_kontor_cache_ = bfs_road_reachable(grid_, kontor_coords_list_);
            road_reach_from_tc_cache_ = bfs_road_reachable(grid_, tc_coords_list_);
        } else {
            reward = PENALTY_DISCONNECTED_HOUSE;
            consecutive_successes_ = 0;
        }

        int num_road_cells = 0;
        for(int r_idx = 0; r_idx < rows_; ++r_idx) {
            for(int c_idx = 0; c_idx < cols_; ++c_idx) {
                if(grid_[r_idx][c_idx] == ROAD) {
                    num_road_cells++;
                }
            }
        }

        if (num_road_cells < HOUSE_H * HOUSE_W ||
            num_houses_placed_ >= (rows_ * cols_) / (HOUSE_H * HOUSE_W) / 2) {
            done = true;
            
            if (episode_end_processing_) {
                int additional_houses = _fill_gaps_greedy();
                if (additional_houses > 0) {
                    reward += additional_houses * REWARD_GREEDY_ADDITIONAL_HOUSE;
                }
            }
        }
        
        return std::make_tuple(_get_reduced_state(), reward, done);
    }

    std::vector<int> get_valid_actions() {
        return _get_valid_actions();
    }

    int get_num_connected_houses() const {
        return num_houses_placed_;
    }

    const std::vector<std::vector<int>>& get_grid() const {
        return grid_;
    }

private:
    int rows_, cols_;
    std::vector<std::vector<int>> grid_;
    std::vector<PlacedBuilding> placed_buildings_info_;
    std::vector<Pos> kontor_coords_list_;
    std::vector<Pos> tc_coords_list_;
    Pos tc_center_;
    int num_houses_placed_;
    int consecutive_successes_;
    Pos last_house_pos_;
    std::vector<Pos> house_positions_;
    std::vector<std::vector<bool>> road_reach_from_kontor_cache_;
    std::vector<std::vector<bool>> road_reach_from_tc_cache_;
    bool episode_end_processing_;
    std::mt19937 rng_;
};

class EnhancedQLearningAgent {
public:
    EnhancedQLearningAgent(double alpha = 0.1, double gamma = 0.99, double epsilon = 1.0, double epsilon_decay_rate = 0.995, double min_epsilon = 0.05) :
        alpha_(alpha), gamma_(gamma), epsilon_(epsilon), epsilon_decay_rate_(epsilon_decay_rate), min_epsilon_(min_epsilon), experience_replay_(5000) {
        std::random_device rd;
        rng_ = std::mt19937(rd());
    }

    std::map<int, double> _get_q_values(const State& state, const std::vector<int>& valid_actions) {
        if (q_table_.find(state) == q_table_.end()) {
            for (int action : valid_actions) {
                q_table_[state][action] = 0.0;
            }
        }
        
        
        for (int action : valid_actions) {
            if (q_table_[state].find(action) == q_table_[state].end()) {
                q_table_[state][action] = 0.0;
            }
        }
        
        std::map<int, double> current_q_values;
        for (int action : valid_actions) {
            current_q_values[action] = q_table_[state][action];
        }
        return current_q_values;
    }

    int choose_action(const State& state, const std::vector<int>& valid_actions) {
        if (valid_actions.empty()) {
            return 0;
        }
        
        std::uniform_real_distribution<> dis(0.0, 1.0);
        if (dis(rng_) < epsilon_) {
            std::uniform_int_distribution<> action_dist(0, valid_actions.size() - 1);
            return valid_actions[action_dist(rng_)];
        }
        
        std::map<int, double> q_values = _get_q_values(state, valid_actions);
        
        
        bool all_same = true;
        if (!q_values.empty()) {
            double first_q = q_values.begin()->second;
            for (const auto& pair : q_values) {
                if (pair.second != first_q) {
                    all_same = false;
                    break;
                }
            }
        }

        if (all_same && !q_values.empty()) {
            
            std::uniform_int_distribution<> action_dist(0, valid_actions.size() - 1);
            return valid_actions[action_dist(rng_)];
        }

        double max_q = -std::numeric_limits<double>::infinity();
        for (const auto& pair : q_values) {
            if (pair.second > max_q) {
                max_q = pair.second;
            }
        }

        std::vector<int> best_actions;
        for (const auto& pair : q_values) {
            if (pair.second == max_q) {
                best_actions.push_back(pair.first);
            }
        }
        std::uniform_int_distribution<> action_dist(0, best_actions.size() - 1);
        return best_actions[action_dist(rng_)];
    }

    void learn(const State& state, int action, double reward, const State& next_state, const std::vector<int>& valid_next_actions) {
        experience_replay_.add(state, action, reward, next_state);
        
        std::vector<ExperienceReplay::Experience> experiences = experience_replay_.sample(std::min((size_t)32, experience_replay_.size()));
        
        for (const auto& exp : experiences) {
            if (q_table_.find(exp.state) == q_table_.end()) {
                q_table_[exp.state] = {};
            }
            if (q_table_[exp.state].find(exp.action) == q_table_[exp.state].end()) {
                q_table_[exp.state][exp.action] = 0.0;
            }
            
            double current_q = q_table_[exp.state][exp.action];
            
            double max_future_q = 0.0;
            if (q_table_.count(exp.next_state) && !q_table_[exp.next_state].empty()) {
                for (const auto& pair : q_table_[exp.next_state]) {
                    if (pair.second > max_future_q) {
                        max_future_q = pair.second;
                    }
                }
            }
            
            double new_q = current_q + alpha_ * (exp.reward + gamma_ * max_future_q - current_q);
            q_table_[exp.state][exp.action] = new_q;
        }
    }

    void decay_epsilon(double episode_reward = 0.0) {
        epsilon_ *= epsilon_decay_rate_;
        epsilon_ = std::max(min_epsilon_, epsilon_);
    }

    double get_epsilon() const {
        return epsilon_;
    }

    size_t get_q_table_size() const {
        return q_table_.size();
    }

private:
    std::map<State, std::map<int, double>> q_table_;
    double alpha_;
    double gamma_;
    double epsilon_;
    double epsilon_decay_rate_;
    double min_epsilon_;
    ExperienceReplay experience_replay_;
    std::mt19937 rng_;
};


bool is_area_road(const std::vector<std::vector<int>>& grid, int r, int c, int h, int w) {
    int rows = grid.size();
    int cols = grid[0].size();
    if (!(r >= 0 && r < rows - h + 1 && c >= 0 && c < cols - w + 1)) {
        return false;
    }
    for (int i = r; i < r + h; ++i) {
        for (int j = c; j < c + w; ++j) {
            if (grid[i][j] != ROAD) {
                return false;
            }
        }
    }
    return true;
}

void place_building_on_grid(std::vector<std::vector<int>>& grid, int building_type_val, int r, int c, int h, int w) {
    for (int i = r; i < r + h; ++i) {
        for (int j = c; j < c + w; ++j) {
            grid[i][j] = building_type_val;
        }
    }
}

std::vector<Pos> get_building_coords(int r, int c, int h, int w) {
    std::vector<Pos> coords;
    coords.reserve(h * w);
    // Use pre-calculated relative coordinates for efficiency if it's a house
    if (h == HOUSE_H && w == HOUSE_W) {
        for (const auto& rel_pos : HOUSE_RELATIVE_COORDS) {
            coords.push_back({r + rel_pos.r, c + rel_pos.c});
        }
    } else {
        // Fallback for other building types if their relative coords are not pre-calculated
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                coords.push_back({r + i, c + j});
            }
        }
    }
    return coords;
}

double calculate_distance_sq(Pos p1, Pos p2) {
    return static_cast<double>((p1.r - p2.r) * (p1.r - p2.r) + (p1.c - p2.c) * (p1.c - p2.c));
}

bool is_within_influence(const std::vector<Pos>& house_coords, const Pos& tc_center) {
    if (tc_center.r == -1) return false; 
    int count = 0;
    for (const auto& coord : house_coords) {
        if (calculate_distance_sq(coord, tc_center) <= TOWN_CENTER_RADIUS_SQ) {
            count++;
        }
    }
    return count >= 5;
}


// Modified BFS function to account for a hypothetical building
std::vector<std::vector<bool>> bfs_road_reachable(
    const std::vector<std::vector<int>>& grid,
    const std::vector<Pos>& start_building_coords,
    const PlacedBuilding* hypothetical_building
) {
    int rows = grid.size();
    int cols = grid[0].size();
    std::vector<std::vector<bool>> visited_roads(rows, std::vector<bool>(cols, false));
    std::deque<Pos> q;

    auto is_hypothetical_cell = [&](int r, int c) {
        if (hypothetical_building) {
            return r >= hypothetical_building->r && r < hypothetical_building->r + hypothetical_building->h &&
                   c >= hypothetical_building->c && c < hypothetical_building->c + hypothetical_building->w;
        }
        return false;
    };

    for (const auto& br_bc : start_building_coords) {
        for (const auto& dir : {Pos{-1,0}, Pos{1,0}, Pos{0,-1}, Pos{0,1}}) {
            int nr = br_bc.r + dir.r;
            int nc = br_bc.c + dir.c;
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == ROAD && !is_hypothetical_cell(nr, nc)) {
                if (!visited_roads[nr][nc]) {
                    visited_roads[nr][nc] = true;
                    q.push_back({nr, nc});
                }
            }
        }
    }
    
    while (!q.empty()) {
        Pos current = q.front();
        q.pop_front();

        for (const auto& dir : {Pos{0,1}, Pos{0,-1}, Pos{1,0}, Pos{-1,0}}) {
            int nr = current.r + dir.r;
            int nc = current.c + dir.c;
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !visited_roads[nr][nc]) {
                if (grid[nr][nc] == ROAD && !is_hypothetical_cell(nr, nc)) {
                    visited_roads[nr][nc] = true;
                    q.push_back({nr, nc});
                }
            }
        }
    }
    return visited_roads;
}

// Modified is_connected_by_road to pass hypothetical building to BFS
bool is_connected_by_road(
    const std::vector<std::vector<int>>& grid,
    const std::vector<Pos>& start_building_coords,
    const std::vector<Pos>& target_building_coords,
    const PlacedBuilding* hypothetical_building
) {
    std::vector<std::vector<bool>> reachable_roads = bfs_road_reachable(grid, start_building_coords, hypothetical_building);
    
    bool any_road_reachable_from_start = false;
    for(const auto& row : reachable_roads) {
        for(bool cell : row) {
            if(cell) {
                any_road_reachable_from_start = true;
                break;
            }
        }
        if(any_road_reachable_from_start) break;
    }
    if (!any_road_reachable_from_start) return false;

    int rows = grid.size();
    int cols = grid[0].size();

    auto is_hypothetical_cell = [&](int r, int c) {
        if (hypothetical_building) {
            return r >= hypothetical_building->r && r < hypothetical_building->r + hypothetical_building->h &&
                   c >= hypothetical_building->c && c < hypothetical_building->c + hypothetical_building->w;
        }
        return false;
    };

    for (const auto& tr_tc : target_building_coords) {
        for (const auto& dir : {Pos{-1,0}, Pos{1,0}, Pos{0,-1}, Pos{0,1}}) {
            int nr = tr_tc.r + dir.r;
            int nc = tr_tc.c + dir.c;
            
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == ROAD && reachable_roads[nr][nc] && !is_hypothetical_cell(nr, nc)) { 
                return true;
            }
        }
    }
    return false;
}


bool check_overlap(int new_r, int new_c, int new_h, int new_w, const std::vector<PlacedBuilding>& placed_buildings_info) {
    for (const auto& existing_building : placed_buildings_info) {
        int r_exist = existing_building.r;
        int c_exist = existing_building.c;
        int h_exist = existing_building.h;
        int w_exist = existing_building.w;

        int x_overlap = std::max(0, std::min(new_c + new_w, c_exist + w_exist) - std::max(new_c, c_exist));
        int y_overlap = std::max(0, std::min(new_r + new_h, r_exist + h_exist) - std::max(new_r, r_exist));
        if (x_overlap > 0 && y_overlap > 0) {
            return true;
        }
    }
    return false;
}


void write_grid_to_file(const std::vector<std::vector<int>>& grid, const std::string& filename, int num_houses) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    outfile << "Best Solution - Houses: " << num_houses << "\n";
    outfile << "Grid Layout:\n";

    for (const auto& row : grid) {
        for (int cell_val : row) {
            switch (cell_val) {
                case EMPTY: outfile << " "; break;
                case HOUSE: outfile << "H"; break;
                case ROAD: outfile << "."; break;
                case TOWN_CENTER: outfile << "T"; break;
                case TRADING_POST: outfile << "P"; break;
                default: outfile << "?"; break;
            }
        }
        outfile << "\n";
    }
    outfile.close();
}


void train_enhanced_rl_agent(EnhancedCityPlanningEnv& env, EnhancedQLearningAgent& agent, int num_episodes) {
    std::vector<std::vector<int>> best_overall_grid;
    int max_overall_houses = -1;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        State state = env.reset();
        bool done = false;
        double total_reward = 0.0;
        int num_actions_this_episode = 0;
        int max_actions_per_episode = 150;

        while (!done && num_actions_this_episode < max_actions_per_episode) {
            std::vector<int> valid_actions = env.get_valid_actions();
            int action = agent.choose_action(state, valid_actions);
            
            State next_state;
            double reward;
            std::tie(next_state, reward, done) = env.step(action);
            
            std::vector<int> next_valid_actions = done ? std::vector<int>() : env.get_valid_actions();
            agent.learn(state, action, reward, next_state, next_valid_actions);
            state = next_state;
            total_reward += reward;
            num_actions_this_episode++;
        }

        agent.decay_epsilon(total_reward);

        int current_houses = env.get_num_connected_houses();
        if (current_houses > max_overall_houses) {
            max_overall_houses = current_houses;
            best_overall_grid = env.get_grid();
        }

        
        if (episode % 100 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - start_time;
            std::cout << "Episode " << episode << "/" << num_episodes
                    << ", Current Houses: " << current_houses
                    << ", Epsilon: " << agent.get_epsilon()
                    << ", Best Overall Houses: " << max_overall_houses
                    << ", Time: " << elapsed.count() << "s"
                    << ", Q-Table Size: " << agent.get_q_table_size() << std::endl;
            start_time = std::chrono::high_resolution_clock::now();
        }
    }

    if (!best_overall_grid.empty() && max_overall_houses >= 0) {
        std::cout << "\nEnhanced QRL-Training completed. Best found solution had " << max_overall_houses << " validly connected houses." << std::endl;
        write_grid_to_file(best_overall_grid, "best_city_layout.txt", max_overall_houses);
        std::cout << "Best grid saved to best_city_layout.txt" << std::endl;
    } else {
        std::cout << "\nEnhanced QRL-Agent could not find or learn a valid placement." << std::endl;
    }
}


int main() {
    int grid_rows = 30;
    int grid_cols = 40;

    EnhancedCityPlanningEnv env(grid_rows, grid_cols);
    EnhancedQLearningAgent agent(
        0.1,
        0.99,
        1.0,
        0.9997,
        0.05
    );

    std::cout << "Enhanced QRL-Training on a " << grid_rows << "x" << grid_cols << " grid..." << std::endl;
    std::cout << "Features: Extended State Space, Reward Shaping (Block-arrangement)," << std::endl;
    std::cout << "          Experience Replay, Continuous Epsilon Decay, Greedy Gap Filling" << std::endl;
    train_enhanced_rl_agent(env, agent, 10000);
    return 0;
}