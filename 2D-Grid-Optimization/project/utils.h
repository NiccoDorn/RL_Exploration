#pragma once

#include <vector>
#include <string>
#include <map>

#include "common.h"

extern std::map<int, Dimensions> BUILDING_TYPES;
extern const std::vector<Pos> HOUSE_RELATIVE_COORDS;

bool is_area_road_flat(const std::vector<int>& grid, int cols, int r, int c, int h, int w);
void place_building_on_grid_flat(std::vector<int>& grid, int cols, int building_type_val, int r, int c, int h, int w);
std::vector<Pos> get_building_coords(int r, int c, int h, int w);
inline double calculate_distance_sq(Pos p1, Pos p2) noexcept;
bool is_within_influence(const std::vector<Pos>& house_coords, const Pos& tc_center);

// Flat grid versions with hypothetical building support
std::vector<std::vector<bool>> bfs_road_reachable_flat(
    const std::vector<int>& grid,
    int rows,
    int cols,
    const std::vector<Pos>& start_building_coords,
    const PlacedBuilding* hypothetical_building = nullptr
);

bool is_connected_by_road_flat(
    const std::vector<int>& grid,
    int rows,
    int cols,
    const std::vector<Pos>& start_building_coords,
    const std::vector<Pos>& target_building_coords,
    const PlacedBuilding* hypothetical_building = nullptr
);

// Legacy 2D grid versions (for compatibility)
std::vector<std::vector<bool>> bfs_road_reachable(
    const std::vector<std::vector<int>>& grid,
    const std::vector<Pos>& start_building_coords,
    const PlacedBuilding* hypothetical_building = nullptr
);

bool is_connected_by_road(
    const std::vector<std::vector<int>>& grid,
    const std::vector<Pos>& start_building_coords,
    const std::vector<Pos>& target_building_coords,
    const PlacedBuilding* hypothetical_building = nullptr
);

bool check_overlap(int new_r, int new_c, int new_h, int new_w, const std::vector<PlacedBuilding>& placed_buildings_info);
void write_grid_to_file(const std::vector<int>& grid, int rows, int cols, const std::string& filename, int num_houses);