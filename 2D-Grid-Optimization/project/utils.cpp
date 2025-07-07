#include <deque>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>

#include "utils.h"

std::map<int, Dimensions> BUILDING_TYPES = {
    {HOUSE, {HOUSE_H, HOUSE_W}},
    {TOWN_CENTER, {TOWN_CENTER_H, TOWN_CENTER_W}},
    {TRADING_POST, {TRADING_POST_H, TRADING_POST_W}}
};

const std::vector<Pos> HOUSE_RELATIVE_COORDS = {
    {0,0}, {0,1}, {0,2},
    {1,0}, {1,1}, {1,2},
    {2,0}, {2,1}, {2,2}
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
    if (h == HOUSE_H && w == HOUSE_W) {
        for (const auto& rel_pos : HOUSE_RELATIVE_COORDS) {
            coords.push_back({r + rel_pos.r, c + rel_pos.c});
        }
    } else {
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
                case EMPTY: outfile << "  "; break;
                case HOUSE: outfile << "H "; break;
                case ROAD: outfile << ". "; break;
                case TOWN_CENTER: outfile << "T "; break;
                case TRADING_POST: outfile << "P "; break;
                default: outfile << "? "; break;
            }
        }
        outfile << "\n";
    }
    outfile.close();
}