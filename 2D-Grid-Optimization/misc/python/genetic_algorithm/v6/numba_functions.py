import datetime
import multiprocessing
import os
import pickle

import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import time

import numba
from numba import njit

EMPTY = 0
HOUSE = 1
ROAD = 2
TOWN_CENTER = 3
TRADING_POST = 4
HOUSE_DIMS = (3, 3)
TOWN_CENTER_DIMS = (7, 5)
TRADING_POST_DIMS = (1, 5)
TOWN_CENTER_RADIUS_SQ = 26 ** 2
BUILDING_TYPES = {
    HOUSE: HOUSE_DIMS,
    TOWN_CENTER: TOWN_CENTER_DIMS,
    TRADING_POST: TRADING_POST_DIMS
}


@njit
def is_area_road_numba(grid, r, c, h, w):
    rows, cols = grid.shape
    if not (0 <= r < rows - h + 1 and 0 <= c < cols - w + 1):
        return False
    for i in range(h):
        for j in range(w):
            if grid[r + i, c + j] != ROAD:
                return False
    return True

def is_area_road(grid, r, c, h, w):
    return is_area_road_numba(grid, r, c, h, w)

@njit
def place_building_on_grid_numba(grid, building_type_val, r, c, h, w):
    for i in range(h):
        for j in range(w):
            grid[r + i, c + j] = building_type_val

def place_building_on_grid(grid, building_type_val, r, c, h, w):
    place_building_on_grid_numba(grid, building_type_val, r, c, h, w)


@njit
def get_building_coords_numba(r, c, h, w):
    coords = np.empty((h * w, 2), dtype=np.int32)
    idx = 0
    for i in range(h):
        for j in range(w):
            coords[idx, 0] = r + i
            coords[idx, 1] = c + j
            idx += 1
    return coords

def get_building_coords(r, c, h, w):
    return get_building_coords_numba(r, c, h, w)

@njit
def calculate_distance_sq_numba(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def calculate_distance_sq(p1, p2):
    return calculate_distance_sq_numba(p1, p2)

@njit
def is_within_influence_numba(house_coords, tc_center):
    if tc_center[0] == -1:  # Use -1 to indicate None
        return False
    count = 0
    for i in range(house_coords.shape[0]):
        if calculate_distance_sq_numba(house_coords[i], tc_center) <= TOWN_CENTER_RADIUS_SQ:
            count += 1
    return count >= 5

def is_within_influence(house_coords, tc_center):
    if tc_center is None:
        return False
    tc_center_array = np.array(tc_center, dtype=np.int32)
    return is_within_influence_numba(house_coords, tc_center_array)


@njit
def check_overlap_numba(new_r, new_c, new_h, new_w, placed_buildings):
    for i in range(placed_buildings.shape[0]):
        r_exist, c_exist, h_exist, w_exist = placed_buildings[i]
        x_overlap = max(0, min(new_c + new_w, c_exist + w_exist) - max(new_c, c_exist))
        y_overlap = max(0, min(new_r + new_h, r_exist + h_exist) - max(new_r, r_exist))
        if x_overlap > 0 and y_overlap > 0:
            return True
    return False

def check_overlap(new_r, new_c, new_h, new_w, placed_buildings_info):
    if not placed_buildings_info:
        return False
    placed_buildings = np.array([(r, c, h, w) for r, c, h, w, _ in placed_buildings_info], dtype=np.int32)
    return check_overlap_numba(new_r, new_c, new_h, new_w, placed_buildings)


@njit
def check_connectivity_numba(grid, start_coords, target_coords):
    """Check if target is connected to start - returns boolean"""
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=numba.boolean)
    queue = np.empty((rows * cols, 2), dtype=np.int32)
    queue_start = 0
    queue_end = 0

    # Add starting positions to queue
    for i in range(start_coords.shape[0]):
        br, bc = start_coords[i]
        # Check adjacent cells
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = br + dr, bc + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == ROAD:
                if not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue[queue_end, 0] = nr
                    queue[queue_end, 1] = nc
                    queue_end += 1

    # BFS
    while queue_start < queue_end:
        r, c = queue[queue_start, 0], queue[queue_start, 1]
        queue_start += 1

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                if grid[nr, nc] == ROAD:
                    visited[nr, nc] = True
                    queue[queue_end, 0] = nr
                    queue[queue_end, 1] = nc
                    queue_end += 1

    # Check if target is connected
    for i in range(target_coords.shape[0]):
        tr, tc = target_coords[i]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = tr + dr, tc + dc
            if 0 <= nr < rows and 0 <= nc < cols and visited[nr, nc] and grid[nr, nc] == ROAD:
                return True
    return False

# Split the BFS function into two separate functions
@njit
def bfs_road_connectivity_visited_numba(grid, start_coords):
    """Returns visited array for BFS connectivity"""
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=numba.boolean)
    queue = np.empty((rows * cols, 2), dtype=np.int32)
    queue_start = 0
    queue_end = 0

    # Add starting positions to queue
    for i in range(start_coords.shape[0]):
        br, bc = start_coords[i]
        # Check adjacent cells
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = br + dr, bc + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == ROAD:
                if not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue[queue_end, 0] = nr
                    queue[queue_end, 1] = nc
                    queue_end += 1

    # BFS
    while queue_start < queue_end:
        r, c = queue[queue_start, 0], queue[queue_start, 1]
        queue_start += 1

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                if grid[nr, nc] == ROAD:
                    visited[nr, nc] = True
                    queue[queue_end, 0] = nr
                    queue[queue_end, 1] = nc
                    queue_end += 1

    return visited


def bfs_road_connectivity(grid, start_building_coords, target_building_coords=None):
    """Wrapper function to handle both connectivity check and visited array return"""
    start_coords = np.array(start_building_coords, dtype=np.int32)

    if target_building_coords is not None:
        target_coords = np.array(target_building_coords, dtype=np.int32)
        return check_connectivity_numba(grid, start_coords, target_coords)
    else:
        return bfs_road_connectivity_visited_numba(grid, start_coords)


@njit
def check_all_buildings_connectivity_numba(current_grid, houses_info_array,
                                           kontor_coords, tc_coords, rows, cols):
    """Numba-compatible version of _check_all_buildings_connectivity"""

    # Check if kontor and town center are connected
    if not check_connectivity_numba(current_grid, kontor_coords, tc_coords):
        return False

    # Get reachability from kontor and town center
    current_kontor_reach = bfs_road_connectivity_visited_numba(current_grid, kontor_coords)
    current_tc_reach = bfs_road_connectivity_visited_numba(current_grid, tc_coords)

    # Check each house connectivity
    for i in range(houses_info_array.shape[0]):
        r, c, h, w, b_type = houses_info_array[i]

        if b_type == HOUSE:
            house_coords = get_building_coords_numba(r, c, h, w)

            # Check connection to kontor
            connected_to_kontor = False
            for j in range(house_coords.shape[0]):
                hr, hc = house_coords[j]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = hr + dr, hc + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                            current_grid[nr, nc] == ROAD and current_kontor_reach[nr, nc]):
                        connected_to_kontor = True
                        break
                if connected_to_kontor:
                    break

            if not connected_to_kontor:
                return False

            # Check connection to town center
            connected_to_tc = False
            for j in range(house_coords.shape[0]):
                hr, hc = house_coords[j]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = hr + dr, hc + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                            current_grid[nr, nc] == ROAD and current_tc_reach[nr, nc]):
                        connected_to_tc = True
                        break
                if connected_to_tc:
                    break

            if not connected_to_tc:
                return False

    return True


@numba.jit(nopython=True)
def generate_random_positions_numba(num_positions, rows, cols, house_dims):
    """Generate multiple random positions at once"""
    positions = np.empty((num_positions, 2), dtype=np.int32)
    for i in range(num_positions):
        positions[i, 0] = np.random.randint(0, rows - house_dims[0])
        positions[i, 1] = np.random.randint(0, cols - house_dims[1])
    return positions