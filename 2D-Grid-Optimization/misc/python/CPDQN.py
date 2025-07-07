import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import copy
import time
import math
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

EMPTY = 0
HOUSE = 1
ROAD = 2
TOWN_CENTER = 3
TRADING_POST = 4

HOUSE_DIMS = (3, 3)
TOWN_CENTER_DIMS = (7, 5)
TRADING_POST_DIMS = (1, 5)

TOWN_CENTER_RADIUS_SQ = 26**2

BUILDING_TYPES = {
    HOUSE: HOUSE_DIMS,
    TOWN_CENTER: TOWN_CENTER_DIMS,
    TRADING_POST: TRADING_POST_DIMS
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_area_road(grid, r, c, h, w):
    rows, cols = grid.shape
    if not (0 <= r < rows - h + 1 and 0 <= c < cols - w + 1): return False
    return np.all(grid[r:r+h, c:c+w] == ROAD)

def place_building_on_grid(grid, building_type_val, r, c, h, w):
    grid[r:r+h, c:c+w] = building_type_val

def get_building_coords(r, c, h, w):
    return [(r + i, c + j) for i in range(h) for j in range(w)]

def calculate_distance_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def is_within_influence(house_coords, tc_center):
    if tc_center is None: return False
    return sum(1 for coord in house_coords if calculate_distance_sq(coord, tc_center) <= TOWN_CENTER_RADIUS_SQ) >= 5

def bfs_road_connectivity(grid, start_building_coords, target_building_coords=None):
    rows, cols = grid.shape
    visited_roads = np.zeros_like(grid, dtype=bool)
    q = deque()

    for br, bc in start_building_coords:
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = br + dr, bc + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == ROAD:
                if not visited_roads[nr, nc]:
                    visited_roads[nr, nc] = True
                    q.append((nr, nc))

    if not q:
        if target_building_coords: return False
        else: return visited_roads

    while q:
        r, c = q.popleft()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited_roads[nr, nc]:
                if grid[nr, nc] == ROAD:
                    visited_roads[nr, nc] = True
                    q.append((nr, nc))

    if target_building_coords:
        for tr, tc in target_building_coords:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = tr + dr, tc + dc
                if 0 <= nr < rows and 0 <= nc < cols and visited_roads[nr, nc] and grid[nr, nc] == ROAD: return True
        return False
    return visited_roads

def check_overlap(new_r, new_c, new_h, new_w, placed_buildings_info):
    for r_exist, c_exist, h_exist, w_exist, _ in placed_buildings_info:
        x_overlap = max(0, min(new_c + new_w, c_exist + w_exist) - max(new_c, c_exist))
        y_overlap = max(0, min(new_r + new_h, r_exist + h_exist) - max(new_r, r_exist))
        if x_overlap > 0 and y_overlap > 0: return True
    return False

class ExperienceReplay:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size=32):
        if len(self.buffer) < batch_size: return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

class EnhancedCityPlanningEnv:
    def __init__(self, grid_dims):
        self.grid_dims = grid_dims
        self.rows, self.cols = grid_dims
        self.grid = None
        self.kontor_coords_list = []
        self.tc_coords_list = []
        self.tc_center = None
        self.placed_buildings_info = []
        self.num_houses_placed = 0
        self.consecutive_successes = 0
        self.last_house_pos = None
        self.house_positions = []
        self.road_reach_from_kontor_cache = None
        self.road_reach_from_tc_cache = None
        self.episode_end_processing = False # end_processing = in leeren 3x3 Räumen Häuserpläatzierung greedy probieren
        self.reset()

    def _get_reduced_state(self):
        # MC Input für CNN
        # Channel 0-4: One-hot encoding für Gebäudetypen
        state_channels = np.zeros((5, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                state_channels[int(self.grid[r, c]), r, c] = 1.0

        # Channel 5: normalisierte euklidische Distanz zum TC center
        tc_dist_channel = np.zeros((self.rows, self.cols), dtype=np.float32)
        if self.tc_center:
            max_dist = math.sqrt(self.rows**2 + self.cols**2)
            for r in range(self.rows):
                for c in range(self.cols):
                    dist = math.sqrt(calculate_distance_sq((r, c), self.tc_center))
                    tc_dist_channel[r, c] = dist / max_dist if max_dist > 0 else 0
        state_channels = np.concatenate((state_channels, tc_dist_channel[np.newaxis, :, :]), axis=0)

        # Channel 6: Straßenverbindung zum Kontor
        kontor_reach_channel = np.zeros((self.rows, self.cols), dtype=np.float32)
        if self.road_reach_from_kontor_cache is not None:
            kontor_reach_channel = self.road_reach_from_kontor_cache.astype(np.float32)
        state_channels = np.concatenate((state_channels, kontor_reach_channel[np.newaxis, :, :]), axis=0)

        # Channel 7: Straßenverbindung zum Dorfzentrum
        tc_reach_channel = np.zeros((self.rows, self.cols), dtype=np.float32)
        if self.road_reach_from_tc_cache is not None:
            tc_reach_channel = self.road_reach_from_tc_cache.astype(np.float32)
        state_channels = np.concatenate((state_channels, tc_reach_channel[np.newaxis, :, :]), axis=0)
        
        return state_channels

    def _is_block_aligned(self, r, c):
        if not self.house_positions: return True
        
        for house_r, house_c in self.house_positions:
            if abs(r - house_r) < HOUSE_DIMS[0] and (c == house_c + HOUSE_DIMS[1] or c == house_c - HOUSE_DIMS[1]):
                return True
            if abs(c - house_c) < HOUSE_DIMS[1] and (r == house_r + HOUSE_DIMS[0] or r == house_r - HOUSE_DIMS[0]):
                return True
        return False

    def _count_potential_block_extensions(self, r, c):
        extensions = 0
        directions = [
            (0, HOUSE_DIMS[1]),
            (0, -HOUSE_DIMS[1]),
            (HOUSE_DIMS[0], 0),
            (-HOUSE_DIMS[0], 0)
        ]
        
        for dr, dc in directions:
            new_r, new_c = r + dr, c + dc
            if (0 <= new_r < self.rows - HOUSE_DIMS[0] + 1 and 
                0 <= new_c < self.cols - HOUSE_DIMS[1] + 1):
                if self._is_valid_position(new_r, new_c):
                    house_coords = get_building_coords(new_r, new_c, HOUSE_DIMS[0], HOUSE_DIMS[1])
                    if is_within_influence(house_coords, self.tc_center): extensions += 1
        
        return extensions

    def _find_all_valid_gaps(self):
        valid_gaps = []
        for r in range(self.rows - HOUSE_DIMS[0] + 1):
            for c in range(self.cols - HOUSE_DIMS[1] + 1):
                if self._is_valid_position(r, c):
                    house_coords = get_building_coords(r, c, HOUSE_DIMS[0], HOUSE_DIMS[1])
                    if is_within_influence(house_coords, self.tc_center):
                        valid_gaps.append((r, c))
        return valid_gaps

    def _fill_gaps_greedy(self):
        houses_added = 0
        max_attempts = 100
        
        for _ in range(max_attempts):
            valid_gaps = self._find_all_valid_gaps()
            if not valid_gaps:break
            block_aligned_gaps = [(r, c) for r, c in valid_gaps if self._is_block_aligned(r, c)]
            best_gap = None
            if block_aligned_gaps: best_gap = max(block_aligned_gaps, key=lambda pos: self._count_potential_block_extensions(pos[0], pos[1]))
            elif valid_gaps: best_gap = max(valid_gaps, key=lambda pos: self._count_potential_block_extensions(pos[0], pos[1]))
            else: break
            
            r, c = best_gap
            house_h, house_w = HOUSE_DIMS
            
            temp_grid = copy.deepcopy(self.grid)
            if self._check_all_buildings_connectivity(temp_grid, r, c):
                place_building_on_grid(self.grid, HOUSE, r, c, house_h, house_w)
                self.placed_buildings_info.append((r, c, house_h, house_w, HOUSE))
                self.house_positions.append((r, c))
                self.num_houses_placed += 1
                houses_added += 1
                self.road_reach_from_kontor_cache = bfs_road_connectivity(self.grid, self.kontor_coords_list)
                self.road_reach_from_tc_cache = bfs_road_connectivity(self.grid, self.tc_coords_list)
            else: break
        return houses_added

    def _is_valid_position(self, r, c):
        house_h, house_w = HOUSE_DIMS
        if check_overlap(r, c, house_h, house_w, self.placed_buildings_info): return False
        if not is_area_road(self.grid, r, c, house_h, house_w): return False
        return True

    def _get_valid_actions(self):
        valid_actions = []
        action_idx = 0
        for r in range(self.rows - HOUSE_DIMS[0] + 1):
            for c in range(self.cols - HOUSE_DIMS[1] + 1):
                if self._is_valid_position(r, c):
                    house_coords = get_building_coords(r, c, HOUSE_DIMS[0], HOUSE_DIMS[1])
                    if is_within_influence(house_coords, self.tc_center):
                        valid_actions.append(action_idx)
                action_idx += 1
        
        return valid_actions if valid_actions else [0]

    def _action_to_position(self, action_idx):
        positions = []
        for r in range(self.rows - HOUSE_DIMS[0] + 1):
            for c in range(self.cols - HOUSE_DIMS[1] + 1):
                positions.append((r, c))
        if action_idx < len(positions): return positions[action_idx]
        return (0, 0)

    def reset(self):
        self.grid = np.full(self.grid_dims, ROAD, dtype=int)
        self.placed_buildings_info = []
        self.kontor_coords_list = []
        self.tc_coords_list = []
        self.tc_center = None
        self.num_houses_placed = 0
        self.consecutive_successes = 0
        self.last_house_pos = None
        self.house_positions = []
        
        kontor_r, kontor_c = 0, 0
        kontor_h, kontor_w = TRADING_POST_DIMS
        place_building_on_grid(self.grid, TRADING_POST, kontor_r, kontor_c, kontor_h, kontor_w)
        self.placed_buildings_info.append((kontor_r, kontor_c, kontor_h, kontor_w, TRADING_POST))
        self.kontor_coords_list = get_building_coords(kontor_r, kontor_c, kontor_h, kontor_w)
        
        tc_h, tc_w = TOWN_CENTER_DIMS
        possible_tc_placements = [
            (r, c) for r in range(self.rows - tc_h + 1)
            for c in range(self.cols - tc_w + 1)
        ]
        random.shuffle(possible_tc_placements)
        tc_placed = False

        for r, c in possible_tc_placements:
            if not check_overlap(r, c, tc_h, tc_w, self.placed_buildings_info):
                if is_area_road(self.grid, r, c, tc_h, tc_w):
                    potential_tc_coords = get_building_coords(r, c, tc_h, tc_w)
                    temp_grid = copy.deepcopy(self.grid)
                    place_building_on_grid(temp_grid, TOWN_CENTER, r, c, tc_h, tc_w)
                    
                    tc_is_connected_to_kontor = bfs_road_connectivity(temp_grid, self.kontor_coords_list, potential_tc_coords)
                    
                    if tc_is_connected_to_kontor:
                        place_building_on_grid(self.grid, TOWN_CENTER, r, c, tc_h, tc_w)
                        self.placed_buildings_info.append((r, c, tc_h, tc_w, TOWN_CENTER))
                        self.tc_coords_list = potential_tc_coords
                        self.tc_center = (r + tc_h // 2, c + tc_w // 2)
                        tc_placed = True
                        break

        if not tc_placed:
            print("Warning: Could not place Town Center. Resetting environment again...")
            return self.reset()

        self.road_reach_from_kontor_cache = bfs_road_connectivity(self.grid, self.kontor_coords_list)
        self.road_reach_from_tc_cache = bfs_road_connectivity(self.grid, self.tc_coords_list)
        return self._get_reduced_state()

    def _check_all_buildings_connectivity(self, current_grid, new_house_r=None, new_house_c=None):
        temp_placed_buildings_info = copy.deepcopy(self.placed_buildings_info)
        if new_house_r is not None:
            house_h, house_w = HOUSE_DIMS
            place_building_on_grid(current_grid, HOUSE, new_house_r, new_house_c, house_h, house_w)
            temp_placed_buildings_info.append((new_house_r, new_house_c, house_h, house_w, HOUSE))

        current_kontor_reach = bfs_road_connectivity(current_grid, self.kontor_coords_list)
        current_tc_reach = bfs_road_connectivity(current_grid, self.tc_coords_list)

        if not bfs_road_connectivity(current_grid, self.kontor_coords_list, self.tc_coords_list):
            return False

        for r, c, h, w, b_type in temp_placed_buildings_info:
            if b_type == HOUSE:
                house_coords = get_building_coords(r, c, h, w)
                connected_to_kontor = False
                for hr, hc in house_coords:
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = hr + dr, hc + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and current_grid[nr, nc] == ROAD and current_kontor_reach[nr, nc]:
                            connected_to_kontor = True
                            break
                    if connected_to_kontor: break
                if not connected_to_kontor: return False

                connected_to_tc = False
                for hr, hc in house_coords:
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = hr + dr, hc + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and current_grid[nr, nc] == ROAD and current_tc_reach[nr, nc]:
                            connected_to_tc = True
                            break
                    if connected_to_tc: break
                if not connected_to_tc: return False
        return True

    def _calculate_reward_shaping(self, r, c):
        reward_bonus = 0
        
        if self._is_block_aligned(r, c): reward_bonus += 3
        
        extensions = self._count_potential_block_extensions(r, c)
        reward_bonus += extensions * 0.5
        
        if self.last_house_pos:
            distance = math.sqrt(calculate_distance_sq((r, c), self.last_house_pos))
            if distance <= 5: reward_bonus += 2
        
        tc_distance = math.sqrt(calculate_distance_sq((r, c), self.tc_center))
        radius = math.sqrt(TOWN_CENTER_RADIUS_SQ)
        tc_bonus = max(0, 2 * (1 - tc_distance / radius))
        reward_bonus += tc_bonus
        
        return reward_bonus

    def step(self, action_idx):
        r, c = self._action_to_position(action_idx)
        house_h, house_w = HOUSE_DIMS
        reward = 0
        done = False
        
        if check_overlap(r, c, house_h, house_w, self.placed_buildings_info):
            reward = -1
            self.consecutive_successes = 0
            return self._get_reduced_state(), reward, done

        if not is_area_road(self.grid, r, c, house_h, house_w):
            reward = -1
            self.consecutive_successes = 0
            return self._get_reduced_state(), reward, done

        house_coords = get_building_coords(r, c, house_h, house_w)
        if not is_within_influence(house_coords, self.tc_center):
            reward = -1
            self.consecutive_successes = 0
            return self._get_reduced_state(), reward, done

        temp_grid = copy.deepcopy(self.grid)

        if self._check_all_buildings_connectivity(temp_grid, r, c):
            place_building_on_grid(self.grid, HOUSE, r, c, house_h, house_w)
            self.placed_buildings_info.append((r, c, house_h, house_w, HOUSE))
            self.house_positions.append((r, c))
            self.num_houses_placed += 1
            self.last_house_pos = (r + house_h // 2, c + house_w // 2)
            reward = 1
            self.consecutive_successes += 1
            reward += self._calculate_reward_shaping(r, c)
            if self.consecutive_successes >= 2: reward += 5
            if self.num_houses_placed > 20: reward += 10
            self.road_reach_from_kontor_cache = bfs_road_connectivity(self.grid, self.kontor_coords_list)
            self.road_reach_from_tc_cache = bfs_road_connectivity(self.grid, self.tc_coords_list)
        else:
            reward = -5
            self.consecutive_successes = 0

        num_road_cells = np.sum(self.grid == ROAD)
        if num_road_cells < HOUSE_DIMS[0] * HOUSE_DIMS[1] or \
            self.num_houses_placed >= (self.rows * self.cols) // (HOUSE_DIMS[0] * HOUSE_DIMS[1]) // 2:
            done = True
            
            if self.episode_end_processing:
                additional_houses = self._fill_gaps_greedy()
                if additional_houses > 0:
                    reward += additional_houses * 5
        
        return self._get_reduced_state(), reward, done

    def get_valid_actions(self):
        return self._get_valid_actions()

    def get_num_connected_houses(self):
        return self.num_houses_placed

class QNetwork(nn.Module):
    def __init__(self, input_channels, grid_rows, grid_cols, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        linear_input_size = 128 * 4 * 4
        
        print(f"QNetwork Init: Using adaptive pooling, linear_input_size={linear_input_size}")
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, input_channels, grid_rows, grid_cols, num_actions,
                alpha=0.0005, gamma=0.99, epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=0.05,
                replay_capacity=5000, batch_size=32, target_update_freq=100):
        
        self.policy_net = QNetwork(input_channels, grid_rows, grid_cols, num_actions).to(DEVICE)
        self.target_net = QNetwork(input_channels, grid_rows, grid_cols, num_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.experience_replay = ExperienceReplay(capacity=replay_capacity)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

    def choose_action(self, state, valid_actions):
        if not valid_actions:return 0

        if random.uniform(0, 1) < self.epsilon: return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_tensor)
                mask = torch.full_like(q_values, -float('inf'))
                for action_idx in valid_actions: mask[0, action_idx] = 0
                masked_q_values = q_values + mask
                return masked_q_values.argmax(1).item()

    def learn(self):
        if self.experience_replay.size() < self.batch_size: return

        experiences = self.experience_replay.sample(self.batch_size)
        states, actions, rewards, next_states = zip(*experiences)
        states = torch.from_numpy(np.array(states)).float().to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(DEVICE)
        next_states = torch.from_numpy(np.array(next_states)).float().to(DEVICE)
        current_q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters(): param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)

def train_dqn_agent(env, agent, num_episodes=5000):
    best_overall_grid = None
    max_overall_houses = -1
    start_time = time.time()
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        num_actions_this_episode = 0
        max_actions_per_episode = 150

        while not done and num_actions_this_episode < max_actions_per_episode:
            valid_actions = env.get_valid_actions()
            action = agent.choose_action(state, valid_actions)

            if not valid_actions and action == 0:
                reward = -10
                done = True
                next_state = state
            else: next_state, reward, done = env.step(action)

            agent.experience_replay.add(state, action, reward, next_state)
            agent.learn()
            state = next_state
            total_reward += reward
            num_actions_this_episode += 1

        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        current_houses = env.get_num_connected_houses()
        if current_houses > max_overall_houses:
            max_overall_houses = current_houses
            best_overall_grid = copy.deepcopy(env.grid)

        if episode % 100 == 0:
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            print(f"Episode {episode}/{num_episodes}, Avg Reward: {avg_reward:.2f}, "
                f"Connected Houses: {current_houses}, Epsilon: {agent.epsilon:.3f}, "
                f"Best Overall Houses: {max_overall_houses}, Time: {elapsed_time:.2f}s, "
                f"Replay Buffer Size: {agent.experience_replay.size()}")
            start_time = time.time()
    return best_overall_grid, max_overall_houses, episode_rewards

def visualize_grid(grid, title="Gebäudeplatzierung"):
    cmap = plt.cm.colors.ListedColormap(['white', 'green', 'grey', 'purple', 'yellow'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(grid.shape[1] * 0.5, grid.shape[0] * 0.5))
    plt.imshow(grid, cmap=cmap, norm=norm, origin='upper', extent=[0, grid.shape[1], grid.shape[0], 0])
    plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(0, grid.shape[1] + 1, 1))
    plt.yticks(np.arange(0, grid.shape[0] + 1, 1))
    plt.xlabel("X-Koordinate (Kacheln)")
    plt.ylabel("Y-Koordinate (Kacheln)")
    plt.title(title)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Wohnhaus'),
        Patch(facecolor='grey', label='Straße'),
        Patch(facecolor='purple', label='Dorfzentrum'),
        Patch(facecolor='yellow', label='Kontor'),
        Patch(facecolor='white', label='Leere Kachel')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_training_progress(episode_rewards):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3)
    window_size = 100
    if len(episode_rewards) >= window_size:
        moving_avg = [np.mean(episode_rewards[i:i+window_size]) for i in range(len(episode_rewards)-window_size+1)]
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress - Episode Rewards')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if len(episode_rewards) >= 1000:
        recent_rewards = episode_rewards[-1000:]
        plt.hist(recent_rewards, bins=50, alpha=0.7)
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title('Recent Episode Rewards Distribution')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    grid_dimensions = (20, 20)
    env = EnhancedCityPlanningEnv(grid_dimensions)
    num_actions = (grid_dimensions[0] - HOUSE_DIMS[0] + 1) * (grid_dimensions[1] - HOUSE_DIMS[1] + 1)
    input_channels = 8

    agent = DQNAgent(
        input_channels=input_channels,
        grid_rows=grid_dimensions[0],
        grid_cols=grid_dimensions[1],
        num_actions=num_actions,
        alpha=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay_rate=0.9995,
        min_epsilon=0.05,
        replay_capacity=10000,
        batch_size=64,
        target_update_freq=200
    )

    print(f"DQN QRL-Training auf einem {grid_dimensions[0]}x{grid_dimensions[1]} Raster mit PyTorch...")
    print("Features: CNN-based Q-Network, Multi-channel Grid State, Experience Replay, Target Network, Adam Optimizer")
    
    best_grid_rl, max_houses_rl, rewards = train_dqn_agent(env, agent, num_episodes=10000)
    
    if best_grid_rl is not None and max_houses_rl > 0:
        print(f"\nDQN QRL-Training abgeschlossen. Beste gefundene Lösung hatte {max_houses_rl} gültig verbundene Wohnhäuser.")
        visualize_grid(best_grid_rl, f"DQN QRL Beste gefundene Lösung ({max_houses_rl} Häuser) - Grid {grid_dimensions[0]}x{grid_dimensions[1]}")
        plot_training_progress(rewards)
    else:
        print("\nDQN QRL-Agent konnte keine gültige Platzierung finden oder lernen.")
        plot_training_progress(rewards)