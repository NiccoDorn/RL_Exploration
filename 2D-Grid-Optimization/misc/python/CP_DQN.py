import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import copy
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
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

def find_connected_houses(grid, tc_center_coords, tp_coords):
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    connected_houses_count = 0
    q = deque()

    start_nodes = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == ROAD:
                is_connected_to_essential = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if (nr >= tc_center_coords[0] - TOWN_CENTER_DIMS[0]//2 and nr < tc_center_coords[0] - TOWN_CENTER_DIMS[0]//2 + TOWN_CENTER_DIMS[0] and \
                            nc >= tc_center_coords[1] - TOWN_CENTER_DIMS[1]//2 and nc < tc_center_coords[1] - TOWN_CENTER_DIMS[1]//2 + TOWN_CENTER_DIMS[1] and \
                            grid[nr, nc] == TOWN_CENTER):
                            is_connected_to_essential = True
                            break
                        if (nr >= tp_coords[0] - TRADING_POST_DIMS[0]//2 and nr < tp_coords[0] - TRADING_POST_DIMS[0]//2 + TRADING_POST_DIMS[0] and \
                            nc >= tp_coords[1] - TRADING_POST_DIMS[1]//2 and nc < tp_coords[1] - TRADING_POST_DIMS[1]//2 + TRADING_POST_DIMS[1] and \
                            grid[nr, nc] == TRADING_POST):
                            is_connected_to_essential = True
                            break
                if is_connected_to_essential:
                    start_nodes.append((r, c))

    for start_node in start_nodes:
        if not visited[start_node[0], start_node[1]]:
            q.append(start_node)
            visited[start_node[0], start_node[1]] = True
            current_component_houses = 0
            component_q = deque([start_node])
            component_visited = np.copy(visited)

            while component_q:
                r, c = component_q.popleft()

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not component_visited[nr, nc]:
                        if grid[nr, nc] == ROAD:
                            component_q.append((nr, nc))
                            component_visited[nr, nc] = True
                        elif grid[nr, nc] == HOUSE:
                            house_top_left_r, house_top_left_c = -1, -1
                            found_house_tl = False
                            for r_check in range(max(0, nr - HOUSE_DIMS[0] + 1), min(rows, nr + 1)):
                                for c_check in range(max(0, nc - HOUSE_DIMS[1] + 1), min(cols, nc + 1)):
                                    if grid[r_check, c_check] == HOUSE:
                                        is_tl = True
                                        for i in range(r_check, r_check + HOUSE_DIMS[0]):
                                            for j in range(c_check, c_check + HOUSE_DIMS[1]):
                                                if not (0 <= i < rows and 0 <= j < cols and grid[i, j] == HOUSE):
                                                    is_tl = False
                                                    break
                                            if not is_tl: break
                                        if is_tl:
                                            house_top_left_r, house_top_left_c = r_check, c_check
                                            found_house_tl = True
                                            break
                                if found_house_tl: break

                            if found_house_tl:
                                for r_h in range(house_top_left_r, house_top_left_r + HOUSE_DIMS[0]):
                                    for c_h in range(house_top_left_c, c_h + HOUSE_DIMS[1]):
                                        if 0 <= r_h < rows and 0 <= c_h < cols and grid[r_h,c_h] == HOUSE and not component_visited[r_h, c_h]:
                                            component_q.append((r_h, c_h))
                                            component_visited[r_h, c_h] = True
                                current_component_houses += 1
            visited = np.logical_or(visited, component_visited)
            connected_houses_count += current_component_houses

    return connected_houses_count


def visualize_grid(grid, title="City Grid"):
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.get_cmap('Paired', 5)
    cmap_colors = cmap.colors
    custom_colors = [
        [0.8, 0.8, 0.8, 1.0],
        cmap_colors[1],
        [0.1, 0.1, 0.1, 1.0],
        cmap_colors[5],
        cmap_colors[3]
    ]
    custom_cmap = plt.matplotlib.colors.ListedColormap(custom_colors)

    plt.imshow(grid, cmap=custom_cmap, origin='upper', vmin=0, vmax=4)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4], fraction=0.046, pad=0.04)
    cbar.set_ticklabels(['EMPTY', 'HOUSE', 'ROAD', 'TOWN CENTER', 'TRADING POST'])
    plt.grid(True, which='both', color='white', linestyle='-', linewidth=0.5)
    plt.show()

class EnhancedCityPlanningEnv:
    def __init__(self, grid_dims):
        self.grid_dims = grid_dims
        self.grid = np.zeros(grid_dims, dtype=int)
        self.tc_coords = None
        self.tp_coords = None
        self.num_houses_placed = 0
        self.last_house_coords = None
        self.consecutive_successes = 0
        self.road_cells_before_action = 0
        self.episode_end_processing = True
        self.reset()

    def reset(self):
        self.grid = np.full(self.grid_dims, EMPTY, dtype=int)
        road_mask = np.random.rand(*self.grid_dims) < 0.5
        self.grid[road_mask] = ROAD

        if not np.any(self.grid == ROAD):
            self.grid[0, 0] = ROAD

        tc_r = random.randint(0, self.grid_dims[0] - TOWN_CENTER_DIMS[0])
        tc_c = random.randint(0, self.grid_dims[1] - TOWN_CENTER_DIMS[1])
        place_building_on_grid(self.grid, TOWN_CENTER, tc_r, tc_c, TOWN_CENTER_DIMS[0], TOWN_CENTER_DIMS[1])
        self.tc_coords = (tc_r + TOWN_CENTER_DIMS[0]//2, tc_c + TOWN_CENTER_DIMS[1]//2)
        tp_placed = False
        attempts = 0
        while not tp_placed and attempts < 100:
            tp_r = random.randint(0, self.grid_dims[0] - TRADING_POST_DIMS[0])
            tp_c = random.randint(0, self.grid_dims[1] - TRADING_POST_DIMS[1])
            temp_grid = np.copy(self.grid)
            overlap = False
            for r_t in range(tp_r, tp_r + TRADING_POST_DIMS[0]):
                for c_t in range(tp_c, tp_c + TRADING_POST_DIMS[1]):
                    if 0 <= r_t < self.grid_dims[0] and 0 <= c_t < self.grid_dims[1]:
                        if temp_grid[r_t, c_t] not in [EMPTY, ROAD]:
                            overlap = True
                            break
                if overlap: break
            
            if not overlap:
                place_building_on_grid(temp_grid, TRADING_POST, tp_r, tp_c, TRADING_POST_DIMS[0], TRADING_POST_DIMS[1])
                self.tp_coords = (tp_r + TRADING_POST_DIMS[0]//2, tp_c + TRADING_POST_DIMS[1]//2)
                self.grid = temp_grid
                tp_placed = True
            attempts += 1
        
        if not tp_placed:
            tp_r = random.randint(0, self.grid_dims[0] - TRADING_POST_DIMS[0])
            tp_c = random.randint(0, self.grid_dims[1] - TRADING_POST_DIMS[1])
            place_building_on_grid(self.grid, TRADING_POST, tp_r, tp_c, TRADING_POST_DIMS[0], TRADING_POST_DIMS[1])
            self.tp_coords = (tp_r + TRADING_POST_DIMS[0]//2, tp_c + TRADING_POST_DIMS[1]//2)

        self.num_houses_placed = 0
        self.last_house_coords = None
        self.consecutive_successes = 0
        self.road_cells_before_action = np.sum(self.grid == ROAD)
        return self._get_state()

    def _get_state(self):
        state = np.zeros((8, *self.grid_dims), dtype=np.float32)
        state[0, :, :] = (self.grid == EMPTY).astype(np.float32)
        state[1, :, :] = (self.grid == HOUSE).astype(np.float32)
        state[2, :, :] = (self.grid == ROAD).astype(np.float32)
        state[3, :, :] = (self.grid == TOWN_CENTER).astype(np.float32)
        state[4, :, :] = (self.grid == TRADING_POST).astype(np.float32)

        tc_r, tc_c = self.tc_coords
        for r in range(self.grid_dims[0]):
            for c in range(self.grid_dims[1]):
                dist_sq = calculate_distance_sq((r, c), (tc_r, tc_c))
                state[5, r, c] = np.sqrt(dist_sq) / np.sqrt((self.grid_dims[0]-1)**2 + (self.grid_dims[1]-1)**2)

        tp_r, tp_c = self.tp_coords
        for r in range(self.grid_dims[0]):
            for c in range(self.grid_dims[1]):
                if self.grid[r, c] == ROAD:
                    manhattan_dist = abs(r - tp_r) + abs(c - tp_c)
                    if manhattan_dist <= 10:
                        state[6, r, c] = 1.0 - (manhattan_dist / 10.0)
                else: state[6, r, c] = 0.0
        
        tc_r, tc_c = self.tc_coords
        for r in range(self.grid_dims[0]):
            for c in range(self.grid_dims[1]):
                if self.grid[r, c] == ROAD:
                    manhattan_dist = abs(r - tc_r) + abs(c - tc_c)
                    if manhattan_dist <= 15:
                        state[7, r, c] = 1.0 - (manhattan_dist / 15.0)
                else: state[7, r, c] = 0.0
        return state

    def _calculate_reward_shaping(self, new_grid, action_r, action_c):
        reward = 0.0
        temp_grid_copy = np.copy(new_grid)
        place_building_on_grid(temp_grid_copy, HOUSE, action_r, action_c, HOUSE_DIMS[0], HOUSE_DIMS[1])
        
        current_house_cells = np.argwhere(temp_grid_copy == HOUSE)
        num_block_aligned = 0
        for hr, hc in current_house_cells:
            if hr + 1 < self.grid_dims[0] and hc + 1 < self.grid_dims[1]:
                if (temp_grid_copy[hr, hc] == HOUSE and
                    temp_grid_copy[hr+1, hc] == HOUSE and
                    temp_grid_copy[hr, hc+1] == HOUSE and
                    temp_grid_copy[hr+1, hc+1] == HOUSE):
                    num_block_aligned += 1
        
        if num_block_aligned > 0: reward += num_block_aligned * 5

        potential_extensions = 0
        for r in range(action_r, action_r + HOUSE_DIMS[0]):
            for c in range(action_c, action_c + HOUSE_DIMS[1]):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_dims[0] and 0 <= nc < self.grid_dims[1]:
                        if new_grid[nr, nc] == ROAD:
                            potential_extensions += 0.5
        reward += potential_extensions

        if self.last_house_coords:
            dist_to_last = math.sqrt(calculate_distance_sq(self.last_house_coords, (action_r, action_c)))
            if dist_to_last <= 5:
                reward += 2.0 / (1.0 + dist_to_last)

        dist_to_tc = math.sqrt(calculate_distance_sq(self.tc_coords, (action_r + HOUSE_DIMS[0]//2, action_c + HOUSE_DIMS[1]//2)))
        if dist_to_tc <= math.sqrt(TOWN_CENTER_RADIUS_SQ):
            reward += 2.0 * (1.0 - (dist_to_tc / math.sqrt(TOWN_CENTER_RADIUS_SQ)))
        
        TRADING_POST_RADIUS_SQ = 10**2
        dist_to_tp = math.sqrt(calculate_distance_sq(self.tp_coords, (action_r + HOUSE_DIMS[0]//2, action_c + HOUSE_DIMS[1]//2)))
        if dist_to_tp <= math.sqrt(TRADING_POST_RADIUS_SQ):
            reward += 1.5 * (1.0 - (dist_to_tp / math.sqrt(TRADING_POST_RADIUS_SQ)))

        if self.consecutive_successes >= 2: reward += self.consecutive_successes * 0.5

        return reward

    def _fill_gaps_greedy(self, current_grid):
        temp_grid = np.copy(current_grid)
        initial_houses = np.sum(temp_grid == HOUSE)
        
        possible_positions = []
        for r in range(self.grid_dims[0] - HOUSE_DIMS[0] + 1):
            for c in range(self.grid_dims[1] - HOUSE_DIMS[1] + 1):
                is_valid = True
                if not is_area_road(temp_grid, r, c, HOUSE_DIMS[0], HOUSE_DIMS[1]):
                    is_valid = False
                
                house_center = (r + HOUSE_DIMS[0]//2, c + HOUSE_DIMS[1]//2)
                if calculate_distance_sq(self.tc_coords, house_center) > TOWN_CENTER_RADIUS_SQ:
                    is_valid = False
                
                if is_valid: possible_positions.append((r, c))

        random.shuffle(possible_positions)

        for r, c in possible_positions:
            if is_area_road(temp_grid, r, c, HOUSE_DIMS[0], HOUSE_DIMS[1]):
                house_center = (r + HOUSE_DIMS[0]//2, c + HOUSE_DIMS[1]//2)
                if calculate_distance_sq(self.tc_coords, house_center) <= TOWN_CENTER_RADIUS_SQ:
                    place_building_on_grid(temp_grid, HOUSE, r, c, HOUSE_DIMS[0], HOUSE_DIMS[1])
        
        houses_after_filling = np.sum(temp_grid == HOUSE)
        additional_houses = houses_after_filling - initial_houses
        reward_for_filling = additional_houses * 5.0

        return temp_grid, reward_for_filling, additional_houses

    def step(self, action):
        row_idx = action // (self.grid_dims[1] - HOUSE_DIMS[1] + 1)
        col_idx = action % (self.grid_dims[1] - HOUSE_DIMS[1] + 1)
        reward = 0.0
        done = False
        temp_grid = np.copy(self.grid)

        if not is_area_road(temp_grid, row_idx, col_idx, HOUSE_DIMS[0], HOUSE_DIMS[1]):
            reward -= 1.0
            self.consecutive_successes = 0
        else:
            house_center = (row_idx + HOUSE_DIMS[0]//2, col_idx + HOUSE_DIMS[1]//2)
            if calculate_distance_sq(self.tc_coords, house_center) > TOWN_CENTER_RADIUS_SQ:
                reward -= 2.0
                self.consecutive_successes = 0
            else:
                place_building_on_grid(self.grid, HOUSE, row_idx, col_idx, HOUSE_DIMS[0], HOUSE_DIMS[1])
                self.num_houses_placed += 1
                self.last_house_coords = (row_idx, col_idx)
                self.consecutive_successes += 1
                reward += 1.0
                reward += self._calculate_reward_shaping(self.grid, row_idx, col_idx)

        valid_actions_exist = False
        for r in range(self.grid_dims[0] - HOUSE_DIMS[0] + 1):
            for c in range(self.grid_dims[1] - HOUSE_DIMS[1] + 1):
                if is_area_road(self.grid, r, c, HOUSE_DIMS[0], HOUSE_DIMS[1]):
                    house_center = (r + HOUSE_DIMS[0]//2, c + HOUSE_DIMS[1]//2)
                    if calculate_distance_sq(self.tc_coords, house_center) <= TOWN_CENTER_RADIUS_SQ:
                        valid_actions_exist = True
                        break
            if valid_actions_exist: break
        
        if not valid_actions_exist and self.num_houses_placed > 0:
            done = True
            reward -= 5.0

        if self.num_houses_placed > 20: reward += 10.0

        final_houses = self.num_houses_placed
        if done and self.episode_end_processing:
            original_grid = np.copy(self.grid)
            filled_grid, filling_reward, additional_houses = self._fill_gaps_greedy(self.grid)
            self.grid = filled_grid
            reward += filling_reward
            final_houses += additional_houses
        
        return self._get_state(), reward, done, final_houses


class QNetwork(nn.Module):
    def __init__(self, input_channels, grid_rows, grid_cols, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(grid_cols)
        convh = conv2d_size_out(grid_rows)
        linear_input_size = convw * convh * 128
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent:
    def __init__(self, input_channels, grid_rows, grid_cols, num_actions, alpha, gamma, epsilon, epsilon_decay_rate, min_epsilon, replay_capacity, batch_size, target_update_freq):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.policy_net = QNetwork(input_channels, grid_rows, grid_cols, num_actions).to(DEVICE)
        self.target_net = QNetwork(input_channels, grid_rows, grid_cols, num_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.memory = deque(maxlen=replay_capacity)
        self.steps_done = 0

    def choose_action(self, state, valid_actions):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        if random.random() < self.epsilon:
            if not valid_actions: return 0
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).squeeze(0)
                mask = torch.full_like(q_values, -float('inf'))
                for action_idx in valid_actions:
                    mask[action_idx] = 0.0
                
                masked_q_values = q_values + mask

                return masked_q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size: return

        transitions = random.sample(self.memory, self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        batch_state = torch.from_numpy(np.array(batch_state)).float().to(DEVICE)
        batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1).to(DEVICE)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        batch_next_state = torch.from_numpy(np.array(batch_next_state)).float().to(DEVICE)
        batch_done = torch.tensor(batch_done, dtype=torch.bool).unsqueeze(1).to(DEVICE)
        state_action_q_values = self.policy_net(batch_state).gather(1, batch_action)
        next_state_max_q_values = self.target_net(batch_next_state).max(1)[0].unsqueeze(1)
        expected_q_values = batch_reward + (~batch_done) * self.gamma * next_state_max_q_values
        loss = F.smooth_l1_loss(state_action_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

def train_dqn_agent(env, agent, num_episodes):
    best_grid = None
    max_houses = 0
    episode_rewards = []
    episode_house_counts = []
    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        num_actions_this_episode = 0
        current_grid_at_best_episode = None

        while not done and num_actions_this_episode < 150:
            valid_actions = []
            for r in range(env.grid_dims[0] - HOUSE_DIMS[0] + 1):
                for c in range(env.grid_dims[1] - HOUSE_DIMS[1] + 1):
                    if env.grid[r:r+HOUSE_DIMS[0], c:c+HOUSE_DIMS[1]].flatten().tolist().count(ROAD) == (HOUSE_DIMS[0] * HOUSE_DIMS[1]):
                        house_center = (r + HOUSE_DIMS[0]//2, c + HOUSE_DIMS[1]//2)
                        if calculate_distance_sq(env.tc_coords, house_center) <= TOWN_CENTER_RADIUS_SQ:
                            action_idx = r * (env.grid_dims[1] - HOUSE_DIMS[1] + 1) + c
                            valid_actions.append(action_idx)
            
            if not valid_actions and env.num_houses_placed > 0:
                reward = -10.0
                done = True
                next_state = state
                final_houses_in_episode = env.num_houses_placed
                if env.episode_end_processing:
                    final_grid_for_eval, _, additional_houses = env._fill_gaps_greedy(env.grid)
                    final_houses_in_episode += additional_houses
                    env.grid = final_grid_for_eval
                
                agent.store_transition(state, 0, reward, next_state, done)
                episode_reward += reward
                break

            action = agent.choose_action(state, valid_actions)
            next_state, reward, done, final_houses_in_episode = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss_val = agent.learn()
            state = next_state
            episode_reward += reward
            num_actions_this_episode += 1

        episode_rewards.append(episode_reward)
        episode_house_counts.append(final_houses_in_episode)

        if final_houses_in_episode > max_houses:
            max_houses = final_houses_in_episode
            best_grid = copy.deepcopy(env.grid)
            current_grid_at_best_episode = copy.deepcopy(env.grid)

        if (episode + 1) % 100 == 0:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            avg_houses_100 = np.mean(episode_house_counts[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, Epsilon: {agent.epsilon:.4f}, "
                f"Avg Reward (last 100): {avg_reward_100:.2f}, "
                f"Avg Houses (last 100): {avg_houses_100:.1f}, "
                f"Max Houses Overall: {max_houses}, "
                f"Time Elapsed: {time.time() - start_time:.2f}s")

    print(f"\nTraining finished after {num_episodes} episodes.")
    print(f"Best solution found: {max_houses} connected houses.")

    return best_grid, max_houses, episode_rewards, episode_house_counts

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
    print("Enhancements: Reward for TP Proximity, Stronger Block Alignment Reward")
    
    best_grid_dqn, max_houses_dqn, rewards, house_counts = train_dqn_agent(env, agent, num_episodes=10000)
    if best_grid_dqn is not None and max_houses_dqn > 0:
        print(f"\nDQN QRL-Training abgeschlossen. Beste gefundene Lösung hatte {max_houses_dqn} gültig verbundene Wohnhäuser.")
        visualize_grid(best_grid_dqn, f"DQN QRL - Best Grid ({max_houses_dqn} Houses)")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Episode Rewards Over Time')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(house_counts)
    plt.xlabel('Episode')
    plt.ylabel('Houses Placed (incl. Greedy Filling)')
    plt.title('DQN Houses Placed Per Episode')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()