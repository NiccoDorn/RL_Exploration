#include <iostream>
#include <chrono>
#include <tuple>
#include <iomanip>
#include "env.h"
#include "agent.h"
#include "utils.h"

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

        agent.decay_epsilon();

        int current_houses = env.get_num_connected_houses();
        if (current_houses > max_overall_houses) {
            max_overall_houses = current_houses;
            best_overall_grid = env.get_grid();
        }

        if (episode % 50 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - start_time;
            std::cout << "Episode " << episode << "/" << num_episodes
                    << " | Epsilon: " << std::fixed << std::setprecision(4) << agent.get_epsilon()
                    << " | Best Overall Houses: " << max_overall_houses
                    << " | Total Reward: " << std::fixed << std::setprecision(2) << total_reward
                    << " | Actions: " << num_actions_this_episode
                    << " | Time: " << std::fixed << std::setprecision(2) << elapsed.count() << "s"
                    << " | Q-Table Size: " << agent.get_q_table_size() << std::endl;
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
        0.2,    // alpha
        0.9,   // gamma
        1.0,    // epsilon
        0.997,  // edr
        0.07    // min_epsilon
    );

    std::cout << "Enhanced QRL-Training on a " << grid_rows << "x" << grid_cols << " grid..." << std::endl;
    std::cout << "Features: Reduced State Space, Simple Reward System," << std::endl;
    std::cout << "          Hybrid Exploration (Epsilon-Greedy + Boltzmann + UCB)," << std::endl;
    std::cout << "          Experience Replay, Incremental Connectivity Checks," << std::endl;
    std::cout << "          Block-Aligned Prioritization, Greedy Gap Filling" << std::endl;

    train_enhanced_rl_agent(env, agent, 10000);
    return 0;
}