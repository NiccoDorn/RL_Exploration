#pragma once

#include <map>
#include <vector>
#include <random>
#include <limits>
#include <cmath>

#include "common.h"
#include "exp_replay.h"

class EnhancedQLearningAgent {
public:
    EnhancedQLearningAgent(double alpha = 0.2, double gamma = 0.99, double epsilon = 1.0, double epsilon_decay_rate = 0.995, double min_epsilon = 0.01);
    int choose_action(const State& state, const std::vector<int>& valid_actions);
    void learn(const State& state, int action, double reward, const State& next_state, const std::vector<int>& valid_next_actions);
    void decay_epsilon();
    double get_epsilon() const;
    size_t get_q_table_size() const;
    void reset_exploration_counts();

private:
    std::map<State, std::map<int, double>> q_table_;
    std::map<State, std::map<int, int>> exploration_counts_;
    double alpha_;
    double gamma_;
    double epsilon_;
    double epsilon_decay_rate_;
    double min_epsilon_;
    double temperature_;
    ExperienceReplay experience_replay_;
    std::mt19937 rng_;
    int total_steps_;
    
    std::map<int, double> _get_q_values(const State& state, const std::vector<int>& valid_actions);
    int _choose_action_boltzmann(const State& state, const std::vector<int>& valid_actions);
    int _choose_action_ucb(const State& state, const std::vector<int>& valid_actions);
    int _choose_action_epsilon_greedy(const State& state, const std::vector<int>& valid_actions);
};