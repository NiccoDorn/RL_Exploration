#include <algorithm>
#include <cmath>

#include "agent.h"

EnhancedQLearningAgent::EnhancedQLearningAgent(double alpha, double gamma, double epsilon, double epsilon_decay_rate, double min_epsilon) :
    alpha_(alpha), gamma_(gamma), epsilon_(epsilon), epsilon_decay_rate_(epsilon_decay_rate), min_epsilon_(min_epsilon),
    temperature_(1.0), experience_replay_(5000), total_steps_(0) {
    std::random_device rd;
    rng_ = std::mt19937(rd());
    q_table_.reserve(10000);
    exploration_counts_.reserve(10000);
}

void EnhancedQLearningAgent::_get_q_values(const State& state, const std::vector<int>& valid_actions, std::unordered_map<int, double>& out_q_values) {
    out_q_values.clear();
    out_q_values.reserve(valid_actions.size());

    auto state_it = q_table_.find(state);
    if (state_it == q_table_.end()) {
        for (int action : valid_actions) {
            q_table_[state][action] = 0.0;
            out_q_values[action] = 0.0;
        }
    } else {
        auto& action_values = state_it->second;
        for (int action : valid_actions) {
            auto action_it = action_values.find(action);
            if (action_it == action_values.end()) {
                q_table_[state][action] = 0.0;
                out_q_values[action] = 0.0;
            } else {
                out_q_values[action] = action_it->second;
            }
        }
    }
}

int EnhancedQLearningAgent::_choose_action_epsilon_greedy(const State& state, const std::vector<int>& valid_actions) {
    if (valid_actions.empty()) {
        return 0;
    }

    std::uniform_real_distribution<> dis(0.0, 1.0);
    if (dis(rng_) < epsilon_) {
        std::uniform_int_distribution<> action_dist(0, valid_actions.size() - 1);
        return valid_actions[action_dist(rng_)];
    }

    thread_local std::unordered_map<int, double> q_values;
    _get_q_values(state, valid_actions, q_values);

    double max_q = -std::numeric_limits<double>::infinity();
    for (const auto& [action, q_val] : q_values) {
        if (q_val > max_q) {
            max_q = q_val;
        }
    }

    thread_local std::vector<int> best_actions;
    best_actions.clear();
    for (const auto& [action, q_val] : q_values) {
        if (q_val == max_q) {
            best_actions.push_back(action);
        }
    }

    std::uniform_int_distribution<> action_dist(0, best_actions.size() - 1);
    return best_actions[action_dist(rng_)];
}

int EnhancedQLearningAgent::_choose_action_boltzmann(const State& state, const std::vector<int>& valid_actions) {
    if (valid_actions.empty()) {
        return 0;
    }

    thread_local std::unordered_map<int, double> q_values;
    _get_q_values(state, valid_actions, q_values);

    thread_local std::vector<double> probabilities;
    probabilities.clear();
    probabilities.reserve(valid_actions.size());

    double sum_exp = 0.0;
    for (int action : valid_actions) {
        double exp_val = std::exp(q_values[action] / temperature_);
        probabilities.push_back(exp_val);
        sum_exp += exp_val;
    }

    for (double& prob : probabilities) {
        prob /= sum_exp;
    }

    std::uniform_real_distribution<> dis(0.0, 1.0);
    double random_val = dis(rng_);
    double cumulative_prob = 0.0;

    for (size_t i = 0; i < valid_actions.size(); ++i) {
        cumulative_prob += probabilities[i];
        if (random_val <= cumulative_prob) {
            return valid_actions[i];
        }
    }

    return valid_actions.back();
}

int EnhancedQLearningAgent::_choose_action_ucb(const State& state, const std::vector<int>& valid_actions) {
    if (valid_actions.empty()) {
        return 0;
    }

    thread_local std::unordered_map<int, double> q_values;
    _get_q_values(state, valid_actions, q_values);

    auto& state_counts = exploration_counts_[state];

    double best_ucb = -std::numeric_limits<double>::infinity();
    thread_local std::vector<int> best_actions;
    best_actions.clear();

    for (int action : valid_actions) {
        int count = state_counts[action];

        double ucb_value;
        if (count == 0) {
            ucb_value = std::numeric_limits<double>::infinity();
        } else {
            double confidence = std::sqrt(2.0 * std::log(total_steps_ + 1) / count);
            ucb_value = q_values[action] + confidence;
        }

        if (ucb_value > best_ucb) {
            best_ucb = ucb_value;
            best_actions.clear();
            best_actions.push_back(action);
        } else if (ucb_value == best_ucb) {
            best_actions.push_back(action);
        }
    }

    std::uniform_int_distribution<> action_dist(0, best_actions.size() - 1);
    return best_actions[action_dist(rng_)];
}

int EnhancedQLearningAgent::choose_action(const State& state, const std::vector<int>& valid_actions) {
    total_steps_++;

    std::uniform_real_distribution<> dis(0.0, 1.0);
    double strategy_choice = dis(rng_);

    int chosen_action;
    if (strategy_choice < 0.1) {
        chosen_action = _choose_action_ucb(state, valid_actions);
    } else if (strategy_choice < 0.3 && epsilon_ > 0.1) {
        chosen_action = _choose_action_boltzmann(state, valid_actions);
    } else {
        chosen_action = _choose_action_epsilon_greedy(state, valid_actions);
    }

    exploration_counts_[state][chosen_action]++;

    return chosen_action;
}

void EnhancedQLearningAgent::learn(const State& state, int action, double reward, const State& next_state, const std::vector<int>& valid_next_actions) {
    experience_replay_.add(state, action, reward, next_state, valid_next_actions);
    std::vector<ExperienceReplay::Experience> experiences = experience_replay_.sample(std::min(size_t(64), experience_replay_.size()));

    for (const auto& exp : experiences) {
        double current_q = q_table_[exp.state][exp.action];

        double max_future_q = 0.0;
        if (!exp.valid_next_actions.empty()) {
            auto next_state_it = q_table_.find(exp.next_state);
            if (next_state_it != q_table_.end()) {
                const auto& next_action_values = next_state_it->second;
                double current_max_q = -std::numeric_limits<double>::infinity();
                bool found_any_valid_q = false;

                for (int next_action : exp.valid_next_actions) {
                    auto next_action_it = next_action_values.find(next_action);
                    if (next_action_it != next_action_values.end()) {
                        if (next_action_it->second > current_max_q) {
                            current_max_q = next_action_it->second;
                            found_any_valid_q = true;
                        }
                    }
                }

                if (found_any_valid_q) {
                    max_future_q = current_max_q;
                }
            }
        }

        double new_q = current_q + alpha_ * (exp.reward + gamma_ * max_future_q - current_q);
        q_table_[exp.state][exp.action] = new_q;
    }
}

void EnhancedQLearningAgent::decay_epsilon() noexcept {
    epsilon_ *= epsilon_decay_rate_;
    epsilon_ = std::max(min_epsilon_, epsilon_);
    temperature_ = std::max(0.1, temperature_ * 0.999);
}

double EnhancedQLearningAgent::get_epsilon() const noexcept {
    return epsilon_;
}

size_t EnhancedQLearningAgent::get_q_table_size() const noexcept {
    return q_table_.size();
}

void EnhancedQLearningAgent::reset_exploration_counts() {
    exploration_counts_.clear();
    total_steps_ = 0;
}
