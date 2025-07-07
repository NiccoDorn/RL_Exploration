#include <algorithm>
#include <cmath>

#include "agent.h"

EnhancedQLearningAgent::EnhancedQLearningAgent(double alpha, double gamma, double epsilon, double epsilon_decay_rate, double min_epsilon) :
    alpha_(alpha), gamma_(gamma), epsilon_(epsilon), epsilon_decay_rate_(epsilon_decay_rate), min_epsilon_(min_epsilon), 
    temperature_(1.0), experience_replay_(5000), total_steps_(0) {
    std::random_device rd;
    rng_ = std::mt19937(rd());
}

std::map<int, double> EnhancedQLearningAgent::_get_q_values(const State& state, const std::vector<int>& valid_actions) {
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

int EnhancedQLearningAgent::_choose_action_epsilon_greedy(const State& state, const std::vector<int>& valid_actions) {
    if (valid_actions.empty()) {
        return 0;
    }

    std::uniform_real_distribution<> dis(0.0, 1.0);
    if (dis(rng_) < epsilon_) {
        std::uniform_int_distribution<> action_dist(0, valid_actions.size() - 1);
        return valid_actions[action_dist(rng_)];
    }

    std::map<int, double> q_values = _get_q_values(state, valid_actions);

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

int EnhancedQLearningAgent::_choose_action_boltzmann(const State& state, const std::vector<int>& valid_actions) {
    if (valid_actions.empty()) {
        return 0;
    }

    std::map<int, double> q_values = _get_q_values(state, valid_actions);
    
    // boltzmann distribution probs
    std::vector<double> probabilities;
    double sum_exp = 0.0;
    
    for (int action : valid_actions) {
        double exp_val = std::exp(q_values[action] / temperature_);
        probabilities.push_back(exp_val);
        sum_exp += exp_val;
    }
    
    // normalize
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

    std::map<int, double> q_values = _get_q_values(state, valid_actions);

    if (exploration_counts_.find(state) == exploration_counts_.end()) {
        for (int action : valid_actions) {
            exploration_counts_[state][action] = 0;
        }
    }

    double best_ucb = -std::numeric_limits<double>::infinity();
    std::vector<int> best_actions;
    
    for (int action : valid_actions) {
        if (exploration_counts_[state].find(action) == exploration_counts_[state].end()) {
            exploration_counts_[state][action] = 0;
        }
        
        double ucb_value;
        if (exploration_counts_[state][action] == 0) {
            ucb_value = std::numeric_limits<double>::infinity();
        } else {
            double confidence = std::sqrt(2.0 * std::log(total_steps_ + 1) / exploration_counts_[state][action]);
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
    
    if (exploration_counts_.find(state) == exploration_counts_.end()) {
        exploration_counts_[state][chosen_action] = 0;
    }
    exploration_counts_[state][chosen_action]++;
    
    return chosen_action;
}

void EnhancedQLearningAgent::learn(const State& state, int action, double reward, const State& next_state, const std::vector<int>& valid_next_actions) {
    experience_replay_.add(state, action, reward, next_state);
    std::vector<ExperienceReplay::Experience> experiences = experience_replay_.sample(std::min((size_t)64, experience_replay_.size()));

    for (const auto& exp : experiences) {
        if (q_table_.find(exp.state) == q_table_.end() || q_table_[exp.state].find(exp.action) == q_table_[exp.state].end()) {
            q_table_[exp.state][exp.action] = 0.0;
        }

        double current_q = q_table_[exp.state][exp.action];

        double max_future_q = 0.0;
        if (!valid_next_actions.empty()) {
            double current_max_q = -std::numeric_limits<double>::infinity();
            bool found_any_valid_q_for_next_state = false;

            for (int next_action : valid_next_actions) {
                double q_val_for_next_action = 0.0;
                if (q_table_.count(exp.next_state) && q_table_[exp.next_state].count(next_action)) {
                    q_val_for_next_action = q_table_[exp.next_state][next_action];
                }
                if (q_val_for_next_action > current_max_q) {
                    current_max_q = q_val_for_next_action;
                    found_any_valid_q_for_next_state = true;
                }
            }
            if (found_any_valid_q_for_next_state) {
                max_future_q = current_max_q;
            }
        }

        double new_q = current_q + alpha_ * (exp.reward + gamma_ * max_future_q - current_q);
        q_table_[exp.state][exp.action] = new_q;
    }
}

void EnhancedQLearningAgent::decay_epsilon() {
    epsilon_ *= epsilon_decay_rate_;
    epsilon_ = std::max(min_epsilon_, epsilon_);
    temperature_ = std::max(0.1, temperature_ * 0.999);
}

double EnhancedQLearningAgent::get_epsilon() const {
    return epsilon_;
}

size_t EnhancedQLearningAgent::get_q_table_size() const {
    return q_table_.size();
}

void EnhancedQLearningAgent::reset_exploration_counts() {
    exploration_counts_.clear();
    total_steps_ = 0;
}