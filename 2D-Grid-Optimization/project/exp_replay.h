#pragma once

#include <deque>
#include <vector>
#include <random>
#include <algorithm>

#include "common.h"

class ExperienceReplay {
public:
    struct Experience {
        State state;
        int action;
        double reward;
        State next_state;
        std::vector<int> valid_next_actions;
    };

    ExperienceReplay(size_t capacity = 5000);
    void add(const State& state, int action, double reward, const State& next_state, const std::vector<int>& valid_next_actions);
    std::vector<Experience> sample(size_t batch_size = 32);
    size_t size() const noexcept;

private:
    std::deque<Experience> buffer_;
    size_t capacity_;
    std::mt19937 rng_;
};