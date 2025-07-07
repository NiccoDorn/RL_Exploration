#include <iterator>

#include "exp_replay.h"

ExperienceReplay::ExperienceReplay(size_t capacity) : capacity_(capacity) {
    std::random_device rd;
    rng_ = std::mt19937(rd());
}

void ExperienceReplay::add(const State& state, int action, double reward, const State& next_state) {
    if (buffer_.size() == capacity_) {
        buffer_.pop_front();
    }
    buffer_.push_back({state, action, reward, next_state});
}

std::vector<ExperienceReplay::Experience> ExperienceReplay::sample(size_t batch_size) {
    if (buffer_.empty()) {
        return {};
    }
    batch_size = std::min(batch_size, buffer_.size());
    std::vector<Experience> samples;
    samples.reserve(batch_size);
    std::sample(buffer_.begin(), buffer_.end(), std::back_inserter(samples), batch_size, rng_);
    return samples;
}

size_t ExperienceReplay::size() const {
    return buffer_.size();
}