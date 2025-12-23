#include <iostream>
#include "env.h"
#include "agent.h"
#include "utils.h"

int main() {
    EnhancedCityPlanningEnv env(30, 40);
    EnhancedQLearningAgent agent(0.1, 0.9, 1.0, 0.995, 0.01);
    
    // Run ONE episode
    State state = env.reset();
    bool done = false;
    int actions = 0;
    int max_actions = 50;
    
    while (!done && actions < max_actions) {
        std::vector<int> valid_actions = env.get_valid_actions();
        if (valid_actions.empty() || valid_actions.size() == 1) break;
        
        int action = agent.choose_action(state, valid_actions);
        auto [next_state, reward, done_flag] = env.step(action);
        state = next_state;
        done = done_flag;
        actions++;
    }
    
    std::cout << "After " << actions << " actions: " << env.get_num_connected_houses() << " houses\n";
    std::cout << "Done: " << done << "\n";
    
    return 0;
}
