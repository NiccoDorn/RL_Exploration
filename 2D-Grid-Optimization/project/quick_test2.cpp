#include <iostream>
#include "env.h"
#include "agent.h"
#include "utils.h"

int main() {
    EnhancedCityPlanningEnv env(30, 40);
    EnhancedQLearningAgent agent(0.1, 0.9, 1.0, 0.995, 0.01);
    
    // Run episode until done OR run exactly 51 actions to trigger greedy filling
    State state = env.reset();
    bool done = false;
    int actions = 0;
    
    while (!done && actions < 51) {  // Changed to 51 to trigger greedy filling
        std::vector<int> valid_actions = env.get_valid_actions();
        if (valid_actions.empty() || valid_actions.size() == 1) {
            std::cout << "No more valid actions at step " << actions << "\n";
            break;
        }
        
        int action = agent.choose_action(state, valid_actions);
        auto [next_state, reward, done_flag] = env.step(action);
        state = next_state;
        done = done_flag;
        actions++;
        
        if (actions == 50) {
            std::cout << "After 50 actions: " << env.get_num_connected_houses() << " houses, done=" << done << "\n";
        }
    }
    
    std::cout << "Final: After " << actions << " actions: " << env.get_num_connected_houses() << " houses, done=" << done << "\n";
    
    return 0;
}
