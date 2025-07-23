# City Planner Project for Automated Blueprint Generation

> The approach of the project does not target solely rectangular 2D-Grid shapes

> The approach is Q-Learning with episodes, decay, Boltzmann & UCB
> Optimizing for a "tame" rectangular 2D-Grid is stupid and not part of this.
> However, I might include it because it may be a "user" feature

> Some inconsistencies may exist due to previous ideas for the rewarding system

> Aim: Generalize problem solving to any 2D shape, thus certain reward heuristics will not be included

> Observations: Balancing alpha & gamma as well as num_actions per episode is hard
> Balancing rewards/penalties is even harder
> Some iterations may yield unexpectedly good results due to the inherent randomness

> Project aims at providing Anno 1800 and Anno 117 basic blueprints for city planning
> n Blueprint solutions per Island Shape and different-sizes rectangular shapes
> (Thus the hardcoding of rectangular optimal solutions might be a thing)

#### Training Flow of Fnction Calls
`main(...)` => `train_enhanced_rl_agent(...)` => `reset()` => `_invalidate_cache()` => `place_building_on_grid(...)`
=> `get_building_coords(...)` => `std::shuffle(...)` => `check_overlap(...)` => `check_overlap(...)` => `place_building_on_grid(...)`
=> `get_building_coords(...)`=> `_update_connectivity_cache()` => `bfs_road_reachable(...)` => `bfs_road_reachable(...)`
=> `_get_reduced_state(...)` => `_is_valid_position(...)` => `check_overlap(...)` => `is_area_road(...)` => `_get_valid_actions_optimized()`
=> `_is_valid_position(...)` => `check_overlap(...)` => `is_area_road(...)` => `get_building_coords(...)` => `is_within_influence(...)`
=> `calculate_distance_sq(...)` => `_check_connectivity_fast(...)` => `get_building_coords(...)` => `_is_block_aligned(...)`
=> `choose_action(...)` => `_choose_action_ucb(...)` || `_get_q_values`

> Flow of function calls needs to be constructed conditionally, else loss of information.