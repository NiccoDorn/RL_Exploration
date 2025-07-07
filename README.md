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