import datetime
import multiprocessing
import os
import pickle

import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import time

from genetic_algorithm.v6.numba_functions import place_building_on_grid, get_building_coords, check_overlap, is_area_road, \
    bfs_road_connectivity, calculate_distance_sq, is_within_influence, check_all_buildings_connectivity_numba

EMPTY = 0
HOUSE = 1
ROAD = 2
TOWN_CENTER = 3
TRADING_POST = 4
HOUSE_DIMS = (3, 3)
TOWN_CENTER_DIMS = (7, 5)
TRADING_POST_DIMS = (1, 5)
TOWN_CENTER_RADIUS_SQ = 26 ** 2
BUILDING_TYPES = {
    HOUSE: HOUSE_DIMS,
    TOWN_CENTER: TOWN_CENTER_DIMS,
    TRADING_POST: TRADING_POST_DIMS
}

class Individual:
    def __init__(self, grid_dims, house_positions=None):
        self.grid_dims = grid_dims
        self.rows, self.cols = grid_dims
        self.house_positions = house_positions if house_positions is not None else []
        self.fitness = 0
        self.grid = None
        self.kontor_coords_list = []
        self.tc_coords_list = []
        self.tc_center = None
        self.placed_buildings_info = []
        self.num_valid_houses = 0
        self.final_grid = None  # Cache for final grid

        # Initialize the base grid with kontor and town center
        self._initialize_base_grid()

        # If no house positions provided, generate random ones
        if house_positions is None:
            self._generate_random_houses()

    def _initialize_base_grid(self):
        self.grid = np.full(self.grid_dims, ROAD, dtype=int)
        self.placed_buildings_info = []
        self.kontor_coords_list = []
        self.tc_coords_list = []

        # Place kontor at (0, 0)
        kontor_r, kontor_c = 0, 0
        kontor_h, kontor_w = TRADING_POST_DIMS
        place_building_on_grid(self.grid, TRADING_POST, kontor_r, kontor_c, kontor_h, kontor_w)
        self.placed_buildings_info.append((kontor_r, kontor_c, kontor_h, kontor_w, TRADING_POST))
        self.kontor_coords_list = get_building_coords(kontor_r, kontor_c, kontor_h, kontor_w)

        # Place town center at fixed position
        r, c = 2, 0
        tc_w, tc_h = TOWN_CENTER_DIMS

        if not check_overlap(r, c, tc_h, tc_w, self.placed_buildings_info) and is_area_road(self.grid, r, c, tc_h, tc_w):
            potential_tc_coords = get_building_coords(r, c, tc_h, tc_w)
            temp_grid = copy.deepcopy(self.grid)
            place_building_on_grid(temp_grid, TOWN_CENTER, r, c, tc_h, tc_w)
            tc_is_connected_to_kontor = bfs_road_connectivity(temp_grid, self.kontor_coords_list, potential_tc_coords)

            if tc_is_connected_to_kontor:
                place_building_on_grid(self.grid, TOWN_CENTER, r, c, tc_h, tc_w)
                self.placed_buildings_info.append((r, c, tc_h, tc_w, TOWN_CENTER))
                self.tc_coords_list = potential_tc_coords
                self.tc_center = (r + tc_h // 2, c + tc_w // 2)

    def _generate_random_houses(self):
        """Generate random house positions for initial population"""
        # Generate more house positions initially - many will be filtered out
        max_houses = min(50, (self.rows * self.cols) // (HOUSE_DIMS[0] * HOUSE_DIMS[1]))
        num_houses = random.randint(max_houses // 2, max_houses)

        self.house_positions = []
        # Generate more positions than needed to account for invalid placements
        for _ in range(num_houses):
            r = random.randint(0, self.rows - HOUSE_DIMS[0])
            c = random.randint(0, self.cols - HOUSE_DIMS[1])
            self.house_positions.append((r, c))

    # Wrapper function (not compiled with Numba)
    def check_all_buildings_connectivity(self, current_grid, houses_info):
        """Wrapper function to convert data and call Numba function"""
        # Convert houses_info to numpy array
        if not houses_info:
            return True

        houses_array = np.array(houses_info, dtype=np.int32)
        kontor_coords = np.array(self.kontor_coords_list, dtype=np.int32)
        tc_coords = np.array(self.tc_coords_list, dtype=np.int32)

        return check_all_buildings_connectivity_numba(
            current_grid, houses_array, kontor_coords, tc_coords, self.rows, self.cols
        )


    def _place_valid_houses(self):
        """Place valid houses and return the count and final grid"""
        if self.tc_center is None:
            return 0, self.grid.copy()

        # Start with base grid
        temp_grid = copy.deepcopy(self.grid)
        houses_info = []
        valid_houses = 0

        # Sort house positions by distance to town center to prioritize closer houses
        sorted_positions = sorted(self.house_positions,
                                  key=lambda pos: calculate_distance_sq(pos, self.tc_center))

        # Try to place each house
        for r, c in sorted_positions:
            house_h, house_w = HOUSE_DIMS

            # Check if house placement is valid
            if (check_overlap(r, c, house_h, house_w, self.placed_buildings_info) or
                    check_overlap(r, c, house_h, house_w, houses_info) or
                    not is_area_road(temp_grid, r, c, house_h, house_w)):
                continue

            # Check if house is within town center influence
            house_coords = get_building_coords(r, c, house_h, house_w)
            if not is_within_influence(house_coords, self.tc_center):
                continue

            # Temporarily place house to check connectivity
            temp_grid_with_house = copy.deepcopy(temp_grid)
            place_building_on_grid(temp_grid_with_house, HOUSE, r, c, house_h, house_w)
            test_houses_info = houses_info + [(r, c, house_h, house_w, HOUSE)]

            if self.check_all_buildings_connectivity(temp_grid_with_house, test_houses_info):
                # House is valid, place it permanently
                place_building_on_grid(temp_grid, HOUSE, r, c, house_h, house_w)
                houses_info.append((r, c, house_h, house_w, HOUSE))
                valid_houses += 1

        return valid_houses, temp_grid

    def evaluate_fitness(self):
        """Calculate fitness score for this individual"""
        if self.tc_center is None:
            self.fitness = -1000
            self.num_valid_houses = 0
            return self.fitness

        # Use the same logic for both fitness evaluation and final grid generation
        valid_houses, final_grid = self._place_valid_houses()
        self.num_valid_houses = valid_houses
        self.final_grid = final_grid  # Cache the final grid

        # Aggressive fitness calculation - heavily reward more houses
        fitness = valid_houses * 1000  # Increased reward for each house

        # Exponential bonus for more houses
        if valid_houses > 0:
            fitness += (valid_houses ** 1.5) * 100

        # Bonus for density - reward having many house positions even if not all are valid
        density_bonus = min(len(self.house_positions) * 5, 200)
        fitness += density_bonus

        # Smaller penalty for invalid houses to encourage exploration
        invalid_houses = len(self.house_positions) - valid_houses
        fitness -= invalid_houses * 2  # Reduced penalty

        self.fitness = fitness
        return self.fitness

    def get_final_grid(self):
        """Get the final grid with all valid houses placed"""
        if self.final_grid is not None:
            return self.final_grid

        # If not cached, generate it
        _, final_grid = self._place_valid_houses()
        self.final_grid = final_grid
        return final_grid

    def get_actual_house_count(self):
        """Get the actual number of houses placed on the grid"""
        final_grid = self.get_final_grid()
        return np.sum(final_grid == HOUSE) // (HOUSE_DIMS[0] * HOUSE_DIMS[1])

    def mutate(self, mutation_rate=0.1):
        """Mutate the individual by modifying house positions"""
        # Clear cache when mutating
        self.final_grid = None

        # Higher chance to add houses
        if random.random() < mutation_rate * 2:  # Double mutation rate for additions
            mutation_type = random.choices(['add', 'remove', 'move'], weights=[0.5, 0.1, 0.4])[0]

            if mutation_type == 'add' and len(self.house_positions) < 60:
                # Add multiple new house positions
                num_to_add = random.randint(1, 3)
                for _ in range(num_to_add):
                    r = random.randint(0, self.rows - HOUSE_DIMS[0])
                    c = random.randint(0, self.cols - HOUSE_DIMS[1])
                    self.house_positions.append((r, c))

            elif mutation_type == 'remove' and len(self.house_positions) > 5:
                # Remove a house position (but keep minimum)
                self.house_positions.pop(random.randint(0, len(self.house_positions) - 1))

            elif mutation_type == 'move' and len(self.house_positions) > 0:
                # Move an existing house position
                idx = random.randint(0, len(self.house_positions) - 1)
                r = random.randint(0, self.rows - HOUSE_DIMS[0])
                c = random.randint(0, self.cols - HOUSE_DIMS[1])
                self.house_positions[idx] = (r, c)

        # Additional mutation: occasionally do a "smart" placement near town center
        if random.random() < mutation_rate and self.tc_center is not None:
            # Try to place a house near the town center
            for _ in range(3):  # 3 attempts
                # Generate position within town center influence
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(0, 20)  # Within reasonable distance
                r = max(0, min(self.rows - HOUSE_DIMS[0],
                               int(self.tc_center[0] + distance * np.cos(angle))))
                c = max(0, min(self.cols - HOUSE_DIMS[1],
                               int(self.tc_center[1] + distance * np.sin(angle))))
                self.house_positions.append((r, c))

    def crossover(self, other):
        """Create offspring through crossover with another individual"""
        # Combine all house positions from both parents
        combined_positions = self.house_positions + other.house_positions

        # Remove duplicates while preserving order
        seen = set()
        unique_positions = []
        for pos in combined_positions:
            if pos not in seen:
                seen.add(pos)
                unique_positions.append(pos)

        # Keep more positions in offspring
        max_houses = min(50, len(unique_positions))
        if max_houses > 0:
            # Take more positions to encourage higher building counts
            num_houses = random.randint(max_houses // 2, max_houses)
            offspring_positions = unique_positions[:num_houses]
        else:
            offspring_positions = []

        # Add some random positions to increase diversity
        for _ in range(random.randint(5, 15)):
            r = random.randint(0, self.rows - HOUSE_DIMS[0])
            c = random.randint(0, self.cols - HOUSE_DIMS[1])
            if (r, c) not in offspring_positions:
                offspring_positions.append((r, c))

        return Individual(self.grid_dims, offspring_positions)

    def save_to_dict(self):
        """Convert individual to dictionary for saving"""
        return {
            'grid_dims': self.grid_dims,
            'house_positions': self.house_positions,
            'fitness': self.fitness,
            'num_valid_houses': self.num_valid_houses,
            'tc_center': self.tc_center,
            'placed_buildings_info': self.placed_buildings_info,
            'kontor_coords_list': self.kontor_coords_list,
            'tc_coords_list': self.tc_coords_list
        }

    @classmethod
    def load_from_dict(cls, data):
        """Create individual from dictionary"""
        individual = cls(data['grid_dims'], data['house_positions'])
        individual.fitness = data['fitness']
        individual.num_valid_houses = data['num_valid_houses']
        individual.tc_center = data['tc_center']
        individual.placed_buildings_info = data['placed_buildings_info']
        individual.kontor_coords_list = data['kontor_coords_list']
        individual.tc_coords_list = data['tc_coords_list']
        return individual


""" Genetic Algorithm """


class GeneticAlgorithm:
    def __init__(self, grid_dims, population_size=100, mutation_rate=0.1, crossover_rate=0.8):
        self.grid_dims = grid_dims
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.generation = 0

        self.fitness_history = []
        self.training_stats = {}

        # Initialize population
        self._initialize_population()

    def _initialize_population(self):
        """Create initial population of random individuals"""
        self.population = []
        for _ in range(self.population_size):
            individual = Individual(self.grid_dims)
            individual.evaluate_fitness()
            self.population.append(individual)

    def _selection(self):
        """Tournament selection"""
        tournament_size = 5
        selected = []

        for _ in range(self.population_size):
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)

        return selected

    def _crossover_and_mutation(self, selected):
        """Create new population through crossover and mutation"""
        new_population = []

        # Keep best individual (elitism)
        if self.best_individual:
            new_population.append(self.best_individual)

        while len(new_population) < self.population_size:
            parent1 = random.choice(selected)

            if random.random() < self.crossover_rate:
                parent2 = random.choice(selected)
                offspring = parent1.crossover(parent2)
            else:
                offspring = Individual(self.grid_dims, parent1.house_positions[:])

            offspring.mutate(self.mutation_rate)
            offspring.evaluate_fitness()
            new_population.append(offspring)

        return new_population

    def evolve(self, generations=100):
        """Run the genetic algorithm for specified generations"""
        start_time = time.time()

        for generation in range(generations):
            self.generation = generation

            # Evaluate all individuals
            for individual in self.population:
                individual.evaluate_fitness()

            # Track best individual
            current_best = max(self.population, key=lambda x: x.fitness)
            if current_best.fitness > self.best_fitness:
                self.best_fitness = current_best.fitness
                # Create a proper copy of the best individual
                self.best_individual = Individual(self.grid_dims, current_best.house_positions[:])
                self.best_individual.evaluate_fitness()

            # Selection
            selected = self._selection()

            # Create new population
            self.population = self._crossover_and_mutation(selected)

            # Progress reporting with actual house count verification
            if self.best_individual:
                actual_houses = self.best_individual.get_actual_house_count()
                reported_houses = self.best_individual.num_valid_houses
            else:
                actual_houses = 0
                reported_houses = 0
            if generation % 10 == 0:
                avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
                elapsed_time = time.time() - start_time
                print(f"Generation {generation}/{generations}, "
                      f"Best Fitness: {self.best_fitness:.2f}, "
                      f"Reported Houses: {reported_houses}, "
                      f"Actual Houses: {actual_houses}, "
                      f"Avg Fitness: {avg_fitness:.2f}, "
                      f"Time: {elapsed_time:.2f}s")
                start_time = time.time()

        return self.best_individual

    def save_model(self, filepath=None):
        """Save the trained model to disk"""
        if filepath is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ga_model_{timestamp}.pkl"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        # Ensure best individual is properly evaluated
        if self.best_individual:
            self.best_individual.evaluate_fitness()

        model_data = {
            'grid_dims': self.grid_dims,
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'best_individual': self.best_individual.save_to_dict() if self.best_individual else None,
            'best_fitness': self.best_fitness,
            'generation': self.generation,
            'fitness_history': self.fitness_history,
            'training_stats': self.training_stats,
            'population': [ind.save_to_dict() for ind in self.population]
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        # Verify the saved model
        if self.best_individual:
            actual_houses = self.best_individual.get_actual_house_count()
            reported_houses = self.best_individual.num_valid_houses
            print(f"Model saved to {filepath}")
            print(f"Best solution: {reported_houses} reported houses, {actual_houses} actual houses")
        else:
            print(f"Model saved to {filepath} (no best individual found)")

        return filepath

    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create new GA instance
        ga = cls(
            grid_dims=model_data['grid_dims'],
            population_size=model_data['population_size'],
            mutation_rate=model_data['mutation_rate'],
            crossover_rate=model_data['crossover_rate']
        )

        # Restore state
        ga.best_fitness = model_data['best_fitness']
        ga.generation = model_data['generation']
        ga.fitness_history = model_data['fitness_history']
        ga.training_stats = model_data['training_stats']

        # Restore best individual
        if model_data['best_individual']:
            ga.best_individual = Individual.load_from_dict(model_data['best_individual'])
            # Re-evaluate to ensure consistency
            ga.best_individual.evaluate_fitness()

        # Restore population
        ga.population = [Individual.load_from_dict(ind_data) for ind_data in model_data['population']]

        # Verify loaded model
        if ga.best_individual:
            actual_houses = ga.best_individual.get_actual_house_count()
            reported_houses = ga.best_individual.num_valid_houses
            print(f"Model loaded from {filepath}")
            print(f"Best solution: {reported_houses} reported houses, {actual_houses} actual houses")
        else:
            print(f"Model loaded from {filepath} (no best individual found)")

        return ga


""" Matplotlib Visualization """


def visualize_grid(grid, title="Gebäudeplatzierung"):
    cmap = plt.cm.colors.ListedColormap(['white', 'green', 'grey', 'purple', 'yellow'])
    # Werte: EMPTY=0, HOUSE=1, ROAD=2, TOWN_CENTER=3, TRADING_POST=4
    bounds = [0, 1, 2, 3, 4, 5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(grid.shape[1] * 0.5, grid.shape[0] * 0.5))
    plt.imshow(grid, cmap=cmap, norm=norm, origin='upper', extent=[0, grid.shape[1], grid.shape[0], 0])
    plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(0, grid.shape[1] + 1, 1))
    plt.yticks(np.arange(0, grid.shape[0] + 1, 1))
    plt.xlabel("X-Koordinate (Kacheln)")
    plt.ylabel("Y-Koordinate (Kacheln)")
    plt.title(title)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Wohnhaus'),
        Patch(facecolor='grey', label='Straße'),
        Patch(facecolor='purple', label='Dorfzentrum'),
        Patch(facecolor='yellow', label='Kontor'),
        Patch(facecolor='white', label='Leere Kachel')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def run_ga_training(grid_dims, population_size, mutation_rate, crossover_rate, loop_number):
    grid_dimensions = (20, 20)
    print(f"GA-Training auf einem {grid_dimensions[0]}x{grid_dimensions[1]} Raster...")

    # Create and run genetic algorithm
    ga = GeneticAlgorithm(
        grid_dims=grid_dimensions,
        population_size=population_size,  # Increased population size
        mutation_rate=mutation_rate,  # Higher mutation rate
        crossover_rate=crossover_rate  # Higher crossover rate
    )
    start_time = time.time()
    best_individual = ga.evolve(generations=1000)

    if best_individual and best_individual.num_valid_houses > 0:
        actual_houses = best_individual.get_actual_house_count()
        reported_houses = best_individual.num_valid_houses
        #print(f"\nGA-Training abgeschlossen.")
        print(f"Beste gefundene Lösung: {reported_houses} reported houses, {actual_houses} actual houses")

        best_grid = best_individual.get_final_grid()
        #visualize_grid(best_grid,
        #                f"GA Beste gefundene Lösung ({actual_houses} Häuser) - Grid {grid_dimensions[0]}x{grid_dimensions[1]}")
    else:
        print("\nGA konnte keine gültige Platzierung finden.")

    ga.save_model(f"my_model_{loop_number+1}.pkl")  # Save model with loop number

    duration = time.time() - start_time
    print(f"\nTraining abgeschlossen. Total duration: {duration:.2f} seconds")


if __name__ == "__main__":
    grid_dimensions = (20, 20)
    processes = []
    for i in range(1):
        process = multiprocessing.Process(target=run_ga_training,
                                          args=(grid_dimensions, 100, 0.5, 0.8, i))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()