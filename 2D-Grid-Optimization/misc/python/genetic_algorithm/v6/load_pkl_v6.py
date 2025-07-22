from genetic_algorithm.v6.genetic_v6 import GeneticAlgorithm, visualize_grid

if __name__ == "__main__":
    grid_dimensions = (20, 20)
    # Load full model
    ga = GeneticAlgorithm.load_model("highscore.pkl")
    print(f"Loaded model with best fitness: {ga.best_fitness}")
    best_individual = ga.best_individual
    if best_individual and best_individual.num_valid_houses > 0:
        actual_houses = best_individual.get_actual_house_count()
        reported_houses = best_individual.num_valid_houses
        print(f"Beste gefundene Lösung: {reported_houses} reported houses, {actual_houses} actual houses")

        best_grid = best_individual.get_final_grid()
        visualize_grid(best_grid,
                        f"GA Beste gefundene Lösung ({actual_houses} Häuser) - Grid {grid_dimensions[0]}x{grid_dimensions[1]}")