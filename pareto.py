import numpy as np
import parameters


class ParetoSelector:
    @staticmethod
    def is_dominated(fitness1, fitness2):
        """Check if fitness1 is dominated by fitness2."""
        return (fitness2[0] >= fitness1[0] and fitness2[1] > fitness1[1]) or (fitness2[0] > fitness1[0] and fitness2[1] >= fitness1[1])

    @staticmethod
    def dominance_count_ranking(population, fitnesses):
        """
        Calculate the dominance count for each individual in the population.

        :param population: List of (individual, individual_id) tuples
        :param fitnesses: Dictionary mapping individual IDs to fitness tuples
        :return: List of tuples (individual, individual_id, dominance_count), sorted by dominance count in descending order
        """
        dominance_counts = {}
        # Initialize dominance counts
        for _, ind_id in population:
            dominance_counts[ind_id] = 0

        # Calculate how many individuals each one dominates
        for _, ind_id in population:
            for _, other_id in population:
                if ind_id != other_id and ParetoSelector.is_dominated(fitnesses[other_id], fitnesses[ind_id]):
                    dominance_counts[ind_id] += 1

        # Create a list of individuals with their dominance count
        ranked_population = [(individual, ind_id, dominance_counts[ind_id]) for individual, ind_id in population]

        # Sort individuals by dominance count (higher is better)
        ranked_population.sort(key=lambda x: x[2], reverse=True)

        return ranked_population

    @staticmethod
    def calculate_crowding_distance(population, fitnesses):
        """
        Calculate the crowding distance for each individual.

        :param population: List of (individual, individual_id) tuples
        :param fitnesses: Dictionary mapping individual IDs to fitness tuples
        :return: Dictionary mapping individual IDs to crowding distances
        """
        distances = {ind_id: 0 for _, ind_id in population}
        num_objectives = len(next(iter(fitnesses.values())))

        for m in range(num_objectives):
            # Sort by the m-th objective
            sorted_population = sorted(population, key=lambda x: fitnesses[x[1]][m])
            min_val = fitnesses[sorted_population[0][1]][m]
            max_val = fitnesses[sorted_population[-1][1]][m]

            # Assign infinite distance to boundary points
            distances[sorted_population[0][1]] = float('inf')
            distances[sorted_population[-1][1]] = float('inf')

            # Calculate crowding distances for intermediate points
            for i in range(1, len(sorted_population) - 1):
                prev_id = sorted_population[i - 1][1]
                next_id = sorted_population[i + 1][1]
                if max_val - min_val != 0:
                    distances[sorted_population[i][1]] += (
                        (fitnesses[next_id][m] - fitnesses[prev_id][m]) / (max_val - min_val)
                    )

        return distances

    @staticmethod
    def select_popgap_individuals(population, fitnesses):
        num_to_select = int(parameters.POP_GAP * parameters.POPULATION_SIZE)

        # Get dominance count rankings
        ranked_population = ParetoSelector.dominance_count_ranking(population, fitnesses)

        # Calculate crowding distances
        crowding_distances = ParetoSelector.calculate_crowding_distance(population, fitnesses)

        # Add crowding distances to the ranked population
        ranked_population = [
            (individual, ind_id, dominance_count, crowding_distances[ind_id])
            for individual, ind_id, dominance_count in ranked_population
        ]

        # Sort by dominance count first (descending), then by crowding distance (descending)
        ranked_population.sort(key=lambda x: (x[2], x[3]), reverse=True)

        # Select the top individuals
        selected_population = ranked_population[:num_to_select]

        # Return only the (individual, individual_id) tuples
        return [(individual, ind_id) for individual, ind_id, _, _ in selected_population]