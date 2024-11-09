import pickle
from copy import deepcopy
from statistics import median
from uuid import uuid4

import minari
import numpy as np
import torch
from tqdm import tqdm
import random
import parameters

from pareto import ParetoSelector
from td3.critic import TD3CriticWrapper

from data_processor import create_offline_dataset_from_minari
from linear_gp import Program, Mutator

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    offline_data = create_offline_dataset_from_minari(
        minari.load_dataset("HalfCheetah-Expert-v2"),
        shuffle=True
    )

    td3_critic = TD3CriticWrapper()

    population = [ (Program(), uuid4()) for _ in range(parameters.POPULATION_SIZE)]

    timesteps = 0

    #min_mse = 20.0
    for batch in offline_data:
        states, _, rewards, _, expert_actions = [
            data.to(device) for data in batch
        ]

        fitnesses = {}
        value_fitnesses = {}
        #mse_fitnesses = {}
        reward_fitnesses = {}
        for individual, individual_id in tqdm(population):
            # Actions are the 6-registers that each individual predicts for a feature set
            actions = [individual.predict(np.array(state.cpu()))[:6] for state in states]
            actions = torch.tensor(actions, device=device)

            # Fitness is the sum Q-value for all state, action pairs.
            value_fitness = td3_critic.predict(states, actions).sum().cpu().detach().numpy() / len(states)
            reward_fitness = rewards.sum().cpu().detach().numpy()

            mse = np.mean([(action.cpu().numpy() - expert_action.cpu().numpy()) ** 2 for action, expert_action in zip(actions, expert_actions)])

            #if mse < min_mse:
            #    adjusted_mse = 1 / mse  # Inverse or scaled-up MSE
            #else:
            #    adjusted_mse = mse

            fitnesses[individual_id] = (value_fitness, -mse)
            value_fitnesses[individual_id] = value_fitness
            reward_fitnesses[individual_id] = mse
            #mse_fitnesses[individual_id] = adjusted_mse

        population = ParetoSelector.select_popgap_individuals(population, fitnesses)

        parents = random.choices(population, k=int(parameters.POP_GAP * parameters.POPULATION_SIZE))
        for parent, parent_id in parents:
            child = deepcopy(parent)
            Mutator.mutateProgram(child)
            population.append((child, uuid4()))

        median_value_fitness = np.median(np.array(list(value_fitnesses.values())))
        best_value_fitness = np.max(np.array(list(value_fitnesses.values())))

        median_reward_fitness = np.median(np.array(list(reward_fitnesses.values())))
        best_reward_fitness = np.min(np.array(list(reward_fitnesses.values())))

        print(f"Timestep {timesteps}, Best Q-Score: {best_value_fitness:.3f}, Best MSE {best_reward_fitness:.3f}")
        timesteps += len(states)

        if timesteps % 10240 == 0:
            with open(f'results/HalfCheetah-v5_{timesteps}.pkl', 'wb') as f:
                pickle.dump({"population": population, "fitnesses": fitnesses}, f)

        if timesteps >= 1_000_000:
            break