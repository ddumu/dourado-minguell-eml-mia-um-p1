from typing import List

import numpy as np

from arms import Bandit
from algorithms import Algorithm


# Modified from source: https://github.com/ldaniel-hm/eml_k_bandit/blob/main/bandit_experiment.ipynb
def run_experiment(bandit: Bandit, algorithms: List[Algorithm], steps: int, runs: int):

    optimal_arm = bandit.optimal_arm  # Necesario para calcular el porcentaje de selecciones óptimas.

    rewards = np.zeros((len(algorithms), steps)) # Matriz para almacenar las recompensas promedio.

    optimal_selections = np.zeros((len(algorithms), steps))  # Matriz para almacenar el porcentaje de selecciones óptimas.

    # Matriz para almacenar el arrepentimiento de cada algoritmo
    regret_accumulated = np.zeros((len(algorithms), steps))

    # Matriz para almacenar estadísticas de los brazos y algoritmos
    arm_stats = np.empty((len(algorithms), len(bandit)), dtype=object)

    for algo_idx in range(len(algorithms)):
        for arm_idx in range(len(bandit)):
            arm_stats[algo_idx, arm_idx] = {
                "avg_reward": 0.0,
                "n_times_selected": 0,
                "is_optimal": arm_idx == optimal_arm
            }

    for run in range(runs):
        current_bandit = Bandit(arms=bandit.arms)

        for algo in algorithms:
            algo.reset() # Reiniciar los valores de los algoritmos.

        total_rewards_per_algo = np.zeros(len(algorithms)) # Acumulador de recompensas por algoritmo. Necesario para calcular el promedio.

        # Arrepentimiento acumulado temporal para el run
        current_regret = np.zeros(len(algorithms))
        
        for step in range(steps):
            for idx, algo in enumerate(algorithms):
                chosen_arm = algo.select_arm() # Seleccionar un brazo según la política del algoritmo.
                reward = current_bandit.pull_arm(chosen_arm) # Obtener la recompensa del brazo seleccionado.
                algo.update(chosen_arm, reward) # Actualizar el valor estimado del brazo seleccionado.

                rewards[idx, step] += reward # Acumular la recompensa obtenida en la matriz rewards para el algoritmo idx en el paso step.
                total_rewards_per_algo[idx] += reward # Acumular la recompensa obtenida en total_rewards_per_algo para el algoritmo idx.

                # Modificar optimal_selections cuando el brazo elegido se corresponda con el brazo óptimo optimal_arm
                optimal_selections[idx, step] += 1 if chosen_arm == optimal_arm else 0

                # Calcular el arrepentimiento del brazo elegido
                optimal_reward = current_bandit.get_expected_value(optimal_arm)
                current_regret[idx] += optimal_reward - reward
                
                # Calcular el arrepentimiento acumulado
                regret_accumulated[idx, step] += current_regret[idx]
    
                # Actualizar las estadísticas del brazo seleccionado para el algoritmo actual
                arm_stats[idx, chosen_arm]["avg_reward"] += reward
                arm_stats[idx, chosen_arm]["n_times_selected"] += 1

    rewards /= runs

    # Calcular el porcentaje de selecciones óptimas y almacenar en optimal_selections
    optimal_selections /= runs

    # Calcular el arrepentimiento y almacenar en regrets
    regret_accumulated /= runs

    # Promediar las estadísticas de los brazos por algoritmo
    for algo_idx in range(len(algorithms)):
          for arm_idx in range(len(arm_stats[algo_idx])):
            arm_stats[algo_idx, arm_idx]["avg_reward"] /= arm_stats[algo_idx, arm_idx]["n_times_selected"]
            arm_stats[algo_idx, arm_idx]["n_times_selected"] //= runs

    return rewards, optimal_selections, regret_accumulated, arm_stats
