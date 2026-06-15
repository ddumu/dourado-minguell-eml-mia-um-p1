"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""
import os
import math
from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from algorithms import Algorithm, EpsilonGreedy, SoftMax, UCB1


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    elif isinstance(algo, SoftMax):
        label += f" (temperature={algo.temperature})"
    elif isinstance(algo, UCB1):
        label += f" (exploration={algo.exploration})"
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm], save_path: str):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param save_path: Dónde guardar la gráfica en disco.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/plot_average_rewards.png")
    
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm], save_path: str):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param save_path: Dónde guardar la gráfica en disco.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Selecciones Óptimas', fontsize=14)
    plt.title('Selecciones Óptimas vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/plot_optimal_selections.png")

    plt.show()


def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], save_path: str):
  """
  Genera la gráfica de Regret Acumulado vs Pasos de Tiempo
  :param steps: Número de pasos de tiempo.
  :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
  :param algorithms: Lista de instancias de algoritmos comparados.
  :param save_path: Dónde guardar la gráfica en disco.
  """
  sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

  plt.figure(figsize=(14, 7))
  for idx, algo in enumerate(algorithms):
      label = get_algorithm_label(algo)
      plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

  plt.xlabel('Pasos de Tiempo', fontsize=14)
  plt.ylabel('Arrepentimiento', fontsize=14)
  plt.title('Arrepentimiento vs Pasos de Tiempo', fontsize=16)
  plt.legend(title='Algoritmos')
  plt.tight_layout()

  os.makedirs(save_path, exist_ok=True)
  plt.savefig(f"{save_path}/plot_regret.png")

  plt.show()


def plot_arm_statistics(arm_stats: np.ndarray, algorithms: List[Algorithm], save_path:str):
    """
    Genera gráficas separadas de selección de brazos: ganancias vs pérdidas para cada algoritmo.
    :param arm_stats: Lista de listas de diccionarios con estadísticas de cada brazo por algoritmo.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param save_path: Dónde guardar la gráfica en disco.
    """
    # Configurar el tema de seaborn
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    # Configurar el número de axes
    n_algos = len(arm_stats)
    n_cols = math.ceil(n_algos / 2)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))
    axes = np.concatenate([axes[0], axes[1]])

    # Ocultar ejes sobrantes si n_algos es impar
    for i in range(n_algos, len(axes)):
        axes[i].set_visible(False)
    
    if n_algos == 1:
        axes = [axes]

    for algo_idx, algo_stats in enumerate(arm_stats):
        ax = axes[algo_idx]

        # Extraer el nombre del algoritmo
        algo_name = get_algorithm_label(algorithms[algo_idx])
        
        # Rango con el número de brazos
        x = range(len(algo_stats))
        
        # Extraer los valores del diccionario
        y_n_selected = [algo_stats[i]["n_times_selected"] for i in x]
        y_avg_reward = [algo_stats[i]["avg_reward"] for i in x]
        y_is_optimal = [algo_stats[i]["is_optimal"] for i in x]
        
        # Array con un 1 en la posición del brazo óptimo
        optimal_index = y_is_optimal.index(True)
        
        # Graficar barras
        bars = ax.bar(x, y_avg_reward, color="blue")
        
        # Cambiar el color del brazo óptimo
        bars[optimal_index].set_color("gold")
        
        # Añadir etiquetas con el número de veces seleccionado de cada brazo
        for i, bar in enumerate(bars):
            ax.text(  # <-- ax.text en lugar de plt.text
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"z={y_n_selected[i]}",
                ha="center",
                va="bottom",
                fontsize=10
            )
            
        # Configurar ejes y título
        ax.set_title(f"{algo_name}", fontsize=14)
        ax.set_xlabel("Brazo", fontsize=12)
        ax.set_ylabel("Recompensa Promedio", fontsize=12)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(i) for i in x])
        
    fig.legend(
        handles=[
            Patch(facecolor="blue", label="Brazo No Óptimo"),
            Patch(facecolor="gold", label="Brazo Óptimo"),
            Patch(facecolor="none", label="z = Número de Veces Seleccionado")
        ],
        fontsize=10,
        loc="center"
    )
        
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/plot_arm_stats.png")

    plt.show()