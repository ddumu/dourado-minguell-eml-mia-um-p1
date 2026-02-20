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
        label += f" (adjust={algo.exploration})"
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
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
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
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
    plt.show()


def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], *args):
  """
  Genera la gráfica de Regret Acumulado vs Pasos de Tiempo
  :param steps: Número de pasos de tiempo.
  :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
  :param algorithms: Lista de instancias de algoritmos comparados.
  :param args: Opcional. Parámetros que consideres. P.e. la cota teórica Cte * ln(T).
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
  plt.show()


def plot_arm_statistics(arm_stats: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera gráficas separadas de selección de brazos: ganancias vs pérdidas para cada algoritmo.
    :param arm_stats: Lista de listas de diccionarios con estadísticas de cada brazo por algoritmo.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    # Configurar el tema de seaborn
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    for algo_idx, algo_stats in enumerate(arm_stats):
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
        plt.figure()
        bars = plt.bar(x, y_avg_reward, color="blue")

        # Cambiar el color del brazo óptimo
        bars[optimal_index].set_color("gold")

        # Añadir etiquetas con el número de veces seleccionado de cada brazo
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"z={y_n_selected[i]}",
                ha="center",
                va="bottom",
                fontsize=10
            )

        # Configurar ejes y título
        plt.title(f"{algo_name}", fontsize=14)
        plt.xlabel("Brazo", fontsize=12)
        plt.ylabel("Recompensa Promedio", fontsize=12)
        plt.xticks(ticks=x, labels=[str(i) for i in x])

        # Crear una leyenda con información
        plt.legend(handles=[
            Patch(facecolor="blue", label="Brazo No Óptimo"),
            Patch(facecolor="gold", label=f"Brazo Óptimo = {optimal_index}"),
            Patch(facecolor="none", label="z = Número de Veces Seleccionado")
        ], fontsize=10)

        # Mostrar la gráfica
        plt.tight_layout()
        plt.show()