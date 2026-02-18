"""
Module: algorithms/ucb1.py
Description: Implementación del algoritmo ucb1 para el problema de los k-brazos.

Date: 2026/02/17

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class UCB1(Algorithm):
    def __init__(self, k: int):
        """
        Inicializa el algoritmo ucb1.

        :param k: Número de brazos.
        """
        super().__init__(k)

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política ucb1.

        :return: índice del brazo seleccionado.
        """

        # Selecciona el brazo con la recompensa promedio estimada más alta
        # https://towardsdatascience.com/the-upper-confidence-bound-ucb-bandit-algorithm-c05c2bf4c13f/
        # https://en.wikipedia.org/wiki/Upper_Confidence_Bound

        ucb_value = self.values + np.sqrt(2*np.log(np.sum(self.counts))/self.counts)
        chosen_arm = np.argmax(ucb_value)
        return int(chosen_arm)
