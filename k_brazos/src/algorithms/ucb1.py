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
    def __init__(self, k: int, exploration: float = 1):
        """
        Inicializa el algoritmo ucb1.

        :param k: Número de brazos.
        :param exploration : parametro para ajustar la exploración
        """

        assert exploration >= 0
        super().__init__(k)
        self.exploration = exploration

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política ucb1.

        :return: índice del brazo seleccionado.
        """

        # Selecciona el brazo con la recompensa promedio estimada más alta
        # https://towardsdatascience.com/the-upper-confidence-bound-ucb-bandit-algorithm-c05c2bf4c13f/
        # https://en.wikipedia.org/wiki/Upper_Confidence_Bound
        idle_arms = np.where(self.counts == 0)[0]
        if len(idle_arms)>0:
            chosen_arm = idle_arms[0]
            # print(f"Brazo elegido {chosen_arm}")
        else:
            ucb_value = self.values + self.exploration * np.sqrt(2*np.log(np.sum(self.counts))/self.counts)
            chosen_arm = np.argmax(ucb_value)
        return int(chosen_arm)
