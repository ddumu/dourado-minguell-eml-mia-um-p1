"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo softmax para el problema de los k-brazos.

Date: 2026/02/17

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

SEED = 42
np.random.seed(SEED)  # Fijar la semilla para reproducibilidad

class SoftMax(Algorithm):
    def __init__(self, k: int, temperature: float = 1):
        """
        Inicializa el algoritmo softmax.

        :param k: Número de brazos.
        :param temperature: Parametro de randomizacion.
        :raises ValueError: Si randomizacion no es >0.
        """
        assert temperature > 0

        super().__init__(k)
        self.temperature = temperature

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política softmax.

        :return: índice del brazo seleccionado.
        """
        # https://en.wikipedia.org/wiki/Softmax_function
        # https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-softmax-algorithm-e1fa4cb0c422
        # https://www.linkedin.com/posts/damienbenveniste_the-softmax-transform-might-be-one-of-the-activity-7301720641559269377-gDdQ/
        # Se escalan los valores ya que generaban overflow -> nan
        idle_arms = np.where(self.counts == 0)[0]
        if len(idle_arms)>0:
            chosen_arm = idle_arms[0]
            # print(f"Brazo elegido {chosen_arm}")
        else: 
            upper_component = np.exp((self.values/self.temperature) - np.max(self.values/self.temperature))
            lower_component = sum(upper_component)
            softmax_value = upper_component/lower_component
            # print(softmax_value)
            chosen_arm = np.random.choice(self.k, p=softmax_value)
        return chosen_arm
    