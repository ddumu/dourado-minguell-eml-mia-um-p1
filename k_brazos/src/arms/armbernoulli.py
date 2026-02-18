"""
Module: arms/armbernoulli.py
Description: Contains the implementation of the ArmBernoulli class for the normal distribution arm.

Date: 2026/02/16

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""


import numpy as np

from arms import Arm

SEED = 42
np.random.seed(SEED)  # Fijar la semilla para reproducibilidad

class ArmBernoulli(Arm):
    def __init__(self, mu: float):
        """
        Inicializa el brazo con distribución de Bernoulli.

        :param mu: Media de la distribución.
        :param sigma: Desviación estándar de la distribución.
        """
        assert (0 <= mu <= 1),  "La media de la distribucion de Bernoulli debe estar entre 0 y 1"

        self.mu = mu
        # self.tries = tries

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución de Bernoulli.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(1, self.mu)
        return float(reward)

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución de Bernoulli.

        :return: Valor esperado de la distribución.
        """

        return self.mu

    def __str__(self):
        """
        Representación en cadena del brazo de Bernoulli.

        :return: Descripción detallada del brazo de Bernoulli.
        """
        return f"ArmBernoulli (mu={self.mu})"

    @classmethod
    def generate_arms(cls, k: int, mu_min: int = 0, mu_max: int = 1):
        """
        Genera k brazos con medias únicas en el rango [mu_min, mu_max].

        :param k: Número de brazos a generar.
        :param mu_min: Valor mínimo de la media.
        :param mu_max: Valor máximo de la media.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert mu_min < mu_max, "El valor de mu_min debe ser menor que mu_max."

        # Generar k- valores únicos de mu con decimales
        mu_values = set()
        while len(mu_values) < k:
            mu = np.random.uniform(mu_min, mu_max)
            mu = round(mu, 2)
            mu_values.add(mu)

        mu_values = list(mu_values)
        # tries = 10
        arms = [ArmBernoulli(mu) for mu in mu_values]

        return arms


