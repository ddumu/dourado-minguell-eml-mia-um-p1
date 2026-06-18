"""
Module: arms/armnormal.py
Description: Contains the implementation of the ArmBinomial class for the normal distribution arm.

Date: 2026/02/16

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""



import numpy as np

from src.arms import Arm


class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución binomial.

        :param p: Media de la distribución.
        :param n: Numero de intentos a partir de los cuales generar la recompensa esperada.
        :param sigma: Desviación estándar de la distribución.
        """
        assert (0 <= p <= 1), "La media de la distribucion binomial debe estar entre 0 y 1"

        self.p = p
        self.n = n

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución binomial.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(self.n, self.p)/self.n
        #reward = np.random.binomial(self.n, self.p)
        return float(reward)

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución binomial.

        :return: Valor esperado de la distribución.
        """

        return self.p

    def __str__(self):
        """
        Representación en cadena del brazo binomial.

        :return: Descripción detallada del brazo binomial.
        """
        return f"ArmBinomial (n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, p_min: int = 0.05, p_max: int = 0.95, n: int=100):
        """
        Genera k brazos con medias únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param n: Número de intentos.
        :param p_min: Valor mínimo de la media.
        :param p_max: Valor máximo de la media.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n > 0, "El número de intentos n debe ser mayor que 0."
        assert p_min < p_max, "El valor de p_min debe ser menor que p_max."

        # Generar k- valores únicos de p
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 2)
            p_values.add(p)

        p_values = list(p_values)
        arms = [ArmBinomial(n, p) for p in p_values]

        return arms


