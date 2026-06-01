import numpy as np

from arms import Arm

class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución de Bernoulli.

        :param p: Media de la distribución.
        """
        assert (0 <= p <= 1),  "La media de la distribucion de Bernoulli debe estar entre 0 y 1"

        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución de Bernoulli.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(1, self.p)
        return float(reward)

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución de Bernoulli.

        :return: Valor esperado de la distribución.
        """

        return self.p

    def __str__(self):
        """
        Representación en cadena del brazo de Bernoulli.

        :return: Descripción detallada del brazo de Bernoulli.
        """
        return f"ArmBernoulli (np={self.p})"

    @classmethod
    def generate_arms(cls, k: int, p_min: int = 0, p_max: int = 1):
        """
        Genera k brazos con medias únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param p_min: Valor mínimo de la media.
        :param p_max: Valor máximo de la media.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert p_min < p_max, "El valor de p_min debe ser menor que p_max."

        # Generar k- valores únicos de p con decimales
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 2)
            p_values.add(p)

        p_values = list(p_values)
        arms = [ArmBernoulli(p) for p in p_values]

        return arms


