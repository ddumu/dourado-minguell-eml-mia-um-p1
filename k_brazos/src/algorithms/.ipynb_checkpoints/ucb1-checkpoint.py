import numpy as np

from .algorithm import Algorithm

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
