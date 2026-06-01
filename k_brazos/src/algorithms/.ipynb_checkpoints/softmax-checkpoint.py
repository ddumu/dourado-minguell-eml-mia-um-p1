import numpy as np

from .algorithm import Algorithm


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
