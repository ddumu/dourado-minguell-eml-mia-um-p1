# 
## Información
- **Alumnos:** Dourado, Denis; Minguell, Diego; 
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2025/2026
- **Grupo:** DouradoMinguell


## Descripción
Este trabajo es una práctica de la asignatura Extensiones de Machine Learning (Máster Universitario en Inteligencia Artificial, Universidad de Murcia, curso 2025/2026), realizada por Denis Dourado y Diego Minguell.

Objetivo: Investigar y experimentar con algoritmos clásicos de aprendizaje por refuerzo en dos escenarios diferenciados.

Partes del trabajo:

 - Bandido de k-brazos: Se comparan estrategias de exploración/explotación (ε-greedy, Softmax y UCB1) sobre tres distribuciones de recompensa (Bernoulli, Binomial y Normal), analizando su convergencia y regret acumulado.
 - Entornos complejos: Se implementan y evalúan algoritmos tabulares (Q-Learning, SARSA, Monte Carlo) y un método basado en redes neuronales (Deep Q-Learning) sobre entornos de Gymnasium, comparando su eficiencia y estabilidad.
 
Los experimentos se ejecutan mediante cuadernos Jupyter/Colab, con la lógica reusable organizada en módulos Python bajo src/.


## Estructura


En la carpeta principal del proyecto se encuentra un cuaderno "main.ipynb" en el cual se encuentran enlaces a sendos cuadernos en los cuales se ha separado el proyecto:
 - Bandido de k-brazos
 - Entornos Complejos.

Dentro de estos cuadernos se puede acceder a cada uno de los experimentos que se han llevado a cabo en este trabajo.

Cada uno de las partes del trabajo contiene su propia carpeta donde se han ubicado todos los scripts adicionales necesarios para la ejecución de los experimentos.


dourado-minguell-eml-mia-um-p1/
├── README.md
├── main.ipynb
├── .venv/
├── k_brazos/                      # Bandidos Multibrazo
│   ├── main.ipynb
│   ├── comparison_experiment.ipynb
│   ├── epsilon_greedy_bernoulli_dist_experiment.ipynb
│   ├── epsilon_greedy_binomial_dist_experiment.ipynb
│   ├── epsilon_greedy_normal_dist_experiment.ipynb
│   ├── softmax_bernoulli_dist_experiment.ipynb
│   ├── softmax_binomial_dist_experiment.ipynb
│   ├── softmax_normal_dist_experiment.ipynb
│   ├── ucb1_bernoulli_dist_experiment.ipynb
│   ├── ucb1_binomial_dist_experiment.ipynb
│   ├── ucb1_normal_dist_experiment.ipynb
│   └── src/
│       ├── algorithms/
│       │   ├── __init__.py
│       │   ├── algorithm.py
│       │   ├── epsilon_greedy.py
│       │   ├── softmax.py
│       │   └── ucb1.py
│       ├── arms/
│       │   ├── __init__.py
│       │   ├── arm.py
│       │   ├── armbernoulli.py
│       │   ├── armbinomial.py
│       │   ├── armnormal.py
│       │   └── bandit.py
│       ├── experiment/
│       │   └── __init__.py
│       └── plotting/
│           ├── __init__.py
│           └── plotting.py
│
└── Entornos_Complejos/            # Aprendizaje por Refuerzo en Entornos Complejos
    ├── main.ipynb
    ├── comparison_tabulars.ipynb
    ├── Deep_Q_Learning.ipynb
    ├── monte_carlo_off_policy_experiment.ipynb
    ├── monte_carlo_on_policy_experiment.ipynb
    ├── q_learning_experiment.ipynb
    ├── sarsa_experiment.ipynb
    ├── SARSA_Semi_Gradiente.ipynb
    └── src/
        ├── experiment/
        │   ├── __init__.py
        │   ├── deep_functions.py
        │   ├── monte_carlo_every_visit.py
        │   ├── q_learning.py
        │   ├── sarsa.py
        │   └── sarsa_functions.py
        └── plotting/
            ├── __init__.py
            └── plotting.py

## Instalación y Uso

El proyecto esta preparado para su ejecución directa haciendo uso de los enlaces proporcionados en los propios cuadernos. Estos enlaces lanzan una isntancia en Google Colab que es ejecutable de manera directa sin más requisitos adicionales.


## Tecnologías Utilizadas

El trabajo se ha realizado haciendo uso de cuadernos Jupyter y scripts en lenguaje de programación Python.
El framework empleado para los entornos complejos ha sido Gymnasium