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



dourado-minguell-eml-mia-um-p1/ <br>
├── README.md  <br>
├── main.ipynb <br>
├── .venv/  <br>
├── k_brazos/                      # Bandidos Multibrazo  <br>
└── Entornos_Complejos/            # Aprendizaje por Refuerzo en Entornos Complejos <br>



## Instalación y Uso

El proyecto esta preparado para su ejecución directa haciendo uso de los enlaces proporcionados en los propios cuadernos. Estos enlaces lanzan una isntancia en Google Colab que es ejecutable de manera directa sin más requisitos adicionales.


## Tecnologías Utilizadas

El trabajo se ha realizado haciendo uso de cuadernos Jupyter y scripts en lenguaje de programación Python.
El framework empleado para los entornos complejos ha sido Gymnasium
