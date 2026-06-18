import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.animation as animation
from IPython.display import HTML

# Función para obtener el vector de características x(s, a) usando codificación one-hot
def get_features(state, action, num_states, nA):
    # Vector de tamaño total: num_states * nA
    x = np.zeros(num_states * nA)
    # Activamos la característica correspondiente al par (estado, acción)
    feature_idx = state * nA + action
    x[feature_idx] = 1.0
    return x

# Función para calcular Q(s, a) a partir de los pesos w
def get_Q_value(w, state, action, num_states, nA):
    x = get_features(state, action, num_states, nA)
    return np.dot(w, x)

# Función para obtener los valores Q de todas las acciones en un estado dado
def get_all_Q_values_for_state(w, state, num_states, nA):
    Q_s = np.zeros(nA)
    for a in range(nA):
        Q_s[a] = get_Q_value(w, state, a, num_states, nA)
    return Q_s

# Política epsilon-soft adaptada a aproximación por función
def random_epsilon_greedy_policy(w, epsilon, state, num_states, nA):
    pi_A = np.ones(nA, dtype=float) * epsilon / nA
    Q_s = get_all_Q_values_for_state(w, state, num_states, nA)
    best_action = np.argmax(Q_s)
    pi_A[best_action] += (1.0 - epsilon)
    return pi_A

# Política epsilon-greedy para la selección de acciones
def epsilon_greedy_policy(w, epsilon, state, num_states, nA):
    pi_A = random_epsilon_greedy_policy(w, epsilon, state, num_states, nA)
    return np.random.choice(np.arange(nA), p=pi_A)

# Política Greedy a partir de los valores Q. Se usa para mostrar la solución.
def pi_star_from_Q(env, Q):
    done = False
    pi_star = np.zeros([env.observation_space.n, env.action_space.n])
    state, info = env.reset() # empezamos arriba a la izquierda = 0
    actions = ""
    while not done:
        # Volvemos a usar la matriz Q directamente como en el cuaderno original
        action = np.argmax(Q[state, :])
        actions += f"{action}, "
        pi_star[state, action] = action
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return pi_star, actions

#@title Algoritmo SARSA Semi-gradiente

def semi_gradient_sarsa(env, num_episodes=5000, epsilon=0.4, decay=False, discount_factor=1.0, alpha=0.1):
  num_states = env.observation_space.n
  nA = env.action_space.n

  # Inicializamos el vector de pesos w de forma aleatoria o en ceros
  # Tamaño: (num_estados * num_acciones)
  w = np.zeros(num_states * nA)

  # Métricas para las gráficas
  stats = 0.0
  list_stats = [stats]
  list_episodes_length = []
  step_display = num_episodes / 10

  for t in tqdm(range(num_episodes)):
      state, info = env.reset(seed=100)
      done = False
      episode_reward = 0.0
      step_count = 0

      if decay:
          epsilon = min(1.0, 1000.0 / (t + 1))

      # 1. Seleccionar la acción inicial A mediante política epsilon-greedy
      action = epsilon_greedy_policy(w, epsilon, state, num_states, nA)

      while not done:
          step_count += 1

          # 2. Ejecutar la acción A, observar R y el nuevo estado S'
          new_state, reward, terminated, truncated, info = env.step(action)
          done = terminated or truncated
          episode_reward += reward # Acumulamos para estadísticas sin descuento para evaluar éxito

          # Calcular x(s, a) y q(s, a, w) actuales
          x_current = get_features(state, action, num_states, nA)
          q_current = np.dot(w, x_current)

          if done:
              # Si es un estado terminal, el valor del siguiente estado-acción es 0
              target = reward
          else:
              # 3. Seleccionar la siguiente acción A' usando la política basada en w
              new_action = epsilon_greedy_policy(w, epsilon, new_state, num_states, nA)

              # Calcular q(s', a', w)
              q_next = get_Q_value(w, new_state, new_action, num_states, nA)
              target = reward + discount_factor * q_next

          # 4. Actualización Semi-gradiente de los pesos:
          # w <- w + alpha * [Target - q(s, a, w)] * Gradiente
          # En aproximación lineal, Gradiente = x(s, a)
          w += alpha * (target - q_current) * x_current

          # Actualizar estado y acción para la siguiente iteración del bucle
          if not done:
              state = new_state
              action = new_action

      # Guardamos datos sobre la evolución (proporción acumulada de éxito)
      stats += episode_reward
      list_stats.append(stats / (t + 1))
      list_episodes_length.append(step_count)

      # Mostrar progreso
      if t % step_display == 0 and t != 0:
          print(f"success rate: {stats / t:.4f}, epsilon: {epsilon:.4f}")
          
  print(f"success rate: {stats / t:.4f}, epsilon: {epsilon:.4f}")
  # Reconstruimos una matriz Q final ficticia solo para mantener la compatibilidad
  # con los prints del cuaderno original.
  Q_final = np.zeros([num_states, nA])
  for s in range(num_states):
      for a in range(nA):
          Q_final[s, a] = get_Q_value(w, s, a, num_states, nA)

  return w, Q_final, list_stats, list_episodes_length


def plot(list_stats):
  # Creamos una lista de índices para el eje x
  indices = list(range(len(list_stats)))

  # Creamos el gráfico
  plt.figure(figsize=(6, 3))
  plt.plot(indices, list_stats)

  # Añadimos título y etiquetas
  plt.title('Proporción de recompensas')
  plt.xlabel('Episodio')
  plt.ylabel('Proporción')

  # Mostramos el gráfico
  plt.grid(True)
  plt.show()

# Define la función para mostrar el tamaño de los episodios
def plot_episodes_length(list_episodes_length):
  # Creamos una lista de índices para el eje x
  indices = list(range(len(list_episodes_length)))

  # Creamos el gráfico
  plt.figure(figsize=(6, 3))
  plt.plot(indices, list_episodes_length)

  # Añadimos título y etiquetas
  plt.title('Tamaño de los Episodios')
  plt.xlabel('Episodio')
  plt.ylabel('Tamaño')

  # Mostramos el gráfico
  plt.grid(True)
  plt.show()


def plot_policy_episodes(Q, episodes=1):
    env = gym.make("FrozenLake-v1", is_slippery=False, map_name="8x8", render_mode="rgb_array")
    frames = []
    for ep in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            frames.append(env.render())

    env.close()
    fig, ax = plt.subplots()

    def _update(i):
        ax.clear()
        ax.imshow(frames[i])
        ax.axis("off")

    ani = animation.FuncAnimation(fig, _update, frames=len(frames))
    plt.close(fig)

    return HTML(ani.to_jshtml())