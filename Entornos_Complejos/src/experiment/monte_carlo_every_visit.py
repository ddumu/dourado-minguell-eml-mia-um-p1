import math
import numpy as np
from tqdm import tqdm

# Original source: https://github.com/ldaniel-hm/eml_tabular/blob/main/MonteCarloTodasLasVisitas.ipynb
def _random_epsilon_greedy_policy(Q, epsilon, state, nA):
    pi_A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    pi_A[best_action] += (1.0 - epsilon)
    return pi_A

# Original source: https://github.com/ldaniel-hm/eml_tabular/blob/main/MonteCarloTodasLasVisitas.ipynb
def monte_carlo_every_visit(
    env,
    on_policy,
    num_episodes,
    discount_factor,
    epsilon_max,
    epsilon_min=None,
    epsilon_decay_slowdown=1,
    seed=42
):
  # Cantidad de acciones y estados
  nA = env.action_space.n
  nS = env.observation_space.n

  # Matriz de valor estado-acción
  Q = np.zeros([nS, nA])
  # Para On-Policy: Contador de visitas del par estado-acción
  # Para Off-Policy: Acumulación de pesos de importancia
  N = np.zeros([nS, nA])

  # Listas para guardar estadísticas
  episodes_rewards = []
  episodes_lengths = []
  episodes_epsilons = []

  # Mostrar progreso cada 10% del entrenamiento
  log_every = max(1, num_episodes // 10)

  # Si se ha establecido decaimiento
  if epsilon_min:
    # El decay rate se calcula de forma que epsilon 0 (aproximado) se alcance al final del entrenamiento
    decay_rate = -math.log(1e-9 / epsilon_max) / (num_episodes * epsilon_decay_slowdown)

  # Inicializar el epsilon con su valor máximo
  epsilon = epsilon_max

  # Por cada episodio
  for t in tqdm(range(num_episodes)):
    # Resetear el entorno
    state, info = env.reset(seed=seed+t)

    # Variables para saber cuando acabar el episodio
    done = False

    # Lista para almacenar estados, acciones y retornos del episodio
    episode = []

    # Si se ha establecido decaimiento
    if epsilon_min:
      # Epsilon decaimiento exponencial del máximo al mínimo
      epsilon = max(epsilon_min, epsilon_max * np.exp(-decay_rate * t))

    # Guardar el epsilon del episodio para ver la evolución
    episodes_epsilons.append(epsilon)

    # Mientras dure el episodio
    while not done:
      # Probabilidades de cada acción según la política de comportamiento
      behaviour_probs = _random_epsilon_greedy_policy(Q, epsilon, state, nA)
      # Acción elegida por la política de comportamiento
      behaviour_action = np.random.choice(np.arange(nA), p=behaviour_probs)
      # Probabilidad que tenía la acción elegida por la política de comportamiento
      behaviour_action_prob = behaviour_probs[behaviour_action]

      # Hacer la acción en el entorno
      new_state, reward, terminated, truncated, info = env.step(behaviour_action)
      # Guardar el episodio
      episode.append((state, behaviour_action, behaviour_action_prob, reward))

      # Cambiar de estado
      state = new_state
      # Marcar el fin del episodio
      done = terminated or truncated

    # Guardar la longitud del episodio para ver la evolución
    episodes_lengths.append(len(episode))

    # Inicializar la ganancia del episodio
    G = 0.0

    # Si es Monte Carlo On-Policy
    if on_policy:
      # Por cada episodio empezando por el final
      for state, action, _, reward in reversed(episode):
        # Calcular ganancia en este paso
        G = reward + discount_factor * G

        # Actualizar el contador del par estado-acción
        N[state, action] += 1

        # Actualizar la estimación para el par estado-acción
        Q[state, action] += (1 / N[state, action]) * (G - Q[state, action])

    # Si es Monte Carlo Off-Policy
    else:
      # Inicializar peso de importancia
      W = 1.0

      # Por cada episodio empezando por el final
      for state, behaviour_action, behaviour_action_prob, reward in reversed(episode):
        # Calcular ganancia en este paso
        G = reward + discount_factor * G

        # Acumular la suma de pesos de importancia para este par estado-acción
        N[state, behaviour_action] += W

        # Actualizar la matriz para el par estado-acción
        Q[state, behaviour_action] += (W / N[state, behaviour_action]) * (G - Q[state, behaviour_action])

        # Acción elegida por la política objetivo
        target_action = np.argmax(Q[state])

        # Si la acción de la política de comportamiento es diferente que la
        # acción de la política objetivo se para la propagación del retorno
        if behaviour_action != target_action:
          break

        # Actualizar la estimación para el par estado-acción
        W *= 1.0 / behaviour_action_prob

    # Guardamos datos sobre la evolución. Promedio de recompensas
    episode_reward = sum(r for _, _, _, r in episode)
    episodes_rewards.append(episode_reward)

    # Para mostrar la evolución.  Comentar si no se quiere mostrar
    if (t % log_every == 0 and t != 0) or t == num_episodes -1:
      print(f"progress {(t/num_episodes)*100}% | mean reward: {np.mean(episodes_rewards)}, epsilon: {epsilon}")

  # Devolver la politica objetivo junto a estadísticas de los episodios
  return Q, episodes_rewards, episodes_lengths, episodes_epsilons