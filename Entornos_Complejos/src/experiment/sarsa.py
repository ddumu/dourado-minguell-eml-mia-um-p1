import math
import numpy as np
from tqdm import tqdm

def _random_epsilon_greedy_policy(Q, epsilon, state, nA):
    pi_A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    pi_A[best_action] += (1.0 - epsilon)
    return pi_A

def sarsa(
    env,
    num_episodes,
    discount_factor,
    alpha,
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

    # Listas para guardar estadísticas
    episodes_rewards = []
    episodes_lengths = []
    episodes_epsilons = []

    # Mostrar progreso cada 10% del entrenamiento
    log_every = max(1, num_episodes // 10)

    # Si se ha establecido decaimiento
    if epsilon_min:
        decay_rate = -math.log(1e-9 / epsilon_max) / (num_episodes * epsilon_decay_slowdown)

    # Inicializar el epsilon con su valor máximo
    epsilon = epsilon_max

    # Por cada episodio
    for t in tqdm(range(num_episodes)):
        # Resetear el entorno
        state, info = env.reset(seed=seed + t)

        # Variables para saber cuando acabar el episodio
        done = False
        episode_reward = 0
        episode_length = 0

        # Si se ha establecido decaimiento
        if epsilon_min:
            epsilon = max(epsilon_min, epsilon_max * np.exp(-decay_rate * t))

        # Guardar el epsilon del episodio para ver la evolución
        episodes_epsilons.append(epsilon)

        # SARSA necesita elegir la primera acción antes del bucle
        probs = _random_epsilon_greedy_policy(Q, epsilon, state, nA)
        action = np.random.choice(np.arange(nA), p=probs)

        # Mientras dure el episodio
        while not done:
            # Hacer la acción en el entorno
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Actualizar estadísticas
            episode_reward += reward
            episode_length += 1

            if not done:
                # Elegir la siguiente acción con la misma política
                next_probs = _random_epsilon_greedy_policy(Q, epsilon, new_state, nA)
                next_action = np.random.choice(np.arange(nA), p=next_probs)
                
                # Añadir la recompesa esperada
                reward += discount_factor * Q[new_state, next_action]
            else:
                # No se usará
                next_action = 0

            # Actualizar las estimaciones para el par estado-acción
            Q[state, action] += alpha * (reward- Q[state, action])

            # Avanzar al siguiente par estado-acción
            state = new_state
            action = next_action

        # Guardar datos sobre la evolución
        episodes_rewards.append(episode_reward)
        episodes_lengths.append(episode_length)

        # Para mostrar la evolución
        if (t % log_every == 0 and t != 0) or t == num_episodes - 1:
            print(f"progress {(t/num_episodes)*100}% | mean reward: {np.mean(episodes_rewards)}, epsilon: {epsilon}")

    return Q, episodes_rewards, episodes_lengths, episodes_epsilons