import os
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# --- RBF Featurization ---
num_centros_pos = 10
num_centros_vel = 10
NUM_FEATURES = num_centros_pos * num_centros_vel

pos_centros = np.linspace(-1.2, 0.6, num_centros_pos)
vel_centros = np.linspace(-0.07, 0.07, num_centros_vel)
CENTROS = np.array([[p, v] for p in pos_centros for v in vel_centros])

# Normalizamos el espacio de estados para aplicar sigma único correctamente
# Rango posición: 1.8, rango velocidad: 0.14
POS_RANGE = 1.8
VEL_RANGE = 0.14
SCALE = np.array([POS_RANGE, VEL_RANGE])  # (2,)

# Estandarizamos centros y usaremos estados estandarizados
CENTROS_SCALED = CENTROS / SCALE  # (100, 2)

SIGMA = 0.3

ACTION_SAMPLES = np.linspace(-1.0, 1.0, num=21)
NUM_ACTIONS = len(ACTION_SAMPLES)


def get_state_features(state):
    """RBF features con sigma balanceado tras normalización por rango."""
    state_scaled = state / SCALE  # shape (2,)
    dists = np.sum((CENTROS_SCALED - state_scaled) ** 2, axis=1)
    phi = np.exp(-dists / (2 * SIGMA ** 2))
    return phi


def greedy_action(w, state):
    phi = get_state_features(state)
    q_valores = np.dot(w, phi)  # w: (21, 100), phi: (100,) -> (21,)
    best_idx = np.argmax(q_valores)
    return best_idx, np.array([ACTION_SAMPLES[best_idx]], dtype=np.float32)


def epsilon_greedy_policy(env, w, epsilon, state):
    if np.random.random() < epsilon:
        random_idx = np.random.choice(NUM_ACTIONS)
        return random_idx, np.array([ACTION_SAMPLES[random_idx]], dtype=np.float32)
    return greedy_action(w, state)


def semi_gradient_sarsa(env, num_episodes=5000, epsilon=0.4, decay=False, discount_factor=0.99, alpha=0.01):
    # Inicialización aleatoria pequeña para romper simetría
    rng = np.random.default_rng(42)
    w = rng.uniform(-0.001, 0.001, (NUM_ACTIONS, NUM_FEATURES))

    stats = 0.0
    list_stats = [stats]
    list_episodes_length = []
    step_display = max(1, num_episodes // 10)

    for t in tqdm(range(num_episodes)):
        state, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        if decay:
            epsilon = max(0.05, epsilon * np.exp(-t / 1500))

        action_idx, action = epsilon_greedy_policy(env, w, epsilon, state)

        while not done:
            step_count += 1

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            phi = get_state_features(state)

            # Q(s,a) para la acción tomada
            q_current = np.dot(w[action_idx], phi)

            if done:
                target = reward
                w[action_idx] += alpha * (target - q_current) * phi
            else:
                new_action_idx, new_action = epsilon_greedy_policy(env, w, epsilon, new_state)
                phi_next = get_state_features(new_state)

                q_next = np.dot(w[new_action_idx], phi_next)
                target = reward + discount_factor * q_next

                w[action_idx] += alpha * (target - q_current) * phi

                state = new_state
                action_idx = new_action_idx
                action = new_action

        stats += episode_reward
        list_stats.append(stats / (t + 1))
        list_episodes_length.append(step_count)

        if t % step_display == 0 and t != 0:
            print(f"Episodio {t} -> Reward medio acumulado: {stats / t:.4f}, Epsilon: {epsilon:.4f}")

    print(f"Proporción final de Reward: {stats / max(1, t):.4f}")
    return w, w, list_stats, list_episodes_length


def pi_star_from_Q(env, w):
    state, info = env.reset()
    done = False
    actions = []
    total_reward = 0

    while not done:
        _, action = greedy_action(w, state)
        actions.append(f"{action[0]:.3f}")
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    actions_str = ", ".join(actions[:15]) + "..." if len(actions) > 15 else ", ".join(actions)
    return f"Política completada con Recompensa Total: {total_reward}", actions_str


def plot(list_stats):
    plt.figure(figsize=(6, 3))
    plt.plot(list_stats)
    plt.title('Proporción Acumulada de Recompensas')
    plt.xlabel('Episodio')
    plt.ylabel('Media de Recompensa')
    plt.grid(True)
    plt.show()


def plot_episodes_length(list_episodes_length):
    plt.figure(figsize=(6, 3))
    plt.plot(list_episodes_length)
    plt.title('Duración de los Episodios')
    plt.xlabel('Episodio')
    plt.ylabel('Pasos')
    plt.grid(True)
    plt.show()


def plot_policy_episodes(w, episodes=1):
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    frames = []
    for ep in range(episodes):
        state, info = env.reset()
        done = False
        while not done:
            _, action = greedy_action(w, state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            frames.append(env.render())
    env.close()

    fig, ax = plt.subplots()
    def _update(i):
        ax.clear()
        ax.imshow(frames[i])
        ax.axis("off")

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), interval=30)
    plt.close(fig)
    return HTML(ani.to_jshtml())