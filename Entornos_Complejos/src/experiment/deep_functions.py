import random
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from IPython.display import HTML

SEED = 2024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env(env_name):
    """Crea una instancia reproducible del entorno según la sección 3.2."""
    env = gym.make(env_name)
    env.reset(seed=SEED)
    return env

class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, state):
        return self.net(state)
    
class ReplayBuffer:
    """Buffer optimizado para transferencias veloces a PyTorch."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action_idx, reward, next_state, terminated):
        self.buffer.append((state, action_idx, reward, next_state, terminated))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminateds = zip(*batch)
        
        # Conversión directa y en bloque a tensores en el dispositivo configurado
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE),
            torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=DEVICE),
            torch.tensor(terminateds, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        )

def select_action(q_network, state, action_grid, epsilon=0.0):
    """Selecciona acción usando Epsilon-Greedy con ruido gaussiano en exploración."""
    if np.random.random() < epsilon:
        action_idx = np.random.randint(len(action_grid))
        base_action = action_grid[action_idx]
        action = np.clip(base_action, action_grid.min(), action_grid.max())
        closest_idx = np.argmin(np.abs(action_grid - action))
        return action_grid[closest_idx], closest_idx
    else:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q_values = q_network(state_t)
            action_idx = torch.argmax(q_values).item()
            
    return action_grid[action_idx], action_idx



def train_dqn(env_name, num_episodes=1000, epsilon_init=1.0, epsilon_min=0.1, 
              decay=True, discount_factor=0.99, batch_size=64, 
              replay_size=100000, target_update=10, learning_rate=1e-3, action_bins=21,
              train_every=4):
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    # print(f"Dimensión del espacio de estados: {env.observation_space}")
    step_display = max(1, num_episodes // 10)

    # Grid de acciones discretizadas
    action_grid = np.linspace(env.action_space.low[0], env.action_space.high[0], action_bins, dtype=np.float32).reshape(-1, 1)

    # Inicialización de Redes Q
    q_network = DQN(state_dim, len(action_grid)).to(DEVICE)
    target_network = DQN(state_dim, len(action_grid)).to(DEVICE)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate, )
    loss_fn = nn.SmoothL1Loss()  # Huber Loss para estabilidad
    replay_buffer = ReplayBuffer(replay_size)

    list_episodes_reward = []
    list_episodes_length = []
    epsilon = epsilon_init

    global_step = 0
    
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset(seed=SEED)
        done = False
        episode_reward = 0.0
        episode_length = 0

        if decay:
            epsilon = max(0.05, epsilon * 0.995)

        while not done:
            action, action_idx = select_action(q_network, state, action_grid, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Recompensas personalizadas para MountainCarContinuous-v0
            custom_reward = reward 
            custom_reward += abs(next_state[1]) * 6.0  # Premia velocidad para favorecer inercia necesaria

            replay_buffer.push(state, action_idx, custom_reward, next_state, terminated)
            state = next_state
            episode_reward += reward
            episode_length += 1
            global_step += 1

            # Saltar entrenamiento si el buffer no está lo suficientemente lleno
            if len(replay_buffer) < batch_size:
                continue

            if global_step % train_every != 0:
                continue

            # Muestreo optimizado
            states_t, actions_t, rewards_t, next_states_t, terminateds_t = replay_buffer.sample(batch_size)

            # Optimización de la Red Q
            q_values = q_network(states_t).gather(1, actions_t)
            with torch.no_grad():
                next_q_values = target_network(next_states_t).max(dim=1, keepdim=True)[0]
                target_q_values = rewards_t + discount_factor * next_q_values * (1.0 - terminateds_t)

            loss = loss_fn(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
            optimizer.step()


        # Sincronización periódica de la red target
        if (episode + 1) % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())

        list_episodes_reward.append(episode_reward)
        list_episodes_length.append(episode_length)
        if episode % step_display == 0 and episode != 0:
            print(f"Episodio {episode} -> Reward medio acumulado: {np.mean(list_episodes_reward[-step_display:]):.4f}, Epsilon: {epsilon:.4f}")

    env.close()
    return q_network, action_grid, list_episodes_reward, list_episodes_length


def pi_star_from_Q(env, q_network, action_grid):
    state, info = env.reset()
    done = False
    actions = []
    total_reward = 0

    while not done:
        # Usamos select_action con epsilon=0.0 para que sea 100% Greedy
        action, _ = select_action(q_network, state, action_grid, epsilon=0.0)
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


def plot_policy_episodes(q_network, action_grid, episodes=1):
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    frames = []
    for ep in range(episodes):
        state, info = env.reset()
        done = False
        while not done:
            action, _ = select_action(q_network, state, action_grid, epsilon=0.0)
            state, _, terminated, truncated, _ = env.step(action)
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
