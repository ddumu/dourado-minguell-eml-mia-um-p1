import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gymnasium as gym
import matplotlib.animation as animation
from IPython.display import HTML


# Red Neuronal para aproximar Q(s, a). 
class DQN_Network(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQN_Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, num_actions)
        )

    def forward(self, x):
        return self.net(x)

# Política Greedy final a partir de la Red Neuronal Q (Sustituye a la anterior)
def pi_star_from_Q(env, q_network):
    done = False
    pi_star = np.zeros([env.observation_space.n, env.action_space.n])
    state, info = env.reset()
    actions = ""
    
    while not done:
        # Creamos un vector de ceros y ponemos un 1 en el estado actual
        state_onehot = np.zeros(env.observation_space.n)
        state_onehot[state] = 1.0
        # Convertimos a tensor float32
        state_t = torch.tensor(state_onehot, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action = torch.argmax(q_network(state_t)).item()
            
        actions += f"{action}, "
        pi_star[state, action] = action
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    return pi_star, actions

def train_dqn(env, num_episodes=5000, epsilon=0.4, decay=False, discount_factor=0.99):
    n_states = env.observation_space.n
    nA = env.action_space.n

    q_network = DQN_Network(nA, n_states) 
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    step_display = num_episodes / 10
    stats = 0.0
    list_stats = []
    list_episodes_length = []

    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        if decay:
          epsilon = min(epsilon, 100.0 / (episode + 1))
        while not done:
            # Convertir estado entero a One-Hot para la red
            state_onehot = np.zeros(n_states)
            state_onehot[state] = 1
            state_t = torch.tensor(state_onehot, dtype=torch.float32).unsqueeze(0)
            
            # Epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(q_network(state_t)).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            # Preparar next_state
            next_state_onehot = np.zeros(n_states)
            next_state_onehot[next_state] = 1
            next_state_t = torch.tensor(next_state_onehot, dtype=torch.float32).unsqueeze(0)
            
            # Target: R + gamma * max Q(s')
            with torch.no_grad():
                target = reward + (discount_factor * torch.max(q_network(next_state_t)) * (1 - int(done)))
            
            # Entrenamiento
            current_q = q_network(state_t)[0, action]
            loss = loss_fn(current_q, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            
        # Estadísticas
        stats += episode_reward
        list_stats.append(stats / (episode + 1))
        list_episodes_length.append(episode_length)
        
        # Mostrar la evolución en el terminal
        if episode % step_display == 0 and episode != 0:
            print(f"success: {stats/episode}, epsilon: {epsilon}")    

    print(f"success: {stats/episode}, epsilon: {epsilon}")    
    return q_network, list_stats, list_episodes_length



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
# Pon aquí tu código.
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
    n_states = env.observation_space.n
    frames = []

    for ep in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            state_onehot = np.zeros(n_states)
            state_onehot[state] = 1.0
            state_t = torch.tensor(state_onehot, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                q_values = Q(state_t)
                action = torch.argmax(q_values).item() 
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 4. Capturar frame
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