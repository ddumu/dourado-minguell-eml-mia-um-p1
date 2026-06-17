import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation

def _moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode="valid")
    
def plot_episodes_lengths(
    episode_lengths,
    label,
    ma_window,
    episode_lengths_2=None,
    label_2=None
):
    plt.figure()

    episode_lengths_ma = _moving_average(episode_lengths, ma_window)
    plt.plot(episode_lengths_ma, label=label)

    if episode_lengths_2 is not None:
        if label_2 is None:
            raise ValueError("You must provide label_2 when using episode_lengths_2")

        episode_lengths_2_ma = _moving_average(episode_lengths_2, ma_window)
        plt.plot(episode_lengths_2_ma, label=label_2)
        
        plt.legend()

    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.show()

def plot_episodes_rewards(
    episodes_rewards,
    label,
    ma_window,
    episodes_rewards_2=None,
    label_2=None
):
    plt.figure()

    episodes_rewards_ma = _moving_average(episodes_rewards, ma_window)
    plt.plot(episodes_rewards_ma, label=label)

    if episodes_rewards_2 is not None:
        if label_2 is None:
            raise ValueError("You must provide label_2 when using episode_lengths_2")

        episodes_rewards_2_ma = _moving_average(episodes_rewards_2, ma_window)
        plt.plot(episodes_rewards_2_ma, label=label_2)
        
        plt.legend()

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

def plot_episodes_epsilons(
    episodes_epsilons,
    label,
    episodes_epsilons_2=None,
    label_2=None
):
    plt.figure()
    plt.plot(range(len(episodes_epsilons)), episodes_epsilons, label=label)

    if episodes_epsilons_2 is not None:
        if label_2 is None:
            raise ValueError("You must provide label_2 when using episode_lengths_2")

        plt.plot(
            range(len(episodes_epsilons_2)),
            episodes_epsilons_2,
            label=label_2
        )
        plt.legend()

    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.show()
    
def plot_policy_episodes(Q, episodes=1):
    env = gym.make("Taxi-v4", render_mode="rgb_array")

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