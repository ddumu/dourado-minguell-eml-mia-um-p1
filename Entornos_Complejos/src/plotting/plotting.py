import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt

def _moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode="valid")

def plot_episodes_lengths(series, ma_window):
    plt.figure()

    for data, label in series:
        ma = _moving_average(data, ma_window)
        plt.plot(ma, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.legend()
    plt.show()

def plot_episodes_rewards(series, ma_window):
    plt.figure()

    for data, label in series:
        ma = _moving_average(data, ma_window)
        plt.plot(ma, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

def plot_episodes_epsilons(series):
    plt.figure()

    for data, label in series:
        plt.plot(range(len(data)), data, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.show()
    
def plot_policy_episodes(env, Q, n_run, n_show, seed):
    frames = []
    rewards = []

    for ep in range(n_run):
        state, info = env.reset(seed=(seed + 1) + ep)
        done = False

        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rewards.append(reward)

            if ep < n_show:
                frames.append(env.render()) 

    env.close()

    fig, ax = plt.subplots()

    def _update(i):
        ax.clear()
        ax.imshow(frames[i])
        ax.axis("off")

    ani = animation.FuncAnimation(fig, _update, frames=len(frames))
    plt.close(fig)

    print("Mean reward:", np.mean(rewards))

    return HTML(ani.to_jshtml())