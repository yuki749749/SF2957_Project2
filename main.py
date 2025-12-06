import os
import sys
import matplotlib.pyplot as plt

import RL as rl
import plotting as pl
import blackjack_extended as bjk
import blackjack_base as bjk_base


def learn_rate_comparison():
    # Learning rate decay comparison
    directory = "{}/data/omega_comparison".format(sys.path[0])
    if not os.path.exists(directory):
        os.makedirs(directory)

    seed = 31233
    n_sims = 10 ** 5  # Number of episodes generated
    epsilon = 0.05  # Probability in epsilon-soft strategy
    init_val = 0.0
    decks = 2
    warmup = n_sims // 10
    # init envs.
    env = bjk.BlackjackEnvExtend(decks=decks, seed=seed)
    sum_env = bjk_base.BlackjackEnvBase(decks=decks, seed=seed)

    omega_vec = [o/10 for o in range(10)]
    # Directory to save plots in
    plot_dir = "{}/figures/omega_comparison".format(sys.path[0])
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for omega in omega_vec:
        print(f"----- omega equal to {omega} -----")
        # Q-learning with expanded state representation
        Q, avg_reward, state_action_count = rl.learn_Q(
            env, n_sims, gamma=1, omega=omega, epsilon=epsilon,
            init_val=init_val, warmup=warmup,
            episode_file=f'{directory}/hand_state_{omega}.txt'
        )
        print("Number of explored states: " + str(len(Q)))
        print("Cumulative avg. reward = " + str(avg_reward))

        # Q-learning with player sum state representation
        sumQ, sum_avg_reward, sum_state_action_count = rl.learn_Q(
            sum_env, n_sims, omega=omega, epsilon=epsilon,
            init_val=init_val, warmup=warmup,
            episode_file=f'{directory}/sum_state_{omega}.txt'
        )
        print("Number of explored states (sum states): " + str(len(sumQ)))
        print("Cumulative avg. reward = " + str(sum_avg_reward))

    # create line plots
    env_types = ["hand"]
    fig, lgd = pl.plot_avg_reward_episode_omega(directory, env_types,
                                                omega_vec)
    fig.savefig(f"{plot_dir}/avgReturnEp_hand_omega.png",
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)

    # create line plots
    env_types = ["sum"]
    fig, lgd = pl.plot_avg_reward_episode_omega(directory, env_types,
                                                omega_vec)
    fig.savefig(f"{plot_dir}/avgReturnEp_sum_omega.png",
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    learn_rate_comparison()
