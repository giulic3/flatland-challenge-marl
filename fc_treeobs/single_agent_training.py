# Import packages for plotting and system
import getopt
import random
import sys
from collections import deque
from datetime import date

# make sure the root path is in system path
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from importlib_resources import path

import matplotlib.pyplot as plt
import numpy as np
import torch

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params

# from flatland.utils.rendertools import RenderTool

import fc_treeobs.nets
from fc_treeobs.dueling_double_dqn import Agent
from fc_treeobs.utils import norm_obs_clip, split_tree_into_feature_groups


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "n:", ["n_episodes="])
    except getopt.GetoptError:
        print('single_agent_training.py -n <n_episodes>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--n_episodes'):
            n_episodes = int(arg)

    random.seed(1)
    np.random.seed(1)

    # Preload an agent
    training = False

    # Initialize a random map
    x_dim = np.random.randint(20, 30)
    y_dim = np.random.randint(20, 30)
    n_agents = 1
    tree_depth = 4

    stochastic_data = {'malfunction_rate': 0,  # Rate of malfunction occurence
                       'min_duration': 0,  # Minimal duration of malfunction
                       'max_duration': 0  # Max duration of malfunction
                       }

    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    # Get an observation builder and predictor
    observation_helper = TreeObsForRailEnv(max_depth=tree_depth, predictor=ShortestPathPredictorForRailEnv())

    env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=sparse_rail_generator(max_num_cities=3,
                                                       seed=1,  # Random seed
                                                       grid_mode=False,
                                                       max_rails_between_cities=2,
                                                       max_rails_in_city=3),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=n_agents,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  obs_builder_object=observation_helper)
    env.reset(True, True)
    # env_renderer = RenderTool(env, gl="PILSVG", )

    handle = env.get_agent_handles()
    features_per_node = env.obs_builder.observation_dim
    nr_nodes = 0
    for i in range(tree_depth + 1):
        nr_nodes += np.power(4, i)
    state_size = features_per_node * nr_nodes
    action_size = 5

    # We set the number of episodes we would like to train on
    if 'n_episodes' not in locals():
        n_episodes = 6000

    # max_steps = int(3 * (env.height + env.width))
    max_steps = env.compute_max_episode_steps(width=env.width, height=env.height)
    eps = 1.
    eps_end = 0.005
    eps_decay = 0.998
    action_dict = dict()
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
    dones_list = []
    action_prob = [0] * action_size
    agent_obs = [None] * env.get_num_agents()
    agent_next_obs = [None] * env.get_num_agents()
    # Initialize the agent
    agent = Agent(state_size, action_size)

    if training:
        with path(fc_treeobs.nets, "avoid_checkpoint6000_round1generators.pth") as file_in:
            agent.qnetwork_local.load_state_dict(torch.load(file_in))

    # Do training over n_episodes
    for episodes in range(1, n_episodes + 1):
        """
        Training Curriculum: In order to get good generalization we change the number of agents
        and the size of the levels every 50 episodes.
        """
        if episodes % 50 == 0:
            x_dim = np.random.randint(20, 30)
            y_dim = np.random.randint(20, 30)

            env = RailEnv(width=x_dim,
                          height=y_dim,
                          rail_generator=sparse_rail_generator(max_num_cities=3,
                                                               # Number of cities in map (where train stations are)
                                                               seed=1,  # Random seed
                                                               grid_mode=False,
                                                               max_rails_between_cities=2,
                                                               max_rails_in_city=3),
                          schedule_generator=sparse_schedule_generator(speed_ration_map),
                          number_of_agents=n_agents,
                          malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                          obs_builder_object=observation_helper)

        # Reset environment
        obs, info = env.reset(True, True)
        # env_renderer.reset()

        # Build agent specific observations
        for a in range(env.get_num_agents()):
            data, distance, agent_data = split_tree_into_feature_groups(obs[a], tree_depth)
            data = norm_obs_clip(data)
            distance = norm_obs_clip(distance)
            agent_data = np.clip(agent_data, -1, 1)
            agent_obs[a] = obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))

        score = 0
        env_done = 0
        # Run episode
        for step in range(max_steps):

            # Action
            for a in range(env.get_num_agents()):
                if info['action_required'][a]:
                    action = agent.act(agent_obs[a], eps=eps)
                    action_prob[action] += 1
                    action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, info = env.step(action_dict)
            # env_renderer.render_env(show=True, show_predictions=True, show_observations=False)
            # Preprocess obs
            for a in range(env.get_num_agents()):
                if next_obs[a]: # Means I'm not done
                    data, distance, agent_data = split_tree_into_feature_groups(next_obs[a], tree_depth)
                    data = norm_obs_clip(data)
                    distance = norm_obs_clip(distance)
                    agent_data = np.clip(agent_data, -1, 1)
                    agent_next_obs[a] = next_obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))

            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):
                if not done[a]:
                    agent.step(agent_obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a])
                score += all_rewards[a] / env.get_num_agents()

            agent_obs = agent_next_obs.copy()
            if done['__all__']:
                env_done = 1
                break
        # Epsilon decay
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        done_window.append(env_done)
        scores_window.append(score / max_steps)  # save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))

        print(
            '\rTraining {} Agent on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                env.get_num_agents(), x_dim, y_dim,
                episodes,
                np.mean(scores_window),
                100 * np.mean(done_window),
                eps, action_prob / np.sum(action_prob)), end=" ")

        if episodes % 1000 == 0:
            print(
                '\rTraining {} Agent.\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                    env.get_num_agents(),
                    episodes,
                    np.mean(scores_window),
                    100 * np.mean(done_window),
                    eps,
                    action_prob / np.sum(action_prob)))
            torch.save(agent.qnetwork_local.state_dict(),
                       './nets/single_agent_navigation' + str(date.today()) + "_" + str(episodes) + '.pth')
            action_prob = [1] * action_size
    plt.plot(scores)
    plt.savefig('single_agent_navigation_scores_train'+str(date.today())+'.png') # First save() and then show() to make it work
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
