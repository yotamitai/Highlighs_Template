import gym
import numpy as np
import argparse
from os import makedirs
from os.path import join, exists
from gym.wrappers import Monitor

from ARCHIVE.utils import FROGGER_CONFIG_DICT, load_agent_config
from interestingness_xrl.learning import write_table_csv
from interestingness_xrl.learning.behavior_tracker import BehaviorTracker
from interestingness_xrl.scenarios import create_helper, create_agent


def video_schedule(config, videos):
    # linear capture schedule
    return lambda e: videos and \
                     (e == config.num_episodes - 1 or e % int(config.num_episodes / config.num_recorded_videos) == 0)


# def load_agent_config(results_dir, trial=0):
#     results_dir = results_dir if results_dir else get_agent_output_dir(DEFAULT_CONFIG, AgentType.Learning, trial)
#     config_file = os.path.join(results_dir, 'config.json')
#     if not os.path.exists(results_dir) or not os.path.exists(config_file):
#         raise ValueError(f'Could not load configuration from: {config_file}.')
#     configuration = EnvironmentConfiguration.load_json(config_file)
#     # if testing, we want to force a seed different than training (diff. test environments)
#     #     configuration.seed += 1
#     return configuration, results_dir


def run_trial(args):
    config, results_dir = load_agent_config(args.results, args.trial)
    helper = create_helper(config)
    output_dir = args.output
    if not exists(output_dir):
        makedirs(output_dir)

    config.save_json(join(output_dir, 'config.json'))  # saves / copies configs to file
    helper.save_state_features(join(output_dir, 'state_features.csv'))
    env_id = '{}-0-v0'.format(config.gym_env_id)
    helper.register_gym_environment(env_id, False, args.fps, False)
    env = gym.make(env_id)  # create environment and monitor
    config.num_episodes = args.num_episodes
    video_callable = video_schedule(config, args.record)
    env = Monitor(env, directory=output_dir, force=True, video_callable=video_callable)
    if video_callable(0):  # adds reference to monitor to allow for gym environments to update video frames
        env.env.monitor = env
    env.seed(config.seed + args.trial)  # initialize seeds (one for the environment, another for the agent)
    agent_rng = np.random.RandomState(config.seed + args.trial)
    # creates the agent
    agent, exploration_strategy = create_agent(helper, agent_t, agent_rng)
    #loads tables from file (some will be filled by the agent during the interaction)
    agent.load(results_dir)


    # runs episodes
    behavior_tracker = BehaviorTracker(config.num_episodes)
    recorded_episodes = []
    for e in range(config.num_episodes):

        # checks whether to activate video monitoring
        env.env.monitor = env if video_callable(e) else None

        # reset environment
        old_obs = env.reset()
        old_s = helper.get_state_from_observation(old_obs, 0, False)

        if args.verbose:
            print(f'Episode: {e}')
            # helper.update_stats_episode(e)
        exploration_strategy.update(e)  # update for learning agent

        t = 0
        done = False
        while not done:
            # select action
            # sample an action based on the softmax probabilities as determined by the q values of the available actions
            a = agent.act(old_s)

            # observe transition
            obs, r, done, _ = env.step(a)
            s = helper.get_state_from_observation(obs, r, done)
            # s is a unique state defined by the elements around the element. combinations of ELEM_LABELS
            r = helper.get_reward(old_s, a, r, s, done)

            # update agent and stats
            agent.update(old_s, a, r, s)
            behavior_tracker.add_sample(old_s, a)
            helper.update_stats(e, t, old_obs, obs, old_s, a, r, s)

            old_s = s
            old_obs = obs
            t += 1

        # adds to recorded episodes list
        if video_callable(e):
            recorded_episodes.append(e)

        # signals new episode to tracker
        behavior_tracker.new_episode()

    # writes results to files
    agent.save(output_dir)
    behavior_tracker.save(output_dir)
    write_table_csv(recorded_episodes, join(output_dir, 'rec_episodes.csv'))
    helper.save_stats(join(output_dir, 'results'), args.clear_results)
    print('\nResults of trial {} written to:\n\t\'{}\''.format(args.trial, output_dir))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent runner')
    parser.add_argument('-a', '--agent', help='agent type', type=int, default=0)
    parser.add_argument('-n', '--num_episodes', help='number of episodes to run', type=int, default=100)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-r', '--results', help='directory from which to load results')
    parser.add_argument('-c', '--config_file_path', help='path to config file')
    parser.add_argument('-t', '--trial', help='trial number to run', type=int, default=0)
    parser.add_argument('-rv', '--record', help='record videos according to linear schedule', action='store_true')
    parser.add_argument('-v', '--verbose', help='print information to the console', action='store_true')
    args = parser.parse_args()

    """experiment parameters"""
    args.agent = 0
    args.trial = 0
    args.num_episodes = 20 # max 2000 (defined in configuration.py)
    args.fps = 2
    args.verbose = True
    args.record = True
    args.show_score_bar = False
    args.clear_results = True
    args.default_frogger_config = FROGGER_CONFIG_DICT['DEFAULT']


    run_trial(args)
