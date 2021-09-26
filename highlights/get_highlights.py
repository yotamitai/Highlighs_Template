import argparse
import json
from datetime import datetime
import random
from pathlib import Path

import gym
import pandas as pd
from os.path import join, basename, abspath

from get_agent import get_agent
from get_traces import get_traces
from utils import create_video, make_clean_dirs, pickle_save, pickle_load
from highlights_state_selection import compute_states_importance, highlights, highlights_div
from get_trajectories import states_to_trajectories, trajectories_by_importance, \
    get_trajectory_images
from ffmpeg import merge_and_fade


def get_highlights(args):

    if args.load_dir:
        """Load traces and state dictionary"""
        traces = pickle_load(join(args.load_dir, 'Traces.pkl'))
        states = pickle_load(join(args.load_dir, 'States.pkl'))
        if args.verbose: print(f"Highlights {15 * '-' + '>'} Traces & States Loaded")
    else:
        env, agent = get_agent(args.load_path)
        env.args = args
        traces, states = get_traces(env, agent, args)
        env.close()
        del gym.envs.registration.registry.env_specs[env.spec.id]

    """highlights algorithm"""
    data = {
        'state': list(states.keys()),
        'q_values': [x.observed_actions for x in states.values()]
    }
    q_values_df = pd.DataFrame(data)

    """importance by state"""
    q_values_df = compute_states_importance(q_values_df, compare_to=args.state_importance)
    highlights_df = q_values_df
    state_importance_dict = dict(zip(highlights_df["state"], highlights_df["importance"]))

    """get highlights"""
    if args.trajectory_importance == "single_state":
        """highlights importance by single state importance"""
        summary_states = highlights(highlights_df, traces, args.num_trajectories,
                                    args.trajectory_length, args.minimum_gap, args.overlay_limit)
        # summary_states = highlights_div(highlights_df, traces, args.num_trajectories,
        #                             args.trajectory_length,
        #                             args.minimum_gap)
        all_trajectories = states_to_trajectories(summary_states, state_importance_dict)
        summary_trajectories = all_trajectories

    else:
        """highlights importance by trajectory"""
        all_trajectories, summary_trajectories = \
            trajectories_by_importance(traces, state_importance_dict, args)

    # random highlights
    # summary_trajectories = random.choices(all_trajectories, k=5)

    # random order
    if args.randomized: random.shuffle(summary_trajectories)

    return






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HIGHLIGHTS')
    parser.add_argument('-a', '--name', help='agent name', type=str, default="Agent-0")
    parser.add_argument('-num_ep', '--num_episodes', help='number of episodes to run', type=int,
                        default=1)
    parser.add_argument('-fps', '--fps', help='summary video fps', type=int, default=1)
    parser.add_argument('-n', '--n_traces', help='number of traces to obtain', type=int,
                        default=10)
    parser.add_argument('-k', '--num_trajectories',
                        help='number of highlights trajectories to obtain', type=int, default=5)
    parser.add_argument('-l', '--trajectory_length',
                        help='length of highlights trajectories ', type=int, default=10)
    parser.add_argument('-v', '--verbose', help='print information to the console',
                        action='store_true')
    parser.add_argument('-overlapLim', '--overlay_limit', help='# overlaping', type=int,
                        default=3)
    parser.add_argument('-minGap', '--minimum_gap', help='minimum gap between trajectories',
                        type=int, default=0)
    parser.add_argument('-rand', '--randomized', help='randomize order of summary trajectories',
                        type=bool, default=True)
    parser.add_argument('-impMeth', '--importance_type',
                        help='importance by state or trajectory', default='single_state')
    parser.add_argument('-impState', '--state_importance',
                        help='method calculating state importance', default='second')
    parser.add_argument('-loadTrace', '--load_last_traces',
                        help='load previously generated traces', type=bool, default=False)
    parser.add_argument('-loadTraj', '--load_last_trajectories',
                        help='load previously generated trajectories', type=bool, default=False)
    args = parser.parse_args()

    # RUN
    get_highlights(args)
