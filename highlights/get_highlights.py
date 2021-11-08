import argparse
import random

import gym
import pandas as pd
from os.path import join

from get_agent import get_agent
from get_traces import get_traces
from utils import pickle_save, pickle_load
from highlights_state_selection import compute_states_importance, highlights, highlights_div
from get_trajectories import states_to_trajectories, trajectories_by_importance


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

    """Save data used for this run in output dir"""
    pickle_save(traces, join(args.output_dir, 'Traces.pkl'))
    pickle_save(states, join(args.output_dir, 'States.pkl'))

    """highlights algorithm"""
    a, b, c = states[(0, 0)].image.shape
    data = {'state': list(states.keys()),
            'q_values': [x.observed_actions for x in states.values()],
            'features': [x.image.reshape(a * b * c) for x in states.values()]}

    if args.highlights_div:
        i = len(traces[0].states) // 2
        threshold = args.div_coefficient * (
            sum(states[(0, i)].image.reshape(a * b * c) - states[(0, i + 1)].image.reshape(
                a * b * c)))
    q_values_df = pd.DataFrame(data)

    """importance by state"""
    q_values_df = compute_states_importance(q_values_df, compare_to=args.state_importance)
    highlights_df = q_values_df
    state_importance_dict = dict(zip(highlights_df["state"], highlights_df["importance"]))

    """get highlights"""
    if args.trajectory_importance == "single_state":
        """highlights importance by single state importance"""
        trace_lengths = {k: len(v.states) for k, v in enumerate(traces)}
        if args.highlights_div:
            summary_states = highlights_div(highlights_df, trace_lengths, args.num_trajectories,
                                            args.trajectory_length, args.minimum_gap,
                                            threshold=threshold)
        else:
            summary_states = highlights(highlights_df, trace_lengths, args.num_trajectories,
                                        args.trajectory_length, args.minimum_gap)
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
    """Save trajectories used for this run in output dir"""
    pickle_save(all_trajectories, join(args.output_dir, 'Trajectories.pkl'))

    return states, summary_trajectories
