import json
from datetime import datetime
from os import makedirs
from os.path import join, abspath
from pathlib import Path

from highlights.ffmpeg import merge_and_fade
from highlights.get_highlights import get_highlights
from highlights.get_trajectories import get_trajectory_images
from highlights.utils import pickle_save, create_video


def save_executions(traces, states, all_trajectories, args):
    """Save data used for this run"""
    pickle_save(traces, join(args.output_dir, 'Traces.pkl'))
    pickle_save(states, join(args.output_dir, 'States.pkl'))
    pickle_save(all_trajectories, join(args.output_dir, 'Trajectories.pkl'))
    if args.verbose: print(f"Highlights {15 * '-' + '>'} Run Configurations Saved")


def save_videos(states, summary_trajectories, args):
    """Save Highlight videos"""
    frames_dir = join(args.output_dir, 'Highlight_Frames')
    videos_dir = join(args.output_dir, "Highlight_Videos")
    height, width, layers = list(states.values())[0].image.shape
    img_size = (width, height)
    get_trajectory_images(summary_trajectories, states, frames_dir)
    create_video(frames_dir, videos_dir, args.num_trajectories, img_size, args.fps)
    if args.verbose: print(f"HIGHLIGHTS {15 * '-' + '>'} Videos Generated")

    """Merge Highlights to a single video with fade in/ fade out effects"""
    fade_out_frame = args.trajectory_length - args.fade_duration
    merge_and_fade(videos_dir, args.num_trajectories, fade_out_frame, args.fade_duration,
                   args.name)


def output_and_metadata(args):
    args.output_dir = join(abspath('results'), '_'.join(
        [datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '_'), args.name]))
    makedirs(args.output_dir)
    with Path(join(args.output_dir, 'metadata.json')).open('w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


def main(args):
    output_and_metadata(args)
    traces, states, all_trajectories, summary_trajectories = get_highlights(args)
    save_executions(traces, states, all_trajectories, args)
    save_videos(states, summary_trajectories, args)
