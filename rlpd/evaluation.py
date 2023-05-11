from typing import Dict

import gym
import numpy as np

from rlpd.wrappers.wandb_video import WANDBVideo


import sys
sys.path.append("/home/dxyang/code/rewardlearning-vid")
from drqv2.video import TrainVideoRecorder

def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, curr_step: int = None, video_recorder: TrainVideoRecorder = None
) -> Dict[str, float]:
    if save_video:
        if video_recorder is None:
            env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    rewards = []
    og_rewards = []
    success = []
    succeeded = []

    for i in range(num_episodes):
        per_episode_rewards = 0
        per_episode_og_rewards = 0
        per_episode_success = False
        per_episode_succeeded = False

        observation, done = env.reset(), False
        if video_recorder is not None and i == 0:
            video_recorder.init(observation["pixels"].squeeze())
        while not done:
            action = agent.eval_actions(observation)
            observation, reward, done, extra = env.step(action)

            per_episode_rewards += reward
            per_episode_og_rewards += extra["og_reward"]
            per_episode_success = extra["success"]
            per_episode_succeeded |= extra["success"]

            if video_recorder is not None:
                video_recorder.record(observation["pixels"].squeeze())

        rewards.append(per_episode_rewards)
        og_rewards.append(per_episode_og_rewards)
        success.append(per_episode_success)
        succeeded.append(per_episode_succeeded)

    if video_recorder is not None:
        video_recorder.save(f'{curr_step}.mp4')

    return {
        "return": np.mean(env.return_queue),
        "length": np.mean(env.length_queue),
        "rewards": np.mean(rewards),
        "og_rewards": np.mean(og_rewards),
        "success": np.mean(success),
        "succeeded": np.mean(succeeded),
    }
