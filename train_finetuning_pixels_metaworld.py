#! /usr/bin/env python
import os
import pickle
from datetime import datetime
from pathlib import Path

import dmcgym
from flax.training import checkpoints
import gym
import numpy as np
import tqdm
from absl import app, flags
from flax.core import FrozenDict
from ml_collections import config_flags

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import wandb
from rlpd.agents import DrQLearner
from rlpd.data import MemoryEfficientReplayBuffer, ReplayBuffer
from rlpd.data.vd4rl_datasets import VD4RLDataset, VMetaworldDataset
from rlpd.evaluation import evaluate
from rlpd.wrappers import WANDBVideo, wrap_pixels

from policy_learning.envs import (
    ImageMetaworldEnv,
    ImageOnlineCustomRewardMetaworldEnv,
)
from policy_learning.rlpd_utils import DMCWrapper
from reward_extraction.reward_functions import generate_expert_data, LearnedImageRewardFunction


FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd_pixels", "wandb project name.")
flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
# flags.DEFINE_string(
#     "dataset_level", "expert", "Dataset level (e.g., random, expert, etc.)."
# )
# flags.DEFINE_string("dataset_path", None, "Path to dataset. If None, uses '~/.vd4rl'.")
flags.DEFINE_integer("dataset_size", 500_000, "How many samples to load")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5e5), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 1, "Stack frames.")
flags.DEFINE_integer(
    "replay_buffer_size", None, "Number of training steps to start training."
)
# flags.DEFINE_integer(
#     "action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values."
# )
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean(
    "memory_efficient_replay_buffer", True, "Use a memory efficient replay buffer."
)
flags.DEFINE_boolean("save_video", True, "Save videos during evaluation and training.")
flags.DEFINE_string("save_dir", None, "Directory to save checkpoints.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
config_flags.DEFINE_config_file(
    "config",
    "configs/drq_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

# added options by dxy
flags.DEFINE_boolean("use_lrf", False, "Use a learned reward function (by default ranking + classifier)")
flags.DEFINE_integer("lrf_update_frequency", 100_000, "Update lrf every x steps")
flags.DEFINE_integer("lrf_train_steps", 500, "Update lrf x times every every lrf_update_frequency steps")

flags.DEFINE_integer("log_video_frequency", 100, "Log a training video every x episodes.")
flags.DEFINE_boolean("skip_relabel_rewards", False, "Do not relabel rewards.")

# flags.DEFINE_integer("train_gail", False, "GAIL style reward function (just classifier)")
# flags.DEFINE_integer("train_airl", False, "AIRL style reward function (classifier rescaled)")
# flags.DEFINE_integer("train_vice", False, "VICE style reward function (just success state classifier)")
# flags.DEFINE_integer("train_soil", False, "state only imitation learning (inverse dynamics learned from interaction)")


def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, FrozenDict):
            if len(v) == 0:
                combined[k] = v
            else:
                combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return FrozenDict(combined)


def main(_):
    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    date_str = datetime.today().strftime('%Y-%m-%d')
    if FLAGS.use_lrf:
        exp_substr = "lrf"
    else:
        exp_substr = "vanilla"

    env_str = "reach"

    if FLAGS.skip_relabel_rewards:
        exp_str = f"rlpd/metaworld_pixels/{date_str}/{env_str}-{exp_substr}-noRelabel"
    else:
        exp_str = f"rlpd/metaworld_pixels/{date_str}/{env_str}-{exp_substr}-lrfUpdate{FLAGS.lrf_train_steps}Every{FLAGS.lrf_update_frequency}"

    log_dir = os.path.join(FLAGS.log_dir, exp_str)
    os.makedirs(log_dir, exist_ok=True)
    chkpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(chkpt_dir, exist_ok=True)
    buffer_dir = os.path.join(log_dir, "buffers")
    os.makedirs(buffer_dir, exist_ok=True)

    action_repeat = 1
    pixel_keys = ('pixels',)

    '''
    Setup the replay buffer (need to generate expert data to populate the rb)
    '''
    dummy_dmc_env = ImageMetaworldEnv(env_str, camera_name="left_cap2", high_res_env=False, rlpd_res=True)
    dummy_env = DMCWrapper(dummy_dmc_env)
    expert_data_path = os.path.join(log_dir, "expert_data.hdf")
    if not os.path.exists(expert_data_path):
        generate_expert_data(env_str, dummy_dmc_env, expert_data_path, 100, False, False)
    ds = VMetaworldDataset(
        dummy_env,
        capacity=100 * 100, #hardcoded because we know we have 100 episodes of length 100
        dataset_path=expert_data_path,
    )
    ds_iterator = ds.get_iterator(
        sample_args={
            "batch_size": int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio),
            "pack_obs_and_next_obs": True,
        }
    )

    replay_buffer_size = FLAGS.replay_buffer_size or FLAGS.max_steps // action_repeat
    if FLAGS.memory_efficient_replay_buffer:
        replay_buffer = MemoryEfficientReplayBuffer(
            dummy_env.observation_space, dummy_env.action_space, replay_buffer_size
        )
        replay_buffer_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": int(
                    FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio)
                ),
                "pack_obs_and_next_obs": True,
            }
        )
    else:
        replay_buffer = ReplayBuffer(
            dummy_env.observation_space, dummy_env.action_space, replay_buffer_size
        )
        replay_buffer_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": int(
                    FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio)
                ),
            }
        )
    replay_buffer.seed(FLAGS.seed)


    '''
    Setup the LRF
    '''
    if FLAGS.use_lrf:
        rb_for_lrf = replay_buffer.get_iterator(
            sample_args={
                "batch_size": int(96 // 2),
            }
        )
        lrf = LearnedImageRewardFunction(
            obs_size=dummy_dmc_env.observation_spec().shape,
            exp_dir=log_dir,
            replay_buffer=rb_for_lrf,
            train_classify_with_mixup=True,
            add_state_noise=True,
            rb_buffer_obs_key="pixels",
            disable_ranking=False,
            train_classifier_with_goal_state_only=False,
            for_rlpd=True
        )
        assert lrf.batch_size == 96 # hardcoded number above

        # relabel the offline dataset rewards with the lrf init'd
        preprocess_batch_size = 100
        all_states = ds.dataset_dict["observations"]["pixels"]
        num_batches = int(len(all_states) / preprocess_batch_size)

        for batch_num in tqdm.tqdm(range(num_batches)):
            start_idx, end_idx = batch_num * preprocess_batch_size, (batch_num + 1) * preprocess_batch_size

            old_reward = np.copy(ds.dataset_dict["rewards"][start_idx: end_idx])
            state_batch = all_states[start_idx: end_idx].transpose(0, 3, 1, 2)
            state_batch_tensor = torch.Tensor(state_batch).float().to(device)
            reward_batch = lrf._calculate_reward(state_batch_tensor)
            ds.dataset_dict["rewards"][start_idx: end_idx] = reward_batch.squeeze()


    '''
    Setup envs
    '''
    if FLAGS.use_lrf:
        dmc_env = ImageOnlineCustomRewardMetaworldEnv(
            env_str, camera_name="left_cap2", high_res_env=False, rlpd_res=True, lrf=lrf, airl_style_reward=False
        )
    else:
        dmc_env = ImageMetaworldEnv(env_str, camera_name="left_cap2", high_res_env=False, rlpd_res=True)
    env = DMCWrapper(dmc_env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    if FLAGS.save_video:
        from drqv2.video import TrainVideoRecorder
        train_recorder = TrainVideoRecorder(Path(log_dir), "train_video", render_size=64)
        eval_recorder = TrainVideoRecorder(Path(log_dir), "eval_video", render_size=64)

    env.seed(FLAGS.seed)

    eval_dmc_env = ImageOnlineCustomRewardMetaworldEnv(
        env_str, camera_name="left_cap2", high_res_env=False, rlpd_res=True, lrf=lrf, airl_style_reward=False
    )
    eval_env = DMCWrapper(eval_dmc_env)
    eval_env.seed(FLAGS.seed + 42)

    '''
    Setup agent
    '''
    # Crashes on some setups if agent is created before replay buffer.
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed,
        env.observation_space,
        env.action_space,
        pixel_keys=pixel_keys,
        **kwargs,
    )

    '''
    Begin training
    '''
    observation, done = env.reset(), False
    if FLAGS.save_video:
        train_recorder.init(observation["pixels"])
        is_train_recording = True
    per_episode_og_reward = 0
    per_episode_success = False
    per_episode_succeeded = False
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps // action_repeat + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)
        per_episode_og_reward += info["og_reward"]
        per_episode_success = info["success"]
        per_episode_succeeded |= info["success"]

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if FLAGS.save_video and is_train_recording:
            train_recorder.record(observation["pixels"])

        if done:
            # save video
            if FLAGS.save_video and is_train_recording:
                train_recorder.save(f'{i}.mp4')
                is_train_recording = False

            # normal reset things
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i * action_repeat)

            # bookkeeping
            wandb.log({f"training/og_return": per_episode_og_reward}, step=i * action_repeat)
            wandb.log({f"training/success": per_episode_success}, step=i * action_repeat)
            wandb.log({f"training/succeeded": per_episode_succeeded}, step=i * action_repeat)
            per_episode_og_reward = 0
            per_episode_success = False
            per_episode_succeeded = False

            # bookkeeping
            if FLAGS.save_video and (i % (100 * FLAGS.log_video_frequency) == 0):
                is_train_recording = True
                train_recorder.init(observation["pixels"])

        if i >= FLAGS.start_training:
            online_batch = next(replay_buffer_iterator)
            offline_batch = next(ds_iterator)
            batch = combine(offline_batch, online_batch)

            # periodically update the classifier in the lrf
            if FLAGS.use_lrf and (i % FLAGS.lrf_update_frequency == 0 or not lrf.seen_on_policy_data):
                lrf.train(FLAGS.lrf_train_steps)

            # relabel the rewards of the batch here
            if FLAGS.use_lrf and not FLAGS.skip_relabel_rewards:
                states_batch = batch['observations']['pixels']
                states_batch = np.transpose(np.squeeze(states_batch[:, :, :, :, 0]), (0, 3, 1, 2))
                states_tensor = torch.Tensor(states_batch).float().to(device)
                new_rewards = lrf._calculate_reward(states_tensor).squeeze()
                new_batch = {}
                for k, v in batch.items():
                    if k == "rewards":
                        new_batch[k] = new_rewards
                    else:
                        new_batch[k] = v
                batch = FrozenDict(new_batch)

            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i * action_repeat)


        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
                curr_step=i,
                video_recorder=eval_recorder
            )
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i * action_repeat)

            # bookkeeping
            # checkpoints.restore_checkpoint(chkpt_dir, target=agent)
            checkpoints.save_checkpoint(
                chkpt_dir, target=agent, step=i * action_repeat, overwrite=True
            )
            try:
                with open(os.path.join(buffer_dir, f"buffer"), "wb") as f:
                    pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)
            except:
                print("Could not save agent buffer.")


if __name__ == "__main__":
    app.run(main)
