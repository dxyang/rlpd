import d4rl
import gym
import numpy as np

from rlpd.data.dataset import Dataset

import sys
sys.path.append("/home/dxyang/code/rewardlearning-vid")
from reward_extraction.data import H5PyTrajDset

class MetaworldDataset(Dataset):
    def __init__(self, h5py_path: str):
        dataset_dict = {}
        self.expert_data_ptr = H5PyTrajDset(h5py_path, read_only_if_exists=True)
        self.expert_data = [d for d in self.expert_data_ptr]

        observations = []
        actions = []
        next_observations = []
        rewards = []
        masks = []
        dones = []

        for traj in self.expert_data:
            observations.append(traj[0][:-1])
            next_observations.append(traj[0][1:])
            actions.append(traj[1][:])
            rewards.append(traj[2][:])

            done = np.full_like(traj[2][:], False, dtype=bool)
            done[-1] = True
            dones.append(done)

            mask = np.full_like(traj[2][:], 1.0, dtype=np.float32)
            mask[-1] = 0.0
            masks.append(mask)

        observations_np = np.concatenate(observations)
        next_observations_np = np.concatenate(next_observations)
        actions_np = np.concatenate(actions)
        rewards_np = np.concatenate(rewards)
        masks_np = np.concatenate(masks)
        dones_np = np.concatenate(dones)

        dataset_dict["observations"] = observations_np.astype(np.float32)
        dataset_dict["next_observations"] = next_observations_np.astype(np.float32)
        dataset_dict["actions"] = actions_np.astype(np.float32)
        dataset_dict["rewards"] = rewards_np.astype(np.float32)
        dataset_dict["masks"] = masks_np.astype(np.float32)
        dataset_dict["dones"] = dones_np.astype(np.bool8)

        import pdb; pdb.set_trace()

        super().__init__(dataset_dict)

class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        dataset_dict = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dones = np.full_like(dataset_dict["rewards"], False, dtype=bool)

        for i in range(len(dones) - 1):
            if (
                np.linalg.norm(
                    dataset_dict["observations"][i + 1]
                    - dataset_dict["next_observations"][i]
                )
                > 1e-6
                or dataset_dict["terminals"][i] == 1.0
            ):
                dones[i] = True

        dones[-1] = True

        dataset_dict["masks"] = 1.0 - dataset_dict["terminals"]
        del dataset_dict["terminals"]

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["dones"] = dones

        '''
        for k, v in dataset_dict.items():
            print(f"{k}: {v.shape}, {v.dtype}")

        observations: (998999, 17), float32
        actions: (998999, 6), float32
        next_observations: (998999, 17), float32
        rewards: (998999,), float32
        masks: (998999,), float32
        dones: (998999,), bool
        '''
        import pdb; pdb.set_trace()

        super().__init__(dataset_dict)
