import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import os
import yaml
from typing import List, Dict, Tuple

class GeneralTrajectoryDataset(Dataset):
    def __init__(self, 
                 folder_path: str,
                 horizon: int,
                 history_length: int = 10,
                 include_current: bool = True,
                 predict_features: List[str] = ['timestamp', 'longitude', 'latitude', 'altitude'],
                 packed_features: List[str] = ['longitude', 'latitude', 'altitude'],
                 normalization: Dict[str, Dict[str, float]] = None):
        
        self.data_dir = folder_path
        self.horizon = horizon
        self.history_length = history_length
        self.include_current = include_current
        self.predict_features = predict_features
        self.features_to_load = predict_features + packed_features
        self.packed_features = packed_features
        self.normalization = normalization or {
            'longitude': {'min': -76.07, 'max': -73.91},
            'latitude': {'min': 40.28, 'max': 41.55},
            'altitude': {'min': 0, 'max': 20000},
        }
        
        self.trajectories = self._load_trajectories()
        self.indices = self._make_indices()


    def _load_trajectories(self) -> List[Dict[str, np.ndarray]]:
        trajectories = []
        self.observation_dim = 0
        self.packed_dim = 0
        for traj_folder in os.listdir(self.data_dir):
            traj_path = os.path.join(self.data_dir, traj_folder)
            if os.path.isdir(traj_path):
                traj_data = {}
                for feature in self.features_to_load:
                    file_path = os.path.join(traj_path, f"{feature}.npz")
                    if os.path.exists(file_path):
                        data = np.load(file_path)['arr_0']
                        if feature in self.normalization:
                            data = self.normalize(data, feature)
                        traj_data[feature] = data
                    else:
                        print(f"Warning: {file_path} not found. Skipping this trajectory.")
                        break
                else:
                    trajectories.append(traj_data)
                    # Calculate observation_dim which consists of the predict_features - currently assumes all predict_features have the same dimension
                    if self.observation_dim == 0:
                        self.observation_dim = len(self.predict_features)

                    if self.packed_dim == 0:
                        self.packed_dim = len(self.packed_features)

        
        print(f"Observation dimension: {self.observation_dim}")
        return trajectories

    def _make_indices(self) -> List[Tuple[int, int, int]]:
        indices = []
        for i, traj in enumerate(self.trajectories):
            traj_length = len(next(iter(traj.values())))
            max_start = traj_length - self.horizon - self.history_length
            for start in range(max_start):
                end = start + self.horizon + self.history_length
                indices.append((i, start, end))
        return indices

    def normalize(self, arr: np.ndarray, feature: str) -> np.ndarray:
        min_val = self.normalization[feature]['min']
        max_val = self.normalization[feature]['max']
        return ((arr - min_val) / (max_val - min_val)) * 2 - 1

    def unnormalize(self, arr: np.ndarray) -> np.ndarray:
        """
        Unnormalize the entire array based on the predict_features.
        
        :param arr: numpy array of shape (batch_size, sequence_length, num_features)
                    or (sequence_length, num_features)
        :return: unnormalized numpy array of the same shape as input
        """
        # Ensure the input array is a numpy array
        arr = np.array(arr)

        # Check if the input is a single sequence or a batch
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # Add batch dimension
        
        assert arr.shape[-1] == len(self.predict_features), "Number of features in array does not match predict_features"
        
        unnormalized = np.zeros_like(arr)
        
        for i, feature in enumerate(self.predict_features):
            if feature in self.normalization:
                min_val = self.normalization[feature]['min']
                max_val = self.normalization[feature]['max']
                unnormalized[..., i] = ((arr[..., i] + 1) / 2) * (max_val - min_val) + min_val
            else:
                unnormalized[..., i] = arr[..., i]  # Keep unnormalized features as is
        
        # Remove batch dimension if it was added
        if unnormalized.shape[0] == 1:
            unnormalized = unnormalized[0]
        
        return unnormalized


    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        traj_idx, start, end = self.indices[idx]
        traj = self.trajectories[traj_idx]
        
        history_end = start + self.history_length
        future_start = history_end if self.include_current else history_end + 1
        
        future = torch.tensor(np.stack([traj[f][future_start:end] for f in self.predict_features], axis=-1)).float()
        history = torch.tensor(np.stack([traj[f][start:history_end] for f in self.packed_features], axis=-1)).float()
        
        cond = (np.array([]), np.array([]))
        return future, history, cond

    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.nn.utils.rnn.PackedSequence]:
        futures, histories, cond = zip(*batch)
        
        padded_futures = pad_sequence(futures, batch_first=True)
        
        history_lengths = [len(h) for h in histories]
        padded_histories = pad_sequence(histories, batch_first=True, padding_value=0)
        packed_histories = pack_padded_sequence(padded_histories, history_lengths, batch_first=True, enforce_sorted=False)
        
        return padded_futures, {"detections": packed_histories}, cond

    def collate_fn_repeat(self, batch: List[Tuple[torch.Tensor, torch.Tensor]], num_samples: int) -> Tuple[torch.Tensor, torch.nn.utils.rnn.PackedSequence]:
        futures, histories, cond = zip(*batch)
        
        # Pad and repeat futures
        padded_futures = pad_sequence(futures, batch_first=True)
        repeated_futures = padded_futures.repeat(num_samples, 1, 1)
        
        # Repeat histories
        repeated_histories = histories * num_samples
        
        # Process repeated histories
        history_lengths = [len(h) for h in repeated_histories]
        padded_histories = pad_sequence(repeated_histories, batch_first=True, padding_value=0)
        packed_histories = pack_padded_sequence(padded_histories, history_lengths, batch_first=True, enforce_sorted=False)
        
        return repeated_futures, {"detections": packed_histories}, cond * num_samples


# Example usage
if __name__ == "__main__":
    dataset = GeneralTrajectoryDataset(
        folder_path='/home/sean/flight_data/N172CK/converted_train',
        horizon=60,
        history_length=10,
        include_current=True,
        features_to_load=['timestamp', 'longitude', 'latitude', 'altitude'],
        packed_features=['longitude', 'latitude', 'altitude'],
        normalization={
            'longitude': {'min': -76.07, 'max': -73.91},
            'latitude': {'min': 40.28, 'max': 41.55},
            'altitude': {'min': 0, 'max': 20000},
        }
    )
    print(f"Observation dimension: {dataset.observation_dim}")
    
    # Example with regular collate_fn
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        num_workers=4, 
        shuffle=True, 
        pin_memory=True, 
        collate_fn=dataset.collate_fn
    )
    
    for future, history in dataloader:
        print(f"Regular - Future shape: {future.shape}")
        print(f"Regular - History shape: {history.data.shape}")
        break
    
    # Example with collate_fn_repeat
    dataloader_repeat = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        num_workers=4, 
        shuffle=True, 
        pin_memory=True, 
        collate_fn=lambda batch: dataset.collate_fn_repeat(batch, num_samples=2)
    )
    
    for future, history in dataloader_repeat:
        print(f"Repeated - Future shape: {future.shape}")
        print(f"Repeated - History shape: {history.data.shape}")
        break