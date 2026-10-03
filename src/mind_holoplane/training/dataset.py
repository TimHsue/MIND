import json
import torch
import numpy as np
import os
from torch.utils.data import Dataset


def load_all_names(file_list_name):
    with open(file_list_name, 'r') as f:
        names = f.readlines()
    names = [name.strip() for name in names if name.strip()]
    if not names:
        raise ValueError('The sample list is empty')
    if any(name.endswith('.npy') for name in names):
        raise ValueError('AE sample lists must contain basenames without .npy')
    return names

def computer_single_weights(data, num_bins=20):
    min_value = np.min(data)
    max_value = np.max(data)
    bins = np.linspace(min_value, max_value, num_bins + 1)
    indices = np.digitize(data, bins) - 1
    indices = np.clip(indices, 0, num_bins - 1)

    single_indices = indices
    total_single_bins = num_bins

    single_counts = np.bincount(single_indices, minlength=total_single_bins)

    freq = (single_counts + 1) / (len(data) + num_bins)
    single_weights = 1.0 / freq

    single_weights = single_weights / np.sum(single_weights)
    sample_weights = single_weights[single_indices]

    return sample_weights

class VoxelOccupancyDataset(Dataset):
    def __init__(self, resolution, file_list_name, voxel_dir, occup_dir, elastic_tensor_dir, points_batch_size,
                 random_sample=True, return_name=False, return_weight=False, return_property=False, return_elastic_tensor=False, dataset_pro=None, device='cpu'):
        self.device = device
        self.random_sample = random_sample
        self.names_dataset = load_all_names(file_list_name)
        self.voxel_dir = voxel_dir
        self.occup_dir = occup_dir
        self.elastic_tensor_dir = elastic_tensor_dir
        self.resolution = resolution
        self.dataset_size = len(self.names_dataset)
        self.points_batch_size = points_batch_size
        self.return_name = return_name
        self.return_property = return_property
        self.return_elastic_tensor = return_elastic_tensor
        self.return_weight = return_weight

        self.voxel_paths = [os.path.join(self.voxel_dir, name + '.npy') for name in self.names_dataset]
        self.occup_paths = [os.path.join(self.occup_dir, name + '.npy') for name in self.names_dataset]
        self.elastic_tensor_paths = [os.path.join(self.elastic_tensor_dir, name + '.npy') for name in self.names_dataset]


        if dataset_pro is not None:
            with open(dataset_pro, 'r') as f:
                metadata = json.load(f)
                if metadata.get('label_space', 'physical') != 'physical':
                    raise ValueError('Joint AE training requires physical conditions')
                labels = metadata['labels']
                dataset_pro = []
                for name in self.names_dataset:
                    now_pro = labels[name + '.npy']
                    dataset_pro.append([now_pro[0], now_pro[1], now_pro[2]])
                self.dataset_pro = np.array(dataset_pro, dtype=np.float32)
                if not np.isfinite(self.dataset_pro).all():
                    raise ValueError("Conditions must be finite")
                weights_1 = computer_single_weights(self.dataset_pro[:, 0])
                weights_2 = computer_single_weights(self.dataset_pro[:, 1])
                weights_3 = computer_single_weights(self.dataset_pro[:, 2])
                self.weights = (weights_1 + weights_2 + weights_3) / 3
            self.pro_size = len(self.dataset_pro[0])


    def __len__(self):
        return self.dataset_size

    def load_voxel(self, idx):
        voxel = np.load(self.voxel_paths[idx], allow_pickle=False)
        if voxel.size != self.resolution ** 3 or not np.isfinite(voxel).all():
            raise ValueError(f"{self.names_dataset[idx]}: expected a finite {self.resolution} cubed voxel SDF")
        voxel = torch.as_tensor(voxel, dtype=torch.float32).view(self.resolution, self.resolution, self.resolution).clone().contiguous()
        voxel_flip_x = torch.flip(voxel, [0])
        voxel = (voxel + voxel_flip_x) / 2
        voxel_flip_y = torch.flip(voxel, [1])
        voxel = (voxel + voxel_flip_y) / 2
        voxel_flip_z = torch.flip(voxel, [2])
        voxel = (voxel + voxel_flip_z) / 2
        return voxel

    def load_occup(self, idx):


        occup = np.load(self.occup_paths[idx], allow_pickle=False)

        if occup.ndim != 2 or occup.shape[1] != 4 or not len(occup) or not np.isfinite(occup).all():
            raise ValueError(f"{self.names_dataset[idx]}: expected finite nonempty [N,4] SDF points")
        occup = torch.as_tensor(occup, dtype=torch.float32).view(-1, 4).clone().contiguous()

        # Columns are x, y, z and signed distance.
        return occup

    def load_elastic_tensor(self, idx):
        elastic_tensor = np.load(self.elastic_tensor_paths[idx], allow_pickle=False)
        if elastic_tensor.shape != (18, 64, 64, 64) or not np.isfinite(elastic_tensor).all():
            raise ValueError(f"{self.names_dataset[idx]}: expected finite [18,64,64,64] displacement references")
        elastic_tensor = torch.as_tensor(elastic_tensor, dtype=torch.float32).permute(1, 2, 3, 0).clone().contiguous()
        return elastic_tensor

    def load_property(self, idx):
        pro = self.dataset_pro[idx]
        pro = torch.as_tensor(pro, dtype=torch.float32)
        return pro

    def load_weights(self, idx):
        weight = self.weights[idx]
        weight = torch.as_tensor(weight, dtype=torch.float32)
        return weight

    def __getitem__(self, idx):
        voxel = self.load_voxel(idx)
        occu_points_all = self.load_occup(idx)

        if self.random_sample:
            num_points = occu_points_all.shape[0]
            sample_indices = torch.randint(0, num_points, size=(self.points_batch_size,))
            sampled_occu_points = occu_points_all[sample_indices]
        else:
            sampled_occu_points = occu_points_all

        return_list = [voxel, sampled_occu_points]
        if self.return_property:
            pro = self.load_property(idx)
            return_list.append(pro)
        if self.return_weight:
            weight = self.load_weights(idx)
            return_list.append(weight)
        if self.return_elastic_tensor:
            elastic_tensor = self.load_elastic_tensor(idx)
            return_list.append(elastic_tensor)
        if self.return_name:
            return_list.append(self.names_dataset[idx])
        return tuple(return_list)
