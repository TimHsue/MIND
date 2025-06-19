

import json
import torch
import numpy as np
import os
from torch.utils.data import Dataset

# Occupancy trains the model better than SDFs
def sdf_to_occ(arr):
    return (arr > 0.5).astype(float)

def load_all_files(data_dir):
    file_names = os.listdir(data_dir)
    # sort
    file_names = sorted(file_names)
    
    # empty np array
    all_data = {}
    
    for file_name in file_names:
        target_dir = os.path.join(data_dir, file_name)
        data = np.load(target_dir)
        
        all_data[file_name] = data

    return all_data

def load_all_names(file_list_name):
    with open(file_list_name, 'r') as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    return names

class VoxelOccupancyDataset_old(Dataset):
    def __init__(self, resolution, dataset_voxel, dataset_occup, points_batch_size, random_sample=True, return_full=False, device='cpu'):
        self.device = device
        self.random_sample = random_sample
        voxel_dataset = load_all_files(dataset_voxel)
        occup_dataset = load_all_files(dataset_occup)
        
        voxel_data_size = len(voxel_dataset)
        occup_data_size = len(occup_dataset)
        assert voxel_data_size == occup_data_size, "Voxel and Occupancy dataset sizes do not match"
        
        # for key in voxel_dataset.keys():
        #     print(key, voxel_dataset[key].shape)
        
        self.dataset_size = voxel_data_size
        self.return_full = return_full
        
        # resolution ^ 3
        voxel_size = list(voxel_dataset.values())[0].shape[0]
        assert resolution ** 3 == voxel_size or resolution == voxel_size, f"Resolution {resolution} does not match voxel size {voxel_size}"
        
        self.voxel_data = torch.stack([torch.Tensor(voxel_dataset[key]).view(resolution, resolution, resolution) for key in voxel_dataset.keys()])
        self.occup_data = torch.stack([torch.Tensor(occup_dataset[key]).view(-1, 4) for key in occup_dataset.keys()])
        # print([key.split('.')[0].split('-')[-2] for key in voxel_dataset.keys()])
        if self.return_full:
            self.obj_name = [key.split('.')[0] for key in voxel_dataset.keys()]

        if random_sample is False:
            for i in range(len(self.occup_data)):
                self.occup_data[i] = self.occup_data[i][torch.randperm(self.occup_data[i].shape[0])]
        
        # print('voxel_data', self.voxel_data.shape)
        # print('occup_data', self.occup_data.shape)
        
        self.points_batch_size = points_batch_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        
        voxel = self.voxel_data[idx]
        occu_points_all = self.occup_data[idx]
        
        # print('voxel', voxel.shape)
        # print('occu_points_all', occu_points_all.shape)
        
        if self.random_sample:
            sample_indices = torch.randint(0, occu_points_all.shape[0], size=(self.points_batch_size,))
            sampled_occu_points = occu_points_all[sample_indices]
        else:
            sampled_occu_points = occu_points_all
        
        # print('sampled_occu_points', sampled_occu_points.shape)

        if self.return_full:
            return voxel, sampled_occu_points, self.obj_name[idx]
        return voxel, sampled_occu_points

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
    def __init__(self, resolution, file_list_name, voxel_dir, occup_dir, consm_dir, points_batch_size,
                 random_sample=True, return_name=False, return_weight=False, return_property=False, return_consm=False, dataset_pro=None, device='cpu'):
        self.device = device
        self.random_sample = random_sample
        self.names_dataset = load_all_names(file_list_name)
        self.voxel_dir = voxel_dir
        self.occup_dir = occup_dir
        self.consm_dir = consm_dir
        self.resolution = resolution
        self.dataset_size = len(self.names_dataset)
        self.points_batch_size = points_batch_size
        self.return_name = return_name
        self.return_property = return_property
        self.return_consm = return_consm
        self.return_weight = return_weight

        self.voxel_paths = [os.path.join(self.voxel_dir, name + '.npy') for name in self.names_dataset]
        self.occup_paths = [os.path.join(self.occup_dir, name + '.npy') for name in self.names_dataset]
        self.consm_paths = [os.path.join(self.consm_dir, name + '.npy') for name in self.names_dataset]


        if dataset_pro is not None:
            with open(dataset_pro, 'r') as f:
                labels = json.load(f)['labels']
                dataset_pro = []
                for name in self.names_dataset:
                    now_pro = labels[name + '.npy']
                    dataset_pro.append([now_pro[0], now_pro[1], now_pro[2]])
                self.dataset_pro = np.array(dataset_pro)
                weights_1 = computer_single_weights(self.dataset_pro[:, 0])
                weights_2 = computer_single_weights(self.dataset_pro[:, 1])
                weights_3 = computer_single_weights(self.dataset_pro[:, 2])
                self.weights = (weights_1 + weights_2 + weights_3) / 3
            self.pro_size = len(self.dataset_pro[0])

        
    def __len__(self):
        return self.dataset_size

    def load_voxel(self, idx):
        # voxel = np.load(self.voxel_paths[idx], mmap_mode='r')
        # voxel = torch.from_numpy(voxel.copy()).float().view(self.resolution, self.resolution, self.resolution)
        voxel = np.load(self.voxel_paths[idx])
        voxel = torch.as_tensor(voxel, dtype=torch.float32).view(self.resolution, self.resolution, self.resolution).clone().contiguous() 
        voxel_flip_x = torch.flip(voxel, [0])
        voxel = (voxel + voxel_flip_x) / 2
        voxel_flip_y = torch.flip(voxel, [1])
        voxel = (voxel + voxel_flip_y) / 2
        voxel_flip_z = torch.flip(voxel, [2])
        voxel = (voxel + voxel_flip_z) / 2
        return voxel

    def load_occup(self, idx):
        
        # occup = np.load(self.occup_paths[idx], mmap_mode='r')
        # occup = torch.from_numpy(occup.copy()).float().view(-1, 4)
        
        occup = np.load(self.occup_paths[idx])

        occup = torch.as_tensor(occup, dtype=torch.float32).view(-1, 4).clone().contiguous() 

        # extract last half of the points (surface)
        totnum = occup.shape[0]
        num = totnum // 2
        # swap x and z
        occup[num:, [0, 2]] = occup[num:, [2, 0]]
        
        return occup
    
    def load_consm(self, idx):
        consm = np.load(self.consm_paths[idx]) # 18, 64, 64, 64
        consm = torch.as_tensor(consm, dtype=torch.float32).permute(1, 2, 3, 0).clone().contiguous()
        # consm = consm - consm.mean()
        return consm

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
            if num_points >= self.points_batch_size:
                sample_indices = torch.randint(0, num_points, size=(self.points_batch_size,))
                sampled_occu_points = occu_points_all[sample_indices]
            else:
                sampled_occu_points = occu_points_all
        else:
            sampled_occu_points = occu_points_all

        return_list = [voxel, sampled_occu_points]
        if self.return_property:
            pro = self.load_property(idx)
            return_list.append(pro)
        if self.return_weight:
            weight = self.load_weights(idx)
            return_list.append(weight)
        if self.return_consm:
            consm = self.load_consm(idx)
            return_list.append(consm)
        if self.return_name:
            return_list.append(self.names_dataset[idx])
        return tuple(return_list)