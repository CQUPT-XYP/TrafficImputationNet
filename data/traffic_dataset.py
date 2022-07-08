import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import numpy as np
import datetime
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class InitDataset():
    def __init__(self, dataset_path) -> None:
        dataset = np.load(dataset_path, allow_pickle=True).tolist()
        time_list = list(dataset.keys())
        self.time_list, self.dataset = self.get_dataset_filter(dataset, time_list)
        random.shuffle(self.dataset)


    def get_dataset(self):
        return self.dataset


    def get_dataset_filter(self, dataset, time_list):
        time_list_filter = []
        dataset_filter = []
        for center_time in time_list:
            try:
                prior_7_days_data = dataset[center_time - datetime.timedelta(days=7)]["real"]
                prior_5_min_data = dataset[center_time - datetime.timedelta(minutes=5)]["real"]
                prior_10_min_data = dataset[center_time - datetime.timedelta(minutes=10)]["real"]
                prior_15_min_data = dataset[center_time - datetime.timedelta(minutes=15)]["real"]

                # 中间为loss的数据
                center_data = dataset[center_time]["loss"]

                next_5_min_data = dataset[center_time + datetime.timedelta(minutes=5)]["real"]
                next_10_min_data = dataset[center_time + datetime.timedelta(minutes=10)]["real"]
                next_15_min_data = dataset[center_time + datetime.timedelta(minutes=15)]["real"]
                next_7_days_data = dataset[center_time + datetime.timedelta(days=7)]["real"]

                # 相关的时间片只要有一个全为0即跳过
                if np.sum(prior_7_days_data) == 0 or np.sum(prior_15_min_data) == 0 or np.sum(
                        prior_10_min_data) == 0 or np.sum(
                        prior_5_min_data) == 0 or np.sum(center_data) == 0 or np.sum(next_5_min_data) == 0 or np.sum(
                    next_10_min_data) == 0 or np.sum(next_15_min_data) == 0 or np.sum(next_7_days_data) == 0:
                    continue
                time_list_filter.append(center_time)
                target = dataset[center_time]["real"]
                loss_indices = dataset[center_time]["loss_indices"]
                loss_mask = dataset[center_time]["loss_mask"]
                block = [prior_7_days_data, prior_15_min_data, prior_10_min_data, prior_5_min_data, center_data, next_5_min_data, next_10_min_data, next_15_min_data, next_7_days_data]
                dataset_filter.append((block, center_data, loss_mask, target, loss_indices))
            except Exception as e:
                pass


        return time_list_filter, dataset_filter



class TrafficDataset(Dataset):
    def __init__(self, dataset, mask_path, stage_name):
        super(TrafficDataset, self).__init__()
        self.mask = np.load(mask_path, allow_pickle=True).tolist()
        self.dataset = dataset
        total_size = len(self.dataset)
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2
        train_size = round(total_size * train_ratio)
        val_size = round(total_size * val_ratio)
        if stage_name == "train":
            self.dataset = dataset[:train_size]
        elif stage_name == "val":
            self.dataset = dataset[train_size : train_size + val_size]
        elif stage_name == "test":
            self.dataset = dataset[train_size + val_size:]
        elif stage_name == "one_day":
            self.dataset = dataset

    def __getitem__(self, index):
        block, center_data, loss_mask, target, loss_indices = self.dataset[index]
        target = torch.Tensor(target).unsqueeze(0)
        block = torch.Tensor(block).unsqueeze(1)
        center_data = torch.Tensor(center_data).unsqueeze(0)
        loss_mask = torch.Tensor(loss_mask).unsqueeze(0)
        mask = torch.Tensor(self.mask).unsqueeze(0)
        loss_indices = torch.LongTensor(loss_indices)
        return block, center_data, loss_mask, target, loss_indices, mask
        

    def __len__(self):
        return len(self.dataset)



# if __name__ == '__main__':
#     path = "dataset/dataset_5_min_add_loss.npy"
#     traffic_dataset = TrafficDataset(path, is_train=False)
#     train_loader = DataLoader(dataset=traffic_dataset, batch_size=20)
#     block, target, loss_indices = next(iter(train_loader))