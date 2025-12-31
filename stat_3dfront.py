import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.Threed_front_dataset import ThreeDFrontDataset
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset_dir = "./datasets/atiss_wo_lat"

if __name__ == "__main__":
    train_dataset = ThreeDFrontDataset(npz_dir=dataset_dir, split='train', padded_length=32)
    stats_loader = DataLoader(train_dataset, batch_size=1, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)

    categories =[ [] for _ in range(32)]
    translates_x = [[] for _ in range(32)]
    translates_z = [[] for _ in range(32)]
    category_dim = 33
    slices = {
            'translate': slice(category_dim, category_dim + 3),
            'size': slice(category_dim + 3, category_dim+6),
            'rotation': slice(category_dim + 6, category_dim + 7),
        }

    for i, sample in tqdm(enumerate(stats_loader)):
        obj_tokens = sample['obj_tokens'][0]
        key_padding_mask = sample['attention_mask'][0]
        for j in range(32):
            obj_j = obj_tokens[j]
            pad_j = key_padding_mask[j]

            if pad_j:
                categories[j].append(32)
                continue
            
            translate_j = obj_j[slices['translate']]
            
            translates_x[j].append(translate_j[0])
            translates_z[j].append(translate_j[2])

            c_j = torch.argmax(obj_j[:category_dim])
            categories[j].append(c_j)

    
    for i in range(32):
        translates_x_i = translates_x[i]
        translates_z_i = translates_z[i]


        plt.figure()
        plt.scatter(translates_x_i, translates_z_i)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"XZ Translate of Obj Index {i+1}")
        plt.savefig(f"./datasets/figures/xz_translate_ind_{i+1}.jpg")

        c_i = categories[i]

        plt.figure()
        plt.hist(c_i, bins=range(min(c_i), max(c_i)+2))
        plt.xlabel("value")
        plt.ylabel("count")
        plt.title(f"Category hist on Obj Index {i+1}")
        plt.savefig(f"./datasets/figures/c_hist_ind_{i+1}.jpg")
        plt.clf()