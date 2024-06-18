import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from collections import Counter


def fit_gmm(df, class_label, n_components=2):
    data = df[df["y"] == class_label]["loss"].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components).fit(data)
    return gmm


def predict_proba(gmm, class_label, df):
    data = df[df["y"] == class_label]["loss"].values.reshape(-1, 1)
    data = (data - data.min()) / (data.max() - data.min())
    probs = gmm.predict_proba(data)
    # select smaller mean class
    return probs[:, np.argmin(gmm.means_)]


def plot_loss_distribution(df, class_label, gmm=None):
    data = df[df["y"] == class_label]["loss"]
    plt.hist(data, bins=30, density=True)

    if gmm:
        x = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
        logprob = gmm.score_samples(x)
        pdf = np.exp(logprob)
        plt.plot(x, pdf, "-k")

    plt.title(f"Loss Distribution for Class {class_label}")
    plt.show()


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class ImageGraphDataset(Dataset):
    def __init__(
        self, image_dir, graph_data, target_transform=None, neighbor_transform=None
    ):
        self.image_dir = image_dir
        self.graph_data = graph_data
        self.target_transform = target_transform
        self.neighbor_transform = neighbor_transform
        self.image_dataset = datasets.ImageFolder(self.image_dir)
        self.neighbors = self.precompute_neighbors()

    def __getitem__(self, index):
        img, label = self.image_dataset[index]
        if self.target_transform:
            img = self.target_transform(img)
        neighbor_imgs = [self.image_dataset[i][0] for i in self.neighbors[index]]
        neighbor_imgs = [
            self.neighbor_transform(img) if self.neighbor_transform else img
            for img in neighbor_imgs
        ]
        return img, neighbor_imgs, label

    def __len__(self):
        return len(self.image_dataset)

    def precompute_neighbors(self):
        edges = self.graph_data.edge_index.t().cpu().numpy()
        num_nodes = len(self.image_dataset)
        neighbors = [[] for _ in range(num_nodes)]
        for i, j in edges:
            neighbors[i].append(j)
        return


def get_class_counts(dataset, args):
    lst = [0 for _ in range(args.num_class)]
    for cls, count in Counter(dataset.train_labels.values()).items():
        lst[cls] = count
    return lst
