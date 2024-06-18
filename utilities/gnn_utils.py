from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops
from torchmetrics import MetricCollection, Accuracy, Recall
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utilities.NearestNeighbors import NearestTreeNeighbors



class CsvImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv, base_path, split, transform=None):
        self.csv = csv
        self.base_path = base_path
        self.split = split
        self.transform = transform
        # find suffix
        for subdir, dirs, files in os.walk(f"{base_path}/{split}"):
            for file in files:
                self.suffix = "." + file.split(".")[-1]
                break

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        internal_id = str(self.csv.iloc[idx]["internal_id"])
        if internal_id.startswith("BLANK"):
            internal_id = f"0000{internal_id[-1]}"
        genus = (
            self.csv.iloc[idx]["tree/genus/genus"].replace("/", "").replace(" ", "_")
        )
        img_name = f"{self.base_path}/{self.split}/{genus}/{internal_id}{self.suffix}"
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, self.csv.iloc[idx]["y"]


def convert_string_column(df, column_name, new_name):
    df[new_name] = pd.Categorical(df[column_name])
    df[new_name] = df[new_name].cat.codes
    return df


def get_features_and_edges(
    dataset,
    nn,
    base_pth,
    net1,
    split,
    device,
    transforms,
    return_preds=False,
    net2=None,
    radius=0.002,
):
    dataset = dataset.sort_values(by=["tree/genus/genus", "internal_id"])
    train_edges = []
    pos = []
    features = []
    ys = []
    all_preds = []
    paths = []
    net1.eval()
    if net2 is not None:
        net2.eval()
    dset = CsvImageDataset(dataset, base_pth, split, transforms)
    loader = DataLoader(dset, batch_size=512, shuffle=False, num_workers=5)
    for idx, (imgs, y) in enumerate(tqdm(loader)):
        imgs = imgs.to(device)
        with torch.no_grad():
            if return_preds:
                preds = net1(imgs)
                if net2 is not None:
                    preds += net2(imgs)
                all_preds.append(preds.cpu())
            features1 = net1.features(imgs)
            features1 = net1.avgpool(features1).cpu().numpy().squeeze()
            if net2 is not None:
                features2 = net2.features(imgs)
                features2 = net2.avgpool(features2).cpu().numpy().squeeze()
                features1 = np.concatenate([features1, features2], axis=1)
        features.append(features1)
    features = np.concatenate(features, axis=0)
    all_preds = torch.cat(all_preds)

    pyg_data = Data(
        x=torch.tensor(features),
        edge_index=torch.tensor([]),
        pos=torch.tensor(dataset[["latitude", "longitude"]].values),
        y=torch.tensor(dataset["y"].values),
    )
    pyg_data = T.RadiusGraph(r=radius, loop=False, max_num_neighbors=50)(pyg_data)
    paths = dataset.apply(
        lambda row: f"{base_pth}/{split}/{row['tree/genus/genus']}/{row['internal_id']}.jpg",
        axis=1,
    ).values

    if return_preds:
        return (
            pyg_data.x,
            pyg_data.edge_index,
            pyg_data.pos,
            pyg_data.y,
            paths,
            all_preds,
        )
    return (pyg_data.x, pyg_data.edge_index, pyg_data.pos, pyg_data.y, paths)


def train(loader, model, device, optimizer, criterion):
    model.train()
    model = model.to(device)
    loss = 0
    idx = 0
    for batch in loader:
        optimizer.zero_grad()  # Clear gradients.
        out = model(
            batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device)
        )  # Perform a single forward pass.
        curr_loss = criterion(
            out[: batch.batch_size], batch.y[: batch.batch_size].to(device)
        )  # Compute the loss solely based on the training nodes.
        loss += curr_loss.item()
        idx += 1
        curr_loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
    return loss / idx


def test(val_data, model, criterion, args):
    model.eval()
    model.cpu()
    preds = model(val_data.x, val_data.edge_index, val_data.edge_attr)
    loss = criterion(preds, val_data.y)
    metrics = MetricCollection(
        [
            MetricCollection(
                Accuracy(
                    num_classes=args.num_class, average="macro", task="multiclass"
                ),
                Recall(num_classes=args.num_class, average="macro", task="multiclass"),
                postfix="_macro",
            ),  # type: ignore
            MetricCollection(
                Accuracy(
                    num_classes=args.num_class, average="micro", task="multiclass"
                ),
                postfix="_micro",
            ),
        ]
    )(preds, val_data.y)
    return metrics, loss.item()


def setup(args, split):
    base_path = args.data_path
    train_csv = pd.read_csv(f"{base_path}/{split}.csv")
    train_csv = train_csv.rename(
        columns={"lat": "latitude", "lon": "longitude"}
    )
    train_csv = convert_string_column(train_csv, "tree/genus/genus", "y").sort_values(
        by=["tree/genus/genus", "internal_id"]
    )
    train_nn = NearestTreeNeighbors(train_csv, 0.002)
    return train_csv, train_nn, base_path


def loader_setup(
    args,
    net,
    main_loader,
    shuffle=True,
    cut="train",
    return_preds=False,
    net2=None,
    radius=0.002,
):
    train_csv, train_nn, base_path = setup(args, cut)

    out = get_features_and_edges(
        train_csv,
        train_nn,
        base_path,
        net,
        cut,
        f"cuda:{args.gpuid}",
        main_loader.transform_test,
        return_preds,
        net2,
        radius,
    )
    if return_preds:
        train_features, train_edges, train_pos, train_ys, train_paths, train_preds = out
    else:
        train_features, train_edges, train_pos, train_ys, train_paths = out

    train_dataset = Data(
        x=train_features,
        edge_index=train_edges.contiguous(),
        pos=train_pos,
        y=train_ys,
        idx=torch.arange(len(train_ys)),
    )

    train_dataset.y = train_dataset.y.long()

    train_dataset.edge_index, train_dataset.edge_attr = T.add_self_loops.add_self_loops(
        train_dataset.edge_index, train_dataset.edge_attr
    )
    transforms = T.Compose([T.Distance(norm=False), T.Cartesian(norm=True)])
    train_dataset = transforms(train_dataset)

    train_loader = NeighborLoader(
        train_dataset,
        [-1],
        batch_size=args.batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )
    if return_preds:
        return train_dataset, train_loader, train_paths, train_preds
    return train_dataset, train_loader, train_paths


def eval_train_gnn(net, criterion, args, epoch, loader):
    train_dataset, train_loader, train_paths = loader_setup(
        args, net, shuffle=False, cut="train", main_loader=loader
    )
    gnn = net.gat
    losses = []
    ys = []
    device = f"cuda:{args.gpuid}"
    with torch.no_grad():
        for idx, batch in enumerate(train_loader):
            pred = gnn(
                batch.x.to(device),
                batch.edge_index.to(device),
                batch.edge_attr.to(device),
            )
            loss = criterion(pred, batch.y.to(device))
            losses.append(loss[: batch.batch_size])
            ys.append(batch.y[: batch.batch_size])

    losses, ys = torch.cat(losses), torch.cat(ys)
    losses = losses.cpu().numpy()
    assert len(losses) == len(ys) == len(train_paths)
    prob = np.zeros(len(ys))
    if epoch < args.class_cond_epoch:
        for y in set(ys.numpy()):
            idx = ys == y
            curr_losses = (losses[idx] - losses[idx].min()) / (
                losses[idx].max() - losses[idx].min()
            )
            curr_losses = curr_losses.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
            gmm.fit(curr_losses)
            curr_prob = gmm.predict_proba(curr_losses)
            prob[idx] = curr_prob[:, gmm.means_.argmin()]
    else:
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        losses = losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=15, reg_covar=5e-4, tol=1e-2)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        prob = prob[:, gmm.means_.argmin()]
    return prob, train_paths


def val_gnn(net, criterion, args, epoch, loader, best_acc, k):
    val_dataset, _, _ = loader_setup(
        args, net, shuffle=False, cut="val", main_loader=loader
    )
    gnn = net.gat
    metrics, _ = test(val_dataset, gnn, criterion, args)

    device = f"cuda:{args.gpuid}"
    net.to(device)

    print("GNN metrics:", metrics)

    acc = metrics["MulticlassAccuracy_micro"]
    if acc > best_acc[k - 1]:
        best_acc[k - 1] = acc
        print("| Saving Best Net%d ..." % k)
        save_point = f"./{args.out_dir}/%s_net%d.pth.tar" % (args.id, k)
        torch.save(net.state_dict(), save_point)

    return acc


def test_with_gnn(net1, net2_with_gnn, args, epoch, loader, test_loader):
    test_dataset, _, _ = loader_setup(
        args, net2_with_gnn, shuffle=False, cut="test", main_loader=loader
    )
    gnn = net2_with_gnn.gat
    gnn.eval()
    device = f"cuda:{args.gpuid}"
    preds_gnn = gnn(
        test_dataset.x.to(device),
        test_dataset.edge_index.to(device),
        test_dataset.edge_attr.to(device),
    )
    torch.cuda.empty_cache()

    preds_reg = []
    with torch.no_grad():
        for idx, (img, y) in enumerate(test_loader):
            preds_reg.append(net1(img.to(f"cuda:{args.gpuid}")).detach())

    preds_reg = torch.cat(preds_reg, dim=0)

    metrics = MetricCollection(
        [
            MetricCollection(
                Accuracy(
                    num_classes=args.num_class, average="macro", task="multiclass"
                ),
                Recall(num_classes=args.num_class, average="macro", task="multiclass"),
                postfix="_macro",
            ),  # type: ignore
            MetricCollection(
                Accuracy(
                    num_classes=args.num_class, average="micro", task="multiclass"
                ),
                postfix="_micro",
            ),
        ]
    )(preds_reg + preds_gnn, test_dataset.y)

    return metrics["MulticlassAccuracy_micro"]


def get_data(data, edge_removal_type):
    data = data.clone()
    data.y = data.y.long()
    data.edge_index, _ = remove_self_loops(data.edge_index)
    del data.edge_attr
    if edge_removal_type == "different":
        data.edge_index = data.edge_index[
            :, data.y[data.edge_index[0]] == data.y[data.edge_index[1]]
        ]
    elif edge_removal_type == "same":
        data.edge_index = data.edge_index[
            :, data.y[data.edge_index[0]] != data.y[data.edge_index[1]]
        ]
    transforms = T.Compose([T.Distance(norm=False), T.Cartesian(norm=True)])
    data = transforms(data)
    return data


def train_gnn(net, args, loader):
    _, train_loader, _ = loader_setup(
        args, net, shuffle=False, cut="train", main_loader=loader
    )

    optimizer = torch.optim.Adam(net.gat.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.5)
    for _ in range(args.gnn_epochs):
        train(train_loader, net.gat, f"cuda:{args.gpuid}", optimizer, criterion)
