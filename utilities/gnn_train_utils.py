import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torchmetrics import MetricCollection, Accuracy, Recall, Precision
from torch_geometric.nn.models import GAT


class NegEntropyLoss(nn.Module):
    def __init__(self):
        """Negative Entropy Regularization.
        Parameters
        * `num_classes` Number of classes in the classification problem.
        """

        super(NegEntropyLoss, self).__init__()

    def forward(self, x):
        r"""Negative Entropy Regularization.
        Args
        * `x` Model's logits, same as PyTorch provided loss functions.
        """
        y_pred = F.softmax(x, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        return -1 * (y_pred * (y_pred + 1e-4).log()).sum(dim=1).mean()


class ELRLoss(nn.Module):
    def __init__(
        self,
        num_examp,
        num_classes=10,
        lam=3,
        beta=0.7,
        label_smoothing=0,
        device="cpu",
        init_target=None,
    ):
        """Early Learning Regularization.
        Parameters
        * `num_examp` Total number of training examples.
        * `num_classes` Number of classes in the classification problem.
        * `lam` Regularization strength; must be a positive float, controling the strength of the ELR.
        * `beta` Temporal ensembling momentum for target estimation.
        """

        super(ELRLoss, self).__init__()
        self.num_classes = num_classes
        if init_target is not None:
            self.target = init_target.to(device)
        else:
            self.target = torch.zeros(num_examp, self.num_classes).to(device)
        self.beta = beta
        self.lam = lam
        self.label_smoothing = label_smoothing
        self.neg_entropy = NegEntropyLoss()

    def forward(self, index, output, label):
        r"""Early Learning Regularization.
        Args
        * `index` Training sample index, due to training set shuffling, index is used to track training examples in different iterations.
        * `output` Model's logits, same as PyTorch provided loss functions.
        * `label` Labels, same as PyTorch provided loss functions.
        """

        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.detach()
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * (
            (y_pred_) / (y_pred_).sum(dim=1, keepdim=True)
        )
        ce_loss = F.cross_entropy(output, label, label_smoothing=self.label_smoothing)
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.lam * elr_reg
        return final_loss



def train(loader, model, device, optimizer, criterion, criterion2):
    model.train()
    model = model.to(device)
    loss = 0
    idx = 0
    for batch in loader:
        # print(batch)
        optimizer.zero_grad()  # Clear gradients.
        # noise
        x = (batch.x).to(device)
        
        out, out_lm, x_enc = model(
            x, batch.edge_index.to(device), batch.edge_attr.to(device)
        )  # Perform a single forward pass.
        curr_loss = criterion(
            batch.idx[: batch.batch_size].to(device),
            out[: batch.batch_size],
            batch.y[: batch.batch_size].to(device),
        )  # Compute the loss solely based on the training nodes.
        loss += curr_loss.item()
        idx += 1
        curr_loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
    return loss / idx


def test(val_data, model, criterion, args):
    model.eval()
    model.cpu()
    preds, _, _ = model(val_data.x, val_data.edge_index, val_data.edge_attr)
    loss = F.cross_entropy(preds, val_data.y)
    metrics = MetricCollection(
        [
            MetricCollection(
                Accuracy(
                    num_classes=args.num_class, average="macro", task="multiclass"
                ),
                Recall(num_classes=args.num_class, average="macro", task="multiclass"),
                Precision(num_classes=args.num_class, average="macro", task="multiclass"),
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


def weighted_sampler(data):
    cls_counts = torch.bincount(data.y)
    filt = cls_counts == 2
    cls_weights = 1/cls_counts**.66
    cls_weights[filt] = 0
    sample_weights = cls_weights[data.y]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def test_bias_corrected_ensemble(
    test_data, baseline_preds, gnn, metrics, num_class, debias_lambda=0.5
):
    gnn.eval()
    gnn_preds, _, _ = gnn(test_data.x, test_data.edge_index)
    gnn_preds = F.normalize(gnn_preds, dim=1)
    gnn_pred_dist = gnn_preds.mean(dim=0).unsqueeze(0)
    gnn_pred_dist = F.normalize(gnn_pred_dist, dim=1)
    baseline_preds = F.normalize(baseline_preds, dim=1)
    
    return metrics(
        gnn_preds - debias_lambda * gnn_pred_dist, 
        test_data.y,
    )


def test_lm(model, device, dataset, criterion, metrics):
    model.eval()
    model = model.to(device)
    preds = model(dataset.x.to(device))
    loss = criterion(preds, dataset.y.to(device))
    return metrics(preds, dataset.y), loss.item()


class CombinedGNNLinear(nn.Module):
    def __init__(self, num_features, num_classes, skip_prob=0.7):
        super(CombinedGNNLinear, self).__init__()
        self.MLP = nn.Sequential(
            nn.Identity(),
        )
        self.skip_prob = skip_prob
        
        self.gnn = GAT(
            in_channels=num_features,
            hidden_channels=1280,
            out_channels=num_classes,
            num_layers=1,
            v2=True,
        )
        
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes),
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        x_enc = self.MLP(x)
        out_lm = self.linear(x_enc)
        if self.training:
            out_gnn = self.gnn(x, edge_index, edge_attr)
        else:
            mask = (torch.rand(x_enc.size(0)) < self.skip_prob).to(x_enc.device)
            x_enc[mask] = x[mask]
            out_gnn = self.gnn(x_enc, edge_index, edge_attr)
        return out_gnn, out_lm, x_enc
