"""Pooled MLP and attention-based MIL, trained using site-level labels only."""
import copy
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from metrics import pr_auc_score


class PooledMLP(nn.Module):
    def __init__(self, hidden=(64,32), dropout=.2):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(74, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


class AttentionMIL(nn.Module):
    def __init__(self, width=32, attention_width=16, dropout=.2):
        super().__init__()
        # Each read is encoded independently using shared neural weights.
        self.encoder = nn.Sequential(nn.Linear(9, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(width, attention_width), nn.Tanh(), nn.Linear(attention_width, 1))
        self.classifier = nn.Sequential(nn.Linear(width + 28, width), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, reads, mask, sequence):
        h = self.encoder(reads)                 # batch x reads x 32
        logits = self.attention(h).squeeze(-1)  # batch x reads
        weights = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1)  # batch x 32
        return self.classifier(torch.cat([pooled, sequence], dim=1)).squeeze(-1)


def build(kind, config=None):
    if kind not in ('mlp', 'mil'):
        raise ValueError(f'Unknown neural model {kind}')
    return PooledMLP(**(config or {})) if kind == 'mlp' else AttentionMIL(**(config or {}))


def fit_normalizer(kind, arrays, fit):
    if kind == 'mlp':
        values = arrays[0][fit].astype(np.float64)
        mean, scale = values.mean(0), values.std(0)
    else:
        # Accumulate only real reads from fitting genes, without a large copy.
        total = np.zeros(9)
        squares = np.zeros(9)
        count = 0
        for index in fit:
            values = arrays[0][index, arrays[1][index]].astype(np.float64)
            total += values.sum(0)
            squares += (values ** 2).sum(0)
            count += len(values)
        mean = total / count
        scale = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
    scale[scale < 1e-8] = 1.
    return mean.astype(np.float32), scale.astype(np.float32)


def tensors(kind, arrays, indices, mean, scale):
    signals = (arrays[0][indices] - mean) / scale
    if kind == 'mlp':
        return (torch.from_numpy(signals.astype(np.float32)),)
    return (torch.from_numpy(signals.astype(np.float32)),
            torch.from_numpy(arrays[1][indices]), torch.from_numpy(arrays[2][indices]))


def probabilities(model, kind, arrays, indices, mean, scale, batch_size=256):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            output.append(torch.sigmoid(model(*tensors(kind, arrays, batch, mean, scale))).numpy())
    return np.concatenate(output) if output else np.empty(0)


def train_network(kind, arrays, y, fit, validation, seed, epochs, batch_size, positive_weight=1., selection_metric="pr_auc", learning_rate=1e-3, patience=0, config=None, weight_decay=1e-4):
    if selection_metric not in ("pr_auc", "ap"):
        raise ValueError("Selection metric must be pr_auc or ap")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or patience < 0 or not np.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError('Invalid neural training settings')
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    model = build(kind, config)
    mean, scale = fit_normalizer(kind, arrays, fit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Weight is derived only from fitting labels; validation remains unweighted.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(positive_weight)))
    best_score, best_state, best_epoch = -1., None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = rng.permutation(fit)
        total_loss = 0.
        for start in range(0, len(permutation), batch_size):
            batch = permutation[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(*tensors(kind, arrays, batch, mean, scale))
            loss = criterion(logits, torch.tensor(y[batch], dtype=torch.float32))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        score = probabilities(model, kind, arrays, validation, mean, scale, batch_size)
        ap = float(average_precision_score(y[validation], score))
        pr_auc = pr_auc_score(y[validation], score)
        value = pr_auc if selection_metric == 'pr_auc' else ap
        record = dict(epoch=epoch, train_loss=total_loss / len(fit), validation_ap=ap, validation_pr_auc=pr_auc)
        history.append(record)
        print(record, flush=True)
        if value > best_score:
            best_score, best_epoch, best_state = value, epoch, copy.deepcopy(model.state_dict())
        if patience and epoch - best_epoch >= patience:
            print(f'Early stopping at epoch {epoch}; best epoch {best_epoch}', flush=True)
            break
    model.load_state_dict(best_state)
    return model, mean, scale, history, best_epoch


def restore(artifact):
    model = build(artifact['kind'], artifact.get('network_config'))
    model.load_state_dict({k: torch.from_numpy(v) for k, v in artifact['state'].items()})
    model.eval()
    return model
