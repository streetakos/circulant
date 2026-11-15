"""
corrupt_utils.py — Dataset loading, evaluation metrics, and calibration utilities.

All loaders accept a single ``data_dir`` root that follows this layout::

    data_dir/
    ├── CIFAR10/          # auto-downloaded by torchvision
    ├── CIFAR100/         # auto-downloaded by torchvision
    ├── Imagenet/
    │   ├── train/
    │   └── val/
    ├── TinyImagenet/
    │   ├── train/
    │   └── val/
    │       └── images/
    ├── Places365/
    │   ├── train/
    │   └── val/
    ├── CIFAR-10-C/       # corrupted — .npy files
    ├── CIFAR-100-C/      # corrupted — .npy files
    ├── TinyImagenet-C/   # corrupted — <corruption>/<severity>/
    └── Imagenet-C/       # corrupted — <corruption>/<severity>/

Pass ``data_dir`` to every loader and paths are resolved automatically.
"""

import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor
from torch.utils.data import Subset
from torchvision import datasets
from torch.utils.data import Subset
import torch.utils.data as data
# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_worker(worker_id: int) -> None:
    """Ensures each DataLoader worker uses a fixed seed."""
    np.random.seed(0)
    random.seed(0)


# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _imagenet_transform(train: bool) -> transforms.Compose:
    base = [transforms.Resize(256), transforms.CenterCrop(224)]
    if train:
        base.append(transforms.RandomHorizontalFlip())
    return transforms.Compose([
        *base,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])



# ---------------------------------------------------------------------------
# Main dataset loader
# ---------------------------------------------------------------------------

def load_image_dataset(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    subset: bool = False,
    subset_size: int = 2000,
):
    """Return (trainloader, testloader) for the requested dataset.

    Args:
        dataset_name: One of CIFAR10 | CIFAR100 | TinyImagenet | Imagenet.
        data_dir:     Root data directory.  Each dataset lives at
                      ``<data_dir>/<dataset_name>/``.
        batch_size:   Batch size for both loaders.
        subset:       If True, evaluate on a random subset of the test set.
        subset_size:  Number of test samples when subset=True.
    """
    g = torch.Generator()
    g.manual_seed(0)
    loader_kwargs = dict(num_workers=2, worker_init_fn=seed_worker, generator=g)

    dset_dir = os.path.join(data_dir, dataset_name)


    if dataset_name == "Imagenet":
        
        trainset = datasets.ImageFolder(os.path.join(data_dir, "ImageNet/train"),
                                        _imagenet_transform(train=True))
        testset  = datasets.ImageFolder(os.path.join(data_dir, "ImageNet/val"),
                                        _imagenet_transform(train=False))

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if subset:
        indices = np.random.choice(len(testset), min(subset_size, len(testset)), replace=False)
        testset = Subset(testset, indices)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    return trainloader, testloader


# ---------------------------------------------------------------------------
# Corrupted dataset loader
# ---------------------------------------------------------------------------

_CORRUPTED_TRANSFORMS = {
    "Imagenet": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
}


def load_corrupted_dataset(
    dname: str,
    severity: int,
    data_dir: str = "./data",
    batch_size: int = 256,
    cuda: bool = True,
    workers: int = 4,
    corruption: str = "impulse_noise",
    subset: bool = True,
    subset_size: int = 50,
):
    """Load a corrupted test set.

    Datasets use the same ``<corruption>/<severity>/`` folder layout::

        data_dir/
        └── ImagenetC/
            └── <corruption>/
                └── <severity>/    ← ImageFolder

    Args:
        dname:       Dataset name: CIFAR10 | CIFAR100 | TinyImagenet | Imagenet.
        severity:    Corruption severity level (1–5).
        data_dir:    Root data directory (see layout above).
        batch_size:  DataLoader batch size.
        cuda:        Pin memory when True.
        workers:     Number of DataLoader workers.
        corruption:  Corruption type, e.g. "impulse_noise".
        subset:      Randomly subsample the dataset when True.
        subset_size: Number of samples when subset=True.
    """
    assert dname in ("Imagenet"), \
        f"Unknown dataset: {dname}"

    folder_name = {
        "Imagenet":    "ImageNetC",#"Imagenet-C",
    }[dname]

    c_dir = os.path.join(data_dir, folder_name, corruption, str(severity))
    dataset = datasets.ImageFolder(c_dir, _CORRUPTED_TRANSFORMS[dname])

    if subset:
        indices = np.random.choice(len(dataset), min(subset_size, len(dataset)), replace=False)
        dataset = Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=cuda,
    )


# ---------------------------------------------------------------------------
# Cross-dataset OOD loader
# ---------------------------------------------------------------------------

_STANDARD_TRANSFORMS = {
    "TinyImagenet": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
}

_TORCHVISION_DATASETS = {
    "MNIST":    datasets.MNIST,
    "Fashion":  datasets.FashionMNIST,
    "KMNIST":   datasets.KMNIST,
    "SVHN":     datasets.SVHN,
    "CIFAR10":  datasets.CIFAR10,
    "CIFAR100": datasets.CIFAR100,
}

_SUPPORTED_DATASETS = ["Imagenet", "Places365"]


def cross_load_dataset(
    dname_source: str,
    dname_target: str,
    data_dir: str = "./data",
    batch_size: int = 64,
    cuda: bool = True,
    workers: int = 4,
    subset: bool = True,
    subset_size: int = 50,
):
    """Return (source_loader, target_loader) for OOD evaluation.

    Both source and target datasets are resolved under ``data_dir``::

        data_dir/
        ├── Imagenet/      → val/
        ├── TinyImagenet/  → val/images/
        ├── Places365/     → val/
        └── CIFAR*/        # auto-downloaded

    Args:
        dname_source: In-distribution dataset name.
        dname_target: OOD dataset name.
        data_dir:     Root data directory.
        batch_size:   DataLoader batch size.
        cuda:         Pin memory when True.
        workers:      Number of DataLoader workers.
        subset:       Subsample datasets when True.
        subset_size:  Samples per split when subset=True.
    """
    assert dname_source in _SUPPORTED_DATASETS, f"Unknown source: {dname_source}"
    assert dname_target in _SUPPORTED_DATASETS, f"Unknown target: {dname_target}"

    def _subset(dset):
        if subset and len(dset) > subset_size:
            idx = np.random.choice(len(dset), subset_size, replace=False)
            return Subset(dset, idx)
        return dset

    def _make_loader(dset):
        return torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=cuda,
        )

    def _load_single(dname):
        dset_dir = os.path.join(data_dir, dname)
        if dname in ("Imagenet", "Places365"):
            if dname == "Imagenet":
                return datasets.ImageFolder(
                    os.path.join(data_dir, "ImageNet/val"),
                    _imagenet_transform(train=False),
                )
            else:
                return datasets.ImageFolder(
                    os.path.join(dset_dir, "places365_standard/val"),
                    _imagenet_transform(train=False),
                )
        
    source_loader = _make_loader(_subset(_load_single(dname_source)))
    target_loader = _make_loader(_subset(_load_single(dname_target)))
    return source_loader, target_loader


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def logmeanexp(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """Numerically stable log(mean(exp(x))) along *dim*."""
    xm, _ = torch.max(x, dim=dim, keepdim=True)
    out = xm + torch.log(torch.mean(torch.exp(x - xm), dim=dim, keepdim=True))
    return out if keepdim else out.squeeze(dim)


def get_preds_targets(model, loader, cuda: bool, mc_samples: int, return_vector: bool = False):
    """Single-forward-pass predictions (no MC sampling)."""
    logprob_vec, target_vec = [], []
    for images, target in loader:
        if cuda:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        log_probs = model(images)
        logprob_vec.append(log_probs.data.cpu())
        target_vec.append(target.data.cpu())
    return torch.cat(logprob_vec), torch.cat(target_vec)


def get_preds_targets_ens(model, loader, cuda: bool, mc_samples: int, return_vector: bool = False):
    """MC-ensemble predictions averaged in log-space."""
    logprob_vec, target_vec = [], []
    for inputs, labels in loader:
        if cuda:
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        outputs = torch.zeros(inputs.shape[0], model.num_classes, mc_samples)
        for j in range(mc_samples):
            outputs[:, :, j] = F.log_softmax(model(inputs), dim=1).data.cpu()
        logprob_vec.append(logmeanexp(outputs, dim=2))
        target_vec.append(labels.data.cpu())
    return torch.cat(logprob_vec), torch.cat(target_vec)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def torch_onehot(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    y = y.long()
    y_onehot = torch.zeros(y.shape[0], n_classes, dtype=y.dtype, device=y.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot


def np_get_one_hot(targets: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes)[targets.reshape(-1)].reshape(list(targets.shape) + [n_classes])


def class_err(y: torch.Tensor, model_out: torch.Tensor) -> float:
    pred = model_out.max(dim=1)[1]
    return pred.ne(y.data).sum().item() / y.shape[0]


def class_ll(y: torch.Tensor, log_probs=None, probs=None, eps: float = 1e-40) -> torch.Tensor:
    assert (log_probs is None) != (probs is None), "Provide exactly one of log_probs or probs."
    if probs is not None:
        log_probs = torch.log(probs.clamp(min=eps))
    return -F.nll_loss(log_probs, y, reduction="mean")


def class_brier(y: torch.Tensor, log_probs=None, probs=None) -> torch.Tensor:
    assert (log_probs is None) != (probs is None), "Provide exactly one of log_probs or probs."
    if log_probs is not None:
        probs = log_probs.exp()
    if y.dim() > 1:
        y = y.squeeze(1)
    y_oh = torch_onehot(y, probs.shape[1]).float()
    return (probs - y_oh).pow(2).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def callibration_bin_probs(
    target_cdf: np.ndarray, n_bins: int, cummulative: bool = False
):
    uniform_bins = np.linspace(0, 1, n_bins + 1)
    uniform_idxs = np.digitize(target_cdf, uniform_bins, right=False)
    bin_counts = np.zeros(n_bins)
    for idx in range(n_bins + 1):
        if idx == n_bins:
            bin_counts[idx - 1] += np.sum(uniform_idxs == idx + 1)
        else:
            bin_counts[idx] = np.sum(uniform_idxs == idx + 1)
    assert bin_counts.sum() == target_cdf.shape[0]
    bin_prop = bin_counts / target_cdf.shape[0]
    if cummulative:
        bin_prop = np.cumsum(bin_prop)
    return bin_prop, bin_counts, uniform_bins


def gauss_callibration(pred_means, pred_stds, targets, n_bins, cummulative=False, two_sided=False):
    if two_sided:
        uniform_pred_err = norm.cdf((pred_means - targets) / pred_stds)
    else:
        uniform_pred_err = norm.cdf(np.abs(pred_means - targets) / pred_stds) * 2 - 1
    bin_prop, bin_counts, uniform_bins = callibration_bin_probs(uniform_pred_err, n_bins, cummulative)
    bin_centers = uniform_bins[1:] - 0.5 / n_bins
    bin_width = 1 / n_bins
    reference = (np.ones(len(bin_centers)) * bin_width if not cummulative
                 else np.arange(0, 1 + 1 / len(bin_centers), 1 / len(bin_centers)))
    return bin_prop, bin_centers, bin_width, bin_counts, reference


def cat_callibration(probs: np.ndarray, y_test: np.ndarray, n_bins: int, top_k=None):
    pred_class = np.argmax(probs, axis=1)
    pred_class_OH    = np_get_one_hot(pred_class, probs.shape[1])
    targets_class_OH = np_get_one_hot(y_test.reshape(-1).astype(int), probs.shape[1])

    if top_k is not None:
        top_k_idx = probs.argsort(axis=1)[:, -top_k:]
        probs            = np.concatenate([probs[r, top_k_idx[r]]            for r in range(len(top_k_idx))])
        targets_class_OH = np.concatenate([targets_class_OH[r, top_k_idx[r]] for r in range(len(top_k_idx))])
        pred_class_OH    = np.concatenate([pred_class_OH[r, top_k_idx[r]]    for r in range(len(top_k_idx))])

    expanded_preds        = probs.reshape(-1)
    pred_class_OH_expand  = pred_class_OH.reshape(-1)
    targets_OH_expand     = targets_class_OH.reshape(-1)
    correct_vec = (targets_OH_expand * (pred_class_OH_expand == targets_OH_expand)).astype(int)

    bin_limits  = np.linspace(0, 1, n_bins + 1)
    bin_step    = bin_limits[1] - bin_limits[0]
    bin_centers = bin_limits[:-1] + bin_step / 2
    bin_idxs    = np.digitize(expanded_preds, bin_limits, right=False) - 1

    bin_counts   = np.ones(n_bins)
    bin_corrects = np.zeros(n_bins)
    for b in range(n_bins + 1):
        if b == n_bins:
            bin_counts[b - 1]   += np.sum(bin_idxs == b)
            bin_corrects[b - 1] += np.sum(correct_vec[bin_idxs == b])
        else:
            bin_counts[b]   = np.sum(bin_idxs == b)
            bin_corrects[b] = np.sum(correct_vec[bin_idxs == b])

    bin_probs = bin_corrects / bin_counts
    bin_probs[bin_counts == 0] = 0

    reference = np.array([expanded_preds[bin_idxs == b].mean() for b in range(n_bins)])
    reference[bin_counts == 0] = 0
    return bin_probs, bin_centers, bin_step, bin_counts, reference


def expected_callibration_error(bin_probs, reference, bin_counts, tail: bool = False) -> float:
    bin_abs_error = np.abs(bin_probs - reference)
    if tail:
        tail_count = bin_counts[0] + bin_counts[-1]
        return (bin_abs_error[0] * bin_counts[0] + bin_abs_error[-1] * bin_counts[-1]) / tail_count
    return (bin_abs_error * bin_counts / bin_counts.sum()).sum()


def class_ECE(y: torch.Tensor, log_probs=None, probs=None, nbins: int = 10, top_k: int = 1) -> float:
    assert (log_probs is None) != (probs is None), "Provide exactly one of log_probs or probs."
    if log_probs is not None:
        probs = log_probs.exp()
    probs = probs.clamp(max=1 - 1e-8)
    bin_probs, _, _, bin_counts, reference = cat_callibration(
        probs.cpu().numpy(), y.cpu().numpy(), nbins, top_k=top_k
    )
    return expected_callibration_error(bin_probs, reference, bin_counts)


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def entropy_from_logprobs(log_probs: torch.Tensor) -> torch.Tensor:
    return -(log_probs.exp() * log_probs).sum(dim=1)


def entropy_from_logprobs_numpy(log_probs: np.ndarray) -> np.ndarray:
    return -(np.exp(log_probs) * log_probs).sum(axis=1)


def categorical_entropy(p: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return -torch.sum(p * torch.log(p + eps), dim=1)


# ---------------------------------------------------------------------------
# Selective prediction & AUROC
# ---------------------------------------------------------------------------

def selective_accuracy(p: np.ndarray, y: np.ndarray) -> float:
    """Area under the selective-prediction accuracy curve."""
    thresholds = np.concatenate([np.linspace(100, 1, 100), [0.1]])
    predictions = p.argmax(-1)
    accuracies  = predictions == y
    scores      = entropy_from_logprobs_numpy(p)

    thresholded_accuracies = []
    for t in thresholds:
        mask = scores <= np.percentile(scores, t)
        thresholded_accuracies.append(np.mean(accuracies[mask]))
    values = np.array(thresholded_accuracies)

    auc_sel = 0.0
    for i in range(len(thresholds) - 1):
        x = (100 - thresholds[i + 1]) if i == 0 else (thresholds[i] - thresholds[i + 1])
        auc_sel += x * (values[i] + values[i + 1]) / 2
    return auc_sel


def selective_accuracy_test_ood(p_id: np.ndarray, p_ood: np.ndarray, y: np.ndarray) -> float:
    """Selective accuracy AUC combining in-distribution and OOD samples."""
    thresholds  = np.concatenate([np.linspace(100, 1, 100), [0.1]])
    accuracies  = np.concatenate([p_id.argmax(-1) == y, np.zeros(p_ood.shape[0])])
    scores      = np.concatenate([entropy_from_logprobs_numpy(p_id),
                                  entropy_from_logprobs_numpy(p_ood)])

    thresholded_accuracies = []
    for t in thresholds:
        mask = scores <= np.percentile(scores, t)
        thresholded_accuracies.append(np.mean(accuracies[mask]))
    values = np.array(thresholded_accuracies)

    auc_sel = 0.0
    for i in range(len(thresholds) - 1):
        x = (100 - thresholds[i + 1]) if i == 0 else (thresholds[i] - thresholds[i + 1])
        auc_sel += x * (values[i] + values[i + 1]) / 2
    return auc_sel


def auroc_logits(
    predicted_logits_test: torch.Tensor,
    predicted_logits_ood:  torch.Tensor,
    score: str = "expected entropy",
) -> float:
    """AUROC score separating in-distribution from OOD samples."""
    test_size = predicted_logits_test.shape[0]
    ood_size  = predicted_logits_ood.shape[0]
    anomaly_targets = np.concatenate([np.zeros(test_size), np.ones(ood_size)])

    if score == "expected entropy":
        scores = torch.cat([
            entropy_from_logprobs(predicted_logits_test),
            entropy_from_logprobs(predicted_logits_ood),
        ]).numpy()
    else:
        raise NotImplementedError(f"Score '{score}' is not implemented.")

    return roc_auc_score(anomaly_targets, scores)
