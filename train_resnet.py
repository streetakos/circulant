"""
ResNet training script with Bayesian / structured covariance layers.
All hyperparameters are controlled via the Config dataclass below.
"""

import os
import sys
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn.modules.dropout import _DropoutNd
from torchvision import models
import argparse
sys.path.append("..")

from corrupt_utils import (
    auroc_logits,
    class_ECE,
    class_brier,
    class_err,
    class_ll,
    cross_load_dataset,
    get_preds_targets_ens,
    load_corrupted_dataset,
    load_image_dataset,
    selective_accuracy,
)
from layers import (
    Conv2dCirculant, Conv2dCirculantFFT,
    Conv2dDiagonal, Conv2dELRG,
    Conv2dKtied, Conv2dRank1,
    Conv2dRank1Circulant,
    LinearCirculant, LinearCirculantFFT,
    LinearDiagonal, LinearELRG,
    LinearKTied, LinearRank1,
    LinearRank1Circulant,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ── Model ──────────────────────────────────────────────────────────────
    cov_type: str = "Deterministic"
    """
    Weight-distribution type.
    Options: Deterministic | Dropout | Circulant | CirculantFFT |
             Diagonal | Ktied | ELRG | RANK1 | RANK1Circulant
    """
    resnet_depth: int = 20          # 20 → BasicBlock, 50 → Bottleneck
    num_classes: int = 1000

    # ── Structured-covariance hyper-params ─────────────────────────────────
    circ_size: int = 20             # Circulant kernel size
    init_scheme: str = "Xavier"     # Xavier | Scaled | 2xScaled
    rank: int = 1                   # ELRG / rank-1 rank
    k_tied: int = 1                 # K-tied number of components
    init_log_var: float = -2.5      # Initial log-variance for Ktied

    # ── KL / ELBO ──────────────────────────────────────────────────────────
    kl_weight: float = 1e-4
    kl_anneal: str = "Scaled"       # Scaled | Linear
    kl_anneal_warm_up: int = 80     # Epochs for linear warm-up

    # ── Training ───────────────────────────────────────────────────────────
    dataset: str = "Imagenet"       # Imagenet | CIFAR100 | TinyImagenet
    data_dir: str = "./data"
    batch_size: int = 16
    num_epochs: int = 150
    max_steps: int = 230_000
    log_steps: int = 5_000

    optimizer: str = "Adam"         # Adam | SGD
    lr: float = 1e-3
    momentum: float = 0.9           # SGD only
    weight_decay: float = 1e-4      # SGD only

    lr_schedule: str = "MultiStepLR"   # MultiStepLR | CosineAnnealingLR | Plain
    milestones: List[int] = field(default_factory=lambda: [80, 120, 160, 200])
    gamma: float = 0.1

    # ── Inference ──────────────────────────────────────────────────────────
    mc_samples_test: int = 2       # Forced to 1 for Deterministic

    # ── I/O ────────────────────────────────────────────────────────────────
    data_dir: str = "./data"
    """Root data directory. Each dataset lives at <data_dir>/<DatasetName>/."""
    save_dir: str = "./checkpoints"
    torch_cache_dir: str = "./torch_cache"  # Override via TORCH_HOME env var

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 20

    # ── GPU ────────────────────────────────────────────────────────────────
    device: str = "cuda:0"

    # ── Corruption / OOD evaluation ────────────────────────────────────────
    corruption_level: int = 1
    ood_dataset: str = "Places365"

    # ── Derived (set automatically) ────────────────────────────────────────
    def __post_init__(self):
        if self.cov_type == "Deterministic":
            self.mc_samples_test = 1

        self.run_name = (
            f"_{self.cov_type}_SEED_{self.seed}"
            f"_KL_{self.kl_weight}_{self.dataset.lower()}_resnet{self.resnet_depth}"
        )
        self.checkpoint_dir = os.path.join(
            self.save_dir, self.dataset + self.run_name
        )


# ---------------------------------------------------------------------------
# Layer factories  (dispatch on cfg.cov_type)
# ---------------------------------------------------------------------------

def make_conv2d_3x3(
    cfg: Config,
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Module:
    common = dict(
        kernel_size=3, stride=stride, padding=dilation,
        groups=groups, bias=True, dilation=dilation,
    )
    ct = cfg.cov_type
    if ct == "Circulant":
        return Conv2dCirculant(in_planes, out_planes,
                               circ_kernel_size=cfg.circ_size,
                               init_scheme=cfg.init_scheme, **common)
    if ct == "CirculantFFT":
        return Conv2dCirculantFFT(in_planes, out_planes,
                                  circ_kernel_size=cfg.circ_size, **common)
    if ct == "Diagonal":
        return Conv2dDiagonal(in_planes, out_planes, **common)
    if ct == "Ktied":
        return Conv2dKtied(in_planes, out_planes, k=cfg.k_tied,
                           init_log_var=cfg.init_log_var, **common)
    if ct in ("Deterministic", "Dropout"):
        return nn.Conv2d(in_planes, out_planes, **common)
    if ct == "ELRG":
        return Conv2dELRG(in_planes, out_planes, rank=cfg.rank, alpha=0.2, **common)
    if ct == "RANK1":
        return Conv2dRank1(in_planes, out_planes, components=7, **common)
    if ct == "RANK1Circulant":
        return Conv2dRank1Circulant(in_planes, out_planes, components=6,
                                    circ_kernel_size=cfg.circ_size,
                                    init_scheme=cfg.init_scheme, **common)
    raise ValueError(f"Unknown cov_type: {ct}")


def make_conv2d_1x1(
    cfg: Config, in_planes: int, out_planes: int, stride: int = 1
) -> nn.Module:
    ct = cfg.cov_type
    if ct == "Circulant":
        return Conv2dCirculant(in_planes, out_planes, kernel_size=1, stride=stride,
                               circ_kernel_size=cfg.circ_size,
                               init_scheme=cfg.init_scheme, bias=True)
    if ct == "CirculantFFT":
        return Conv2dCirculantFFT(in_planes, out_planes, kernel_size=1, stride=stride,
                                  circ_kernel_size=cfg.circ_size, bias=True)
    if ct == "Diagonal":
        return Conv2dDiagonal(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)
    if ct == "Ktied":
        return Conv2dKtied(in_planes, out_planes, kernel_size=1, stride=stride,
                           k=cfg.k_tied, init_log_var=cfg.init_log_var, bias=True)
    if ct in ("Deterministic", "Dropout"):
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)
    if ct == "ELRG":
        return Conv2dELRG(in_planes, out_planes, kernel_size=1, stride=stride,
                          rank=cfg.rank, alpha=0.2, bias=True)
    if ct == "RANK1":
        return Conv2dRank1(in_planes, out_planes, kernel_size=1, stride=stride,
                           components=7, bias=True)
    if ct == "RANK1Circulant":
        return Conv2dRank1Circulant(in_planes, out_planes, kernel_size=1, stride=stride,
                                    components=6, circ_kernel_size=cfg.circ_size,
                                    init_scheme=cfg.init_scheme, bias=True)
    raise ValueError(f"Unknown cov_type: {ct}")


def make_linear(cfg: Config, in_features: int, out_features: int) -> nn.Module:
    ct = cfg.cov_type
    if ct == "Circulant":
        return LinearCirculant(in_features, out_features,
                               kernel_size=cfg.circ_size,
                               init_scheme=cfg.init_scheme, bias=True)
    if ct == "CirculantFFT":
        return LinearCirculantFFT(in_features, out_features,
                                  kernel_size=cfg.circ_size, bias=True)
    if ct == "Diagonal":
        return LinearDiagonal(in_features, out_features, bias=True)
    if ct == "Ktied":
        return LinearKTied(in_features, out_features, k=cfg.k_tied,
                           init_log_var=cfg.init_log_var, bias=True)
    if ct in ("Deterministic", "Dropout"):
        return nn.Linear(in_features, out_features, bias=True)
    if ct == "ELRG":
        return LinearELRG(in_features, out_features, bias=True,
                          rank=cfg.rank, alpha=0.2)
    if ct == "RANK1":
        return LinearRank1(in_features, out_features, bias=True, components=7)
    if ct == "RANK1Circulant":
        return LinearRank1Circulant(in_features, out_features,
                                    kernel_size=cfg.circ_size,
                                    init_scheme=cfg.init_scheme,
                                    components=6, bias=True)
    raise ValueError(f"Unknown cov_type: {ct}")


# ---------------------------------------------------------------------------
# MC-Dropout helper
# ---------------------------------------------------------------------------

class MC_Dropout2d(_DropoutNd):
    """Always-on spatial dropout (active at eval time too)."""
    def forward(self, x: Tensor) -> Tensor:
        return F.dropout2d(x, self.p, training=True, inplace=self.inplace)


# ---------------------------------------------------------------------------
# ResNet building blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        cfg: Config,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        norm_layer = norm_layer or nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = make_conv2d_3x3(cfg, inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = make_conv2d_3x3(cfg, planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride
        self.drop = MC_Dropout2d(p=0.1, inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.cfg.cov_type == "Dropout":
            out = self.drop(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        cfg: Config,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = make_conv2d_1x1(cfg, inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = make_conv2d_3x3(cfg, width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = make_conv2d_1x1(cfg, width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu3(out + identity)
        return out


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------

class ResNet(nn.Module):

    def __init__(
        self,
        cfg: Config,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self._norm_layer = norm_layer or nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1

        self.conv1 = make_conv2d_3x3(cfg, 3, self.inplanes)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = make_linear(cfg, 512 * block.expansion, cfg.num_classes)

        self._init_weights()
        self.num_classes  = cfg.num_classes

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (Conv2dCirculant, Conv2dDiagonal, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                make_conv2d_1x1(self.cfg, self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        block_kwargs = dict(
            cfg=self.cfg,
            groups=1,
            base_width=64,
            dilation=self.dilation,
            norm_layer=norm_layer,
        )

        layers = [block(inplanes=self.inplanes, planes=planes,
                        stride=stride, downsample=downsample, **block_kwargs)]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes, **block_kwargs))

        return nn.Sequential(*layers)

    def get_kl(self) -> Tensor:
        """Sum KL divergences from all Bayesian layers."""
        kl = torch.tensor(0.0)
        for module in self.modules():
            if hasattr(module, "kl_reg"):
                kl = kl + module.kl_reg()
        return kl

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_resnet(cfg: Config) -> ResNet:

    

    """Return a ResNet-20 (BasicBlock) or ResNet-50 (Bottleneck)."""
    if cfg.resnet_depth == 20:
        return ResNet(cfg, BasicBlock, [2, 2, 2, 2])
    if cfg.resnet_depth == 50:
        return ResNet(cfg, Bottleneck, [3, 4, 6, 3])
    raise ValueError(f"Unsupported resnet_depth: {cfg.resnet_depth}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_pretrained_imagenet(model: ResNet, cfg: Config) -> None:
    """Download and load ImageNet-pretrained ResNet-18 weights into *model* (best-effort).

    Weights are cached under ``cfg.torch_cache_dir`` (default: ``./torch_cache``).
    Override by setting the ``TORCH_HOME`` environment variable before launching.
    """
    

   
    # Use the modern weights API (replaces 'pretrained=True')

    
    cache_dir = os.environ.get("TORCH_HOME", cfg.torch_cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCH_HOME"] = cache_dir

    
    # Force PyTorch to use this directory for models and hub downloads
    os.environ['TORCH_HOME'] = cache_dir
    os.environ['XDG_CACHE_HOME'] = cache_dir # Extra safety for some environments
    # Use the modern weights API (replaces 'pretrained=True')
    
    weights_path = os.path.join(cache_dir, "resnet18_imagenet1k_v1.pth")
    if not os.path.exists(weights_path):
        pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        torch.save(pretrained.state_dict(), weights_path)

    # Temporarily swap layers to standard Conv2d / Linear for weight loading
    model.fc = nn.Linear(512, cfg.num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.load_state_dict(torch.load(weights_path), strict=False)
    print(f"ImageNet weights loaded from {weights_path}.")


def compute_kl_weight(cfg: Config, current_step: int) -> float:
    if cfg.kl_anneal == "Linear":
        return min(1.0, current_step / cfg.kl_anneal_warm_up) * cfg.kl_weight
    # "Scaled" – fixed weight, scaling handled by NLL coefficient
    return cfg.kl_weight


def evaluate(
    model: nn.Module,
    loader,
    cfg: Config,
    device: torch.device,
) -> dict:
    model.eval()
    with torch.no_grad():
        logprobs, targets = get_preds_targets_ens(
            model, loader, True, cfg.mc_samples_test, return_vector=False
        )
    _, predicted = torch.max(logprobs, 1)
    accuracy = 100.0 * (predicted == targets).sum().item() / targets.size(0)
    return dict(
        accuracy=accuracy,
        error=class_err(targets, logprobs),
        ll=class_ll(targets, log_probs=logprobs).item(),
        brier=class_brier(targets, log_probs=logprobs).item(),
        ece=class_ECE(targets, log_probs=logprobs),
        selective=selective_accuracy(logprobs.cpu().numpy(), targets.cpu().numpy()),
        logprobs=logprobs,
        targets=targets,
    )


def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model, path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_resnet(cfg).to(device)

    if cfg.dataset == "Imagenet":
        load_pretrained_imagenet(model, cfg)
        model.to(device)

    # ── Loss / Optimiser / Scheduler ───────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    if cfg.lr_schedule == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs
        )
    elif cfg.lr_schedule == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.milestones, gamma=cfg.gamma
        )
    else:
        scheduler = None

    trainloader, testloader = load_image_dataset(
        cfg.dataset, cfg.data_dir, cfg.batch_size, subset = True
    )

    # ── Training loop ──────────────────────────────────────────────────────
    is_bayesian = cfg.cov_type not in ("Deterministic", "Dropout")
    best_error = float("inf")
    best_ll = float("-inf")
    training_steps = 0

    print(f"\nStarting training | cov_type={cfg.cov_type} | dataset={cfg.dataset}\n")

    for epoch in range(cfg.num_epochs):
        model.train()

        for batch in trainloader:
            training_steps += 1
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            nll = criterion(outputs, labels)

            if is_bayesian:
                kl = model.get_kl()
                kl_w = compute_kl_weight(cfg, epoch)
                loss = nll * 50_000 + kl_w * kl 
            else:
                loss = nll
                kl = torch.tensor(0.0)

            loss.backward()
            optimizer.step()

            # ── Periodic logging & checkpoint ──────────────────────────────
            if training_steps % cfg.log_steps == 0:
                metrics = evaluate(model, testloader, cfg, device)
                print(
                    f"Step {training_steps:>7d} | "
                    f"Acc {metrics['accuracy']:.2f}% | "
                    f"LL {metrics['ll']:.4f} | "
                    f"Err {metrics['error']:.4f} | "
                    f"Brier {metrics['brier']:.4f} | "
                    f"ECE {metrics['ece']:.4f} | "
                    f"NLL {nll.item():.4f}" +
                    (f" | KL*w {(kl_w * kl).item():.4f}" if is_bayesian else "")
                )

                # Step checkpoint
                save_checkpoint(
                    model,
                    os.path.join(cfg.checkpoint_dir, f"step_{training_steps}.pt"),
                )

                # Best-error checkpoint
                if metrics["error"] < best_error:
                    best_error = metrics["error"]
                    save_checkpoint(
                        model, os.path.join(cfg.checkpoint_dir, "best_error.pt")
                    )

                # Best-LL checkpoint
                if metrics["ll"] > best_ll:
                    best_ll = metrics["ll"]
                    save_checkpoint(
                        model, os.path.join(cfg.checkpoint_dir, "best_ll.pt")
                    )

                model.train()
                

            if training_steps >= cfg.max_steps:
                break

        if scheduler is not None:
            scheduler.step()

        if training_steps >= cfg.max_steps:
            break

    print("Training complete.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoints(cfg: Config) -> None:
    device = torch.device(cfg.device)
    checkpoints = [
        os.path.join(cfg.checkpoint_dir, "best_ll.pt"),
        os.path.join(cfg.checkpoint_dir, "best_error.pt"),
    ]

    for ckpt_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_path}")
        print(f"{'='*60}")
        model = torch.load(ckpt_path).to(device)

        # ── In-distribution ────────────────────────────────────────────────
        _, testloader = load_image_dataset(
            cfg.dataset, cfg.data_dir, cfg.batch_size, subset=False, subset_size=2000
        )
        m = evaluate(model, testloader, cfg, device)
        print(
            f"[In-dist]  Acc={m['accuracy']:.2f}%  LL={m['ll']:.4f}  "
            f"Err={m['error']:.4f}  Brier={m['brier']:.4f}  "
            f"ECE={m['ece']:.4f}  Sel={m['selective']:.4f}"
        )

        # ── Corrupted ──────────────────────────────────────────────────────
        corrupt_loader = load_corrupted_dataset(
            cfg.dataset, cfg.corruption_level,
            data_dir=cfg.data_dir, batch_size=cfg.batch_size, cuda=True, workers=2, subset=False,subset_size=2000
        )
        m_c = evaluate(model, corrupt_loader, cfg, device)
        print(
            f"[Corrupted] Acc={m_c['accuracy']:.2f}%  LL={m_c['ll']:.4f}  "
            f"Err={m_c['error']:.4f}  Brier={m_c['brier']:.4f}  "
            f"ECE={m_c['ece']:.4f}  Sel={m_c['selective']:.4f}"
        )

        # ── OOD / AUROC ────────────────────────────────────────────────────
        source_loader, target_loader = cross_load_dataset(
            cfg.dataset, cfg.ood_dataset,
            data_dir=cfg.data_dir, batch_size=cfg.batch_size,
            cuda=True, workers=1, subset=False,subset_size = 2000
        )
        model.eval()
        with torch.no_grad():
            lp_in, _ = get_preds_targets_ens(
                model, source_loader, True, cfg.mc_samples_test, return_vector=False
            )
            lp_out, _ = get_preds_targets_ens(
                model, target_loader, True, cfg.mc_samples_test, return_vector=False
            )
        auroc = auroc_logits(lp_in.cpu(), lp_out.cpu(), score="expected entropy")
        print(f"[OOD AUROC vs {cfg.ood_dataset}] {auroc:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run model with custom configuration.")
    
    # 2. Add the arguments
    parser.add_argument("--cov_type",  type=str,  default="Diagonal",  help="Type of covariance (default: Diagonal)" )
    parser.add_argument("--do_eval",   action="store_true",  help="Enable evaluation mode (default: False)" )
    parser.add_argument("--seed",      type=int,       default=60,  help="Random seed for reproducibility (default: 60)" )    

    # 3. Parse the arguments from the command line
    args = parser.parse_args()


    cfg = Config(
        #cov_type="Deterministic",
        cov_type      = args.cov_type,  #,Ktied, ELRG Circulant, Diagonal, Deterministic 
        init_scheme   = "Xavier", 
        resnet_depth=20,
        num_classes=1000,
        dataset="Imagenet",
        data_dir="/netdisk/data/CV",   
        batch_size=16,
        num_epochs=150,
        lr=0.0001,
        kl_weight=0.0001, 
        seed=args.seed, #10,50 20 ||| 
        device="cuda:0",
    )

    print("Config:", cfg)

    train(cfg)

    # Use 10 MC samples for the final evaluation
    if cfg.cov_type != "Deterministic":
        cfg.mc_samples_test = 10

    if args.do_eval:
        evaluate_checkpoints(cfg)

