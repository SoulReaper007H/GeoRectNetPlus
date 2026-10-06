# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 0 — Install · Imports · Seed · WHU dataset guard      ║
# ╚══════════════════════════════════════════════════════════════╝
# Run this cell first every session.
# It installs missing packages, sets the global random seed,
# detects GPU, and verifies the WHU dataset is attached.

import subprocess, sys

_REQUIRED = [
    "timm>=0.9", "tqdm", "Pillow", "scipy",
    "scikit-image", "opencv-python-headless",
]
for _pkg in _REQUIRED:
    _name = _pkg.split(">=")[0].replace("-", "_")
    try:
        __import__(_name)
    except ImportError:
        print(f"Installing {_pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pkg])

# ── standard imports ─────────────────────────────────────────────
import math, os, glob, random, csv, time, shutil, hashlib
import threading, contextlib
from dataclasses import dataclass
from typing import Dict, Tuple, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from tqdm.auto import tqdm
import timm

# ── reproducibility ───────────────────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[init] device = {device}")
if torch.cuda.is_available():
    print(f"[init] GPU = {torch.cuda.get_device_name(0)}"
          f"  VRAM = {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── WHU dataset guard ─────────────────────────────────────────────
WHU_ROOT = os.environ.get("WHU_ROOT", "/kaggle/input/datasets/sengulgs/whu-building-dataset/WHU")
if not os.path.isdir(WHU_ROOT):
    print(f"[init] WARNING: WHU dataset not found at {WHU_ROOT}. Set WHU_ROOT before training/testing.")
print(f"[init] WHU dataset OK: {WHU_ROOT}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Config                                             ║
# ║                                                              ║
# ║  SMOKE_TEST = True  -> 5-minute sanity run (tiny data)        ║
# ║  SMOKE_TEST = False -> real training (WHU 1/8 labels)         ║
# ╚══════════════════════════════════════════════════════════════╝

SMOKE_TEST = True   # ← flip to False for real training

# ── smoke-test hard caps (ignored when SMOKE_TEST=False) ─────────
_SM_TRAIN   = 20    # total training images (labeled + unlabeled)
_SM_VAL     = 8     # validation images
_SM_PSEUDO  = 5     # max unlabeled batches in Stage 2
_SM_EPOCHS  = (1, 1, 1)   # (S1, S2, S3) epochs for smoke
_SM_WARMUP  = 1           # warmup epochs for smoke

# ── real-training defaults ────────────────────────────────────────
_RT_EPOCHS  = (15, 10, 5) # (S1, S2, S3)
_RT_WARMUP  = 3

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    # ── dataset ──────────────────────────────────────────────────
    data_root: str   = os.environ.get("WHU_ROOT", "/kaggle/input/datasets/sengulgs/whu-building-dataset/WHU")
    img_size: int    = 512

    # ── label fraction (1/8 = 12.5% — primary paper result) ──────
    label_frac: float = 0.01 if SMOKE_TEST else 0.125

    # ── batch / workers ──────────────────────────────────────────
    batch_size: int   = 2 if SMOKE_TEST else 4
    num_workers: int  = 0 if SMOKE_TEST else 2

    # ── AMP + async gates ─────────────────────────────────────────
    use_amp: bool         = True
    use_async_gates: bool = not SMOKE_TEST      # CPU thread only in real mode
    async_warmup_batches: int = 5 if SMOKE_TEST else 50

    # ── epochs ───────────────────────────────────────────────────
    epochs_stage1: int  = _SM_EPOCHS[0] if SMOKE_TEST else _RT_EPOCHS[0]
    epochs_stage2: int  = _SM_EPOCHS[1] if SMOKE_TEST else _RT_EPOCHS[1]
    epochs_stage3: int  = _SM_EPOCHS[2] if SMOKE_TEST else _RT_EPOCHS[2]
    warmup_freeze: int  = _SM_WARMUP    if SMOKE_TEST else _RT_WARMUP

    # ── pseudo-label budget ───────────────────────────────────────
    # Stage 2 uses at most (pseudo_budget_N × labeled_count) unlabeled images.
    # Keeps Stage 2 proportional to supervision level; prevents 8000-batch runs.
    max_pseudo_batches: int = _SM_PSEUDO if SMOKE_TEST else 0  # 0 = use budget below
    pseudo_budget_N: int    = 3    # real mode: 3× labeled count as pseudo budget

    # ── smoke-test data caps ──────────────────────────────────────
    smoke_max_train: int = _SM_TRAIN
    smoke_max_val: int   = _SM_VAL

    # ── learning rates ───────────────────────────────────────────
    lr_backbone: float     = 1e-5
    lr_head: float         = 3e-4
    lr_loss_weights: float = 5e-5

    # ── pseudo-label thresholds (annealed each epoch) ─────────────
    thr_start: float = 0.65
    thr_end: float   = 0.50

    # ── loss weights ──────────────────────────────────────────────
    loss_w_in: float       = 1.0
    loss_w_out: float      = 1.5
    loss_dice_w: float     = 1.0
    boundary_weight: float = 2.5
    conf_loss_weight: float = 0.3

    # ── early stopping / saving ───────────────────────────────────
    patience: int     = 3  if SMOKE_TEST else 12
    viz_every: int    = 1  if SMOKE_TEST else 2
    pred_thr: float   = 0.5
    preview_every: int = 0
    preview_samples: int = 2

    # ── paths ─────────────────────────────────────────────────────
    save_path: str       = os.environ.get("GEORECT_SAVE_PATH", "checkpoints/best_model.pt")
    results_dir: str     = os.environ.get("GEORECT_RESULTS_DIR", "results")
    checkpoint_path: str = os.environ.get("GEORECT_CHECKPOINT_PATH", "checkpoints/training_checkpoint.pt")
    persist_dir: str     = os.environ.get("GEORECT_PERSIST_DIR", "checkpoints/persist")

    # ── EDL ──────────────────────────────────────────────────────
    edl_loss_weight: float    = 0.5
    edl_annealing_epochs: int = 2 if SMOKE_TEST else 10

    # ── CAM-Guided CLAAM v2 ───────────────────────────────────────
    claam_loss_weight: float      = 0.4
    claam_cam_weight: float       = 0.5
    claam_contrast_weight: float  = 0.5
    claam_entropy_weight: float   = 0.05
    claam_high_margin: float      = 0.7
    claam_low_margin: float       = 0.3
    claam_diversity_weight: float = 0.0
    claam_thr_start: float        = 0.55
    claam_thr_end: float          = 0.40
    cam_freeze_epochs: int        = 0 if SMOKE_TEST else 5

    # ── CLAC ──────────────────────────────────────────────────────
    clac_loss_weight: float = 0.2

    # ── TVR ───────────────────────────────────────────────────────
    tvr_window: int         = 5
    tvr_accept_ratio: float = 0.6

    # ── GOP ───────────────────────────────────────────────────────
    gop_rectilinear_thr: float = 0.45

    # ── CAFCG ─────────────────────────────────────────────────────
    cafcg_disagreement_thr: float = 0.3
    cafcg_keep_q: float           = 0.70

    # ── SDR ───────────────────────────────────────────────────────
    sdr_loss_weight: float = 0.3
    sdr_max_dist: float    = 50.0

    # ── pseudo misc ───────────────────────────────────────────────
    pseudo_ramp_start: float     = 0.1
    pseudo_freeze_backbone: bool = True
    pseudo_edge_erode_px: int    = 5

    # ── cross-dataset ─────────────────────────────────────────────
    inria_root: str             = "/kaggle/input/inria-building/AerialImageDataset"
    cross_dataset_img_size: int = 512

    log_every_n_batches: int = 0

cfg = Config()
os.makedirs(cfg.persist_dir, exist_ok=True)
os.makedirs(cfg.results_dir, exist_ok=True)

print(f"[config] SMOKE_TEST = {SMOKE_TEST}")
print(f"[config] label_frac = {cfg.label_frac}  "
      f"epochs S1/S2/S3 = {cfg.epochs_stage1}/{cfg.epochs_stage2}/{cfg.epochs_stage3}")
print(f"[config] batch = {cfg.batch_size}  "
      f"async_gates = {cfg.use_async_gates}  "
      f"max_pseudo_batches = {cfg.max_pseudo_batches}")
print(f"[config] persist_dir = {cfg.persist_dir}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Image utilities                                    ║
# ║  rgb_to_gray · hh_wavelet_channel · EdgeBranch               ║
# ╚══════════════════════════════════════════════════════════════╝

def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """Luminance-weighted grayscale  [B,3,H,W] -> [B,1,H,W]."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def hh_wavelet_channel(x: torch.Tensor) -> torch.Tensor:
    """Haar HH (diagonal edge) channel  [B,3,H,W] -> [B,1,H,W].
    Highlights fine-grained boundary textures for the edge branch."""
    gray   = rgb_to_gray(x)
    kernel = torch.tensor([[1., -1.], [-1., 1.]],
                          device=gray.device).view(1, 1, 2, 2)
    hh = F.conv2d(gray, kernel, stride=1, padding=1)
    return hh[:, :, :x.shape[-2], :x.shape[-1]]


class EdgeBranch(nn.Module):
    """Lightweight 3-layer CNN that produces a soft edge probability map.
    Input: 4 channels (RGB + HH wavelet).  Output: 1-channel logit map."""
    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32,    32, 3, padding=1)
        self.conv3 = nn.Conv2d(32,     1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)     # raw logit — caller applies sigmoid if needed

print("[cell2] image utilities defined")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3 — DABLCNet architecture                              ║
# ║  Backbone : ViT-B/16 (timm, 512-res, 32×32 patch grid)       ║
# ║  Neck     : MultiLayerFusion (4 ViT layers) + ASPP           ║
# ║  Decoder  : EdgeGuidedDecoder + SpatialEncoder skips         ║
# ║  Heads    : logits · EDL (Dirichlet) · CLAAM · SDR           ║
# ╚══════════════════════════════════════════════════════════════╝

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

class SpatialEncoder(nn.Module):
    """Lightweight CNN for multi-scale spatial features -> decoder skip connections.
    Gives the decoder high-res boundary info that ViT's 32×32 bottleneck loses.
    Only ~160K params — negligible vs ViT's 86M."""
    def __init__(self):
        super().__init__()
        # 512 -> 256, 32ch
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        # 256 -> 128, 64ch
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

    def forward(self, x):
        s1 = self.layer1(x)   # [B, 32, H/2, W/2]  = 256
        s2 = self.layer2(s1)  # [B, 64, H/4, W/4]  = 128
        return s1, s2


class ViTAttentionExtractor:
    """Capture ViT CLS-to-patch attention from selected transformer blocks."""
    def __init__(self, vit_encoder, layers=(2, 5, 8, 11), detach=True):
        self.layers = list(layers)
        self.detach = detach
        self._maps = {}
        self._hooks = []
        self._register(vit_encoder)

    def _register(self, vit_encoder):
        for idx in self.layers:
            block = vit_encoder.blocks[idx]
            hook = block.attn.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            scale = getattr(module, "scale", q.shape[-1] ** -0.5)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            self._maps[layer_idx] = attn.detach() if self.detach else attn
        return hook

    def clear(self):
        self._maps = {}

    def get_spatial_maps(self, patch_h, patch_w):
        spatial = []
        for idx in self.layers:
            if idx not in self._maps:
                raise RuntimeError(f"Attention map for ViT layer {idx} was not captured")
            attn = self._maps[idx]
            cls_attn = attn[:, :, 0, 1:]
            expected = patch_h * patch_w
            if cls_attn.shape[-1] != expected:
                grid = int(math.sqrt(cls_attn.shape[-1]))
                if grid * grid != cls_attn.shape[-1]:
                    raise ValueError(f"Cannot reshape {cls_attn.shape[-1]} patch tokens into a grid")
                patch_h = patch_w = grid
            spatial.append(cls_attn.reshape(cls_attn.shape[0], cls_attn.shape[1], patch_h, patch_w))
        return spatial

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class ViTEncoder(nn.Module):
    """512-resolution ViT: 512/16 = 32x32 tokens.
    Returns the same multi-layer feature maps as Phase-2, and stores
    attention maps for CAM-Guided CLAAM.
    """
    def __init__(self):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm not available.")
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=True,
            num_classes=0, img_size=512)
        try:
            self.vit.set_grad_checkpointing(enable=True)
        except AttributeError:
            pass
        self.patch_size = 16
        self.embed_dim = 768
        self.hook_layers = [2, 5, 8, 11]
        self._features = {}
        self._last_attn_maps = None
        self._register_hooks()
        self.attn_extractor = ViTAttentionExtractor(self.vit, layers=self.hook_layers)

    def _register_hooks(self):
        for idx in self.hook_layers:
            block = self.vit.blocks[idx]
            block.register_forward_hook(self._make_hook(idx))

    def _make_hook(self, idx):
        def hook_fn(module, input, output):
            self._features[idx] = output
        return hook_fn

    def forward(self, x: torch.Tensor):
        mean = IMAGENET_MEAN.to(x.device)
        std = IMAGENET_STD.to(x.device)
        x = (x - mean) / std

        self._features = {}
        self.attn_extractor.clear()
        _ = self.vit.forward_features(x)

        b = x.shape[0]
        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size

        multi_feats = []
        for idx in self.hook_layers:
            feat = self._features[idx]
            if feat.shape[1] == grid_h * grid_w + 1:
                feat = feat[:, 1:, :]
            feat_map = feat.transpose(1, 2).reshape(b, self.embed_dim, grid_h, grid_w)
            multi_feats.append(feat_map)

        self._last_attn_maps = self.attn_extractor.get_spatial_maps(grid_h, grid_w)
        return multi_feats

    def get_attention_maps(self):
        if self._last_attn_maps is None:
            raise RuntimeError("Run the ViT encoder before requesting attention maps")
        return self._last_attn_maps


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLabV3+ style).
    Captures multi-scale context at the bottleneck — buildings come in many sizes.
    Parallel dilated convolutions at rates 6, 12, 18 + global average pooling."""
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.atrous6 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.atrous12 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.atrous18 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout(0.1))

    def forward(self, x):
        h, w = x.shape[-2:]
        f1 = self.conv1x1(x)
        f2 = self.atrous6(x)
        f3 = self.atrous12(x)
        f4 = self.atrous18(x)
        f5 = self.gap(x)
        f5 = F.interpolate(f5, size=(h, w), mode="bilinear", align_corners=False)
        out = torch.cat([f1, f2, f3, f4, f5], dim=1)
        return self.project(out)


class MultiLayerFusion(nn.Module):
    """Fuse 4 ViT layers (each 768-ch @ 32×32) into a single feature map.
    Projects each layer to 256-ch, concatenates, then compresses to out_ch."""
    def __init__(self, embed_dim=768, out_ch=768):
        super().__init__()
        self.proj3  = nn.Sequential(nn.Conv2d(embed_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.proj6  = nn.Sequential(nn.Conv2d(embed_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.proj9  = nn.Sequential(nn.Conv2d(embed_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.proj12 = nn.Sequential(nn.Conv2d(embed_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.fuse = nn.Sequential(
            nn.Conv2d(256 * 4, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, multi_feats):
        p3  = self.proj3(multi_feats[0])
        p6  = self.proj6(multi_feats[1])
        p9  = self.proj9(multi_feats[2])
        p12 = self.proj12(multi_feats[3])
        return self.fuse(torch.cat([p3, p6, p9, p12], dim=1))


class EdgeGuidedDecoder(nn.Module):
    """Decoder for 32×32 bottleneck -> 512. All learned upsampling — no bilinear blur.
    32->64->128(+skip@128)->256(+skip@256)->512 with edge fusion."""
    def __init__(self, in_ch: int = 256):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 256, 2, stride=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.skip_fuse2 = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.skip_fuse3 = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(32, 1, 1)

    def _decode(self, feat, out_size, skips=None):
        """Shared decode path: 32->64->128->256->512."""
        x = self.up1(feat)
        x = self.up2(x)
        if skips is not None:
            s2 = F.interpolate(skips[1], size=x.shape[-2:],
                               mode="bilinear", align_corners=False)
            x = self.skip_fuse2(torch.cat([x, s2], dim=1))
        x = self.up3(x)
        if skips is not None:
            s1 = F.interpolate(skips[0], size=x.shape[-2:],
                               mode="bilinear", align_corners=False)
            x = self.skip_fuse3(torch.cat([x, s1], dim=1))
        x = self.up4(x)
        if x.shape[-2] != out_size[0] or x.shape[-1] != out_size[1]:
            x = F.interpolate(x, size=out_size, mode="nearest")
        x = self.refine(x)
        return x

    def forward(self, feat: torch.Tensor, edge: torch.Tensor,
                out_size: Tuple[int, int], skips=None) -> torch.Tensor:
        x = self._decode(feat, out_size, skips)
        edge_resized = F.interpolate(edge, size=out_size, mode="nearest")
        x = torch.cat([x, edge_resized], dim=1)
        x = self.fuse(x)
        return self.out_conv(x)

    def get_features(self, feat: torch.Tensor, out_size: Tuple[int, int],
                     skips=None) -> torch.Tensor:
        """Return 32-ch feature map for confidence head (no edge fusion)."""
        return self._decode(feat, out_size, skips)


class ConfidenceHead(nn.Module):
    """Original sigmoid confidence head (baseline / fallback when EDL disabled)."""
    def __init__(self, in_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        temp = torch.exp(self.log_temp) + 1e-6
        return torch.sigmoid(x / temp)


# ═══════════════════════════════════════════════════════════════════
#  CONTRIBUTION 1: Evidential Dirichlet Confidence Head (EDL-Conf)
# ═══════════════════════════════════════════════════════════════════

class EvidentialHead(nn.Module):
    """Outputs Dirichlet concentration params α = evidence + 1 per class.
    Epistemic uncertainty u = K/S where S = Σα, K = num_classes."""
    def __init__(self, in_ch: int = 32, num_classes: int = 2):
        super().__init__()
        self.K = num_classes
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, num_classes, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        evidence = F.softplus(self.conv3(x))
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        belief = evidence / S
        uncertainty = float(self.K) / S
        confidence = 1.0 - uncertainty
        return {
            "alpha": alpha,
            "belief": belief,
            "uncertainty": uncertainty,
            "confidence": confidence.clamp(0, 1),
            "strength": S,
        }


def kl_dirichlet(alpha: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """KL divergence between Dir(α) and Dir(1,...,1)."""
    ones = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_ones = ones.sum(dim=1, keepdim=True)
    kl = (torch.lgamma(S_alpha) - torch.lgamma(S_ones)
          - (torch.lgamma(alpha) - torch.lgamma(ones)).sum(dim=1, keepdim=True)
          + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=1, keepdim=True))
    return kl.mean()


def edl_loss(alpha, target, epoch, total_epochs, annealing_epochs=10):
    """EDL loss = Bayes risk + annealed KL."""
    target_1h = torch.cat([1.0 - target, target], dim=1)
    S = alpha.sum(dim=1, keepdim=True)
    bayes_risk = (target_1h * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
    annealing_coef = min(1.0, epoch / max(1, annealing_epochs))
    alpha_tilde = target_1h + (1.0 - target_1h) * (alpha - 1.0) + 1.0
    kl = kl_dirichlet(alpha_tilde, num_classes=2)
    return bayes_risk.mean() + annealing_coef * kl


# ═══════════════════════════════════════════════════════════════════
#  CONTRIBUTION 2: Cross-Layer Attention Agreement Map (CLAAM / MLAA)
# ═══════════════════════════════════════════════════════════════════

class CAMGuidedCLAAM(nn.Module):
    """v2: removes per-image min-max norm, adds per-layer projection
    and global tanh-gain. Output keys are unchanged."""
    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.cam_generator = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim // 4, embed_dim // 16, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 16),
            nn.GELU(),
            nn.Conv2d(embed_dim // 16, 1, kernel_size=1),
        )

        # NEW v2: per-layer projection forces specialization between layers
        # whose ViT features are nearly identical (DeepViT attention collapse).
        self.layer_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_heads, num_heads, kernel_size=1,
                          groups=num_heads, bias=False),
                nn.BatchNorm2d(num_heads),
                nn.GELU(),
            ) for _ in range(num_layers)
        ])

        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Refine WITHOUT trailing Sigmoid; output gets gain-tanh below.
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        # NEW v2: global gain+bias REPLACE per-image min-max normalization.
        self.out_gain = nn.Parameter(torch.tensor(2.0))
        self.out_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, attn_maps_per_layer, feature_map, out_size):
        cam_logits = self.cam_generator(feature_map)
        cam = torch.sigmoid(cam_logits)

        projected = [self.layer_proj[i](a)
                     for i, a in enumerate(attn_maps_per_layer)]
        attn_stack = torch.stack(projected, dim=1)
        attn_cam = attn_stack * cam.unsqueeze(1)

        head_w = self.head_weights.softmax(dim=0)
        layer_w = self.layer_weights.softmax(dim=0)
        attn_hw = attn_cam * head_w.view(1, 1, -1, 1, 1)
        attn_lw = attn_hw * layer_w.view(1, -1, 1, 1, 1)

        layer_consensus = attn_lw.mean(dim=2)
        layer_std = layer_consensus.std(dim=1)
        layer_mean = layer_consensus.mean(dim=1)

        temp = self.temperature.clamp(min=0.1)
        raw_agreement = layer_mean / (1.0 + layer_std / temp)
        refined = self.refine(raw_agreement.unsqueeze(1))

        # KEY CHANGE: global tanh-gain instead of per-image min-max normalization
        agreement = 0.5 * (torch.tanh(self.out_gain * refined + self.out_bias) + 1.0)

        variance = layer_std.unsqueeze(1)
        mean_pred = layer_mean.unsqueeze(1)

        if agreement.shape[-2:] != out_size:
            agreement = F.interpolate(agreement, size=out_size, mode="bilinear", align_corners=False)
            variance = F.interpolate(variance, size=out_size, mode="bilinear", align_corners=False)
            mean_pred = F.interpolate(mean_pred, size=out_size, mode="bilinear", align_corners=False)
            cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
            cam_logits = F.interpolate(cam_logits, size=out_size, mode="bilinear", align_corners=False)

        return {
            "agreement": agreement,
            "variance": variance,
            "cam": cam,
            "cam_logits": cam_logits,
            "mean_pred": mean_pred,
            "head_weights": self.head_weights.softmax(dim=0),
            "layer_weights": self.layer_weights.softmax(dim=0),
        }

CLAAM = CAMGuidedCLAAM


def claam_consistency_loss(layer_logits, target):
    """Force ALL ViT layers to agree with GT."""
    loss = torch.tensor(0.0, device=target.device)
    for logit in layer_logits:
        loss = loss + F.binary_cross_entropy_with_logits(logit, target)
    return loss / len(layer_logits)


# ═══════════════════════════════════════════════════════════════════
#  IDEA 2: Cross-Layer Attention Consistency Loss (CLAC-Loss)
# ═══════════════════════════════════════════════════════════════════

def clac_loss(multi_feats: List[torch.Tensor]) -> torch.Tensor:
    """Penalizes spatial activation disagreement between adjacent ViT layers."""
    loss = torch.tensor(0.0, device=multi_feats[0].device)
    for i in range(len(multi_feats) - 1):
        act_i = multi_feats[i].mean(dim=1, keepdim=True)
        act_j = multi_feats[i + 1].mean(dim=1, keepdim=True)
        act_i = (act_i - act_i.amin(dim=(2, 3), keepdim=True)) / \
                (act_i.amax(dim=(2, 3), keepdim=True) - act_i.amin(dim=(2, 3), keepdim=True) + 1e-6)
        act_j = (act_j - act_j.amin(dim=(2, 3), keepdim=True)) / \
                (act_j.amax(dim=(2, 3), keepdim=True) - act_j.amin(dim=(2, 3), keepdim=True) + 1e-6)
        loss = loss + F.mse_loss(act_i, act_j)
    return loss / (len(multi_feats) - 1)


# ═══════════════════════════════════════════════════════════════════
#  IDEA 4: Gradient Orientation Prior (GOP)
# ═══════════════════════════════════════════════════════════════════

def gradient_orientation_gate(probs: torch.Tensor, thr: float = 0.5,
                              rectilinear_thr: float = 0.45) -> torch.Tensor:
    """Reject pseudo-label components whose boundary gradient orientations
    deviate from rectilinear (0°/90°) angles."""
    import cv2
    B = probs.shape[0]
    gate = torch.ones_like(probs, dtype=torch.bool)

    for b in range(B):
        mask_np = (probs[b, 0].detach().cpu().numpy() > thr).astype(np.uint8) * 255
        if mask_np.sum() == 0:
            continue

        # AMP fix: probs may be float16 under autocast; cv2.Sobel needs float32/64.
        prob_np = probs[b, 0].detach().float().cpu().numpy()
        gx = cv2.Sobel(prob_np, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(prob_np, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.degrees(np.arctan2(gy, gx)) % 180

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        reject_mask = np.zeros_like(mask_np, dtype=np.uint8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            comp_mask = np.zeros_like(mask_np)
            cv2.drawContours(comp_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(comp_mask, kernel, iterations=2)
            eroded = cv2.erode(comp_mask, kernel, iterations=1)
            boundary = ((dilated > 0) & ~(eroded > 0))

            boundary_mag = mag[boundary]
            boundary_angle = angle[boundary]
            if boundary_mag.size == 0:
                continue
            strong = boundary_mag > (boundary_mag.mean() + 1e-8)

            if strong.sum() < 5:
                continue

            angles_strong = boundary_angle[strong]
            rectilinear = (
                (angles_strong < 15) | (angles_strong > 165) |
                ((angles_strong > 75) & (angles_strong < 105))
            )
            rect_frac = rectilinear.sum() / len(angles_strong)

            if rect_frac < rectilinear_thr:
                cv2.drawContours(reject_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        if reject_mask.any():
            reject_t = torch.from_numpy(reject_mask > 0).to(probs.device)
            gate[b, 0] = gate[b, 0] & (~reject_t)

    return gate


# ═══════════════════════════════════════════════════════════════════
#  IDEA 5a: Confidence-Aware Frequency Consistency Gate (CAFCG)
# ═══════════════════════════════════════════════════════════════════

def cafcg_gate(probs: torch.Tensor, disagreement_thr: float = None,
               keep_quantile: float = None) -> torch.Tensor:
    """CAFCG v2: ratio-based detector + adaptive per-image quantile threshold.
    The legacy `disagreement_thr` argument is kept for back-compat but ignored;
    use `keep_quantile` (or cfg.cafcg_keep_q) to control selectivity."""
    if keep_quantile is None:
        keep_quantile = getattr(cfg, "cafcg_keep_q", 0.70)
    B, C, H, W = probs.shape
    ksize, sigma = 9, 2.0
    x = torch.arange(ksize, device=probs.device).float() - ksize // 2
    g1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).view(1, 1, ksize, ksize)
    low_freq = F.conv2d(probs, g2d, padding=ksize // 2)
    high_freq = (probs - low_freq).abs()
    ratio = high_freq / (low_freq.clamp_min(0.05))
    flat = ratio.flatten(2)
    thr = torch.quantile(flat, keep_quantile, dim=2, keepdim=True).unsqueeze(-1)
    return ratio <= thr


class SignedDistanceHead(nn.Module):
    """Predicts normalized signed distance field (SDF) from decoder features.
    Output range: tanh -> [-1, +1] where +1 = deep interior, -1 = far outside,
    0 = on building boundary."""
    def __init__(self, in_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        return torch.tanh(self.conv2(x))  # [-1, +1]


def compute_sdf_target(mask: torch.Tensor, max_dist: float = 50.0) -> torch.Tensor:
    """Compute normalized signed distance field from binary GT mask.
    Args:
        mask: [B, 1, H, W] binary building mask
        max_dist: clip and normalize distances to [-1, +1]
    Returns:
        sdf: [B, 1, H, W] where +1 = deep inside, -1 = far outside, 0 = boundary
    """
    from scipy.ndimage import distance_transform_edt
    B = mask.shape[0]
    sdf_batch = torch.zeros_like(mask)
    for b in range(B):
        m = mask[b, 0].cpu().numpy().astype(np.float64)
        if m.sum() > 0:
            dist_in = distance_transform_edt(m)
        else:
            dist_in = np.zeros_like(m)
        if (1 - m).sum() > 0:
            dist_out = distance_transform_edt(1 - m)
        else:
            dist_out = np.zeros_like(m)
        sdf = dist_in - dist_out  # + inside, - outside, 0 on boundary
        sdf = np.clip(sdf / max_dist, -1.0, 1.0)
        sdf_batch[b, 0] = torch.from_numpy(sdf).float()
    return sdf_batch.to(mask.device)


# ═══════════════════════════════════════════════════════════════════
#  Updated DABLCNet: integrates EDL, CLAAM, SDR, and all novel modules
# ═══════════════════════════════════════════════════════════════════

class DABLCNet(nn.Module):
    """DABLCNet + Phase-2 contributions with CAM-Guided CLAAM swapped in."""
    def __init__(self):
        super().__init__()

        # Core architecture
        self.encoder = ViTEncoder()
        self.layer_fuse = MultiLayerFusion(768, out_ch=768)
        self.aspp = ASPP(in_ch=768, out_ch=256)
        self.spatial_enc = SpatialEncoder()
        self.edge_branch = EdgeBranch(in_ch=4)
        self.decoder = EdgeGuidedDecoder(in_ch=256)

        # Contribution 1: Evidential Dirichlet head
        self.edl_head = EvidentialHead(in_ch=32, num_classes=2)

        # Contribution 2: CAM-Guided CLAAM
        self.claam = CAMGuidedCLAAM(embed_dim=768, num_heads=12, num_layers=4)

        # SDR: Signed Distance Regression head
        self.sdr_head = SignedDistanceHead(in_ch=32)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        hh = hh_wavelet_channel(x)
        edge_in = torch.cat([x, hh], dim=1)
        edge_map = self.edge_branch(edge_in)

        multi_feats = self.encoder(x)
        attn_maps = self.encoder.get_attention_maps()

        fused = self.layer_fuse(multi_feats)
        bottleneck = self.aspp(fused)
        skips = self.spatial_enc(x)
        logits = self.decoder(bottleneck, edge_map,
                              out_size=x.shape[-2:], skips=skips)

        dec_feat = self.decoder.get_features(bottleneck,
                                              out_size=x.shape[-2:], skips=skips)

        edl_out = self.edl_head(dec_feat)
        conf_map = edl_out["confidence"]

        claam_out = self.claam(attn_maps, multi_feats[-1], out_size=x.shape[-2:])
        sdf_pred = self.sdr_head(dec_feat)

        return {
            "logits": logits,
            "edge": edge_map,
            "conf": conf_map,
            "edl": edl_out,
            "claam": claam_out,
            "multi_feats": multi_feats,
            "sdf_pred": sdf_pred,
        }

print('[cell3] DABLCNet architecture defined')


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4 — DABLCLoss                                          ║
# ║                                                              ║
# ║  Asymmetric BCE  : w_in / w_out learned via log-ratio        ║
# ║    -> gradient flows through _logit_ratio (sum-constrained)   ║
# ║    -> NOT passed as weight= to F.bce (non-differentiable)     ║
# ║  Gap-aware boundary loss : penalises inter-building merging   ║
# ║  EDL loss        : Bayes risk + annealed KL (Contribution 1) ║
# ║  CLAAM loss      : CAM-BCE + contrastive + entropy floor      ║
# ║  CLAC loss       : cross-layer activation consistency         ║
# ║  SDR loss        : signed-distance regression                 ║
# ╚══════════════════════════════════════════════════════════════╝

class DABLCLoss(nn.Module):
    """DABL-C loss + Novel contributions:
    - Original: asymmetric BCE + Dice + Boundary sharpening + Confidence supervision
    - NEW: Constrained learnable weights (ratio-based, prevents collapse)
    - NEW: Gap-aware boundary loss (penalizes inter-building merging)
    - NEW: EDL Bayes risk + KL (Contribution 1)
    - NEW: CLAAM consistency + diversity regularization (Contribution 2 / MLAA)
    - NEW: CLAC spatial consistency regularization (Idea 2)
    """
    def __init__(self, w_in_init=1.0, w_out_init=1.0, dice_weight=1.0,
                 boundary_weight=0.5, conf_loss_weight=0.3):
        super().__init__()
        # Constrained parameterization: learn the LOG-RATIO of w_in/w_out
        # Total sum is fixed at (w_in_init + w_out_init), only the balance is learned.
        # This prevents both weights from collapsing to zero.
        self._total_w = w_in_init + w_out_init  # fixed constant
        # logit_ratio: sigmoid(logit) = w_in / (w_in + w_out)
        init_ratio = w_in_init / (w_in_init + w_out_init)
        init_logit = math.log(init_ratio / (1.0 - init_ratio + 1e-8) + 1e-8)
        self._logit_ratio = nn.Parameter(torch.tensor(float(init_logit)))
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.conf_loss_weight = conf_loss_weight

    @property
    def w_in(self):
        """Building weight: derived from learned ratio, sum-constrained."""
        ratio = torch.sigmoid(self._logit_ratio)
        return self._total_w * ratio

    @property
    def w_out(self):
        """Background weight: derived from learned ratio, sum-constrained."""
        ratio = torch.sigmoid(self._logit_ratio)
        return self._total_w * (1.0 - ratio)

    def _boundary(self, x):
        """Gradient-based boundary extraction."""
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return (gx.abs() + gy.abs()).clamp(0, 1)

    def _gap_weight_map(self, target):
        """Create weight map emphasizing inter-building gaps.
        Uses GPU-based max-pool dilation to find gap pixels between
        adjacent buildings — these are the regions where merging errors
        occur and boundary accuracy is most critical.
        Returns: weight map [B, 1, H, W] with higher values at gaps."""
        # Dilate GT buildings on GPU using max pooling (equivalent to morphological dilation)
        dilated = F.max_pool2d(target, kernel_size=11, stride=1, padding=5)
        # Gap pixels: near buildings but not buildings themselves
        gap = (dilated - target).clamp(0, 1)
        # Building boundary pixels: buildings minus eroded buildings
        eroded = -F.max_pool2d(-target, kernel_size=5, stride=1, padding=2)  # min-pool = erosion
        boundary = (target - eroded).clamp(0, 1)
        # Weight map: base=1, gap regions=5x, building boundaries=3x
        weight = 1.0 + 4.0 * gap + 2.0 * boundary
        return weight

    def _claam_diversity_loss(self, layer_logits):
        """Penalize excessive agreement among CLAAM per-layer predictions.
        Encourages each ViT layer head to capture different building aspects.
        Uses pairwise cosine similarity on downsampled predictions."""
        if len(layer_logits) < 2:
            return torch.tensor(0.0, device=layer_logits[0].device)
        flat = [F.adaptive_avg_pool2d(torch.sigmoid(l), (16, 16)).reshape(l.shape[0], -1)
                for l in layer_logits]
        sim_sum = 0.0
        count = 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                sim_sum = sim_sum + F.cosine_similarity(flat[i], flat[j], dim=-1).mean()
                count += 1
        return sim_sum / max(count, 1)

    def forward(self, logits, target, conf=None, model_output=None,
                epoch=0, total_epochs=45, is_pseudo=False):
        probs = torch.sigmoid(logits)
        eps = 1e-6
        # NEW: track per-loss components for paper logging
        self.last_components = {}
        w_in = self.w_in     # derived from constrained ratio
        w_out = self.w_out

        # Dynamic pos_weight: small buildings get drowned by background
        pos_pixels = target.sum() + eps
        neg_pixels = (1 - target).sum() + eps
        pos_weight = (neg_pixels / pos_pixels).clamp(max=5.0)

        # Asymmetric BCE with area-based reweighting
        bce = -(w_in * pos_weight * target * torch.log(probs + eps) +
                w_out * (1 - target) * torch.log(1 - probs + eps))
        if conf is not None:
            bce = bce * conf
        bce_loss = bce.mean()
        self.last_components["bce"] = bce_loss.detach().item()

        # Dice loss
        inter = (probs * target).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = (1.0 - (2.0 * inter + eps) / (union + eps)).mean()
        self.last_components["dice"] = dice_loss.detach().item()

        # Gap-aware boundary loss: steep sigmoid -> near-hard edges
        # Weight boundary errors MORE heavily in inter-building gap regions
        # This directly penalizes the under-segmentation / merging problem
        sharp_probs = torch.sigmoid(logits * 10.0)
        pred_bd = self._boundary(sharp_probs)
        gt_bd = self._boundary(target)
        gap_weights = self._gap_weight_map(target)
        bd_loss = (gap_weights * (pred_bd - gt_bd) ** 2).mean()
        self.last_components["boundary"] = bd_loss.detach().item()

        total = bce_loss + self.dice_weight * dice_loss + self.boundary_weight * bd_loss

        # Scale factor for novel loss terms during pseudo-label training
        novel_scale = 0.3 if is_pseudo else 1.0

        # ── CONTRIBUTION 1: EDL Loss (Bayes risk + annealed KL) ──
        if model_output is not None and model_output.get("edl") is not None:
            alpha = model_output["edl"]["alpha"]
            l_edl = edl_loss(alpha, target, epoch, total_epochs,
                             annealing_epochs=cfg.edl_annealing_epochs)
            total = total + cfg.edl_loss_weight * novel_scale * l_edl
            self.last_components["edl"] = l_edl.detach().item()

        # ── CONTRIBUTION 2: CAM-Guided CLAAM v2 (contrastive + entropy floor) ──
        if model_output is not None and model_output.get("claam") is not None:
            claam_out = model_output["claam"]
            cam_logits = claam_out["cam_logits"]
            agreement = claam_out["agreement"]
            if cam_logits.shape[-2:] != target.shape[-2:]:
                cam_logits = F.interpolate(cam_logits, size=target.shape[-2:],
                                           mode="bilinear", align_corners=False)
            if agreement.shape[-2:] != target.shape[-2:]:
                agreement = F.interpolate(agreement, size=target.shape[-2:],
                                          mode="bilinear", align_corners=False)
            target_f = target.float()

            # (a) CAM supervision (BCE)
            cam_loss = F.binary_cross_entropy_with_logits(cam_logits.float(), target_f)

            # (b) Margin-based contrastive: agreement HIGH on FG, LOW on BG.
            # This is the anti-collapse term — without contrast, the agreement
            # map saturates to a constant.
            fg_pixels = target_f.sum().clamp_min(1.0)
            bg_pixels = (1.0 - target_f).sum().clamp_min(1.0)
            high = getattr(cfg, "claam_high_margin", 0.7)
            low  = getattr(cfg, "claam_low_margin",  0.3)
            fg_violation = F.relu(high - agreement) * target_f
            bg_violation = F.relu(agreement - low) * (1.0 - target_f)
            contrast_loss = (fg_violation.sum() / fg_pixels +
                             bg_violation.sum() / bg_pixels) * 0.5

            # (c) Entropy floor: penalize spatially uniform agreement maps
            sp_var = agreement.flatten(2).var(dim=2).mean()
            entropy_loss = F.relu(0.01 - sp_var) / 0.01

            l_claam = (cfg.claam_cam_weight * cam_loss
                       + cfg.claam_contrast_weight * contrast_loss
                       + cfg.claam_entropy_weight * entropy_loss)
            total = total + cfg.claam_loss_weight * novel_scale * l_claam
            self.last_components["claam_cam"] = cam_loss.detach().item()
            self.last_components["claam_contrast"] = contrast_loss.detach().item()
            self.last_components["claam_entropy"] = entropy_loss.detach().item()

        # ── IDEA 2: CLAC-Loss (cross-layer spatial consistency regularization) ──
        if model_output is not None and model_output.get("multi_feats") is not None:
            l_clac = clac_loss(model_output["multi_feats"])
            total = total + cfg.clac_loss_weight * l_clac
            self.last_components["clac"] = l_clac.detach().item()

        # ── IDEA 5: SDR Loss (signed distance field regression) ──
        if model_output is not None and model_output.get("sdf_pred") is not None:
            # FIX 10: skip SDR loss when pseudo-label has insufficient FG (junk SDF).
            fg_pixels_count = float(target.sum().item())
            if (not is_pseudo) or fg_pixels_count >= 256:
                sdf_pred = model_output["sdf_pred"]
                sdf_target = compute_sdf_target(target, max_dist=cfg.sdr_max_dist)
                l_sdr = F.smooth_l1_loss(sdf_pred, sdf_target)
                total = total + cfg.sdr_loss_weight * novel_scale * l_sdr
                self.last_components["sdr"] = l_sdr.detach().item()

        return total

print('[cell4] DABLCLoss defined')


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5 — 7-gate pseudo-label filter + TVR buffer            ║
# ║                                                              ║
# ║  Gate 1 : Probability threshold (annealed)                   ║
# ║  Gate 2 : EDL uncertainty  (low uncertainty = trustworthy)   ║
# ║  Gate 3 : Edge proximity   (high edge = boundary pixel)      ║
# ║  Gate 4 : Geometry         (compactness / rectangularity)    ║
# ║  Gate 5 : CLAAM agreement  (cross-layer consensus)           ║
# ║  Gate 6 : GOP              (rectilinear boundary prior)      ║
# ║  Gate 7 : CAFCG            (frequency consistency)           ║
# ║                                                              ║
# ║  All 7 gates evaluated at COMPONENT level (not pixel level)  ║
# ║  TVR rolling window (bit-packed) also feeds into Gate 5      ║
# ╚══════════════════════════════════════════════════════════════╝


def progressive_threshold(epoch: int, total_epochs: int, start: float, end: float) -> float:
    if total_epochs <= 1:
        return end
    t = epoch / (total_epochs - 1)
    return start + t * (end - start)


def _geometry_gate(probs: torch.Tensor, thr: float = 0.5,
                   compact_min: float = 0.20, rect_min: float = 0.50,
                   max_aspect: float = 6.0, min_area: int = 20) -> torch.Tensor:
    """Geom v2: stricter rules - rejects tiny / non-rectangular / overly elongated comps."""
    import cv2
    B = probs.shape[0]
    gate = torch.ones_like(probs, dtype=torch.bool)
    for b in range(B):
        mask_np = (probs[b, 0].detach().cpu().numpy() > thr).astype(np.uint8) * 255
        if mask_np.sum() == 0:
            continue
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        reject_mask = np.zeros_like(mask_np, dtype=np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                cv2.drawContours(reject_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                continue
            peri = cv2.arcLength(cnt, True)
            compactness = (4 * math.pi * area) / (peri * peri + 1e-6)
            x, y, w, h = cv2.boundingRect(cnt)
            rectangularity = area / (w * h + 1e-6)
            aspect = max(w, h) / max(min(w, h), 1)
            if (compactness < compact_min or rectangularity < rect_min
                    or aspect > max_aspect):
                cv2.drawContours(reject_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        if reject_mask.any():
            reject_t = torch.from_numpy(reject_mask > 0).to(probs.device)
            gate[b, 0] = gate[b, 0] & (~reject_t)
    return gate


def _erode_pseudo_labels(pseudo: torch.Tensor, erode_px: int) -> torch.Tensor:
    """Erode pseudo-label boundaries by erode_px pixels using GPU min-pooling.
    This PREVENTS the merging echo-chamber: by shrinking each predicted building
    a few pixels inward, we ensure the model never reinforces the 'bleed' across
    narrow inter-building gaps from its own noisy predictions.
    Only the confident interior pixels survive as pseudo-labels."""
    if erode_px <= 0:
        return pseudo
    k = 2 * erode_px + 1
    # Min-pooling = erosion for binary masks: shrinks foreground by erode_px
    eroded = -F.max_pool2d(-pseudo, kernel_size=k, stride=1, padding=erode_px)
    return eroded


# ═══════════════════════════════════════════════════════════════════
#  IDEA 3: Temporal Voting across Rotations (TVR)
# ═══════════════════════════════════════════════════════════════════

class TemporalVotingBuffer:
    """Rolling TVR history stored as binary masks.

    The first version kept full float32 masks in memory and on disk, which can
    make best_model_tvr.pt many GB. This version stores uint8 masks in RAM and
    bit-packs them for checkpoints while keeping the same voting behavior.
    """
    def __init__(self, window: int = 5, accept_ratio: float = 0.6):
        self.window = window
        self.accept_ratio = accept_ratio
        self.buffer = {}

    @staticmethod
    def _normalize_key(path_or_name):
        base = os.path.basename(str(path_or_name))
        return os.path.splitext(base)[0]

    def update(self, image_ids: list, predictions: torch.Tensor):
        from collections import deque
        preds = (predictions.detach().cpu() > 0.5).to(torch.uint8)
        for i, img_id in enumerate(image_ids):
            key = self._normalize_key(img_id)
            if key not in self.buffer:
                self.buffer[key] = deque(maxlen=self.window)
            self.buffer[key].append(preds[i:i+1].contiguous())

    def get_stable_mask(self, image_ids: list, device: torch.device) -> torch.Tensor:
        masks = []
        for img_id in image_ids:
            key = self._normalize_key(img_id)
            if key not in self.buffer or len(self.buffer[key]) < 2:
                masks.append(torch.ones(1, 1, 1, 1))
            else:
                stacked = torch.cat([m.float() for m in self.buffer[key]], dim=0)
                vote_frac = stacked.mean(dim=0, keepdim=True)
                stable = (vote_frac >= self.accept_ratio).float()
                masks.append(stable)
        return [m.to(device) for m in masks]

    @staticmethod
    def _pack_mask(mask: torch.Tensor) -> dict:
        arr = (mask.detach().cpu().numpy() > 0).astype(np.uint8)
        shape = tuple(arr.shape)
        packed = np.packbits(arr.reshape(-1))
        return {"shape": shape, "packed": torch.from_numpy(packed.copy())}

    @staticmethod
    def _unpack_mask(item) -> torch.Tensor:
        if isinstance(item, dict) and "packed" in item:
            shape = tuple(item["shape"])
            packed = item["packed"].detach().cpu().numpy().astype(np.uint8)
            flat = np.unpackbits(packed, count=int(np.prod(shape))).astype(np.uint8)
            return torch.from_numpy(flat.reshape(shape)).to(torch.uint8)
        if isinstance(item, torch.Tensor):
            return (item.detach().cpu() > 0.5).to(torch.uint8)
        return (torch.tensor(item) > 0.5).to(torch.uint8)

    def state_dict(self):
        return {
            "format": "tvr_packbits_v1",
            "window": self.window,
            "accept_ratio": self.accept_ratio,
            "buffer": {
                img_id: [self._pack_mask(pred) for pred in preds]
                for img_id, preds in self.buffer.items()
            },
        }

    def load_state_dict(self, state):
        from collections import deque
        self.window = int(state.get("window", self.window))
        self.accept_ratio = float(state.get("accept_ratio", self.accept_ratio))
        self.buffer = {}
        for img_id, preds in state.get("buffer", {}).items():
            q = deque(maxlen=self.window)
            for pred in preds:
                q.append(self._unpack_mask(pred).contiguous())
            self.buffer[self._normalize_key(img_id)] = q
        return self


def get_tvr_save_path(model_path=None):
    model_path = model_path or cfg.save_path
    root, ext = os.path.splitext(model_path)
    ext = ext or ".pt"
    return f"{root}_tvr{ext}"


def save_tvr_buffer(buffer=None, path=None):
    buffer = buffer or tvr_buffer
    path = path or get_tvr_save_path()
    torch.save(buffer.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 ** 2) if os.path.exists(path) else 0.0
    print(f"  TVR compact save: {len(buffer.buffer)} ids, {size_mb:.1f} MB -> {path}")
    return path


def load_tvr_buffer(path=None, buffer=None):
    buffer = buffer or tvr_buffer
    path = path or get_tvr_save_path()
    if not os.path.exists(path):
        print(f"No TVR buffer found at {path}")
        return False
    state = torch.load(path, map_location="cpu")
    buffer.load_state_dict(state)
    print(f"Loaded TVR buffer from {path} ({len(buffer.buffer)} image ids)")
    return True


tvr_buffer = TemporalVotingBuffer(
    window=cfg.tvr_window,
    accept_ratio=cfg.tvr_accept_ratio
)


# ═══════════════════════════════════════════════════════════════════
#  Legacy triple_gate_accept (kept for backward compat / ablation)
# ═══════════════════════════════════════════════════════════════════

def triple_gate_accept(probs: torch.Tensor, conf: torch.Tensor,
                       edge: torch.Tensor, thr: float) -> torch.Tensor:
    gate_conf = conf >= thr
    edge_strength = torch.sigmoid(edge)
    gate_edge = edge_strength <= 0.6
    gate_geom = _geometry_gate(probs, thr=0.5)
    gate = gate_conf & gate_edge & gate_geom
    b = probs.shape[0]
    pseudo = torch.zeros_like(probs)
    for i in range(b):
        p = probs[i]
        p_mean = p.mean()
        p_std = p.std()
        adaptive_thr = max(p_mean + 0.5 * p_std, 0.2)
        pseudo[i] = (p >= adaptive_thr).float()
    pseudo = pseudo * gate.float()
    return pseudo


# ═══════════════════════════════════════════════════════════════════
#  NEW: Multi-Uncertainty Gate (7 gates — all contributions combined)
# ═══════════════════════════════════════════════════════════════════

def _component_level_all7_from_gates(probs: torch.Tensor,
                                     edl_unc: torch.Tensor,
                                     edge_strength: torch.Tensor,
                                     claam_agree: torch.Tensor,
                                     tvr_map: torch.Tensor,
                                     gop_gate: torch.Tensor,
                                     cafcg_gate_map: torch.Tensor,
                                     prob_thresh: float = 0.5,
                                     min_area: int = 16) -> torch.Tensor:
    """Component-level all-7 pseudo-label filter.

    Raw predicted connected components are accepted/rejected as objects. Each
    gate contributes a component-level score, avoiding pixel-wise anti-correlation
    between interior gates (EDL/Edge) and boundary gates (GOP/CAFCG).
    """
    from scipy import ndimage

    quorums = {
        "prob": 0.68,   # Component mean probability confidence
        "edl": 0.65,    # Require most component pixels to be low-uncertainty
        "edge": 0.50,   # Adaptive edge threshold handles scale
        "geom": 0.18,   # Remove thin line artifacts
        "claam": 0.01,  # Harmless until CLAAM is retrained out of collapse
        "tvr": 0.50,
        "gop": 0.30,    # Require stronger rectilinear evidence at component level
        "cafcg": 0.40,  # CAFCG signal is weak, so keep this slightly loose
    }

    final = torch.zeros_like(probs)
    for b in range(probs.shape[0]):
        prob_np = probs[b, 0].detach().cpu().float().numpy()
        raw_mask = prob_np >= prob_thresh
        labeled, n_comps = ndimage.label(raw_mask)
        if n_comps == 0:
            continue

        edl_np = edl_unc[b, 0].detach().cpu().float().numpy()
        edge_np = edge_strength[b, 0].detach().cpu().float().numpy()
        claam_np = claam_agree[b, 0].detach().cpu().float().numpy()
        tvr_np = tvr_map[b, 0].detach().cpu().float().numpy()
        gop_np = gop_gate[b, 0].detach().cpu().float().numpy()
        cafcg_np = cafcg_gate_map[b, 0].detach().cpu().float().numpy()

        edge_thresh = np.percentile(edge_np[raw_mask], 85) if raw_mask.any() else 0.7

        edl_pass = edl_np <= getattr(cfg, "edl_thr", 0.60)
        edge_pass = edge_np <= edge_thresh
        claam_pass = claam_np >= getattr(cfg, "claam_thr_end", 0.40)
        tvr_pass = tvr_np >= getattr(cfg, "tvr_thr", 0.50)
        gop_pass = gop_np >= getattr(cfg, "gop_thr", 0.25)
        # cafcg_gate_map is a pass mask here (1=pass, 0=reject), not raw disagreement.
        cafcg_pass = cafcg_np >= 0.5

        final_np = np.zeros_like(raw_mask, dtype=np.float32)
        for comp_id in range(1, n_comps + 1):
            comp = labeled == comp_id
            area = int(comp.sum())
            if area < min_area:
                continue

            rows, cols = np.where(comp)
            bbox_h = int(rows.max() - rows.min()) + 1
            bbox_w = int(cols.max() - cols.min()) + 1
            compactness = area / max(float(bbox_h * bbox_w), 1.0)

            prob_score = float(prob_np[comp].mean())
            passes = (
                prob_score >= quorums["prob"] and
                edl_pass[comp].mean() >= quorums["edl"] and
                edge_pass[comp].mean() >= quorums["edge"] and
                compactness >= quorums["geom"] and
                claam_pass[comp].mean() >= quorums["claam"] and
                tvr_pass[comp].mean() >= quorums["tvr"] and
                gop_pass[comp].mean() >= quorums["gop"] and
                cafcg_pass[comp].mean() >= quorums["cafcg"]
            )
            if passes:
                # Component-level decision, high-precision pixel emission.
                final_np[comp & edl_pass] = 1.0

        final[b, 0] = torch.from_numpy(final_np).to(probs.device, dtype=probs.dtype)
    return final


def multi_uncertainty_gate(probs: torch.Tensor, model_output: dict,
                           thr: float, epoch: int, total_epochs: int,
                           image_ids: list = None) -> torch.Tensor:
    """7-gate component-level pseudo-label acceptance.

    All seven gates are retained, but they are evaluated over each predicted
    building component instead of intersected pixel-by-pixel.
    """
    # AMP fix: under autocast, probs/edge/edl/claam are float16. Downstream gate
    # ops (cv2.Sobel, F.conv2d with float32 kernels) require float32. Cast once
    # at the entry point so every gate is safe.
    probs = probs.float()
    edge = model_output.get("edge")
    if edge is not None:
        edge = edge.float()
    edl_out = model_output.get("edl")
    if edl_out is not None and "uncertainty" in edl_out:
        edl_out = {**edl_out, "uncertainty": edl_out["uncertainty"].float()}
    claam_out = model_output.get("claam")
    if claam_out is not None and "agreement" in claam_out:
        claam_out = {**claam_out, "agreement": claam_out["agreement"].float()}
    B = probs.shape[0]

    edl_unc = edl_out["uncertainty"]
    edge_strength = torch.sigmoid(edge)
    claam_agreement = claam_out["agreement"]

    if image_ids is not None:
        tvr_masks = tvr_buffer.get_stable_mask(image_ids, probs.device)
        gate_tvr_list = []
        for i in range(B):
            m = tvr_masks[i] if i < len(tvr_masks) else torch.ones(1, 1, 1, 1, device=probs.device)
            if m.shape[-2:] != probs.shape[-2:]:
                m = torch.ones(1, 1, probs.shape[2], probs.shape[3], device=probs.device)
            gate_tvr_list.append(m.float())
        tvr_map = torch.cat(gate_tvr_list, dim=0)
    else:
        tvr_map = torch.ones_like(probs)

    # Existing GOP/CAFCG implementations return pass/fail masks; component-level
    # quorum turns those pixel masks into object-level scores.
    gop_gate = gradient_orientation_gate(probs, thr=0.5, rectilinear_thr=0.25).float()
    cafcg_gate_map = cafcg_gate(probs, disagreement_thr=max(cfg.cafcg_disagreement_thr, 0.40)).float()

    pseudo = _component_level_all7_from_gates(
        probs=probs,
        edl_unc=edl_unc,
        edge_strength=edge_strength,
        claam_agree=claam_agreement,
        tvr_map=tvr_map,
        gop_gate=gop_gate,
        cafcg_gate_map=cafcg_gate_map,
        prob_thresh=0.5,
        min_area=16,
    )

    if image_ids is not None:
        tvr_buffer.update(image_ids, pseudo)

    return pseudo

print('[cell5] 7-gate filter + TVR buffer defined')


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Evaluation metrics                                 ║
# ║                                                              ║
# ║  boundary_iou   BIoU — PRIMARY metric (best-model trigger)   ║
# ║  standard_iou   pixel-level IoU / Jaccard                    ║
# ║  hausdorff_distance  HD95 (boundary accuracy)                ║
# ║  assd_distance  Average Symmetric Surface Distance           ║
# ║  ece_score      Expected Calibration Error                   ║
# ╚══════════════════════════════════════════════════════════════╝

from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt


def boundary_iou(pred: torch.Tensor, target: torch.Tensor,
                 thr: float = 0.5, dilation_px: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """Boundary-IoU following Cheng et al. 2021.
    Uses morphological dilation to extract boundary regions at fixed pixel tolerance,
    then computes IoU only within those boundary strips."""
    pred_bin = (pred >= thr).float()
    b = pred_bin.shape[0]
    struct = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1))
    biou_vals = []
    for i in range(b):
        p = pred_bin[i, 0].cpu().numpy()  # [H, W]
        t = target[i, 0].cpu().numpy()
        # Boundary = dilated XOR original (the boundary strip)
        p_dilated = binary_dilation(p, structure=struct).astype(np.float32)
        t_dilated = binary_dilation(t, structure=struct).astype(np.float32)
        p_eroded = binary_erosion(p, structure=struct).astype(np.float32)
        t_eroded = binary_erosion(t, structure=struct).astype(np.float32)
        p_boundary = np.clip((p_dilated - p) + (p - p_eroded), 0, 1)
        t_boundary = np.clip((t_dilated - t) + (t - t_eroded), 0, 1)
        # IoU within boundary regions
        inter = (p_boundary * t_boundary).sum()
        union = p_boundary.sum() + t_boundary.sum() - inter
        biou_vals.append((inter + eps) / (union + eps))
    return torch.tensor(np.mean(biou_vals), device=pred.device)


def standard_iou(pred: torch.Tensor, target: torch.Tensor,
                 thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Standard pixel-level IoU (Jaccard index)."""
    pred_bin = (pred >= thr).float()
    inter = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean()


def hausdorff_distance(pred: torch.Tensor, target: torch.Tensor,
                       thr: float = 0.5, percentile: float = 95.0) -> torch.Tensor:
    """Hausdorff distance via boundary distance transforms (robust: uses percentile).
    Uses boundary pixels (not filled regions) for standard HD95 computation.
    Properly handles empty pred/GT cases."""
    pred_bin = (pred >= thr).float()
    b = pred_bin.shape[0]
    hd_vals = []
    for i in range(b):
        p = pred_bin[i, 0].cpu().numpy().astype(bool)
        t = target[i, 0].cpu().numpy().astype(bool)
        # Both empty -> perfect agreement -> HD = 0
        if not p.any() and not t.any():
            hd_vals.append(0.0)
            continue
        # One empty, other not -> max possible distance as penalty (capped)
        if not p.any() or not t.any():
            hd_vals.append(float(np.sqrt(p.shape[0]**2 + p.shape[1]**2)))
            continue
        # Extract boundaries for standard HD95
        p_boundary = p ^ binary_erosion(p)
        t_boundary = t ^ binary_erosion(t)
        if not p_boundary.any():
            p_boundary = p  # single-pixel components
        if not t_boundary.any():
            t_boundary = t
        dt_pred = distance_transform_edt(~p_boundary)
        dt_tgt = distance_transform_edt(~t_boundary)
        d_t2p = dt_pred[t_boundary]   # distances from GT boundary to nearest pred boundary
        d_p2t = dt_tgt[p_boundary]    # distances from pred boundary to nearest GT boundary
        hd = max(np.percentile(d_t2p, percentile), np.percentile(d_p2t, percentile))
        hd_vals.append(float(hd))
    return torch.tensor(np.mean(hd_vals), device=pred.device)


def assd_distance(pred: torch.Tensor, target: torch.Tensor,
                  thr: float = 0.5) -> torch.Tensor:
    """Average Symmetric Surface Distance via distance transforms.
    ASSD = mean of all surface-to-surface distances in both directions.
    Properly handles empty pred/GT cases."""
    pred_bin = (pred >= thr).float()
    b = pred_bin.shape[0]
    assd_vals = []
    for i in range(b):
        p = pred_bin[i, 0].cpu().numpy().astype(bool)
        t = target[i, 0].cpu().numpy().astype(bool)
        # Both empty -> perfect agreement -> ASSD = 0
        if not p.any() and not t.any():
            assd_vals.append(0.0)
            continue
        # One empty, other not -> max penalty
        if not p.any() or not t.any():
            assd_vals.append(float(np.sqrt(p.shape[0]**2 + p.shape[1]**2)))
            continue
        p_boundary = p ^ binary_erosion(p)
        t_boundary = t ^ binary_erosion(t)
        if not p_boundary.any():
            p_boundary = p
        if not t_boundary.any():
            t_boundary = t
        dt_pred = distance_transform_edt(~p_boundary)
        dt_tgt = distance_transform_edt(~t_boundary)
        d_t2p = dt_pred[t_boundary].mean() if t_boundary.any() else 0.0
        d_p2t = dt_tgt[p_boundary].mean() if p_boundary.any() else 0.0
        assd_vals.append(float((d_t2p + d_p2t) / 2.0))
    return torch.tensor(np.mean(assd_vals), device=pred.device)


def ece_score(probs: torch.Tensor, target: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    conf = probs.view(-1)
    t = target.view(-1)
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        in_bin = (conf >= bins[i]) & (conf < bins[i + 1])
        if in_bin.any():
            acc = t[in_bin].mean()
            avg_conf = conf[in_bin].mean()
            ece += (in_bin.float().mean()) * (acc - avg_conf).abs()
    return ece


def reliability_bins(probs: torch.Tensor, target: torch.Tensor, n_bins: int = 15):
    conf = probs.view(-1)
    t = target.view(-1)
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    bin_acc, bin_conf, bin_frac = [], [], []
    for i in range(n_bins):
        in_bin = (conf >= bins[i]) & (conf < bins[i + 1])
        if in_bin.any():
            bin_acc.append(t[in_bin].mean().item())
            bin_conf.append(conf[in_bin].mean().item())
            bin_frac.append(in_bin.float().mean().item())
        else:
            bin_acc.append(0.0)
            bin_conf.append(0.0)
            bin_frac.append(0.0)
    return bin_acc, bin_conf, bin_frac


print('[cell6] metrics defined  (BIoU · IoU · HD95 · ASSD · ECE)')


# Portable project directories
os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)
os.makedirs(os.path.dirname(cfg.checkpoint_path) or ".", exist_ok=True)
os.makedirs(cfg.results_dir, exist_ok=True)
os.makedirs(cfg.persist_dir, exist_ok=True)



import argparse
import glob
import csv
from tqdm.auto import tqdm

def evaluate_test_set_standalone(model, test_img_paths, test_mask_paths, device, batch_size=2):
    test_ds = WHUDataset(test_img_paths, test_mask_paths, img_size=cfg.img_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    vals = {k: [] for k in ["IoU","BIoU","Dice","ECE","Precision","Recall"]}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test"):
            imgs, masks = batch[0].to(device), batch[1].to(device)
            probs = torch.sigmoid(model(imgs)["logits"])
            pred = (probs >= cfg.pred_thr).float()
            eps = 1e-6
            tp = (pred * masks).sum(dim=(2,3))
            fp = (pred * (1-masks)).sum(dim=(2,3))
            fn = ((1-pred) * masks).sum(dim=(2,3))
            inter = tp
            union = pred.sum(dim=(2,3)) + masks.sum(dim=(2,3)) - inter
            vals["Precision"].append((tp/(tp+fp+eps)).mean().item())
            vals["Recall"].append((tp/(tp+fn+eps)).mean().item())
            vals["IoU"].append((inter/(union+eps)).mean().item())
            vals["Dice"].append((2*inter/(pred.sum(dim=(2,3))+masks.sum(dim=(2,3))+eps)).mean().item())
            vals["BIoU"].append(boundary_iou(probs, masks).item())
            vals["ECE"].append(ece_score(probs, masks).item())

    results = {k: float(np.mean(v)) for k,v in vals.items()}
    os.makedirs(cfg.results_dir, exist_ok=True)
    out_csv = os.path.join(cfg.results_dir, "test_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric","value"])
        for k,v in results.items(): w.writerow([k, f"{v:.6f}"])

    print("\nTEST RESULTS")
    for k,v in results.items():
        print(f"{k:12s}: {v:.6f}")
    print(f"Saved: {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("WHU_ROOT", cfg.data_root))
    parser.add_argument("--checkpoint", default=os.environ.get("GEORECT_SAVE_PATH", cfg.save_path))
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    cfg.data_root = args.data_root
    cfg.save_path = args.checkpoint

    test_img_dir = os.path.join(cfg.data_root, "test", "Image")
    test_mask_dir = os.path.join(cfg.data_root, "test", "Mask")
    exts = ["*.png","*.tif","*.jpg","*.jpeg","*.bmp"]
    imgs = sorted(sum([glob.glob(os.path.join(test_img_dir,e)) for e in exts], []))
    masks = sorted(sum([glob.glob(os.path.join(test_mask_dir,e)) for e in exts], []))
    n = min(len(imgs), len(masks))
    if n == 0:
        raise FileNotFoundError(f"No paired test images/masks found under {test_img_dir} and {test_mask_dir}")

    model = DABLCNet().to(device)
    if not os.path.exists(cfg.save_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.save_path}")
    model.load_state_dict(torch.load(cfg.save_path, map_location=device))
    evaluate_test_set_standalone(model, imgs[:n], masks[:n], device, args.batch_size)

if __name__ == "__main__":
    main()
