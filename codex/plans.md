# RangeMamba-C Implementation Notes

This document summarizes what was requested in the teacher's `plans.md`, what has been implemented in this codebase, what has been verified, and what should be done next.

## 1. Original Goal

The teacher's plan proposes a new architecture named **RangeMamba-C**, where `C` means **Circular**.

The intended design is not a small modification of the original RangeRet model. It is a new Mamba-based extension that keeps the successful parts of the RangeRet paper and replaces the RetNet/Circular Retention backbone with a Circular Axial Mamba backbone.

The target high-level pipeline is:

```text
Point cloud
-> range projection
-> 5-channel range image
-> convolutional stem
-> hierarchical patch/token encoder
-> Circular Axial Mamba backbone
-> FPN/U-Net style decoder
-> 2D semantic logits
-> reprojection to 3D points
-> kNN postprocessing
```

The key design requirements from the teacher plan are:

- Preserve the 360-degree circular continuity of LiDAR range images.
- Use Mamba for long-range sequence/context modeling.
- Keep CNN/local geometry modeling in the early stem and inside each block.
- Add a stronger decoder than the original RangeRet semantic head.
- Keep the existing range-view pipeline, losses, range augmentations, and kNN refinement at first for fair comparison.

## 2. Baseline Codebase State Before Implementation

The repository already had these model paths:

- `RangeRet` in `network/rangeret.py`
- `MambaRV` in `network/mambarv.py`

`RangeRet` uses:

```text
ConvStem
-> VisionEmbedding
-> RetNet/Circular Retention backbone
-> simple semantic head
```

`MambaRV` uses:

```text
CNN encoder
-> Mamba bottleneck over flattened 2D feature map
-> CNN decoder
```

The current `MambaRV` is useful as a simple Mamba experiment, but it does not implement the teacher's requested architecture because:

- It does not use a hierarchical 3-stage Mamba encoder.
- It does not separate horizontal and vertical scans.
- It does not implement explicit circular row scanning.
- It does not use an FPN/U-Net style multi-scale decoder as described in the plan.

Therefore, the correct approach was to add a new model instead of replacing `RangeRet` or extending `MambaRV`.

## 3. What Was Implemented

### 3.1 New Model File

Added:

```text
network/rangemambac.py
```

This file contains the new `RangeMambaC` model and supporting modules.

Main modules added:

- `RMSNorm2D`
- `ConvBNAct`
- `ContextBlock`
- `ResidualConvBlock`
- `RangeMambaStem`
- `PatchEmbed2D`
- `Downsample2D`
- `MambaSequenceLayer`
- `CircularRowBiMamba`
- `ColBiMamba`
- `ConvFFN`
- `CircularAxialMambaBlock`
- `DecoderFuse`
- `FPNDecoder`
- `SegHead`
- `BoundaryHead`
- `RangeMambaC`

### 3.2 Implemented Architecture Flow

For SemanticKITTI, the intended shape flow is:

```text
Input:
  [B, 5, 64, 1024]

Stem:
  [B, 96, 64, 1024]

Stage 1 patch embedding:
  [B, 128, 16, 256]

Stage 1 Circular Axial Mamba blocks:
  [B, 128, 16, 256]

Downsample to Stage 2:
  [B, 192, 8, 128]

Stage 2 Circular Axial Mamba blocks:
  [B, 192, 8, 128]

Downsample to Stage 3:
  [B, 256, 4, 64]

Stage 3 Circular Axial Mamba blocks:
  [B, 256, 4, 64]

FPN decoder:
  [B, 96, 64, 1024]

Segmentation head:
  [B, 20, 64, 1024]
```

The model returns the same standardized output interface used by the rest of the repository:

```python
return {"logits": logits, "aux": aux}
```

This is important because the existing `Trainer`, `User`, and `network.interfaces.get_logits()` already expect this style.

### 3.3 Circular Row Mamba

The horizontal scan path is implemented by `CircularRowBiMamba`.

Input shape:

```text
[B, C, H, W]
```

It reshapes each row into a sequence:

```text
[B, C, H, W]
-> [B * H, W, C]
```

Then it applies circular padding:

```text
[last k tokens] + [original row tokens] + [first k tokens]
```

Then it applies:

- forward Mamba
- backward Mamba over the reversed sequence
- crop back to original width
- linear fusion of forward and backward features

This implements the teacher plan's recommended "Option A: cyclic padding before row scan".

### 3.4 Vertical Column Mamba

The vertical scan path is implemented by `ColBiMamba`.

Input shape:

```text
[B, C, H, W]
```

It reshapes each column into a sequence:

```text
[B, C, H, W]
-> [B * W, H, C]
```

Then it applies:

- forward Mamba
- backward Mamba over the reversed sequence
- linear fusion

No circular padding is used vertically, because vertical LiDAR rows do not wrap around.

### 3.5 Local Depthwise Convolution Branch

Each `CircularAxialMambaBlock` also includes a local depthwise convolution branch:

```python
self.local_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
```

This preserves local spatial modeling, following the teacher plan's recommendation and matching the RangeRet paper's observation that local geometry matters.

### 3.6 Fusion in Each Block

Each `CircularAxialMambaBlock` computes:

```text
local branch
row circular Mamba branch
column Mamba branch
```

Then it fuses them with:

```python
Conv2d(dim * 3, dim, kernel_size=1)
BatchNorm2d
GELU
```

This is the simpler fusion option from the teacher plan. The more complex gated fusion is not implemented yet.

### 3.7 ConvFFN

Each block ends with a convolutional feed-forward path:

```text
1x1 expand
-> GELU
-> depthwise 3x3 conv
-> dropout
-> 1x1 project
```

This is implemented by `ConvFFN`.

### 3.8 FPN/U-Net Style Decoder

The decoder is implemented by `FPNDecoder`.

It uses skip connections from:

- Stage 2
- Stage 1
- Stem output

Decoder flow:

```text
F3 -> project -> upsample to F2 size -> concat with F2 -> fuse
-> project -> upsample to F1 size -> concat with F1 -> fuse
-> project -> upsample to F0/stem size -> concat with F0 -> fuse
-> segmentation head
```

This is stronger than the original RangeRet simple semantic head and follows the teacher plan.

### 3.9 Optional Boundary Head

The code includes `BoundaryHead`, and `RangeMambaC` can create it if:

```yaml
use_boundary_head: true
```

However, the first config sets:

```yaml
use_boundary_head: false
```

Reason: the repository's current `BoundaryLoss` is not designed for a separate predicted boundary map. It currently works from semantic probabilities and labels. If we enable a boundary prediction head, we must also add target generation and a new loss path. That is intentionally left for a later phase.

## 4. Factory Integration

Updated:

```text
network/factory.py
```

Added support for:

```yaml
model:
  name: "rangemambac"
```

Aliases added:

```python
{"rangemambac", "range_mamba_c", "rangemamba_c"}
```

This means existing entrypoints can build the model without changing `train.py` or `infer.py`.

Supported command path:

```text
train.py
-> modules.trainer.Trainer
-> network.factory.build_model
-> network.rangemambac.RangeMambaC
```

## 5. New Config

Added:

```text
config/RangeMambaC-semantickitti.yaml
```

Key model settings:

```yaml
model:
  name: "rangemambac"
  pretrained_component: "full"
  params:
    input_dim: 5
    stem_dim: 96

    stage_dims: [128, 192, 256]
    stage_depths: [3, 4, 6]

    patch_kernel: 7
    patch_stride: 4
    patch_padding: 3

    row_pad: [16, 8, 4]
    mamba:
      d_state: 16
      d_conv: 4
      expand: 2

    ffn_expand: 2
    dropout: 0.0
    use_boundary_head: false
```

This matches the teacher plan's first recommended `RangeMamba-C-Small` version.

The config keeps the paper-style SemanticKITTI training setup:

```yaml
train:
  epochs: 64
  learning_rate: 1.0e-2
  weight_decay: 0.05
  optimizer: AdamW
  batch_size: 8
  range_aug: True
  scheduler:
    name: "WarmupCosine"
```

The config also keeps:

- CE + boundary + Lovasz loss bundle
- kNN postprocess
- SemanticKITTI 64 x 1024 range image size
- 20 training classes including unlabeled/ignore class

## 6. Requirements Update

Updated:

```text
requirements.txt
```

Added:

```text
mamba-ssm
```

Reason: both the existing `network/mambarv.py` and the new `network/rangemambac.py` import:

```python
from mamba_ssm import Mamba
```

The exact version is intentionally not pinned yet because `mamba-ssm` compatibility depends strongly on the installed PyTorch/CUDA version. The README says this codebase was tested with:

- PyTorch 2.2.2 + CUDA 12.1
- PyTorch 2.4.0 + CUDA 12.4

The install version should be chosen to match the user's actual CUDA/PyTorch environment.

## 7. Script Update

Updated:

```text
script.md
```

Added the RangeMamba-C training command:

```bash
python train.py  --dataset ../dataset/SemanticKitti/data_odometry_velodyne/dataset/  --data ./config/labels/semantic-kitti.yaml  --config ./config/RangeMambaC-semantickitti.yaml  --log ./log/rangemambac_kitti  --fp16
```

Important dataset path note:

`--dataset` must point to the dataset root folder that contains `sequences/`, not the `sequences/` folder itself.

Correct:

```text
../dataset/SemanticKitti/data_odometry_velodyne/dataset/
```

Incorrect:

```text
../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences/
```

The parser internally appends `sequences`.

## 8. Verification Status

Static verification completed:

```bash
python -m py_compile network/rangemambac.py network/factory.py
```

Result:

```text
passed
```

YAML verification completed:

```bash
python -c "import yaml; yaml.safe_load(open('config/RangeMambaC-semantickitti.yaml', encoding='utf-8')); print('yaml ok')"
```

Result:

```text
yaml ok
```

Full model-build / forward verification was not completed in the PowerShell environment because that Python environment does not have `torch` installed:

```text
ModuleNotFoundError: No module named 'torch'
```

Attempting to call WSL from the PowerShell sandbox failed with:

```text
Access is denied.
Error code: Wsl/Service/CreateInstance/E_ACCESSDENIED
```

Therefore, the next model-build and forward test must be run by the user in the actual WSL/GPU training environment.

## 9. Immediate Next Steps

### Step 1: Confirm Dependencies in WSL

Run:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python3 -c "import mamba_ssm; print('mamba_ssm ok')"
python3 -c "import timm; print(timm.__version__)"
```

If `mamba_ssm` is missing, install a version compatible with your Torch/CUDA setup.

Start with:

```bash
pip install mamba-ssm
```

If build/install fails, the version may need to be matched to your CUDA/PyTorch environment.

### Step 2: Run a Model Build Test

Run:

```bash
python3 - <<'PY'
import yaml
from network.factory import build_model

arch = yaml.safe_load(open("config/RangeMambaC-semantickitti.yaml", encoding="utf-8"))
model = build_model(arch, (64, 1024), arch["dataset"]["num_classes"])
print(type(model).__name__)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")
PY
```

Expected:

```text
RangeMambaC
<some parameter count> M params
```

### Step 3: Run a Small Forward Test

Run this on GPU if possible:

```bash
python3 - <<'PY'
import yaml
import torch
from network.factory import build_model
from network.interfaces import get_logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arch = yaml.safe_load(open("config/RangeMambaC-semantickitti.yaml", encoding="utf-8"))
model = build_model(arch, (64, 1024), arch["dataset"]["num_classes"]).to(device).eval()
x = torch.randn(1, 5, 64, 1024, device=device)

with torch.inference_mode():
    out = model(x)
    logits = get_logits(out)

print(logits.shape)
PY
```

Expected:

```text
torch.Size([1, 20, 64, 1024])
```

### Step 4: Benchmark Memory and Runtime

Use the existing benchmark script:

```bash
python3 benchmark.py \
  --config config/RangeMambaC-semantickitti.yaml \
  --device cuda \
  --batch-size 1 \
  --warmup 10 \
  --iters 30 \
  --fp16
```

If memory is too high, reduce the model:

```yaml
stage_dims: [96, 160, 224]
stage_depths: [2, 3, 4]
row_pad: [8, 4, 2]
```

### Step 5: Run a Short Training Smoke Test

Before launching a full 64-epoch run, train briefly with a temporary debug config:

```yaml
train:
  epochs: 1
  batch_size: 1
  workers: 0
```

Command:

```bash
python3 train.py \
  --dataset ../dataset/SemanticKitti/data_odometry_velodyne/dataset/ \
  --data ./config/labels/semantic-kitti.yaml \
  --config ./config/RangeMambaC-semantickitti.yaml \
  --log ./log/rangemambac_debug \
  --fp16
```

Only after the debug run succeeds should we launch:

```bash
python3 train.py \
  --dataset ../dataset/SemanticKitti/data_odometry_velodyne/dataset/ \
  --data ./config/labels/semantic-kitti.yaml \
  --config ./config/RangeMambaC-semantickitti.yaml \
  --log ./log/rangemambac_kitti \
  --fp16
```

## 10. Known Risks and Design Choices

### Risk 1: Mamba Memory and Speed

The architecture uses separate forward and backward Mamba modules for both rows and columns.

At Stage 1:

```text
row sequences: B * 16 sequences of length 256
column sequences: B * 256 sequences of length 16
```

This should be manageable, but it is still heavier than a simple bottleneck-only Mamba model.

If memory is too high, reduce:

- `stage_depths`
- `stage_dims`
- `row_pad`
- batch size

### Risk 2: `mamba-ssm` Installation

`mamba-ssm` can be sensitive to CUDA, PyTorch, and compiler versions.

If install fails, resolve the package/environment first before debugging the model.

### Risk 3: Boundary Head Is Not Yet Trained

The model supports an optional boundary head, but the config disables it.

Do not enable:

```yaml
use_boundary_head: true
```

until we implement:

- semantic boundary target generation from labels
- boundary loss function for the predicted boundary map
- trainer/loss bundle support for `outputs["aux"]["boundary"]`

### Risk 4: This Is a New Architecture, Not Paper-Reproduction RangeRet

The RangeRet paper's `65.7` SemanticKITTI validation mIoU corresponds to the original RangeRet/CiR architecture with range augmentations.

RangeMamba-C is a new proposed extension. It may perform better or worse initially. It should be treated as an experiment and evaluated with ablations.

## 11. Suggested Ablation Plan

Run experiments in this order so we know what actually helps.

### Experiment A: Base RangeMamba-C

Config:

```text
RangeMambaC-semantickitti.yaml
```

Settings:

- full CE + boundary + Lovasz loss bundle
- range augmentations enabled
- kNN postprocess enabled for final validation through `infer.py`
- no boundary head

Purpose:

Establish whether the model trains and reaches a meaningful baseline.

### Experiment B: Smaller RangeMamba-C

Use this if memory/runtime is too high:

```yaml
stage_dims: [96, 160, 224]
stage_depths: [2, 3, 4]
row_pad: [8, 4, 2]
```

Purpose:

Find a stable, efficient version that can train fully.

### Experiment C: No Column Mamba

Modify the block to remove `ColBiMamba`, or temporarily make the column output zero/identity.

Purpose:

Measure whether vertical scan modeling helps.

### Experiment D: No Circular Row Padding

Set:

```yaml
row_pad: [0, 0, 0]
```

Purpose:

Measure the value of circular row context.

### Experiment E: Simple Fusion vs Gated Fusion

Current implementation uses:

```text
Conv1x1([row, col, local])
```

Teacher plan also proposed gated fusion:

```text
G = sigmoid(Conv1x1([row, col, local]))
X_mix = G1 * row + G2 * col + G3 * local
```

Purpose:

Check whether gated fusion improves performance.

### Experiment F: Boundary Head

Only after base training is stable:

- enable `use_boundary_head`
- implement boundary target generation
- add BCE or Dice loss
- test whether boundary refinement improves poles/signs/bicycles/person boundaries

## 12. Evaluation Procedure

Training validation inside `Trainer.validate()` does not apply kNN postprocess.

For paper-style final evaluation, use `infer.py` after training:

```bash
python3 infer.py \
  --dataset ../dataset/SemanticKitti/data_odometry_velodyne/dataset/ \
  --data ./config/labels/semantic-kitti.yaml \
  --config ./config/RangeMambaC-semantickitti.yaml \
  --model ./log/rangemambac_kitti/rangemambac-best.pt \
  --split valid \
  --log ./out/rangemambac_kitti_valid \
  --fp16
```

Note: the checkpoint filename is based on `self.model_name` from the config. With:

```yaml
model:
  name: "rangemambac"
```

the expected checkpoint path is:

```text
./log/rangemambac_kitti/rangemambac-best.pt
```

## 13. Final Current Status

Current implementation state:

- RangeMamba-C model file added.
- Factory integration added.
- SemanticKITTI config added.
- Training command added to `script.md`.
- `mamba-ssm` added to requirements.
- Syntax checks passed.
- YAML parsing passed.
- Full forward test still needs to be run in WSL/GPU environment.

Recommended next action:

Run the model build and dummy forward tests in WSL. If they pass, run a 1-epoch debug training run before launching the full 64-epoch experiment.
