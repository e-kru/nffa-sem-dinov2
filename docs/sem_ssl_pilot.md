# SEM SSL pilot with Meta DINOv2

This integration imports Meta's original `DINOLoss`, `iBOTPatchLoss`, `KoLeoLoss`,
`DINOHead`, ViT-S/14 backbone, augmentation, mask generator, collator, parameter
groups and cosine scheduler. No mathematical loss implementation is copied into
this repository. The commands below run bounded checks, not full training.

For a detailed German explanation of the data flow, training semantics and
responsibility of each file, see [sem_ssl_pipeline.md](sem_ssl_pipeline.md).
The user-run A100 synthetic CUDA/BF16 smoke test has now passed; its results are
documented there with links to the saved reports.

## Dependency and pretrained weights

- Upstream: <https://github.com/facebookresearch/dinov2>
- Exact revision: [`7764ea0f912e53c92e82eb78a2a1631e92725fc8`](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8).
- Checkout on this workstation: `/path/to/workspace/external/dinov2`, outside
  the project Git repository, detached at that revision. `DINOV2_ROOT` overrides
  the location. Every entry point rejects a different revision, a dirty checkout,
  or a shadowing `dinov2` package. Do not edit the external code.
- Source pin and weight identity: `dinov2-upstream.json`.
- Released backbone: <https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth>.
- SHA-256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
  A fresh HTTPS download from Meta matched the existing cached file byte for byte
  on 2026-11-11. Both pilot entry points require this artifact identity.

Initialize the checkout from the project root:

```bash
python3 -m scripts.setup_dinov2_external
# Or: python3 -m scripts.setup_dinov2_external --directory /your/external/dinov2
export DINOV2_ROOT=/path/to/workspace/external/dinov2
```

The helper requires Python 3.10 or later; the isolated environment below provides
it. If the cached weight file is unavailable on the A100 host, download the
pinned release to an external weights directory and use that path for
`--checkpoint`:

```bash
mkdir -p ../external/weights
curl --fail --location \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
  --output ../external/weights/dinov2_vits14_pretrain.pth
sha256sum ../external/weights/dinov2_vits14_pretrain.pth
```

The setup helper never resets dirty code or modifies the project's Git index.
An existing clone must contain the pinned revision. Source code is Apache-2.0;
consult the upstream license and model card for the released weights.

## Current environment: shared sem-gpu

For a shared filesystem, both machines can use the same data directory. The example environment is
prepared directly at:

`sem-gpu`

No additional Conda environment is needed or was created for this preparation.
Activate the existing environment on either host:

```bash
cd /path/to/nffa-sem-dinov2
conda activate sem-gpu
export DINOV2_ROOT=/path/to/workspace/external/dinov2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
```

Existing packages were reused unchanged: Python 3.11.17, PyTorch
`2.13.0+cu126`, torchvision `0.28.0+cu126`, CUDA toolkit/runtime packages for
12.6, cuDNN 9.10.2.21, Triton 3.7.1, NumPy 2.4.6, Pillow 12.3.0, PyYAML 6.0.3
and pytest 9.2.0. This is a deviation from Meta's historical PyTorch 2.0/CUDA 11.7
training stack, verified here through CPU integration rather than a downgrade.

Exactly these missing packages were added, with dependency resolution disabled:

| Package | Installed version |
| --- | --- |
| omegaconf | 2.3.0 |
| antlr4-python3-runtime | 4.9.3 |
| fvcore | 0.1.5.post20221221 |
| iopath | 0.1.10 |
| yacs | 0.1.8 |
| termcolor | 2.4.0 |
| tabulate | 0.9.0 |
| portalocker | 2.10.1 |

OmegaConf and ANTLR support config loading. fvcore/iopath and their small helpers
support the original trainer/checkpointer imports and reference tests. Existing
PyYAML, NumPy, Pillow, tqdm and typing-extensions satisfy the remaining runtime
dependencies. No xFormers build was installed: native PyTorch SDPA attention and
Meta's own iBOT PyTorch loss fallback are used. `torchmetrics`/`cuml-cu11` belong
to evaluation paths and `submitit` to the unused Slurm launcher; they are not
required for this direct single-GPU SSL pilot. Optional `tifffile` is unnecessary
for the curated JPEG SEM input and synthetic tests.

The external checkout is resolved by the project's guarded `activate()` helper
and `DINOV2_ROOT`; no global `.pth` file or editable install changes other model
loaders. The external source remains unchanged and pinned.

## Inventory and package-change audit

Before any package installation, a complete inventory was saved under
`outputs/environment_audits/sem-gpu_2026-11-11/before/`. It contains
`pip freeze --all`, pip JSON, `pip check`, Conda JSON, explicit Conda package URLs,
environment YAML, Conda history, every Conda metadata record, and distribution
metadata/RECORD hashes. The same capture is in `after/`.

`changes.json` confirms all **100 existing Python distributions** retain identical
versions and metadata/RECORD hashes, all Conda metadata files are unchanged, and
only the eight packages above were added directly inside the shared environment.
`pip check` passed both before and after. No existing package was removed or
reinstalled. These are package/metadata integrity checks, not a byte-for-byte
backup of every environment file.

Downloaded source archives, eight built/downloaded pure-Python wheels and SHA-256
hashes, plus pip's dry-run/install JSON reports, are retained in the audit folder.
The reviewed installation used `--no-deps --no-index` and the local wheel folder.
`environments/requirements-micronet-ssl.txt` lists exactly those additions; inspect
the current inventory before using it on another environment, installing only
missing entries. Do not apply Meta's historical full requirements to this shared
environment: they pin a different PyTorch/CUDA stack. The old environment files
are historical reproduction references, not the current setup instructions.

For a future inventory, choose a new directory:

```bash
python -m scripts.inventory_ssl_environment \
  --output outputs/environment_audits/sem-gpu_before_future_changes
```

## First A100 smoke test on gpu-host

Shared storage does not supply GPU hardware to the workstation. CPU checks here
do not establish A100 driver compatibility, CUDA kernel execution, BF16 support or
GPU memory use. Codex has not connected to `gpu-host` or launched GPU training.
The user has since run the prepared smoke test successfully on an A100 80 GB:
three native BF16 steps, a correct checkpoint/export round trip and approximately
1.4 GiB PyTorch peak allocated memory at batch 2. This does not establish
long-run stability or throughput for a real SEM training job.
Run this **yourself on gpu-host**, selecting one allocated A100 80 GB:

```bash
cd /path/to/nffa-sem-dinov2
CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_dinov2_a100.sh
```

The script uses the existing `sem-gpu/bin/python` directly and refuses to run
on another host. It creates a new timestamped output root, tests Meta's complete
official trainer for two synthetic steps first, then uses a failed report from
that exact A100/PyTorch stack for a three-step BF16 native-adapter pilot, batch 2,
zero data workers. It then reloads the adapter checkpoint and verifies the raw
backbone export on CPU. Reports show finite losses, strict initialization,
student updates, frozen teacher and CUDA peak allocated memory. No real SEM
images or complete training schedule are run by this smoke script.

With the prepared environment, the complete official trainer is expected to
report `xFormers is required for training`; this dependency is required by its
FSDP/nested-tensor execution path. The native adapter retains Meta's original
objectives and avoids that dependency. If the official probe unexpectedly
passes, the wrapper reports that result and stops without running the adapter.
Direct commands and the real SEM continuation are described below.

## First try the complete official trainer

On the A100 host, expose exactly one GPU and run this first, using a **new**
output directory. The command exits unsuccessfully on any failure and writes
`report.json` plus `traceback.txt`. There is no automatic adapter fallback.

```bash
export CUDA_VISIBLE_DEVICES=0
python -m scripts.probe_dinov2_official \
  --checkpoint /path/to/workspace/cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth \
  --steps 2 --output outputs/sem_ssl_checks/official_a100
```

This calls the original `SSLMetaArch`, `prepare_for_distributed_training` and
`do_train`. It uses Meta's original optimizer, FP16/FSDP precision policy, gradient
clipping, center updates, parameter EMA, sampler and checkpointing. Single-rank
NCCL/FSDP is enabled by Meta's own initialization; no DDP or multi-GPU job is
launched. The official code does support initialization of a single-rank job;
there is no evidence here that it inherently requires multiple GPUs.

The probe changes only configuration, strict pretrained bootstrap and dataset
injection: ViT-S/14, batch 2, two local crops, 518/98 pixel crops, two iterations,
no LR warmup or last-layer freeze, no evaluation. Prototype counts remain 65,536.
The 518 global crop gives `build_model_from_cfg` the positional grid of the
released checkpoint. The official `student.pretrained_weights` branch expects a
`{"model": ...}` wrapper and uses `strict=False`; the released checkpoint is a
raw backbone state dict. We instead strictly load it into `student.backbone`
before official preparation copies the student into the teacher. No upstream
initialization or training methods are patched. The synthetic dataset is supplied
through the loop's dataset factory; losses and optimization stay upstream.

If it passes, use the complete official trainer for a **two-step real SEM check**:

```bash
python -m scripts.probe_dinov2_official --sem \
  --checkpoint /path/to/workspace/cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth \
  --steps 2 --output outputs/sem_ssl_checks/official_sem_a100
```

The official probe deliberately permits at most three steps. Its saved config
is a diagnostic schedule, not a post-training recipe. Passing requires finite
metrics from the complete official training loop. It does not establish useful
representation learning from such a short run.

## Adapter only when the official probe fails

`src/ssl/adapter.py` is a single-device orchestration layer. It imports the
unchanged upstream components and does not need FSDP, NCCL or nested xFormers
attention. CUDA runs require a failed official-probe report from the same
upstream revision, PyTorch version and GPU model. A passing official report is
rejected: prefer the complete official path when it works.

```bash
python -m scripts.pilot_dinov2 \
  --checkpoint /path/to/workspace/cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth \
  --official-report outputs/sem_ssl_checks/official_a100/report.json \
  --device cuda --precision bf16 --steps 10 --num-workers 8 \
  --output outputs/sem_ssl_checks/adapter_sem_a100
```

The default is real SEM train data, batch 8, 224-pixel globals, eight 98-pixel
locals, full 65,536-prototype shared head and all three objectives. It permits
2–20 optimizer steps, never resumes or overwrites an existing run, and enforces
one visible A100. Use `--synthetic` for a data-independent smoke test first.
`--batch-size` and `--num-workers` are configurable. CPU checks require explicit
`--device cpu --precision fp32`.

The SEM defaults read `outputs/dataset_v3/manifest.csv`, split B and the additional
split A holdouts. Only included `crop_v2` images in training in both protocols
survive; entire B groups are removed if any selected member is held out in A.
Labels are not used. No splits or source images are changed. The V2 metadata-strip
crop runs once on the original RGB image before Meta's random multicrop
augmentation. No downstream center crop or preliminary resize is applied.
Override locations with `--manifest`, `--split`, repeated `--holdout-split`, and
`--data-root` (or the existing `NFFA_SEM_ROOT`). When supplying another dataset,
provide its complete evaluation holdouts explicitly. Groups crossing partitions,
duplicate IDs, unsupported crop modes, missing files, and paths escaping the
data root are rejected.

## Semantics and deviations

The invariant teacher/student semantics are:

- Strictly initialize **all** student backbone tensors from the pinned release;
  copy the entire student backbone and freshly initialized head into the teacher.
  The release does not provide SSL head weights. Both use the original backbone
  and head classes. The 518 positional grid is retained and interpolated by
  Meta's own backbone at each input size.
- Teacher is always in evaluation mode with gradients disabled. Teacher globals
  are unmasked; student globals use Meta's block masks; locals are unmasked.
- Reverse the two teacher CLS views for global DINO cross-view matching. Every
  local student view matches both global teacher views. Patch targets retain
  their original view order and use exactly the same flattened mask indices and
  per-image mask weights as Meta's collator.
- Call Meta's `softmax_center_teacher`, `update_center`, `DINOLoss.forward`,
  `iBOTPatchLoss.forward_masked` and `KoLeoLoss.forward` unchanged. Centering has
  the upstream one-step-delayed update. DINO normalization is `2 + 2*n_local`;
  the global term has factor 2; iBOT's factors `2 * (1/2)` cancel. KoLeo is
  evaluated independently within each of the two global views, weighted by 0.1.
- Use Meta's AdamW parameter groups, layerwise LR decay, patch LR multiplier,
  sqrt batch-size LR rule, zero LR for frozen last layers, cosine LR/weight decay,
  teacher momentum and temperature schedules. Clip gradients separately for
  backbone and head. Update teacher parameters as `m*teacher + (1-m)*student`
  **after** the optimizer step, including trainable weight-norm parameters.

All adapter deviations from official execution/configuration:

| Area | Adapter behavior and implication |
| --- | --- |
| Device execution | Plain modules and one optimizer; no FSDP wrappers, sharding, stream synchronization, distributed gradient reduction or scaler. No gradient accumulation. |
| Crop execution | Two sequential student backbone calls instead of xFormers nested tensors. `XFORMERS_DISABLED=1` selects Meta's own native PyTorch SDPA attention. The stochastic-depth branch/RNG sequence differs from the nested implementation; probabilities and the backbone code remain upstream. GPU numerical parity remains unverified. |
| Head packing | Concatenate actual CLS/masked tokens with `torch.cat` instead of xFormers packing and padded buffers. The original head has no batch normalization or dropout, so unused padded rows do not affect selected outputs or gradients. CPU reference loss/gradient parity is tested. |
| Precision | FP32 master parameters and loss/center inputs, BF16 autocast only for backbone/head on A100. CPU uses FP32. Meta's official loop uses FP16 FSDP and a sharded gradient scaler by default. The upstream iBOT implementation selects its own fused cross entropy if xFormers is installed, otherwise its original PyTorch fallback. |
| Teacher normalization | Official default `centering`, not the `sinkhorn_knopp` option used by some large-scale configs. Sinkhorn mode and a separate iBOT head are rejected rather than approximated. |
| Backbone | Released distilled ViT-S/14, no registers, unchunked MLP blocks, drop path 0.3; no ViT-L/G, SwiGLU, Cell-DINO or channel adaptation. Teacher and student use this same architecture. |
| Crop/data policy | SEM V2 strip crop and train-only group-aware manifest replace ImageNet loading; Meta photometric/geometric augmentation is unchanged, including color jitter, blur and solarization. These SEM augmentation choices remain an experimental assumption. 224/98 follows patch-14 geometry rather than the generic defaults 224/96. |
| Pilot schedule | `configs/sem_ssl_pilot.yaml` merges over the exact upstream defaults: batch 8, base LR 2e-4, 2,000-step schedule horizon, 100-step LR/temperature warmup and last-layer freeze. A bounded pilot runs only its first few steps, so the last layer remains frozen. The horizon does not launch training; these are diagnostic settings, not tuned SEM hyperparameters. |
| Sampling | Seeded shuffled finite PyTorch DataLoader with drop-last and explicit worker seeding; cycling if needed replaces Meta's sharded infinite sampler. No trajectory-equivalence claim across samplers or worker counts. |
| Metrics | `total_loss` is the actual scalar backpropagated. iBOT and weighted KoLeo metrics show their full contributions; Meta logs half of each. DINO metrics are unweighted as upstream. No downstream/test evaluation occurs. |
| Checkpoints | Save plain student/teacher/loss state, AdamW, config, iteration and RNG states, plus raw teacher backbone export. Flush the final pending center update before saving so Python-only pending fields are not lost. No resume CLI or claim of exact worker/prefetch replay. No periodic evaluation or best-checkpoint selection in this bounded pilot. |
| Reproducibility | Save full merged config, input manifest/split/checkpoint hashes, selected image IDs/groups, source hashes, seed, package versions, project Git state, metrics, teacher-gradient/backbone-update checks and CUDA peak memory when available. Image file bytes are not all hashed. |

## Current sem-gpu compatibility and verification

The synthetic CPU pilot in the prepared shared environment completed three
steps with the pinned released backbone and full 65,536-prototype head, batch 2,
224/98 pixel crops and FP32. Initialization matched every released backbone
tensor, the initial student/teacher states matched, all losses were finite, the
student backbone changed, and the teacher had no gradients. Total losses were
16.471920, 16.528072 and 16.550316. These are smoke-test results, not evidence of
improved SEM representations. Config, source/package versions, optimizer state,
student/teacher state, RNG state and raw teacher backbone were saved under
`outputs/sem_ssl_checks/micronet_synthetic_cpu_2026-11-11/`.

Verification in `sem-gpu` passed:

- Full `python -m pytest -q`: **132 passed, 2 skipped, 3 subtests passed**.
  The two skips are the existing sandbox IPC/socket limitations. The eight SSL
  tests also passed separately, including loss and all-student-gradient comparison
  with the original `SSLMetaArch.forward_backward`, teacher/EMA and center state.
- The full pilot checkpoint strictly reloaded with every model tensor identical
  and finite; AdamW state restored successfully.
- The raw exported backbone matched the checkpoint teacher exactly and loaded
  strictly into a fresh original Meta backbone. Both CLS and patch-feature
  maximum absolute errors were **0.0** on a fixed CPU input.
- `pip check` passed before and after installation. Python source syntax,
  `bash -n scripts/smoke_dinov2_a100.sh` and `git diff --check` passed.

Small evidence files are retained in
lokales Prüfprotokoll,
including both package freezes, the package delta, training metrics and checkpoint
verification. The complete before/after environment inventories and installation
archives remain in the shared audit folder described above. Large checkpoints
are retained only in the ignored run directory.

The original complete trainer was also imported using `sem-gpu`: fvcore,
iopath, the original losses/head/backbone and FSDP imports succeeded up to the
explicit `AssertionError: xFormers is required for training`. The report is in
`outputs/sem_ssl_checks/micronet_official_cpu_2026-11-11/report.json`. This is the
reason to use the native adapter with this existing environment, rather than
replacing its PyTorch/CUDA stack. On A100 the official probe runs again so its
failure report records the actual GPU and identical shared packages.

Meta's original `weight_norm` head and `torch.cuda.amp.autocast` in KoLeo emit
deprecation warnings under PyTorch 2.13. They execute successfully in the CPU
smoke run; their implementations have not been rewritten to silence warnings.
Without a compatible xFormers build, the complete official FSDP loop remains
unavailable. The user-run A100 smoke test now verifies three native CUDA/BF16
steps at batch 2 on an A100 80 GB, followed by a successful checkpoint/export
check. Real SEM GPU training, longer-run stability and larger-batch memory use
remain to be measured. See the [pipeline explanation](sem_ssl_pipeline.md) for
the recorded GPU values and report links.

CPU reproduction in the activated `sem-gpu`, always using new output paths:

```bash
python -m pytest -q
python -m scripts.pilot_dinov2 \
  --checkpoint /path/to/workspace/cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth \
  --synthetic --device cpu --precision fp32 --batch-size 2 --num-workers 0 --steps 3 \
  --output outputs/sem_ssl_checks/micronet_synthetic_cpu_repeat
python -m scripts.verify_dinov2_checkpoint \
  --run outputs/sem_ssl_checks/micronet_synthetic_cpu_repeat
```

The checkpoint verifier loads our own trusted full checkpoint with
`weights_only=False` because it contains Python/NumPy RNG state. It strictly
reloads model and optimizer state, checks finiteness and equality of model
tensors, then loads the raw export with `weights_only=True` into a fresh original
Meta backbone. CLS and patch outputs must match the restored teacher exactly.
It does not implement or promise exact DataLoader/prefetch resume.

## Initial isolated-CPU verification on 2026-11-11 (historical)

The attached workstation (`workstation`) exposes no NVIDIA device or CUDA runtime.
No A100 job or full post-training run was launched. The isolated environment was
created locally before the subsequent authorized preparation of `sem-gpu`.

- The complete official trainer was attempted first in the ready CPU environment:
  it stopped at `AssertionError: xFormers is required for training`. CUDA is also
  unavailable. This is a local dependency/hardware blocker, **not evidence that
  official single-A100 FSDP cannot work**.
- Eight SSL tests passed, including original component identity, teacher eval/no
  gradients, parameter EMA after optimizer, center serialization, schedule/freezing,
  group holdouts, pin enforcement and empty-mask rejection.
- The reference test executes the original `SSLMetaArch.forward_backward` on CPU
  and compares both losses and **all student parameter gradients** against the
  adapter. Only device transfer, FSDP resharding and xFormers tensor packing are
  replaced in that test. It uses FP32, drop path zero and a smaller head for speed.
  It does not validate xFormers kernels, FSDP internals, stochastic depth, BF16 or
  CUDA. No such substitutions are used in the official runtime probe.
- A three-step CPU synthetic run used the real released ViT-S/14, original full
  65,536-prototype head, two 224 globals, eight 98 locals, batch 2 and the unchanged
  objectives. Strict pretrained equality and initial student/teacher equality
  passed; the backbone updated and the teacher had no gradients. Total losses:
  16.498936, 16.531553, 16.573664. These are smoke-test values, not learning results.
- The real SEM data path also passed a three-step CPU check with the same full
  model/head and batch 2. The corpus selection retained 8,061 images after both
  evaluation protocols' holdouts and conservative group exclusions. Only six
  sampled images were used by the three optimization steps. Total losses:
  16.534630, 16.535049, 16.541397. Initialization, backbone-update and teacher-gradient
  checks passed, and the complete pilot state plus raw teacher backbone were saved.
- Existing downstream regression tests passed: 124 passed, 2 skipped because the
  sandbox blocks IPC sockets. No existing benchmark or held-out evaluation ran.
- These test totals come from two completed runs (SSL tests and existing tests
  separately). A later combined rerun was interrupted during shared-filesystem
  Git stalls before producing test results. Final Python syntax and whitespace
  checks passed for all eight new Python source/test files.

Small reports, configs, metrics and package freezes are retained in
lokales Prüfprotokoll.
Large checkpoints and the selected-image list remain under the ignored
`outputs/sem_ssl_checks/adapter_{synthetic,sem}_cpu` directories. Source hashes in
the SEM report identify the code actually exercised; subsequent CLI-only changes
add the A100 memory-size guard and support config paths outside the repository.

The official single-A100 import probe and native synthetic GPU smoke check have
now been run by the user. The next acceptance check is a short real SEM GPU
pilot with the native adapter; the complete official loop still requires xFormers.
Inspect finite losses, backbone updates, teacher gradients, CUDA peak memory and
the recorded failure reason before deciding on a separate full-training recipe.
