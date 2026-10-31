# Aligned two-stage training

The default full recipe reuses the trained eight-task **head + wrist** WAVE-0 world model. Stage 1 trains the independent tactile predictor, Belief Updater and Future Corrector while freezing the visual world model and its action/state encoders. Stage 2 freezes the tactile predictor and visual world modules, and trains each task policy jointly with its Updater and Corrector. No contact-state classification labels or separate tactile pretraining phase are required. See [the complete recipe](tactile_training.md) for initialization, losses, data, commands and validation.

The pinned reference in `vendor/aligned_reference` supplies preprocessing, task statistics and segment boundaries. Actions use `q[t+i+1] - q[t]` for seven arm coordinates and absolute gripper position, padded to 32 dimensions for flow supervision. Shared-world statistics come from the selected pretrained WM. Current and future head/wrist frames use identical time indices and PIL resize-pad; camera order is head, wrist. Policy vision learns in Stage 2; the unaugmented visual target tower and AnyTouch2 remain frozen.

| Stage | Per-GPU batch | Gradient accumulation | Global batch | Maximum updates |
| --- | ---: | ---: | ---: | ---: |
| Shared contact Stage 1, four GPUs | 32 | 2 | 256 | 5,000 |
| Each task Stage 2, two GPUs (four-GPU recipe) | 8 | 4 | 64 | 10,000 |
| Each task Stage 2, four GPUs (eight-GPU recipe) | 8 | 2 | 64 | 10,000 |

The default `configs/aligned_4xa800.yaml` needs four GPUs throughout: Stage 1 uses all four, then the two task policies run concurrently on two GPUs each. `configs/aligned_8xa800.yaml` uses eight GPUs for Stage 1 and four per task for Stage 2. Stage 2 exhausted 80-GB A800 memory at batch 32 and again at batch 16 with unsharded Adam; both recipes now use batch 8 with optimizer state sharding. Memory sufficiency still requires a GPU run; the four-GPU recipe holds twice as much optimizer state per rank as the eight-GPU recipe. Changing GPU count, optimizer layout or accumulation requires a new run rather than exact resumption of an existing run. Use `--stage1-checkpoint /path/to/completed/shared_wm/stage1/checkpoints/step_XXXXXXXX` to initialize new policies from a completed teacher without retraining Stage 1. This verifies the selected teacher's model hash, latent configuration and base encoder weights; it preserves the original provenance instead of rewriting it for the new recipe.

Stage 1 uses learning rate 1e-4, warmup 500, final 2.5e-6 and weight decay 0.01. It validates every 500 updates and selects by tactile plus corrected-visual error, averaged equally over tasks. Stage 2 uses 2.5e-5 for policy and Belief Updater, 1e-4 for other new parameters, warmup 1,000 and weight decay 1e-10. EMA decay is 0.99. Formal training retains FP32 trainable parameters, BF16 autocast, SDPA, fused AdamW and gradient checkpointing.

Stage 2 sets `shard_optimizer_state: true` and `ema_device: cuda`. PyTorch ZeroRedundancyOptimizer distributes Adam moments across the ranks of each task; parameters and gradients remain replicated under DDP. This follows the successful dual-view WAVE-0 run (`wave0-shared-8task-head-wrist-gpu-003`), which used two ranks per task, micro batch 8, accumulation 4 and GPU EMA, reaching about 73.16 GiB peak allocated memory. ContactWAM retains two differentiable recurrent time steps, so that peak cannot be used as its memory estimate. EMA is a separate smoothed set of trainable weights used at export/inference, not an input to the training loss. Keeping it on GPU avoids transferring approximately 12.59 GiB of parameters to CPU each update.

Every rank writes its own `optimizer.rankNN.pt` shard. The complete marker includes all shard hashes and appears only after every shard is saved; resume verifies layout, hashes, rank count and parameter mapping without consolidating full Adam state onto GPU 0. `memory.rankNN.jsonl` records initialization, the first two updates' phases, periodic peaks and the phase of any OOM. These logs include the local estimated Adam allocation and EMA placement; an early forward alone does not establish memory sufficiency for subsequent optimizer updates.

The sequence forward reuses frozen world-vision features when the same sample/frame appears as both a future target and a later observation, or as left padding. Exact decoded-frame indices account for clipping at the end of an episode; the cache is local to one batch and never shares features across samples. It does not cache the trainable, augmented policy vision tower. Independent tactile history warmup and the final inference contact prediction call the tactile predictor directly, skipping visual futures and Corrector outputs that were discarded. Both differentiable time steps, all losses, GPU EMA and gradient checkpointing are retained. For a full five-observation Stage 2 window, this reduces visual WM and Corrector forwards from seven to four and frozen two-camera image encodings from fourteen batches to twelve; shorter/padded histories have different savings. This is a compute-count comparison, not a measured A800 speedup.

The reuse changes are covered by reduced-width FP32/BF16 loss/gradient comparisons and frame-alignment tests. After changing runtime sources, start a new output directory and reuse the completed Stage 1 checkpoint; an old Stage 2 run fails the exact-resume source-identity check. Do not rewrite its provenance to bypass that check.

`wave0_recovery` reuses the same frozen visual checkpoint and skips contact Stage 1. It trains policies using the same task-specific data and action masks. It never consumes the full method's contact checkpoint. Both variants need independent output directories; use `--resume auto` only for the same recipe, data and code identity.

To perform bounded real-model integration checks on an available GPU:

```bash
export CBW_PYTHON=/zhe_li/tactile-wam/contact-belief-wam/.venv/bin/python
bash scripts/train.sh debug --config configs/workspace.local.yaml \
  --variant full --output /absolute/new/debug-run \
  --device cuda --optimizer-device cpu --updates 2 --micro-batch 2
```

The debug checks real recurrent data, gradients, frozen parameters, optimizer/RNG restoration, export and fixed-noise inference after reload. Debug settings do not change the formal recipe. Completed debug checks do not establish convergence, multi-GPU memory sufficiency or policy success rate.

Final policies are under `<run-root>/<variant>/tasks/<task>/export/final`. Serving loads their camera mode and the simulator adapter supplies both views. Independent tactile checkpoints retain their trained k; do not override k for timing tests. CPU serving remains FP32 unless `CBWAM_DEBUG_CPU_BF16=1` is explicitly enabled. Use independent simulation seeds for the final comparison.
