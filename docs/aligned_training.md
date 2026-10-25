# Aligned Contact-Belief training

The public entry is now `python -m contact_belief_wam train` or `bash scripts/train.sh`. The default `full` recipe adds an independent `tactile` phase before the WAVE-0 stages described below. See [tactile_training.md](tactile_training.md) for its objective, transfer rules, and validation. Old coupled checkpoints remain loadable but cannot resume this new recipe.

The aligned pipeline uses `wave0-univtac-fast` as its training reference and changes the Contact/tactile method. It preserves the reference model initialization, data transforms, optimizer and schedule conventions, validation cadence, and stage budgets. `full` adds the aligned Contact-Belief method. `wave0_recovery` provides the WAVE-0 comparison with the same recovery-data setting.

The pinned reference files and their hashes are in `vendor/aligned_reference/SOURCE.json`. The aligned input uses `q[t+i+1] - q[t]` for the first seven action coordinates, absolute gripper position, and 32-dimensional flow supervision including the padded coordinates. Policy normalization is copied from the reference task statistics; world inputs use its shared eight-task statistics. Historical executed actions use the previous interval's anchor. No statistics are recomputed from recovery data. The exact SentencePiece state prompt and PIL resize-pad are shared by training and aligned deployment. Stage 2 trains the policy vision tower and uses a separate frozen, unaugmented vision tower for future supervision.

Stage 1 uses 720 original training episodes plus 100 recovery episodes, with the same 80 original validation episodes as the reference. Each local batch of 32 has four sequences from each task; the two insertion tasks contribute two original and two recovery sequences each. Stage 2 uses all 100 original and 50 recovery episodes per insertion task, sampled 1:1. Perturbation and blocked-hold actions remain available to the world model but are masked out of policy supervision for both variants. The full method retains random history lengths 0–4, the final two differentiable updates, and its Contact-specific losses and switches; the WAVE-0 baseline has no recurrent Contact history.

The default recipe is `configs/aligned_8xa800.yaml`:

| Stage | Data and models | Per-GPU batch | Global batch | Updates |
|---|---|---:|---:|---:|
| Tactile (`full` only) | Independent current-observation/action-conditioned tactile predictor; no belief or predicted visual input | 32 on 2 GPUs | 64 | 5,000 |
| Shared Stage 1 | Eight-task shared world model; original demonstrations and insertion-task recovery episodes | 32 on 8 GPUs | 256 | 5,000 |
| Stage 2: `insert_hole` | Independent policy initialized from the shared checkpoint | 32 on 2 GPUs | 64 | 10,000 |
| Stage 2: `insert_HDMI` | Independent policy initialized from the same shared checkpoint | 32 on 2 GPUs | 64 | 10,000 |

Gradient accumulation is one. Stage 1 uses the reference `1e-4` learning rate, 500-step warmup, cosine decay to `2.5e-6`, and `0.01` weight decay. It validates 128 samples per task every 500 updates, selects by the common visual-world objective, and follows the reference early stopping rule (patience four, minimum 2,000 updates, maximum 5,000). Stage 2 uses policy and new-module learning rates `2.5e-5` and `1e-4`, 1,000-step warmup, cosine decay, and `1e-10` weight decay. Both policies use the same Stage 1 model and normalization inputs, then train independently. Stage 2 mixes original and recovery examples from its first update. EMA decay is `0.99`, Stage 2 only; trainable parameters use FP32, and EMA state lives on rank-zero GPU during formal A800 training. SDPA, fused AdamW, BF16 autocast and gradient checkpointing are enabled. Stage 2 runs the two tasks concurrently on GPUs 0–1 and 2–3; the remaining four GPUs are idle.

## Prepare and bounded debug

Prepare the real manifests, normalization statistics, and frozen tactile features with the recipe's data paths:

```bash
bash /zhe_li/tactile-wam/contact-belief-wam/scripts/train.sh prepare \
  --config /zhe_li/tactile-wam/contact-belief-wam/configs/aligned_8xa800.yaml
```

Run the bounded checks separately for each variant. The debug output path is separate from the formal run path in the config. At most two optimizer updates are made for each model: the independent tactile predictor (full only), shared Stage 1 model and each of the two Stage 2 policies. The checks use real HDF5 windows and cached tactile features. Stage 1 includes two original windows from the same task to exercise task-preserving ranking, along with a recovery window. Each Stage 2 debug checks an original and a recovery window for that policy's task. A smaller micro-batch and CPU optimizer state may be used to fit the complete model on the developer GPU; these are debug-only execution settings.

```bash
bash /zhe_li/tactile-wam/contact-belief-wam/scripts/train.sh debug \
  --config /zhe_li/tactile-wam/contact-belief-wam/configs/aligned_8xa800.yaml \
  --variant full \
  --output /zhe_li/tactile-wam/contact-belief-wam/outputs/aligned_debug \
  --device cuda --optimizer-device cpu --updates 2 --micro-batch 2

bash /zhe_li/tactile-wam/contact-belief-wam/scripts/train.sh debug \
  --config /zhe_li/tactile-wam/contact-belief-wam/configs/aligned_8xa800.yaml \
  --variant wave0_recovery \
  --output /zhe_li/tactile-wam/contact-belief-wam/outputs/aligned_debug \
  --device cuda --optimizer-device cpu --updates 2 --micro-batch 2
```

Stage 1 uses a two-example, same-task original micro-batch for task-preserving ranking and also processes one recovery example. It requires two sequence positions for full-mode BPTT and one for `wave0_recovery`. Each Stage-2 policy processes one original and one recovery example. Debug verifies finite real-batch losses and gradients, active Contact-Belief gradients, frozen-parameter stability, recurrent sequence backpropagation, model/EMA export, optimizer and LR-schedule-position restoration, RNG restoration, and identical fixed-noise inference between the in-memory update and checkpoint reload. The report records the exact debug settings, sampled tasks and sources, update traces, restored-state checks, and that formal training was not launched. The CPU optimizer and offloaded temporary BF16 model copies are debug-only; the formal recipe remains unchanged. Do not use a debug checkpoint to initialize an A800 run.

The report and logs are written under `outputs/aligned_debug/<variant>/`. CPU-master offload disables PyTorch's memory-efficient SDPA backend for that debug call because its backward kernel rejects the compact attention-mask row stride produced by activation offload; math and Flash SDPA remain available. Formal training keeps its configured SDPA path. A completed model is not updated again when the same debug command is rerun. If a process is interrupted after a complete checkpoint is saved, rerunning the identical command finishes its saved restore/inference checks without another optimizer update. Missing or inconsistent checkpoints are rejected; inspect such an incomplete report before choosing a fresh debug output.

## Formal A800 recipe

The commands below document the intended A800 job; this repository change does not launch formal training. Run each variant in its own output directory and use `--resume none` only for a new run:

```bash
bash /zhe_li/tactile-wam/contact-belief-wam/scripts/train.sh train \
  --config /zhe_li/tactile-wam/contact-belief-wam/configs/aligned_8xa800.yaml \
  --variant full --resume none

bash /zhe_li/tactile-wam/contact-belief-wam/scripts/train.sh train \
  --config /zhe_li/tactile-wam/contact-belief-wam/configs/aligned_8xa800.yaml \
  --variant wave0_recovery --resume none
```

The formal recipe keeps the reference GPU counts, micro-batches, global batches, update counts, FP32 trainable parameters, and GPU EMA. It does not use the developer debug's CPU optimizer state or reduced micro-batch. For the default full configuration, independent tactile training runs first; Stage 1 then loads and freezes its predictor, and each task-specific Stage 2 starts from the completed shared checkpoint. The baseline skips tactile training. Training and simulation results should be reported separately: a completed debug or simulator smoke is an integration check, not a training-quality result.

The two commands are separate jobs. The baseline trains its own Stage 1 teacher from the same official world initialization; it never consumes the full method's teacher. Output directories default to `runs/aligned-8xa800/full` and `runs/aligned-8xa800/wave0_recovery`. Set `--output /absolute/new/run-root` to choose a new root; the variant suffix is appended automatically. No upload package or new weight download is needed with the existing mounts.

To resume the same run, repeat the corresponding command with `--resume auto`. The pipeline restores the latest complete checkpoint per stage/task, including optimizer, schedule position, EMA, random state and sampler position. Finished Stage 1 is reused. Do not resume an old absolute-action run with the aligned entry.

The final policies are under `<run-root>/<variant>/tasks/<task>/export/final`. Inference selects EMA by default and retains the world action encoder for contact prediction. A local evaluation command is:

```bash
/zhe_li/tactile-wam/contact-belief-wam/.venv/bin/python -m contact_belief_wam eval \
  --task insert_hole \
  --checkpoint /zhe_li/tactile-wam/contact-belief-wam/runs/aligned-8xa800/full/tasks/insert_hole/export/final \
  --output /zhe_li/tactile-wam/contact-belief-wam/outputs/aligned_trained_eval/insert_hole \
  --seeds 1000000 --device cpu --port 18781
```

The policy device in this example is CPU; UniVTAC uses the local GPU. `--k 5 --max-policy-steps 10` requests a short timing/alignment check rather than a complete default-budget episode. Formal A800 memory use and throughput still need confirmation in the submitted eight-card job.

Aligned CPU inference remains FP32 by default. For a local simulator smoke test on a CPU with BF16 acceleration, set `CBWAM_DEBUG_CPU_BF16=1` before starting the evaluator. The aligned service reports `effective_autocast` in inference diagnostics (`cpu:bfloat16`, `cuda:bfloat16`, or `disabled`). This switch affects serving only; it does not change the formal training precision or recipe.
