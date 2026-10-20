# Implementation record

Local work includes data preparation, recovery collection, frozen AnyTouch2 feature caching, bounded CPU/debug runs, and simulator smoke evaluations. Formal training is intended for the user's A800 environment and has not started.

## Implemented

- Vendored official WAVE-0 and its pinned Transformers fork; source hashes are recorded.
- Belief, Corrector, Contact Predictor, four modes, five condition switches, shared `k`, strict world loading and export.
- HDF5 manifests, next-frame absolute actions, train-only quantiles, shared tactile preprocessing and frozen encoder.
- Random history with two-update truncated gradients, two-stage Accelerate orchestration, checkpoint resume and validation metrics.
- Standalone policy service, per-step tactile ring buffer, UniVTAC recovery collector/evaluation adapter, and direct-path A800 train/resume scripts.

## Eight-A800 correction (2026-10-18, Asia/Shanghai)

The user's current cloud allocation is **8×A800**. The default shared pipeline now points to `configs/shared_8xa800_bs64.yaml`: per-GPU batch 64, global batch 512, GA=1, with one eight-rank job at a time (shared WM, then each of two independent task policies). All 900 episodes train, validation remains disabled, and formal GPU AdamW / no gradient checkpointing remains unchanged. The prior three-GPU file and recorded debug results are historical.

The active WAVE-0 job `outputs/wave0-shared-8task-mb32-002` was inspected read-only through its pipeline state, saved recipe and training log. It uses eight GPUs with micro 32/global 256 in Stage 1, then four parallel two-GPU policies per wave, each micro 32/global 64. Its original-only data, relative actions, validation/early-stopping and Stage-2 EMA settings are distinct from this Contact-Belief recipe.

## Shared training pipeline update (2026-10-17)

The current command/configuration is documented in [`docs/shared_training.md`](docs/shared_training.md). It supersedes the single-task training recipe below. Formal training has not been launched.

- Stage 1 trains one shared visual/tactile WM and recurrent belief on eight tasks (5,000 steps). Two independent Stage 2 policies then each train for 10,000 steps, starting from that same teacher and the existing official unfinetuned π0.5 base. The first 1,000 Stage 2 steps sample original demonstrations, then mix original/recovery 1:1.
- Per the latest user instruction, all 900 episodes (800 original + 100 recovery, 201,225 frames) participate in training. There is no validation split or validation pass in the formal configuration. All-train manifests are in `data/manifests/all_train/`; both stages use `data/stats/shared.all_train.quantiles.json`, computed from all 900 episodes. The old split manifests/statistics remain available to old experiments.
- All 900 frozen AnyTouch2 caches passed shape, dtype, finite-value, encoder metadata, frame-index and HDF5-step checks. Final dataset input checks confirm 900 Stage 1 episodes and 150 episodes in each task policy, with validation disabled. Reports: `reports/shared_cache_audit.json`, `reports/all_train_input_audit.json`.
- Forced perturbation and blocked-hold targets are masked from flow matching and corresponding planner imitation losses. Those real actions remain available to WM supervision and belief history. Task/source sampling, continuous histories, TBPTT, shared teacher handoff, and optimizer/sampler resume are implemented.
- Formal training uses 3×A800, per-GPU batch 64, global batch 192, GA=1, BF16, GPU AdamW, and **no gradient checkpointing or CPU optimizer offload**. Module freezing follows the training method; no additional layers are frozen to save memory.

| 4090 debug model | Updates | Peak allocated GPU memory | Reload output max difference |
|---|---:|---:|---:|
| Shared Stage 1 | 2 | 3.82 GiB | 0.0 |
| insert_hole Stage 2 | 2 | 16.30 GiB | 0.0 |
| insert_HDMI Stage 2 | 2 | 16.30 GiB | 0.0 |

All three checks used real batches, had finite effective gradients, kept frozen parameters unchanged, and restored model/optimizer/scheduler/RNG state. CPU optimizer state and gradient checkpointing were used only for these bounded 4090 checks. HDMI's first attempt hit competing GPU use before any update; it was retried after the other job finished. Its zero-update failure record is preserved. No model exceeded two debug updates.

The already-running GPU debug retained its original per-case inputs and statistics (before the user's all-data change), rather than repeating updates. The final all-data manifests, recomputed statistics, policy inputs and disabled-validation path were subsequently checked separately. Full results are in `outputs/debug_shared_4090/debug_summary.json` and `reports/shared_training_summary.json`.

The CPU suite recorded 89 passed and one CUDA-only skip; that test subsequently passed on the actual 4090. After the all-data change, 18 relevant data/model/pipeline tests passed. The three-stage cloud launch command was checked with `--dry-run`. No A800 batch-64 memory/throughput test or formally trained success-rate claim is made.

## Dated status snapshot (2026-10-17 12:01 UTC)

Dynamic counts and completion claims live here so the README and operating guides can stay focused on design and commands. Times below are UTC. Recovery totals count completed collection records only when they have `success: true`, at least one measured blockage, no training exclusion, and an existing corresponding HDF5 file. Per-episode records were atomically written during collection.

### Data and recovery collection

- Each recovery collector reached 50 eligible successful episodes with measured blockage and a corresponding HDF5 file. Untrained attempts are excluded. Each final mixed manifest contains 100 original and 50 recovery episodes; the per-source split is 90/10 for original data and 45/5 for recovery data.
- Quantile statistics are finite and 8-dimensional for `next_frame_absolute_first8`. The cache audit passed for every manifest row: features are finite `[N,8,512]` `float16`, and `raw_step` and index agree with HDF5.
- Frame totals are 16,813 original + 28,104 recovery for `insert_hole`, and 11,800 original + 20,241 recovery for `insert_HDMI`.
- Six sample recovery video and curve pointers are indexed in [`reports/recovery_preview/index.json`](reports/recovery_preview/index.json).

### Runtime and model checks

- Strict official WAVE-0 world loading passed: 167 tensors loaded, 3 converted, 0 reinitialized, and 0 unused (`reports/official_world_load.json`).
- The project `.venv` uses Python 3.10.22 and imports Torch 2.7.1+cu126 with patched Transformers 4.53.3. The completed full CPU suite in `reports/tests.log` records 54 passed, 1 warning, in 16.43 seconds; the report was last written at 2026-10-17 08:10:25 UTC. An editable `pip --no-index --no-deps -e .` install resolves `contact_belief_wam` to the current `src/contact_belief_wam/` checkout.
- The bounded CPU debug using `models/pi05_base_local_debug/` completed one Stage 1 and one Stage 2 update, with checkpoint reload max absolute difference `0.0` (`outputs/debug_local_base/debug_report.json`). The completed HDMI mixed-data CPU debug report was last written at 2026-10-17 07:23:59 UTC; it records one update per stage, reload difference `0.0`, and `complete: true` (`outputs/debug_HDMI_mixed/debug_report.json`). These are bounded debug runs, not training convergence evidence.
- A BF16 warmup-cache bug that dropped gradients was fixed. Zero-update real-episode audits for both stages show no missing contact gradients and confirm BPTT=2 (`reports/gradient_audit_stage1.json`, `reports/gradient_audit_stage2.json`). These are gradient audits, not optimizer updates.
- With fixed noise and identical exported parameters, wrapper outputs exactly match the vendor sampler for both `pi05` and `wave0` (`reports/baseline_sampler_parity.json`, max absolute difference `0.0`). This checks sampler parity, not trained baseline quality.
- `models/pi05_base_local_debug/model.safetensors` initializes a newly constructed PyTorch model. It came from the official, unfinetuned JAX `pi05_base`, converted with the official tensor converter. The converter initialized the unused `paligemma_with_expert.gemma_expert.lm_head.weight` vocabulary head because the JAX action expert has no vocabulary head. This is not the older task-finetuned JAX baseline. No separate official Hugging Face checkpoint download is planned or required.

### Simulation and training

- The model-driven task-level smoke runs are complete. `insert_hole` has three complete k=10 rollouts (one timeout, two early failures) plus one short, complete k=5 check (`outputs/smoke/insert_hole-k10/evaluation.json`, `outputs/smoke/insert_hole-k5/evaluation.json`). HDMI has three complete 600-policy-step timeouts; that run used CPU policy inference, CPU tactile encoding, and GPU simulation (`outputs/smoke/insert_HDMI-k10/evaluation.json`, last written at 2026-10-17 08:02:00 UTC). The untrained model did not insert the objects in these smoke runs; those rollout outcomes do not leave this round's task-level simulation check incomplete.
- Local model, debug, checkpoint, data, and smoke work is complete as recorded above. Formal A800 training has not started. The cloud upload bundle and upload inventory generator were cancelled and deleted at the user's request; the data `prepare manifest` command remains part of this pipeline, and direct-path A800 train/resume scripts are documented in [`docs/cloud.md`](docs/cloud.md).

## Remaining work

- The local implementation and dataset handoff are complete. Formal A800 training is the user's later manual step, not unfinished development work.

## Environment notes

- Raw UniVTAC HDF5 stores 9 joints; state/action use the first 8 with next-frame targets.
- The official WAVE-0 archive is vendored; `vendor/SOURCE.json` records its exact archive digest.
- The converted local base is a verified PyTorch initialization source; its provenance and unused vocabulary-head initialization must remain explicit. It is distinct from the older task-finetuned JAX baseline.
