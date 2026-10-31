# HDMI natural-failure takeover pilot

Run `cbwam collect-takeover --preflight-only`, then `cbwam collect-takeover` with
the existing model Python environment. Native simulation runs through `scripts/sim.sh`.
The selected checkpoint is the head+wrist WAVE-0 `head-wrist-gpu-003` HDMI Stage-2
10k EMA. The new collector delegates inference to the existing dual-view server,
and uses the existing `RecordedController` and 2-physics-step execution function.
It does not enable the old, single-view-only verification RNG pathway.

Outputs default to `data/recovery_wave0_v1/insert_HDMI/pilot`. `pilot.json` is the
durable attempt ledger; `index.html` links the videos, curves, paired branches,
causal input snapshots and full rejection reports. Existing training manifests
are not changed. This command has no expansion mode and never launches training.

The fixed pilot allows 20 seeds starting at 3000000, at most 10 takeover branches,
and stops for user review at 3 qualifying recoveries plus 2 normal successes or
the attempt limit. Every attempt is reserved before launching the simulator.
Interrupted attempts consume their reservation. A restart cannot refund quotas
or resume a pilot in `awaiting_user_review`. A runtime error stops for diagnosis
instead of burning the remaining seeds. Re-running resumes other eligible seeds.
A `diagnostic_hold.json` also prevents new rollouts, including after restart. It
is written when an offline audit disproves the progress feedback assumption. An
audit exiting with code 3 is a completed diagnostic, not a corrupted trajectory.

The natural rollout runs to completion without intervention. The causal detector
uses measured joint FK along world -Z, command FK, and raw tactile observations.
Its initial 8 observations calibrate tactile noise (floor .01, 3 sigma); offline
geometry must subsequently establish that this baseline preceded external contact.
At each 10-action boundary a requested advance of at least .2 mm, actual/requested
progress below .2 and 3 consecutive tactile observations above threshold confirm
a blockage in one or two chunks. Three observations are not three chunks.

Only naturally failed rollouts may be replayed. Each observation in the entire
prefix, including object-relative-to-gripper pose and both tactile windows, must
pass the existing strict replay tolerances. Mismatches are not qualifying recovery
data. The teacher retreats 2 mm, then probes .5 mm / 1 degree within 3 mm lateral
and 3 degree angular bounds. It has 12 probes including ambiguous repeats and the
remaining portion of the original 600-action budget. Seed chooses only initial
candidate order. Feedback determines continuation, withdrawal and direction changes.
After unloading, the teacher first returns to the previously observed contact
depth, then scores the final .5 mm probe. Free-space travel is not evidence of
improved alignment. Initial exploratory commands remain excluded from BC labels.
No object/slot GT pose enters the feedback controller. Native surface geometry and
actual object insertion progress are recorded separately for acceptance.

Trace observations number N+1, executed commands number N. `training_contract.json`
separates continuous context from `policy_supervision_ranges`. Use `policy_prefix`
to construct targets before flow/noise processing; masking loss alone is not enough.
Initial random exploration and unsuccessful transitions are not BC labels. Do not
reset tactile/Belief context at takeover or exclude valid WM transitions.

AnyTouch2 is frozen, uses the same preprocessing as TWAM, and is evaluated offline
on causal indices t-6,t-4,t-2,t. `decision_input.npz` reconstructs the model-visible
input and prior 40 actions. A latent change is evidence of changed input, not proof
that a trained policy uses tactile feedback. Native mesh proximity within the UIPC
.5 mm contact distance is an audit proxy rather than a force measurement; surface
evidence, tactile change, progress and video still require human review before use.
Recovery acceptance compares raw images and latent features with the eight
observations immediately preceding external contact, so an earlier grip change
cannot masquerade as newly available contact evidence. Pilot traces store raw,
lossless camera arrays; newer attempts use uncompressed chunks to reduce recorder
overhead. Source hashes are kept with attempts and the offline auditor.

Validation: `python -m pytest -q tests/test_takeover_pilot.py`. For real collection,
the collector acquires the existing evaluation GPU lock and requires 20 GiB free.
All artifacts remain outside Git; no checkpoint or old recovery data is overwritten.

## First native validation: feedback assumption requires revision

The first complete natural failure (seed 3000001) exposed a limitation of the
requested FK stall rule. Between strategy observations 40 and 50 the measured
end effector advanced 2.496 mm, while the connector advanced only 0.027 mm relative
to the slot. The connector moved 2.471 mm relative to the gripper. The detector
reported normal command tracking, so no takeover was authorized. This is a
missed object stall, not a valid negative example for the detector.

The existing WAVE-0 execution path calls `take_action(..., force=True)`;
`UniVTAC/envs/robot/robot.py:set_arm` directly sets DOF positions. Consequently,
joint FK can track the requested motion despite object blockage and grasp slip.
Gripper opening alone also cannot certify a stable grasp. The initial feedback
Teacher must not be considered validated by the unit tests or by arm movement.

The pilot retains this failure and the subsequent normal success without intervention
(seed 3000002, 547 actions), then stops for diagnosis. It has not produced the
planned three recovery demonstrations. The strict replay and Teacher path have
not yet been validated on a real accepted takeover. The preexisting training
manifests remain untouched.

Before further recovery collection, replace the FK-only progress assumption with
an observable estimate of connector/slot progress and grasp slip from the actual
camera and tactile inputs. Retain FK for commanded and executed arm motion. Check
the proposed observable feedback against the saved failure and normal-contact
success first; do not substitute simulator GT lateral/rotational error, reduce
thresholds to force takeovers, or change the baseline's physical execution mode.
The native poses remain an offline falsification and acceptance reference only.

## Explicitly authorized oracle diagnostic (2026-10-27)

The user subsequently requested one ground-truth-assisted recovery preview for
the existing natural failure seed 3000001, without expansion or training. This
is a separate diagnostic exception to the observation-only Teacher above. The
automatic collector's `diagnostic_hold.json` remains in place.

Run one branch with the model Python environment and GPU access:

```bash
PYTHONPATH=src:/zhe_li/tactile-wam/pi05-univtac/src python -m contact_belief_wam.oracle_trial --takeover-step 40
```

The command replays and checks every observation of the recorded prefix before
any correction, including relative grasp pose and both tactile windows. An
oracle uses current connector/slot poses to unload, make bounded lateral
corrections, and test actual connector progress. It never teleports actors or
changes the execution mode, native success predicate, or 600-action budget.
Every action uses two physics steps (one is possible on terminal success).

Each explicit retry reserves a separate `oracle/3000001/attempt-NN` directory and
consumes the shared ten-recovery-attempt allowance, including interrupted or
rejected replays. There is no automatic retry loop. Once an oracle succeeds,
the launcher rejects further oracle collection for this seed pending review.
All oracle action labels are disabled; results are marked privileged and
excluded from training regardless of success. Original trajectories, manifests,
and checkpoints are unchanged.

`python -m contact_belief_wam.oracle_preview PATH_TO_RESULT --device cuda`
audits the recorded causal AnyTouch windows and makes paired videos/curves. The
report distinguishes physical recovery from evidence for an observation-only
Teacher; even a successful oracle is not counted as one of the three qualified
feedback recovery samples. The new invariants are exercised by
`tests/test_oracle_recovery.py` alongside the existing takeover tests.

The first real attempt was rejected before any policy action: rebuilding seed
3000001 produced a different CuRobo pre-move endpoint (maximum joint difference
0.000957 rad) and depth edge RMSE of 0.084–0.091 mm. Existing tolerances were not
changed. `oracle_trial --paired-live` therefore reproduces this same seed using
the actual dual-view WAVE-0 service, captures a full same-instance snapshot at
the boundary, lets the baseline finish, and runs a ten-action restoration probe
before the oracle. The new reference is explicitly labeled as a live
reproduction, not an exact replay of the earlier stored failure. The shared
prefix is copied when it actually executes, retaining N actions and N+1
observations and independent continuous histories in both branches.

The second attempt could not load the policy after an unrelated GPU workload
appeared. It executed zero strategy actions and is retained as an infrastructure
failure. `--paired-live --policy-device cpu` keeps the same checkpoint, EMA,
dual-view preprocessing, action contract, and GPU physics while performing model
inference on CPU. The service records this device explicitly. Such a run is a
fresh reference; it is not asserted numerically identical to CUDA inference.
JSON persistence explicitly uses UTF-8 so native simulation's ASCII locale cannot
hide an error containing the policy name or a Chinese diagnostic.

Attempt 03 reproduced a 600-action natural failure and restored physical state
within 0.0003 mm, with matching commands and tactile marker motion. Strict
observation replay nevertheless failed: RGB temporal rendering and depth-edge
pixels differed. This failure remains recorded and its tolerances are unchanged.
`--physical-diagnostic` is a separate oracle-only acceptance scope: it retains
every joint, pose, relative grasp, marker and command check and additionally
compares native object/slot surface vertices. It may demonstrate physical
recoverability despite an explicitly failed image replay, but cannot certify
an observation-equivalent pair or qualify a natural-failure recovery sample.
Both strict and physical audit results must be shown in the preview. This option
is unavailable to the automatic observation-only collection workflow.

Attempt 04 passed both physical restoration audits but stopped before its first
oracle action because the pre-move CuRobo solver's captured CUDA Graph could not
change problem shape. Oracle waypoints now use a separate, graph-disabled IK
solver with the existing robot/collision model and FK/bounds verification. A real
GPU preflight of 2 mm unload and consecutive 0.5 mm corrections achieved roughly
0.005 mm FK error. This leaves the WAVE-0 action path and pre-move planner intact.

## Existing pair prepared for training-data review

The user accepted the effect of the attempt-05 recovery video and requested a
frame-by-frame training sample audit, without collection or training. Run:

```bash
PYTHONPATH=src /zhe_li/tactile-wam/contact-belief-wam/.venv/bin/python -m contact_belief_wam.takeover_sample_review
```

This reads only the existing oracle 3000001 attempt-05 and natural success
3000002. Outputs are isolated in
`data/recovery_review_v1/insert_HDMI/pair_3000001_3000002/`. Open `index.html` for
the two cameras, both causal tactile windows, actual commands, label masks,
and continuous execution history. CSV files cover every observation; selected
`samples/*.npz` preserve exact loader tensors. `validation.json` and
`loader_checks.json` record the checks and source hashes.

The recovery has 198 commands / 199 observations. Its first 40 commands remain
context/WM-only; 158 subsequent commands are proposed as privileged-teacher BC
labels, split at 15 actual feedback decisions. The normal success has 547
commands / 548 observations, all with valid proposed policy supervision.
These proposals do not modify the source trace's quarantined policy flags and
do not certify observation-only Teacher behavior.

`AlignedDataset` supports the recorded `wave0-hdmi-contact-v1` format. It uses
`o[t] -> actual_command[t] -> o[t+1]`, rather than substituting future measured
joint positions for dispatched commands. Policy targets are sanitized before
flow inputs are constructed; WM transitions and historical executions retain
their true commands across takeover/label boundaries. History starts at the
episode start, independently of policy ranges. Both cameras are mandatory.
Source hashes, observation/action clocks, causal tactile windows, encoder
identity, and feedback boundaries are checked on load.

The review manifest has `split=review` and `training_enabled=false`. It can only
be inspected using the explicit `allow_review=True` dataset option; the normal
training entry point does not set this option. No active manifest is changed.
Privileged demonstrations can be used for imitation learning after qualification;
the oracle collector's blanket exclusion is a diagnostic quarantine, not a
theoretical prohibition. Learning useful recovery and relying on tactile
feedback still require subsequent closed-loop evaluation and ablations.
