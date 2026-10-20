from __future__ import annotations

from contextlib import nullcontext
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file

from .data import UniVTACDataset, ReplaySampler
from .model import create_policy, group_name, gradient_report
from .runtime import atomic_json, digest, object_digest, project_root


def lr_at(step, peak, final, warmup, steps):
    if step < warmup:
        start = peak/(warmup+1)
        return start+(peak-start)*step/max(1, warmup)
    fraction = min(1., (step-warmup)/max(1, steps-warmup))
    return final+(peak-final)*.5*(1+math.cos(math.pi*fraction))


def make_optimizer(policy, run, stage):
    cfg = run[stage]
    performance = performance_config(run)
    groups = {"policy": [], "new": []}
    for name, p in policy.named_parameters():
        if p.requires_grad:
            groups["new" if group_name(name) in ("world", "future_condition") else "policy"].append(p)
    params = []
    for name, values in groups.items():
        if values:
            peak = cfg.get("lr_new", cfg["lr"]) if name=="new" else cfg["lr"]
            params.append({"params": values, "peak": peak, "name": name, "lr": peak})
    device = next((p.device for p in policy.parameters()), torch.device("cpu"))
    fused = bool(performance.get("fused_adamw", False)) and device.type == "cuda"
    optimizer_options = {"betas": (.9, .95), "eps": 1e-8,
                         "weight_decay": cfg["weight_decay"]}
    if fused:
        optimizer_options.update(fused=True, foreach=None)
    else:
        optimizer_options["foreach"] = False
    return torch.optim.AdamW(params, **optimizer_options)


def set_lr(optimizer, cfg, step):
    for g in optimizer.param_groups:
        final = cfg["final_lr"]*g["peak"]/cfg["lr"]
        g["lr"] = lr_at(step, g["peak"], final, cfg["warmup"], cfg["steps"])


class EMA:
    """Float32 EMA stored on the selected device, normally CPU."""
    def __init__(self, policy, decay, device="cpu", *, initialize=True):
        self.decay = decay
        self.device = torch.device(device)
        self.values = ({n: p.detach().to(device=self.device, dtype=torch.float32).clone()
                        for n, p in policy.named_parameters() if p.requires_grad}
                       if initialize else {})

    @torch.no_grad()
    def update(self, policy):
        if self.device.type == "cuda":
            names, params = [], []
            for name, parameter in policy.named_parameters():
                if name in self.values:
                    names.append(name)
                    params.append(parameter.detach().to(device=self.device, dtype=torch.float32))
            values = [self.values[name] for name in names]
            if values:
                torch._foreach_mul_(values, self.decay)
                torch._foreach_add_(values, params, alpha=1-self.decay)
            return
        for n,p in policy.named_parameters():
            if n in self.values:
                value = p.detach().to(device=self.device, dtype=torch.float32)
                self.values[n].mul_(self.decay).add_(value, alpha=1-self.decay)

    def load(self, path, policy):
        # Load directly onto the selected device so resume does not allocate an
        # initialization EMA and then replace it, or stage a second full copy.
        values = load_file(str(path), device=str(self.device))
        expected = {n: p for n, p in policy.named_parameters() if p.requires_grad}
        if set(values) != set(expected):
            raise ValueError("EMA checkpoint parameter keys differ from the policy")
        for name, value in values.items():
            if value.dtype != torch.float32 or value.shape != expected[name].shape:
                raise ValueError(f"EMA checkpoint tensor differs for {name}")
        self.values = values

    def save(self, path):
        # safetensors writes CPU tensors. On CPU, .cpu() is a view and avoids a
        # duplicate model allocation; on CUDA, the copy happens only at save time.
        save_file({n: v.detach().cpu().contiguous() for n, v in self.values.items()}, str(path))


def performance_config(run):
    performance = run.get("performance", {})
    if performance is None:
        return {}
    if not isinstance(performance, dict):
        raise ValueError("performance must be a mapping")
    return performance


def select_ema_device(run, device, *, cpu_offload=False):
    requested = performance_config(run).get("ema_device", "cpu")
    if requested not in ("cpu", "cuda"):
        raise ValueError("performance.ema_device must be 'cpu' or 'cuda'")
    device = torch.device(device)
    if requested == "cuda" and device.type == "cuda" and not cpu_offload:
        return device
    return torch.device("cpu")


def ddp_find_unused_parameters(run):
    # Retain legacy behavior unless the opt-in performance section says otherwise.
    return performance_config(run).get("find_unused_parameters", True)


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_due(enabled, step, log_every, limit):
    return enabled and (step % log_every == 0 or step == limit)


def assert_finite(value, message):
    finite = torch.isfinite(value).all()
    if value.device.type == "cuda":
        # Avoid a host read on every CUDA update while still surfacing invalid
        # values through the current stream's asynchronous assertion.
        torch._assert_async(finite, message)
    elif not finite:
        raise FloatingPointError(message)


def identity(run, debug):
    cfg = {k:v for k,v in run.items() if k not in ("workspace", "base", "world_init", "tokenizer", "baseline", "output",
                                                 "workflow_root", "wm_norm", "stage1_checkpoint")}
    root = project_root()
    hashes = {str(p.relative_to(root)): digest(p) for p in sorted((root/"src").rglob("*.py"))}
    for task in run.get("tasks", [run["task"]]):
        for p in sorted((root/"baseline"/task).glob("*.json")):
            hashes[str(p.relative_to(root))] = digest(p)
    if run.get("wm_norm"):
        hashes["shared_wm_normalization"] = digest(run["wm_norm"])
    if run.get("stage1_checkpoint"):
        cfg["shared_teacher"] = json.loads((Path(run["stage1_checkpoint"])/"complete.json").read_text())
    for p in sorted((root/"vendor/wave0/src/lerobot/policies/wave0").glob("*.py")):
        hashes[str(p.relative_to(root))] = digest(p)
    hashes["tokenizer"] = digest(run["tokenizer"])
    # Explicit source receipts bind initialization without rehashing 14 GB at every resume.
    for key, receipt in (("base", "conversion.json"), ("world_init", "DOWNLOAD_SOURCE.json")):
        if run.get(key):
            p = Path(run[key])/receipt
            hashes[key+"_receipt"] = digest(p) if p.exists() else None
            hashes[key+"_bytes"] = (Path(run[key])/"model.safetensors").stat().st_size
    result = {"recipe": cfg, "hashes": hashes, "debug": debug}
    result["fingerprint"] = object_digest(result)
    return result


def latest(output, stage):
    paths = sorted((Path(output)/stage/"checkpoints").glob("step_*/complete.json"))
    return paths[-1].parent if paths else None


def deployment_contract(run):
    if run["task"]=="shared_wm":
        return {}
    path = Path(run["baseline"])/"deployment_contract.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"prompt": json.loads((Path(run["baseline"])/"instruction.json").read_text())["seen"][0],
            "policy_step_budget": 317 if run["task"]=="insert_hole" else 600}


def check_shared_teacher(run, debug):
    if not run.get("stage1_checkpoint"):
        raise ValueError("Shared Stage 2 requires --stage1-checkpoint from the completed shared WM run")
    path = Path(run["stage1_checkpoint"])
    meta = json.loads((path/"complete.json").read_text())
    saved = json.loads((path/"run_identity.json").read_text())
    if meta["stage"]!="stage1" or saved["recipe"]["task"]!="shared_wm" or saved["debug"]!=debug:
        raise ValueError("Teacher must be a shared Stage-1 checkpoint of the same debug/formal mode")
    if saved["recipe"]["tasks"] != run["tasks"] or digest(path/"wm_norm_stats.json") != digest(run["wm_norm"]):
        raise ValueError("Teacher tasks or world normalization differ")
    if digest(path/"pretrained_model/model.safetensors") != meta["model_sha256"]:
        raise ValueError("Teacher weight integrity check failed")
    if not debug:
        result = json.loads((path.parents[1]/"result.json").read_text())
        if not result["complete"] or result["identity"]!=saved["fingerprint"] or Path(result["selected_checkpoint"])!=path:
            raise ValueError("Shared WM selection has not completed")
    return path


def prune_checkpoints(run, stage, selected=None):
    keep = run.get("keep_checkpoints", 0)
    if not keep:
        return
    paths = sorted((Path(run["output"])/stage/"checkpoints").glob("step_*/complete.json"))
    for p in paths[:-keep]:
        if selected is None or p.parent != Path(selected):
            shutil.rmtree(p.parent)


def validate_world(policy, run, device, world, rank, debug=False):
    """Fixed, task-balanced held-out episodes; no train transforms or RNG drift."""
    from .shared import SharedDataset
    from torch.utils.data._utils.collate import default_collate
    ds = SharedDataset(run, validation=True)
    count = 2 if debug else run["stage1"]["validation_samples_per_task"]
    # All tasks contribute equally; every physical task batch has same-task negatives.
    local = max(2, count//world)
    if not debug and (count % world or count//world < 2):
        raise ValueError("Validation requires >=2 samples/task/rank and an exact partition")
    totals = torch.zeros(len(ds.datasets), dtype=torch.float64, device=device)
    previous, was_training = rng_state(), policy.training
    policy.eval()
    torch.manual_seed(1729+rank)
    try:
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            for task, child in enumerate(ds.datasets):
                order = np.random.default_rng(1729+task).permutation(len(child))
                if local*world > len(order):
                    raise ValueError("Validation sample budget exceeds available held-out frames")
                indices = order[rank*local:(rank+1)*local]
                for start in range(0, local, min(local, run["stage1"]["micro_batch"])):
                    chosen = indices[start:start+run["stage1"]["micro_batch"]]
                    raw = default_collate([child[int(i)] for i in chosen])
                    batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()}
                    loss, _ = policy(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite held-out WM loss")
                    totals[task] += loss.double()*len(chosen)/local
        if world > 1:
            dist.all_reduce(totals)
            totals /= world
        return {"loss": float(totals.mean()), "per_task": dict(zip(run["tasks"], totals.cpu().tolist())),
                "samples_per_task": local*world, "source_episode_ids": run["wm_validation_episode_ids"]}
    finally:
        ds.close()
        policy.train(was_training)
        restore_rng(previous)


def rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def tensor_witness(tensor):
    value = tensor.detach().float().flatten()
    # Every parameter contributes a small deterministic witness, without duplicating the model.
    return value[:8].cpu().tolist()


def witnesses(policy, optimizer, ema):
    return {"parameters": {n:tensor_witness(p) for n,p in policy.named_parameters() if p.requires_grad},
            "optimizer": {str(i):{k:tensor_witness(v) for k,v in state.items() if torch.is_tensor(v)}
                          for i,state in enumerate(optimizer.state.values())},
            "ema": {n:tensor_witness(v) for n,v in ema.values.items()} if ema else None}


def save_checkpoint(policy, optimizer, ema, run, stage, step, run_identity, world, rank, audit):
    state = rng_state()
    states = [None]*world
    if world > 1:
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    path = Path(run["output"])/stage/"checkpoints"/f"step_{step:08d}"
    if rank == 0:
        if path.exists():
            raise FileExistsError(f"Will not overwrite checkpoint: {path}")
        pending = path.with_name(path.name+".partial")
        if pending.exists():
            # Preserve failed writes for inspection while allowing one-command resume.
            pending.rename(pending.with_name(pending.name+f".abandoned-{time.time_ns()}"))
        pending.mkdir(parents=True)
        policy.save_pretrained(pending/"pretrained_model")
        torch.save({"optimizer": optimizer.state_dict(), "rng": states, "step": step, "world_size": world}, pending/"training.pt")
        if ema:
            ema.save(pending/"ema.safetensors")
        norm = Path(run["wm_norm"]) if run["task"]=="shared_wm" else Path(run["baseline"])/"norm_stats.json"
        shutil.copy2(norm, pending/"norm_stats.json")
        if run.get("shared_wm"):
            shutil.copy2(run["wm_norm"], pending/"wm_norm_stats.json")
        shutil.copy2(run["tokenizer"], pending/"tokenizer.model")
        atomic_json(pending/"run_identity.json", run_identity)
        atomic_json(pending/"deployment.json", {
            "task": run["task"], "camera": "head_only", "chunk_size": 50, "execute_steps": 10,
            "strategy_hz": 60, "physics_hz": 120, "physical_steps_per_action": 2,
            "action": "first7_delta_from_chunk_observation_plus_absolute_gripper",
            "normalization": "baseline_q01_q99_range_plus_1e-6", "tokenized_state_dim": 8,
            "future_offset_frames": 10, "official_inclusive_action_horizon": 9,
            "policy_step_budget": deployment_contract(run).get("policy_step_budget"),
            "evaluation_seeds": [1000000, 1000049], "use_ema": stage=="stage2",
            "prompt": deployment_contract(run).get("prompt"),
            "shared_world_normalization": bool(run.get("shared_wm")),
            "baseline_camera_comparison": deployment_contract(run).get("baseline_camera_comparison"),
            "formal_training": not run_identity["debug"]})
        if run_identity["debug"]:
            atomic_json(pending/"witnesses.json", witnesses(policy, optimizer, ema))
        atomic_json(pending/"audit.json", audit)
        meta = {"step": step, "stage": stage, "world_size": world, "identity": run_identity["fingerprint"]}
        if stage=="stage1" and run.get("shared_wm"):
            meta["model_sha256"] = digest(pending/"pretrained_model/model.safetensors")
        atomic_json(pending/"complete.json", meta)
        pending.rename(path)
    if world > 1:
        dist.barrier()
    return path


def probe(policy, batch, device, seed=123, cpu_offload=False):
    previous = rng_state()
    was_training = policy.training
    policy.eval()
    torch.manual_seed(seed)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=not cpu_offload):
        if cpu_offload:
            from .debug_offload import offloaded_forward
            loss, metrics = offloaded_forward(policy, batch, device, execution_dtype=torch.bfloat16)
        else:
            loss, metrics = policy(batch)
    policy.train(was_training)
    restore_rng(previous)
    return {"loss": float(loss), "metrics": metrics}


def train(run, stage, *, debug=False, updates=1, resume=False, device="cuda", threads=8, verify_only=False,
          debug_cpu_offload=False):
    if debug and not 1 <= updates <= 2:
        raise ValueError("Local debug is limited to 1–2 optimizer updates per stage")
    if verify_only and not resume:
        raise ValueError("verify-only requires --resume")
    if not debug and (device != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("Formal training requires CUDA; use debug for bounded local CPU checks")
    performance = performance_config(run)
    log_every = int(performance.get("log_every", run.get("log_every", 100)))
    if log_every <= 0:
        raise ValueError("log_every must be a positive integer")
    profile_updates = bool(performance.get("profile_updates", False))
    torch.set_num_threads(threads)
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if debug_cpu_offload:
        if not debug or device != "cuda" or world != 1:
            raise ValueError("CPU offload is only supported for single-GPU CUDA debug")
        run = {**run, "debug_cpu_offload": True}
    cpu_offload = debug_cpu_offload and stage == "stage2"
    if not debug and world != run["num_gpus"]:
        raise ValueError("Launch with the configured number of torchrun processes")
    if device == "cuda":
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if dev.type=="cuda" else "gloo")
    out = Path(run["output"])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/f".{stage}.lock").open("w") if rank==0 else None
    if lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shared_stage1 = run.get("shared_wm") and stage=="stage1"
    if run.get("shared_wm") and ((stage=="stage1") != (run["task"]=="shared_wm")):
        raise ValueError("Shared Stage 1 uses all tasks; Stage 2 requires --task")
    stage1 = check_shared_teacher(run, debug) if run.get("shared_wm") and stage=="stage2" else None
    ident = identity(run, debug)
    checkpoint = latest(out, stage) if resume else None
    if not resume and (out/stage).exists():
        raise FileExistsError("Stage output exists; use --resume or a fresh --output")
    attempt = out/stage/"run_identity.json"
    if attempt.exists() and json.loads(attempt.read_text()) != ident:
        raise ValueError("Recipe/code/data identity changed; use a new output directory")
    # Every rank must finish checking the old directory state before rank 0
    # creates it. Otherwise a late rank rejects this launch's own output.
    if world > 1:
        dist.barrier()
    if rank==0:
        atomic_json(attempt, ident)
    if checkpoint:
        saved = json.loads((checkpoint/"run_identity.json").read_text())
        if ident != saved:
            raise ValueError("Recipe/code/data identity changed; use a new output directory")
    if stage=="stage2" and not run.get("shared_wm"):
        stage1 = latest(out, "stage1")
        if stage1 is None:
            raise ValueError("Complete Stage 1 first")
        meta = json.loads((stage1/"complete.json").read_text())
        if not debug and meta["step"] != run["stage1"]["steps"]:
            raise ValueError("Stage 1 has not completed its configured training budget")
        if json.loads((stage1/"run_identity.json").read_text()) != ident:
            raise ValueError("Stage 1 and Stage 2 must share the same recipe identity")
    result_path = out/stage/"result.json"
    if resume and not verify_only and checkpoint and result_path.exists():
        old_result = json.loads(result_path.read_text())
        if (old_result.get("identity")==ident["fingerprint"] and old_result.get("complete")
                and (not debug or old_result["step"]>=updates)):
            if world>1:
                dist.destroy_process_group()
            return {**old_result, "skipped_completed": True}
    torch.manual_seed(run["seed"])
    np.random.seed(run["seed"])
    random.seed(run["seed"])
    started = time.monotonic()
    print(json.dumps({"event": "loading", "task": run["task"], "stage": stage, "device": str(dev), "debug": debug}), flush=True)
    policy, loading = create_policy(run, stage, stage1/"pretrained_model" if stage1 else None,
                                    checkpoint/"pretrained_model" if checkpoint else None)
    policy.to("cpu" if cpu_offload else dev)
    optimizer = make_optimizer(policy, run, stage)
    ema_device = select_ema_device(run, dev, cpu_offload=cpu_offload)
    ema = EMA(policy, run["ema_decay"], ema_device, initialize=checkpoint is None) \
        if stage=="stage2" and rank==0 else None
    start = 0
    saved_rng = None
    restore_audit = None
    if checkpoint:
        state = torch.load(checkpoint/"training.pt", map_location="cpu", weights_only=False)
        if state["world_size"] != world:
            raise ValueError("Resume requires the same world size")
        optimizer.load_state_dict(state["optimizer"])
        if ema:
            ema.load(checkpoint/"ema.safetensors", policy)
        start = state["step"]
        saved_rng = state["rng"][rank]
        if debug and rank==0:
            old = json.loads((checkpoint/"witnesses.json").read_text())
            if witnesses(policy, optimizer, ema) != old:
                raise AssertionError("Parameter/optimizer/EMA restore witness differs")
            restore_audit = {"parameters_optimizer_ema_restored": True}
        del state  # Do not retain a second CPU copy of every optimizer slot on GPU resume.
    if world > 1:
        wrapped = DistributedDataParallel(policy, device_ids=[local_rank] if dev.type=="cuda" else None,
                                          find_unused_parameters=ddp_find_unused_parameters(run),
                                          gradient_as_bucket_view=True,
                                          broadcast_buffers=False)
    else:
        wrapped = policy
    if shared_stage1:
        from .shared import SharedDataset, BalancedReplaySampler
        dataset = SharedDataset(run)
    else:
        dataset = UniVTACDataset(run)
    from torch.utils.data._utils.collate import default_collate
    diagnostic = default_collate([child[i] for child in dataset.datasets for i in (0, len(child)//2)]) if shared_stage1 else default_collate([dataset[0], dataset[len(dataset)//2]])
    diagnostic = {k:v.to(dev) if torch.is_tensor(v) else v for k,v in diagnostic.items()}
    if checkpoint and debug:
        restored_probe = probe(policy, diagnostic, dev, cpu_offload=cpu_offload)
        old_probe = json.loads((checkpoint/"audit.json").read_text())["probe"]
        difference = abs(restored_probe["loss"]-old_probe["loss"])
        if difference > 1e-6:
            raise AssertionError(f"Checkpoint forward changed: {difference}")
        if rank==0:
            restore_audit["probe_max_abs_difference"] = difference
    if verify_only:
        if not checkpoint or not debug:
            raise ValueError("verify-only requires debug resume")
        if rank==0:
            atomic_json(out/stage/"restore_report.json", restore_audit)
        dataset.close()
        if world>1:
            dist.destroy_process_group()
        return restore_audit
    # Never carry open h5py handles across DataLoader worker forks.
    dataset.close()
    limit = updates if debug else run[stage]["steps"]
    global_batch = (16 if shared_stage1 else 2)*world if debug else run[stage].get("global_batch", run["global_batch"])
    micro = (16 if shared_stage1 else 2) if debug and stage=="stage1" else 1 if debug else run[stage]["micro_batch"]
    if global_batch % (micro*world):
        raise ValueError("Global batch must divide micro_batch * world_size")
    accumulation = global_batch//(micro*world)
    sampler = BalancedReplaySampler(dataset.sizes, run["seed"], global_batch, world, rank, micro, start, limit) if shared_stage1 else ReplaySampler(len(dataset), run["seed"], global_batch, world, rank, start, limit)
    loader = DataLoader(dataset, batch_size=micro, sampler=sampler, num_workers=0 if debug else run["workers"],
                        generator=torch.Generator().manual_seed(run["seed"]+rank), pin_memory=dev.type=="cuda")
    policy.train()
    if saved_rng:
        restore_rng(saved_rng)
    else:
        torch.manual_seed(run["seed"]+rank)
    optimizer.zero_grad(set_to_none=True)
    audit = {"loading": loading, "restore": restore_audit, "updates": [], "debug": debug}
    selection = {"best_loss": None, "best_checkpoint": None, "stale": 0, "stopped_early": False}
    if checkpoint and shared_stage1:
        selection = json.loads((checkpoint/"audit.json").read_text()).get("selection", selection)
    if shared_stage1:
        audit["training_pairs_per_task"] = dict(zip(run["tasks"], dataset.sizes))
    step, micro_index, loss_sum = start, 0, None
    last_log_time, last_log_step = time.monotonic(), start
    iterator = iter(loader)
    while step < limit and not selection["stopped_early"]:
        next_step = step + 1
        log_due = debug or next_step % log_every == 0 or next_step == limit
        profile_update = profile_due(profile_updates, next_step, log_every, limit)
        if profile_update:
            # Drain prior asynchronous work before measuring this sampled update.
            sync_if_cuda(dev)
        update_started = time.monotonic()
        phase_seconds = {"data_h2d": 0., "forward_backward": 0., "optimizer": 0., "ema": 0.} \
            if profile_update else None
        set_lr(optimizer, run[stage], step)
        before = {n:tensor_witness(p) for n,p in policy.named_parameters() if p.requires_grad} if debug else None
        for micro_index in range(accumulation):
            phase_started = time.monotonic() if profile_update else None
            raw = next(iterator)
            if shared_stage1:
                from collections import Counter
                if min(Counter(raw["world_model_task"]).values()) < 2 or set(raw["world_model_task"]) != set(run["tasks"]):
                    raise AssertionError("Shared WM physical batch lost task balance or same-task negatives")
            batch = {k:v.to(dev) if torch.is_tensor(v) else v for k,v in raw.items()}
            if profile_update:
                sync_if_cuda(dev)
                phase_seconds["data_h2d"] += time.monotonic() - phase_started
            phase_started = time.monotonic() if profile_update else None
            if stage=="stage2":
                # Upstream increments per forward. Pin to optimizer updates so
                # gradient accumulation does not accelerate WAM schedules.
                policy.model.stage2_step.fill_(step)
            sync = nullcontext() if world==1 or micro_index==accumulation-1 else wrapped.no_sync()
            with sync, torch.autocast(dev.type, dtype=torch.bfloat16, cache_enabled=not cpu_offload):
                if cpu_offload:
                    from .debug_offload import offloaded_forward
                    loss, metrics = offloaded_forward(policy, batch, dev, backward_scale=accumulation,
                                                      execution_dtype=torch.bfloat16)
                else:
                    loss, metrics = wrapped(batch)
                    assert_finite(loss, "Nonfinite training loss")
                    (loss/accumulation).backward()
            loss_part = loss.detach()/accumulation
            loss_sum = loss_part if loss_sum is None else loss_sum + loss_part
            if profile_update:
                sync_if_cuda(dev)
                phase_seconds["forward_backward"] += time.monotonic() - phase_started
        phase_started = time.monotonic() if profile_update else None
        gradients = gradient_report(policy) if debug else None
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), run["grad_clip"])
        assert_finite(grad_norm, "Nonfinite gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if stage=="stage2":
            policy.model.stage2_step.fill_(step)
        if profile_update:
            sync_if_cuda(dev)
            phase_seconds["optimizer"] += time.monotonic() - phase_started
        phase_started = time.monotonic() if profile_update else None
        if ema:
            ema.update(policy)
        if profile_update:
            sync_if_cuda(dev)
            phase_seconds["ema"] += time.monotonic() - phase_started
        needs_row_values = debug or log_due
        loss_value = float(loss_sum) if needs_row_values else None
        grad_norm_value = float(grad_norm) if needs_row_values else None
        update_seconds = time.monotonic() - update_started
        interval_seconds = time.monotonic() - last_log_time if log_due else None
        interval_steps = step - last_log_step if log_due else None
        step_per_second = interval_steps / interval_seconds if interval_seconds and interval_seconds > 0 else None
        row = {"step": step, "loss": loss_value, "grad_norm": grad_norm_value,
               "lr": {g["name"]:g["lr"] for g in optimizer.param_groups}, "metrics": metrics,
               "update_seconds": update_seconds,
               "step_per_second": step_per_second,
               "global_samples_per_second": (global_batch*interval_steps/interval_seconds
                                              if interval_seconds and interval_seconds > 0 else None),
               "phase_seconds": phase_seconds,
               "timing_sampled": profile_update,
               "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(dev)/1024**3 if dev.type=="cuda" else None,
               "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(dev)/1024**3 if dev.type=="cuda" else None}
        if debug:
            row["debug_cpu_offload"] = cpu_offload
            row["gradients"] = gradients
            row["changed_groups"] = sorted({group_name(n) for n,p in policy.named_parameters()
                                             if p.requires_grad and tensor_witness(p)!=before[n]})
            required = {"world"} if stage=="stage1" else {"policy_vision", "language", "action_expert", "future_condition"}
            if not required.issubset(row["changed_groups"]):
                raise AssertionError(f"No update in required groups: {required-set(row['changed_groups'])}")
            if any(g["missing"] for g in gradients.values()):
                raise AssertionError(f"Trainable tensors without gradient: {gradients}")
            audit["updates"].append(row)
        if rank==0 and log_due:
            log = out/stage/"train.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as handle:
                handle.write(json.dumps(row)+"\n")
            print(json.dumps({"event":"update", "task":run["task"], "stage":stage, "step":step,
                              "loss":loss_value, "step_per_second":step_per_second,
                              "phase_seconds":phase_seconds,
                              "cuda_peak_allocated_gib":row["cuda_peak_allocated_gib"],
                              "cuda_peak_reserved_gib":row["cuda_peak_reserved_gib"],
                              "seconds":time.monotonic()-started}), flush=True)
        if log_due:
            last_log_time, last_log_step = time.monotonic(), step
        loss_sum = None
        validation_due = shared_stage1 and (debug or step%run[stage]["validate_every"]==0 or step==limit)
        if validation_due:
            validation = validate_world(policy, run, dev, world, rank, debug)
            validation["step"] = step
            if rank==0:
                with (out/stage/"validation.jsonl").open("a") as handle:
                    handle.write(json.dumps(validation)+"\n")
                print(json.dumps({"event": "validation", **validation}), flush=True)
            if selection["best_loss"] is None or validation["loss"] < selection["best_loss"]:
                selection.update(best_loss=validation["loss"], stale=0,
                                 best_checkpoint=str(out/stage/"checkpoints"/f"step_{step:08d}"))
            else:
                selection["stale"] += 1
            selection["stopped_early"] = bool(not debug and step>=run[stage]["early_stop_min_steps"]
                                              and selection["stale"]>=run[stage]["early_stop_patience"])
            audit["validation"] = validation
            audit["selection"] = selection.copy()
        if debug or validation_due or step%run["save_every"]==0 or step==limit:
            if debug:
                audit["probe"] = probe(policy, diagnostic, dev, cpu_offload=cpu_offload)
            checkpoint = save_checkpoint(policy, optimizer, ema, run, stage, step, ident, world, rank, audit)
            if rank==0 and not debug:
                prune_checkpoints(run, stage, selection["best_checkpoint"])
    dataset.close()
    result = {"task":run["task"], "stage":stage, "step":step, "checkpoint":str(checkpoint),
              "formal_training":not debug,"seconds":time.monotonic()-started,
              "world_size":world,"global_batch":global_batch, "micro_batch":micro,
              "accumulation":accumulation, "complete":True}
    result["identity"] = ident["fingerprint"]
    if shared_stage1:
        result.update(selected_checkpoint=selection["best_checkpoint"], selection=selection)
    if rank==0:
        atomic_json(out/stage/"result.json", result)
    if world>1:
        dist.destroy_process_group()
    return result
