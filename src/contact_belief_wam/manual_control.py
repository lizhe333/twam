"""Small, Isaac-free protocol shared by the browser server and simulator worker.

The HTTP side only reads snapshots and queues one command. Only the worker may
advance physics. A request id is durable and cannot execute twice after a retry.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import threading

import numpy as np

from .paths import atomic_json


class CommandError(ValueError):
    pass


def validate_command(value):
    if not isinstance(value, dict):
        raise CommandError("指令必须是 JSON 对象")
    allowed = {"id", "revision", "step", "kind", "axis", "sign", "amount", "duration", "exploration", "note"}
    if value.keys() - allowed:
        raise CommandError("未知指令字段")
    if not isinstance(value.get("id"), str) or not re.fullmatch(r"[a-zA-Z0-9_-]{8,80}", value["id"]):
        raise CommandError("无效指令编号")
    for key in ("step", "revision"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise CommandError(f"无效 {key}")
    kind = value.get("kind")
    if kind not in {"takeover", "replay", "translate", "rotate", "gripper", "hold", "finish"}:
        raise CommandError("未知动作")
    note = value.get("note", "")
    if not isinstance(note, str) or len(note) > 300:
        raise CommandError("备注最多 300 字")
    if type(value.get("exploration", False)) is not bool:
        raise CommandError("试探标记必须是布尔值")
    if kind in {"translate", "rotate", "gripper", "hold"}:
        if type(value.get("duration")) is not int or value["duration"] not in (1, 5, 10, 20):
            raise CommandError("插值步数必须为 1、5、10 或 20")
    if kind in {"translate", "rotate"}:
        if type(value.get("axis")) is not int or value["axis"] not in (0, 1, 2):
            raise CommandError("无效坐标轴")
        if type(value.get("sign")) is not int or value["sign"] not in (-1, 1):
            raise CommandError("无效方向")
        amount = value.get("amount")
        choices = (.1, .25, .5, 1., 2.) if kind == "translate" else (.25, .5, 1.)
        if type(amount) not in (int, float) or not math.isfinite(amount) or amount not in choices:
            raise CommandError("无效小步幅度")
    if kind == "gripper" and value.get("sign") not in (-1, 1):
        raise CommandError("夹爪方向必须为开或合")
    return dict(value)


def command_cost(command):
    if command["kind"] in {"takeover", "finish"}:
        return 0
    return 10 if command["kind"] == "replay" else command["duration"]


def check_state(command, state):
    if state.get("status") != "ready":
        raise CommandError("仿真尚未暂停就绪，请等待当前操作结束")
    if command["step"] != state["step"] or command["revision"] != state["revision"]:
        raise CommandError("画面已更新，请查看新反馈后重新操作")
    kind = command["kind"]
    if state["mode"] == "replay" and kind not in {"replay", "takeover", "finish"}:
        raise CommandError("请先点击“开始人工接管”")
    if state["mode"] == "manual" and kind in {"replay", "takeover"}:
        raise CommandError("已接管的分支不能倒退或继续原策略")
    if command_cost(command) > state["budget"] - state["step"]:
        raise CommandError("剩余动作预算不足；请减少插值步数或结束保存")


def movement_target(pose, origin, command, *, lateral_mm=3., rotation_deg=3., axial_mm=30.):
    """Only observed EE pose, NEVER object/slot ground-truth correction errors."""
    from scipy.spatial.transform import Rotation
    target = np.array(pose, dtype=np.float64, copy=True)
    base = np.asarray(origin, dtype=np.float64)
    kind, axis = command["kind"], command["axis"]
    if kind == "translate":
        target[axis] += command["sign"] * command["amount"] / 1000
    else:
        vector = np.zeros(3)
        vector[axis] = np.deg2rad(command["sign"] * command["amount"])
        quat = (Rotation.from_rotvec(vector) * Rotation.from_quat(target[[4, 5, 6, 3]])).as_quat()
        target[3:] = np.r_[quat[3], quat[:3]]
    angle = np.rad2deg((Rotation.from_quat(target[[4, 5, 6, 3]]) *
                       Rotation.from_quat(base[[4, 5, 6, 3]]).inv()).magnitude())
    if (np.linalg.norm(target[:2] - base[:2]) * 1000 > lateral_mm + 1e-5 or
            abs(target[2] - base[2]) * 1000 > axial_mm + 1e-5 or angle > rotation_deg + 1e-5):
        raise CommandError("超出本分支局部范围：横向 3 mm、转角 3°、轴向 30 mm；指令未执行")
    return target


class CommandBus:
    """One unqueued motion at a time, including across browser tabs/reloads."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "requests").mkdir(exist_ok=True)
        self.pending = self.root / "pending.json"
        self.active = self.root / "active.json"
        self.snapshot = self.root / "snapshot.json"
        self.lock = threading.Lock()

    def read(self):
        try:
            return json.loads(self.snapshot.read_text())
        except FileNotFoundError:
            return dict(status="starting", mode="replay", step=0, revision=0, budget=600,
                        message="正在启动仿真并验证原轨迹回放…")

    def submit(self, command):
        command = validate_command(command)
        with self.lock:
            receipt = self.root / "requests" / f"{command['id']}.json"
            if receipt.exists():
                if json.loads(receipt.read_text()) != command:
                    raise CommandError("指令编号已被另一请求使用")
                return dict(accepted=True, id=command["id"], duplicate=True)
            if self.pending.exists() or self.active.exists():
                raise CommandError("已有指令执行中，不接受排队动作")
            check_state(command, self.read())
            atomic_json(receipt, command)
            with receipt.open("rb") as handle:
                os.fsync(handle.fileno())
            atomic_json(self.pending, command)
            return dict(accepted=True, id=command["id"], duplicate=False)

    def claim(self):
        """Called ONLY by the simulator thread; does not touch simulation/RNG."""
        try:
            self.pending.replace(self.active)
        except FileNotFoundError:
            return None
        return json.loads(self.active.read_text())

    def complete(self, event):
        with (self.root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.active.unlink(missing_ok=True)

