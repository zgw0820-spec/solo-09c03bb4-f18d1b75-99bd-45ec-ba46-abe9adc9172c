"""异步训练引擎。

并发与早停规则（与需求一一对应）：

1. 每轮训练完成后，用当时的权重快照异步计算验证集损失；
2. 验证协程带随机延迟，结果允许乱序返回；
3. 已发起训练但尚未判定的轮次最多 3 个（信号量控位，有空位才发起新一轮）；
4. 判定严格按轮号顺序：轮 N 未判定前，轮 N+1 即使先返回也保持 returned；
5. 连续两轮验证损失未刷新历史最低，立即停止发起新一轮，触发轮之后的
   在途结果一律忽略（ignored）；
6. 早停或跑满最后一轮后，输出判定范围内损失最低的那轮权重（并列取最早轮）。

实现要点：用 asyncio.TaskGroup 做结构化并发；信号量在生产者发起轮次前
acquire、在 judge 判定（或忽略）该轮后 release，因此信号量计数恒等于
“已发起但未判定”的轮次数，从结构上保证不超过 3，而非依赖时序。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import model

MAX_IN_FLIGHT = 3
PATIENCE = 2  # 连续两轮未刷新最低损失即早停

# 轮次状态
PENDING = "pending"        # 尚未开始训练
TRAINING = "training"      # 训练中
VALIDATING = "validating"  # 训练完，验证协程在途
RETURNED = "returned"      # 验证已返回，等待按轮号判定
JUDGED = "judged"          # 已判定（有效结果）
IGNORED = "ignored"        # 早停触发后被忽略的结果
CANCELLED = "cancelled"    # 早停时未完成的在途任务 / 未发起轮次


@dataclass
class EpochRecord:
    epoch: int
    state: str = PENDING
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    started_at: Optional[float] = None
    returned_at: Optional[float] = None
    judged_at: Optional[float] = None
    note: str = ""


@dataclass
class TrainConfig:
    max_epochs: int
    val_delay: float          # 每轮验证延迟基线（秒）
    jitter: bool = True       # 是否在延迟上加随机抖动（产生乱序）
    lr: float = model.DEFAULT_LR
    seed: int = 42


@dataclass
class TrainSession:
    id: str
    config: TrainConfig
    dataset: model.Dataset
    records: list[EpochRecord] = field(default_factory=list)
    status: str = "running"   # running / early_stopped / completed / failed
    best_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None
    best_weights: Optional[np.ndarray] = None
    best_accuracy: Optional[float] = None
    stop_reason: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    peak_in_flight: int = 0
    stale_streak: int = 0
    # 内部同步对象（不参与序列化）
    _results: dict[int, float] = field(default_factory=dict)
    _events: dict[int, asyncio.Event] = field(default_factory=dict)
    _snapshots: dict[int, np.ndarray] = field(default_factory=dict)
    _semaphore: Optional[asyncio.Semaphore] = field(default=None, repr=False)
    _stop_event: Optional[asyncio.Event] = field(default=None, repr=False)
    _lock: Optional[asyncio.Lock] = field(default=None, repr=False)
    _in_flight: list[int] = field(default_factory=lambda: [0], repr=False)
    _peak: list[int] = field(default_factory=lambda: [0], repr=False)

    def snapshot(self) -> dict:
        """供 API/前端读取的可 JSON 序列化状态。"""
        return {
            "id": self.id,
            "status": self.status,
            "max_epochs": self.config.max_epochs,
            "val_delay": self.config.val_delay,
            "jitter": self.config.jitter,
            "lr": self.config.lr,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_accuracy": self.best_accuracy,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "stale_streak": self.stale_streak,
            "peak_in_flight": self.peak_in_flight,
            "n_train": self.dataset.n_train,
            "n_val": self.dataset.n_val,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "epochs": [
                {
                    "epoch": r.epoch,
                    "state": r.state,
                    "train_loss": r.train_loss,
                    "val_loss": r.val_loss,
                    "note": r.note,
                }
                for r in self.records
            ],
        }


DelayFn = Callable[[int], float]


async def run_session(
    session: TrainSession,
    dataset: Optional[model.Dataset] = None,
    delay_fn: Optional[DelayFn] = None,
) -> TrainSession:
    """执行整个训练会话，直到早停或跑满最后一轮。"""
    ds = dataset or session.dataset
    cfg = session.config
    rng = np.random.default_rng(cfg.seed + 999)
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(MAX_IN_FLIGHT)
    stop_event = asyncio.Event()
    lock = asyncio.Lock()
    # 已占空位（已发起、等待判定）的轮号；judge 在轮次终态时移除并释放
    slot_owned: set[int] = set()
    session._semaphore = semaphore
    session._stop_event = stop_event
    session._lock = lock
    session.records = [
        EpochRecord(epoch=i + 1) for i in range(cfg.max_epochs)
    ]

    def actual_delay(epoch: int) -> float:
        if delay_fn is not None:
            return max(0.0, delay_fn(epoch))
        base = max(0.0, cfg.val_delay)
        if not cfg.jitter:
            return base
        # 抖动范围大，足以让后发起的轮次先返回（制造乱序）
        return base * (0.15 + 1.7 * float(rng.random()))

    async def epoch_task(epoch: int, weights_in: np.ndarray) -> None:
        """单轮：训练一步 -> 快照权重 -> 带延迟异步验证。

        并发位由生产者（run_epochs）统一 acquire，且要等到 judge 判定完
        该轮（judged/ignored）才释放——信号量计数即“已发起但未判定”的
        轮次数，因此该数字在结构上恒不超过 MAX_IN_FLIGHT，而非依赖时序。
        """
        rec = session.records[epoch - 1]
        rec.state = TRAINING
        rec.started_at = time.time()
        async with lock:
            session._in_flight[0] += 1
            session._peak[0] = max(session._peak[0], session._in_flight[0])
            # 实时写回，保证状态翻转为早停/完成时峰值已可读取
            session.peak_in_flight = session._peak[0]
        weights, train_loss = await loop.run_in_executor(
            None, model.train_step, weights_in, ds.X_train, ds.y_train, cfg.lr
        )
        rec.train_loss = train_loss

        snapshot = weights.copy()
        session._snapshots[epoch] = snapshot
        rec.state = VALIDATING
        async with lock:
            session._peak[0] = max(session._peak[0], session._in_flight[0])
            session.peak_in_flight = session._peak[0]

        event = asyncio.Event()
        session._events[epoch] = event
        try:
            # 验证延迟：若会话被外部取消，TaskGroup 会在此抛 CancelledError
            await asyncio.sleep(actual_delay(epoch))
            val_loss = await loop.run_in_executor(
                None, model.bce_loss, snapshot, ds.X_val, ds.y_val
            )
            session._results[epoch] = val_loss
            rec.val_loss = val_loss
            rec.returned_at = time.time()
            rec.state = RETURNED
            rec.note = "已返回，等待按轮号判定"
        finally:
            # 无论正常返回还是被取消都发信号；空位由 judge 统一释放
            event.set()

    async def run_epochs() -> None:
        """生产者：先占空位再发起轮次；早停后剩余轮次标为取消、不再发起。"""
        w = model.initial_weights(ds.X_train.shape[1])
        acquire_task: Optional[asyncio.Task] = None
        stop_task = asyncio.create_task(stop_event.wait())
        try:
            async with asyncio.TaskGroup() as tg:
                for rec in session.records:
                    if stop_event.is_set():
                        rec.state = CANCELLED
                        rec.note = "早停已触发，未发起"
                        continue
                    # 空位控制：满 3 个（已发起未判定）时等待；停止事件可抢先唤醒
                    acquire_task = asyncio.create_task(semaphore.acquire())
                    await asyncio.wait(
                        {acquire_task, stop_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if stop_event.is_set():
                        if not acquire_task.done():
                            acquire_task.cancel()
                        else:
                            semaphore.release()
                        rec.state = CANCELLED
                        rec.note = "早停已触发，未发起"
                        continue
                    slot_owned.add(rec.epoch)
                    # 用上一轮结束时的权重发起本轮，任务内部完成本轮梯度更新
                    tg.create_task(epoch_task(rec.epoch, w.copy()))
                    # 延续权重链：同步计算下一步权重（与该轮训练使用相同输入）
                    w, _ = await loop.run_in_executor(
                        None, model.train_step, w, ds.X_train, ds.y_train, cfg.lr
                    )
        finally:
            stop_task.cancel()

    async def judge() -> None:
        """消费者：严格按轮号判定，连续 PATIENCE 轮未刷新即早停。"""
        best_epoch: Optional[int] = None
        best_loss = float("inf")
        best_weights: Optional[np.ndarray] = None
        best_acc: Optional[float] = None
        stale = 0
        stop_trigger_epoch: Optional[int] = None
        judged = 0

        async def close_slot(epoch: int) -> None:
            """轮次进入终态（判定/忽略/取消）时归还空位，每轮恰好一次。"""
            if epoch in slot_owned:
                slot_owned.discard(epoch)
                async with lock:
                    session._in_flight[0] = max(0, session._in_flight[0] - 1)
                semaphore.release()

        while judged < cfg.max_epochs:
            epoch = judged + 1
            rec = session.records[epoch - 1]

            # 早停后未发起的轮次：无事件、无空位，直接跳过
            if rec.state == CANCELLED:
                judged += 1
                continue

            event = session._events.get(epoch)
            if event is None:
                # 该轮还没创建（生产者在等空位），稍后重试
                await asyncio.sleep(0.005)
                continue
            await event.wait()

            # 在途任务被 TaskGroup 取消（会话级取消）
            if rec.state in (IGNORED, CANCELLED):
                await close_slot(epoch)
                judged += 1
                continue

            val_loss = session._results.get(epoch)
            if val_loss is None:
                rec.state = IGNORED
                rec.note = rec.note or "早停触发，结果忽略"
                await close_slot(epoch)
                judged += 1
                continue

            # 早停触发轮之后的结果一律忽略
            if stop_trigger_epoch is not None and epoch > stop_trigger_epoch:
                rec.state = IGNORED
                rec.note = f"早停在第 {stop_trigger_epoch} 轮触发，结果忽略"
                await close_slot(epoch)
                judged += 1
                continue

            # 有效判定（严格按轮号顺序）
            rec.state = JUDGED
            rec.judged_at = time.time()
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_weights = session._snapshots[epoch].copy()
                best_acc = model.accuracy(best_weights, ds.X_val, ds.y_val)
                stale = 0
                rec.note = "已判定，刷新最低验证损失"
            else:
                stale += 1
                if not np.isfinite(val_loss):
                    rec.note = f"已判定，损失发散（NaN/Inf），连续 {stale} 轮未刷新"
                else:
                    rec.note = f"已判定，未刷新最低损失（连续 {stale} 轮）"
            judged += 1
            await close_slot(epoch)

            session.best_epoch = best_epoch
            session.best_val_loss = best_loss if best_epoch else None
            session.best_weights = best_weights
            session.best_accuracy = best_acc
            session.stale_streak = stale

            if stale >= PATIENCE:
                # 连续两轮未刷新：停止发起新一轮；触发轮之后的在途结果将被忽略
                stop_trigger_epoch = epoch
                stop_event.set()
                session.status = "early_stopped"
                session.stop_reason = (
                    f"第 {epoch} 轮判定后连续 {PATIENCE} 轮未刷新最低验证损失"
                )

    try:
        async with asyncio.TaskGroup() as outer:
            outer.create_task(run_epochs())
            outer.create_task(judge())
    except Exception as exc:  # pragma: no cover - 防御性
        session.status = "failed"
        session.error = repr(exc)
        session.finished_at = time.time()
        raise

    session.finished_at = time.time()
    if session.status == "running":
        session.status = "completed"
        session.stop_reason = f"已跑完最后一轮（第 {cfg.max_epochs} 轮）"
    return session


def new_session(cfg: TrainConfig, dataset: model.Dataset) -> TrainSession:
    return TrainSession(id=uuid.uuid4().hex[:12], config=cfg, dataset=dataset)
