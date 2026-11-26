"""FastAPI 应用：训练控制、进度查询、最佳权重下载。"""

from __future__ import annotations

import asyncio
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import engine, model

WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"

DATASET: Optional[model.Dataset] = None
SESSIONS: dict[str, engine.TrainSession] = {}
BACKGROUND_TASKS: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global DATASET
    DATASET = model.build_dataset()
    yield


app = FastAPI(title="UPS 过热预警模型训练页", lifespan=lifespan)


class StartRequest(BaseModel):
    max_epochs: int = Field(default=15, ge=1, le=100)
    val_delay: float = Field(default=0.6, ge=0.0, le=30.0)
    jitter: bool = True
    lr: float = Field(default=model.DEFAULT_LR, gt=0.0, le=20.0)
    seed: int = Field(default=42, ge=0, le=10_000_000)


@app.get("/api/dataset")
async def dataset_info():
    ds = DATASET
    return {
        "feature_names": ds.feature_names,
        "n_train": ds.n_train,
        "n_val": ds.n_val,
        "train_positive": int(ds.y_train.sum()),
        "val_positive": int(ds.y_val.sum()),
        "n_flipped": ds.n_flipped,
        "label_noise_ratio": model.TRAIN_LABEL_NOISE,
        "means": ds.raw_means.round(3).tolist(),
        "stds": ds.raw_stds.round(3).tolist(),
    }


@app.post("/api/train")
async def start_training(req: StartRequest):
    # 单会话：已有训练在跑则拒绝，避免并发会话互相干扰
    active = next(
        (s for s in SESSIONS.values() if s.status == "running"), None
    )
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"训练 {active.id} 仍在进行中，请等待结束后再发起",
        )

    cfg = engine.TrainConfig(
        max_epochs=req.max_epochs,
        val_delay=req.val_delay,
        jitter=req.jitter,
        lr=req.lr,
        seed=req.seed,
    )
    session = engine.new_session(cfg, DATASET)
    SESSIONS[session.id] = session

    task = asyncio.create_task(engine.run_session(session, DATASET))
    BACKGROUND_TASKS.add(task)
    task.add_done_callback(BACKGROUND_TASKS.discard)
    return {"session_id": session.id, **session.snapshot()}


@app.get("/api/sessions/{session_id}")
async def session_status(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="训练会话不存在")
    return session.snapshot()


@app.get("/api/sessions")
async def list_sessions():
    return [s.snapshot() for s in SESSIONS.values()]


@app.get("/api/sessions/{session_id}/weights")
async def download_weights(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="训练会话不存在")
    if session.status == "running":
        raise HTTPException(
            status_code=409, detail="训练尚未结束，无法下载权重"
        )
    if session.best_weights is None:
        raise HTTPException(status_code=409, detail="尚无有效判定轮次")

    w = session.best_weights
    names = session.dataset.feature_names + ["偏置"]
    buf = io.BytesIO()
    np.savez(
        buf,
        weights=w,
        feature_names=np.array(names, dtype=object),
        best_epoch=np.int64(session.best_epoch),
        best_val_loss=np.float64(session.best_val_loss),
        best_accuracy=np.float64(session.best_accuracy or 0.0),
        status=np.array(session.status),
        stop_reason=np.array(session.stop_reason),
    )
    buf.seek(0)
    filename = f"ups_lr_weights_{session.id}_epoch{session.best_epoch}.npz"
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/")
async def index():
    return FileResponse(WEB_INDEX)
