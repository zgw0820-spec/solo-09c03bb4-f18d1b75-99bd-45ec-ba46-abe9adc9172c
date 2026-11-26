"""内置 UPS 样本与逻辑回归（纯 NumPy 实现，全批量梯度下降）。

样本为合成的 UPS 运行数据，标签 1 表示过热预警，特征包括：
内部温度、负载率、风扇转速、电池温度、环境温度、电池老化度。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = [
    "内部温度(°C)",
    "负载率(%)",
    "风扇转速(rpm)",
    "电池温度(°C)",
    "环境温度(°C)",
    "电池老化度(%)",
]

# 学习率偏大：训练集含误标注时，验证损失会在前几轮见底后反弹，便于演示早停。
DEFAULT_LR = 1.1
# 训练样本中的误标注比例（模拟现场误报/漏报记录），验证集保持干净
TRAIN_LABEL_NOISE = 0.22
NOISE_SEED = 1
EPS = 1e-12


@dataclass(frozen=True)
class Dataset:
    X_train: np.ndarray  # 已标准化，含偏置列，形状 (n_train, 7)
    y_train: np.ndarray  # 含误标注噪声
    X_val: np.ndarray
    y_val: np.ndarray    # 干净标签，用于客观评估
    feature_names: list[str]
    raw_means: np.ndarray
    raw_stds: np.ndarray
    n_flipped: int

    @property
    def n_train(self) -> int:
        return self.y_train.shape[0]

    @property
    def n_val(self) -> int:
        return self.y_val.shape[0]


def _build_raw(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_per_class = 210
    # 正常机组
    normal = np.column_stack(
        [
            rng.normal(45, 6, n_per_class),       # 内部温度
            rng.normal(55, 15, n_per_class),      # 负载率
            rng.normal(2800, 300, n_per_class),   # 风扇转速
            rng.normal(38, 5, n_per_class),       # 电池温度
            rng.normal(28, 6, n_per_class),       # 环境温度
            rng.normal(30, 15, n_per_class),      # 电池老化度
        ]
    )
    # 过热机组
    overheat = np.column_stack(
        [
            rng.normal(72, 7, n_per_class),
            rng.normal(82, 10, n_per_class),
            rng.normal(3400, 250, n_per_class),
            rng.normal(58, 7, n_per_class),
            rng.normal(35, 6, n_per_class),
            rng.normal(65, 16, n_per_class),
        ]
    )
    X = np.vstack([normal, overheat])
    y = np.concatenate(
        [np.zeros(n_per_class), np.ones(n_per_class)]
    ).astype(np.float64)
    # 打乱
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def build_dataset(seed: int = 42, val_ratio: float = 0.3) -> Dataset:
    X_raw, y = _build_raw(seed)
    n_val = int(X_raw.shape[0] * val_ratio)
    X_train_raw, X_val_raw = X_raw[:-n_val], X_raw[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]

    means = X_train_raw.mean(axis=0)
    stds = X_train_raw.std(axis=0)
    X_train = (X_train_raw - means) / stds
    X_val = (X_val_raw - means) / stds
    # 追加偏置列
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

    # 仅对训练标签注入固定随机误标注，验证集保持干净
    rng_noise = np.random.default_rng(NOISE_SEED)
    flip = rng_noise.random(y_train.shape[0]) < TRAIN_LABEL_NOISE
    y_train_noisy = y_train.copy()
    y_train_noisy[flip] = 1.0 - y_train_noisy[flip]

    return Dataset(
        X_train=X_train,
        y_train=y_train_noisy,
        X_val=X_val,
        y_val=y_val,
        feature_names=FEATURE_NAMES,
        raw_means=means,
        raw_stds=stds,
        n_flipped=int(flip.sum()),
    )


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """平均二元交叉熵损失；权重出现 NaN（学习率过大发散）时返回 +inf。"""
    if not np.all(np.isfinite(w)):
        return float("inf")
    p = sigmoid(X @ w)
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def train_step(
    w: np.ndarray, X: np.ndarray, y: np.ndarray, lr: float = DEFAULT_LR
) -> tuple[np.ndarray, float]:
    """全批量梯度下降一步，返回新权重（新数组）与该轮训练损失。"""
    p = sigmoid(X @ w)
    grad = X.T @ (p - y) / X.shape[0]
    w_new = w - lr * grad
    if not np.all(np.isfinite(w_new)):
        # 发散时冻结权重，保证后续轮次仍可判定（损失不再刷新 -> 触发早停）
        w_new = w.copy()
    return w_new, bce_loss(w_new, X, y)


def accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    pred = (sigmoid(X @ w) >= 0.5).astype(np.float64)
    return float(np.mean(pred == y))


def initial_weights(n_features: int) -> np.ndarray:
    return np.zeros(n_features, dtype=np.float64)
