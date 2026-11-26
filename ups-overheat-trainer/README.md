# UPS 过热预警模型训练页

基于 **FastAPI + NumPy** 的逻辑回归训练演示：在合成的 UPS（不间断电源）内置样本上
做全批量梯度下降，网页可配置训练轮数上限、每轮验证延迟与模拟报错轮，实时查看训练/
验证进度，并完整演示**异步乱序验证、验证结果与报错统一按轮号判定、连续两轮未刷新
即早停、报错失败即作废**的过程。

## 功能与规则对照

| 需求 | 实现位置 |
| --- | --- |
| FastAPI + NumPy 训练逻辑回归 | `app/model.py`（sigmoid / BCE / 全批量梯度下降，纯 NumPy） |
| 内置样本，且能演示早停 | `app/model.py` 生成 420 条样本；训练集注入 22% 误标注，验证集保持干净，验证损失前几轮见底后反弹 |
| 网页设置训练轮数上限、各轮验证延迟 | `web/index.html` 表单 → `POST /api/train`（`max_epochs`、`val_delay`，另有抖动开关/学习率/模拟报错轮） |
| 查看训练与验证进度 | 前端每 400ms 轮询 `GET /api/sessions/{id}`，表格展示每轮状态、训练/验证损失与说明 |
| 每轮用当时权重**异步**计算验证损失 | `app/engine.py::epoch_task`，验证协程带延迟、在线程池算 BCE |
| 验证结果可乱序返回，判定按轮号 | 延迟加随机抖动；结果先到先标记 `returned`，`judge` 严格按 1..N 顺序判定 |
| 验证报错也按轮号判定，后轮报错不得抢跑 | `epoch_task` 把报错记为该轮的“报错返回值”（`error_returned`），与结果一起排队；不会再拖垮整场训练 |
| 已开始训练但未判定的轮次最多 3 个，有空位才接着跑 | 信号量在发起前 acquire、**判定后才 release**，计数即“已发起未判定”窗口，结构性 ≤ 3 |
| 连续两轮未刷新最低损失即停止发起新一轮 | `PATIENCE = 2`，触发后结论立即固化、`stop_event` 置位，生产者不再创建任务 |
| 先判到早停：立即开放最佳轮下载，之后报错不改结论/权重 | 判定时即保存 `best_weights`；触发轮之后的结果/报错无论何时返回都标 `ignored` |
| 先判到报错：本次训练失败、不再发起新轮次 | `failed_epoch` 记录失败轮号，`status = failed`，`stop_event` 置位 |
| 失败后网页标出失败轮号并禁用下载，接口也拒绝 | `web/index.html` 失败轮卡片/红色行/禁用按钮；`GET .../weights` 对 `status == failed` 返回 `409`，即使此前已有最佳轮 |
| 训练页可指定一轮模拟验证报错 | 表单“模拟验证报错” → `simulate_error_epoch`，走与真实报错完全相同的判定路径 |
| 下载判定范围内损失最低轮的权重，并列取最早轮 | `GET /api/sessions/{id}/weights` 下载 `.npz`；`best_loss` 用严格小于比较，首次达到即保留 |
| Docker Compose 启动 | `docker compose up --build`，映射 8000 端口 |

轮次状态机：`pending → training → validating → returned | error_returned → judged | failed`；
结论固化后的轮次为 `ignored`（已发起，结果/报错忽略）或 `cancelled`（未发起）。

## 两条演示路径

1. **早停**（默认参数，`lr=1.1`）：连续两轮未刷新最低损失 → `early_stopped`，
   最佳轮权重立即可下载；之后即便触发轮后某轮验证报错，结论与权重都不变。
2. **失败**：例如 `max_epochs=12, lr=1.1, simulate_error_epoch=4` → 第 4 轮报错被
   按轮号判到后 `failed`；网页标出第 4 轮并禁用下载；权重接口返回 `409`
   （即使第 4 轮之前已有最佳轮）。

## 快速开始（Docker Compose）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

停止：`docker compose down`。

## 本地直接运行（无 Docker 时）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 训练页 |
| GET | `/api/dataset` | 内置样本概况（特征、样本量、误标注数） |
| POST | `/api/train` | 发起训练，body：`max_epochs`、`val_delay`、`jitter`、`lr`、`seed`、`simulate_error_epoch`（可选） |
| GET | `/api/sessions/{id}` | 查询会话与逐轮进度（含 `failed_epoch`、每轮 `error`） |
| GET | `/api/sessions/{id}/weights` | 训练结束后下载最佳轮权重（`.npz`）；失败会话返回 `409` |

同一时间只允许一个运行中的会话，重复发起返回 `409`；训练未结束时下载权重也返回 `409`；
训练失败（任一已判定轮验证报错）后下载权重返回 `409`，即使此前已有最佳轮。

权重文件内容（`numpy.savez`）：

- `weights`：6 个特征权重 + 1 个偏置，形状 `(7,)`
- `feature_names`：权重对应名称（末位为“偏置”）
- `best_epoch` / `best_val_loss` / `best_accuracy`：最佳轮信息
- `status` / `stop_reason`：结束状态（`early_stopped` / `completed`）与原因
（失败会话不会生成权重文件）

## 测试

```bash
python -m unittest discover -s tests -v
```

- `tests/test_engine.py`：乱序返回但按序判定、并发窗口结构性 ≤3、脚本化早停、
  并列取最早轮、跑满完成、内置样本真实早停、30 组随机配置模糊测试；
  新增：先早停后报错不改结论、先报错即失败且不再发起、后轮报错按轮号排队、
  含随机模拟报错的模糊测试
- `tests/test_api.py`：真实 uvicorn 线程上端到端验证（含乱序轮询、409、权重下载校验、
  模拟报错导致失败并拒绝下载、早停后晚报错仍可下载）

## 目录结构

```
app/
  model.py    # 样本生成、逻辑回归、损失函数
  engine.py   # 异步训练引擎（并发控制/乱序/判定/早停）
  main.py     # FastAPI 路由
web/
  index.html  # 训练页（原生 HTML/JS，无构建步骤）
tests/        # unittest 测试
Dockerfile
docker-compose.yml
```
