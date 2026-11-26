"""HTTP 端到端测试：真实 uvicorn 线程 + urllib，不引入 httpx 依赖。"""

import json
import socket
import time
import urllib.request
import urllib.error

import numpy as np
import uvicorn
import unittest

from app.main import app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        config = uvicorn.Config(
            app, host="127.0.0.1", port=cls.port, log_level="warning"
        )
        cls.server = uvicorn.Server(config)
        import threading

        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{cls.port}/api/dataset", timeout=1)
                return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("测试服务器未在 15s 内就绪")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True

    def _get(self, path):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}{path}", timeout=10
        ) as r:
            return r.status, json.loads(r.read())

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_full_flow_with_out_of_order_polling(self):
        status, info = self._get("/api/dataset")
        self.assertEqual(status, 200)
        self.assertEqual(len(info["feature_names"]), 6)
        self.assertGreater(info["n_train"], info["n_val"])

        status, created = self._post(
            "/api/train",
            {"max_epochs": 12, "val_delay": 0.15, "jitter": True, "lr": 3.0},
        )
        self.assertEqual(status, 200)
        sid = created["session_id"]

        # 训练中重复发起应被 409 拒绝
        status2, _ = self._post(
            "/api/train", {"max_epochs": 5, "val_delay": 0.0}
        )
        self.assertEqual(status2, 409)

        # 轮询直到结束，并确认进度状态机合法
        saw_returned = False
        final = None
        deadline = time.time() + 60
        valid_states = {
            "pending", "training", "validating", "returned", "error_returned",
            "judged", "failed", "ignored", "cancelled",
        }
        while time.time() < deadline:
            _, snap = self._get(f"/api/sessions/{sid}")
            for ep in snap["epochs"]:
                self.assertIn(ep["state"], valid_states)
                if ep["state"] == "returned":
                    saw_returned = True
            final = snap
            if snap["status"] in ("early_stopped", "completed", "failed"):
                break
            time.sleep(0.05)

        self.assertIn(final["status"], ("early_stopped", "completed"))
        self.assertLessEqual(final["peak_in_flight"], 3)
        self.assertTrue(saw_returned, "轮询期间应能观察到 returned（乱序/待判定）状态")
        self.assertIsNotNone(final["best_epoch"])

        # 下载权重并校验 npz 内容
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/api/sessions/{sid}/weights", timeout=10
        ) as r:
            self.assertEqual(r.status, 200)
            data = r.read()
            self.assertIn("attachment", r.headers["Content-Disposition"])
        bundle = np.load(__import__("io").BytesIO(data), allow_pickle=True)
        self.assertEqual(bundle["weights"].shape, (7,))
        self.assertEqual(int(bundle["best_epoch"]), final["best_epoch"])
        self.assertAlmostEqual(
            float(bundle["best_val_loss"]), final["best_val_loss"], places=10
        )
        self.assertEqual(str(bundle["status"]), final["status"])

    def test_simulated_error_fails_epoch_and_blocks_download(self):
        """指定轮报错：页面可见失败轮号，接口拒绝权重，即使此前已有最佳轮。"""
        # lr=1.1 时第 1 轮即可刷新最低损失（此前有最佳轮）；第 3 轮模拟报错
        _, created = self._post(
            "/api/train",
            {
                "max_epochs": 12, "val_delay": 0.15, "jitter": True,
                "lr": 1.1, "simulate_error_epoch": 3,
            },
        )
        sid = created["session_id"]
        self.assertEqual(created["simulate_error_epoch"], 3)
        final = None
        deadline = time.time() + 60
        while time.time() < deadline:
            _, snap = self._get(f"/api/sessions/{sid}")
            final = snap
            if snap["status"] in ("early_stopped", "completed", "failed"):
                break
            time.sleep(0.05)

        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["failed_epoch"], 3)
        # 失败前已判定出最佳轮（lr=1.1 时第 2 轮），权重仍在内存，但接口拒绝提供
        self.assertIsNotNone(final["best_epoch"])
        self.assertLess(final["best_epoch"], final["failed_epoch"])
        self.assertIn("第 3 轮", final["error"])
        ep3 = final["epochs"][2]
        self.assertEqual(ep3["state"], "failed")
        self.assertIn("验证报错", ep3["error"])
        # 失败轮之后的在途轮次只能是 ignored/cancelled
        for ep in final["epochs"][3:]:
            self.assertIn(ep["state"], ("ignored", "cancelled"))
        # 报错之前已判定的轮次保留结论
        self.assertEqual(final["epochs"][0]["state"], "judged")

        # 接口拒绝提供本次权重（即使 best_epoch 存在）
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/api/sessions/{sid}/weights",
                timeout=10,
            )
            self.fail("训练失败后下载权重应被拒绝")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 409)
            detail = json.loads(e.read())["detail"]
            self.assertIn("第 3 轮", detail)
            self.assertIn("失败", detail)

    def test_early_stop_then_late_error_still_downloadable(self):
        """早停结论先固化：之后第 6 轮报错返回也不改结论，权重仍可下载。"""
        _, created = self._post(
            "/api/train",
            {
                "max_epochs": 12, "val_delay": 0.3, "jitter": False,
                "lr": 1.1, "simulate_error_epoch": 9,
            },
        )
        sid = created["session_id"]
        final = None
        deadline = time.time() + 60
        while time.time() < deadline:
            _, snap = self._get(f"/api/sessions/{sid}")
            final = snap
            if snap["status"] in ("early_stopped", "completed", "failed"):
                break
            time.sleep(0.05)

        self.assertEqual(final["status"], "early_stopped")  # 真实样本 lr=1.1 第 7 轮早停
        self.assertIsNone(final["failed_epoch"])
        self.assertIsNotNone(final["best_epoch"])
        # 早停触发后返回的第 9 轮报错应被忽略，不得改成失败
        self.assertEqual(final["epochs"][8]["state"], "ignored")
        self.assertIn("早停", final["epochs"][8]["note"])

        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/api/sessions/{sid}/weights", timeout=10
        ) as r:
            self.assertEqual(r.status, 200)
            data = r.read()
        bundle = np.load(__import__("io").BytesIO(data), allow_pickle=True)
        self.assertEqual(str(bundle["status"]), "early_stopped")
        self.assertEqual(int(bundle["best_epoch"]), final["best_epoch"])

    def test_simulate_error_epoch_out_of_range_rejected(self):
        """模拟报错轮号超过训练轮数上限：直接 422，不创建会话。"""
        status, body = self._post(
            "/api/train",
            {"max_epochs": 4, "val_delay": 0.0, "simulate_error_epoch": 9},
        )
        self.assertEqual(status, 422)
        self.assertIn("超出", body["detail"])

    def test_running_weights_download_rejected(self):
        _, created = self._post(
            "/api/train", {"max_epochs": 10, "val_delay": 0.2, "jitter": True}
        )
        sid = created["session_id"]
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/api/sessions/{sid}/weights", timeout=5
            )
            self.fail("训练中下载权重应被拒绝")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 409)
        # 等它结束，避免会话残留影响其它用例
        deadline = time.time() + 60
        while time.time() < deadline:
            _, snap = self._get(f"/api/sessions/{sid}")
            if snap["status"] in ("early_stopped", "completed", "failed"):
                break
            time.sleep(0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
