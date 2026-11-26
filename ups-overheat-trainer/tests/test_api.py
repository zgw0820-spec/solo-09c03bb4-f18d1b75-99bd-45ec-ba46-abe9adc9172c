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
            "pending", "training", "validating", "returned",
            "judged", "ignored", "cancelled",
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
