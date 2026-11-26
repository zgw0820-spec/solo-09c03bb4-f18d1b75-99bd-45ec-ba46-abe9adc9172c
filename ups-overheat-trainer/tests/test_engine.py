"""训练引擎规则测试：乱序返回、并发上限、早停、并列取最早、忽略规则。"""

import asyncio
import unittest
from unittest import mock

from app import engine, model


def make_session(max_epochs=8, val_delay=0.0, jitter=False, lr=1.5, seed=7):
    cfg = engine.TrainConfig(
        max_epochs=max_epochs, val_delay=val_delay, jitter=jitter, lr=lr, seed=seed
    )
    return engine.new_session(cfg, model.build_dataset(seed=seed))


def async_run(coro):
    return asyncio.run(coro)


class EngineRuleTests(unittest.TestCase):
    def test_out_of_order_return_but_judged_in_epoch_order(self):
        """乱序返回：后轮先返回时只能停在 returned，判定严格按轮号。"""
        session = make_session(max_epochs=6)
        # 轮号越大延迟越短 -> 后轮先返回
        delays = {1: 0.18, 2: 0.15, 3: 0.12, 4: 0.09, 5: 0.06, 6: 0.03}
        observed_out_of_order = False

        orig_validate_bce = model.bce_loss

        async def main():
            nonlocal observed_out_of_order
            task = asyncio.create_task(
                engine.run_session(session, delay_fn=lambda e: delays[e])
            )
            # 在运行过程中观察是否出现“后轮 returned 而前轮仍未 judged”
            while not task.done():
                await asyncio.sleep(0.005)
                returned = {
                    r.epoch for r in session.records if r.state == engine.RETURNED
                }
                judged = {
                    r.epoch for r in session.records if r.state == engine.JUDGED
                }
                for ep in returned:
                    earlier = set(range(1, ep))
                    if earlier - judged:
                        observed_out_of_order = True
            await task

        async_run(main())
        self.assertTrue(
            observed_out_of_order, "应能观察到验证结果乱序返回（returned 先于前轮判定）"
        )
        # 所有进入判定的轮次，judged_at 必须按轮号单调
        judged_records = [r for r in session.records if r.state == engine.JUDGED]
        times = [r.judged_at for r in judged_records]
        self.assertEqual(times, sorted(times))
        self.assertEqual([r.epoch for r in judged_records], list(range(1, len(judged_records) + 1)))

    def test_in_flight_never_exceeds_three(self):
        session = make_session(max_epochs=10, val_delay=0.05, jitter=True)
        async_run(engine.run_session(session))
        self.assertLessEqual(session.peak_in_flight, engine.MAX_IN_FLIGHT)
        self.assertEqual(session.peak_in_flight, engine.MAX_IN_FLIGHT)

    def test_early_stop_with_scripted_losses(self):
        """脚本化验证损失：0.9,0.8,0.85,0.87 -> 第4轮后连续2轮未刷新，早停。"""
        session = make_session(max_epochs=8)
        scripted = {1: 0.9, 2: 0.8, 3: 0.85, 4: 0.87, 5: 0.5, 6: 0.4}
        ds = session.dataset
        call_log = {"n": 0}

        def fake_val_loss(w, X, y):
            # train_step 已被 mock，这里只有验证集调用；取消前可能多调，给 inf
            if X is ds.X_val:
                call_log["n"] += 1
                return scripted.get(call_log["n"], float("inf"))
            return 0.7

        def fake_train_step(w, X, y, lr):
            return w.copy(), 0.7

        with mock.patch.object(model, "bce_loss", side_effect=fake_val_loss), \
             mock.patch.object(model, "train_step", side_effect=fake_train_step):
            async_run(engine.run_session(session, delay_fn=lambda e: 0.01))

        self.assertEqual(session.status, "early_stopped")
        self.assertEqual(session.best_epoch, 2)
        self.assertAlmostEqual(session.best_val_loss, 0.8)
        judged = [r.epoch for r in session.records if r.state == engine.JUDGED]
        self.assertEqual(judged, [1, 2, 3, 4])
        # 触发轮（4）之后的结果必须全部忽略，即使 5/6 的脚本损失更低
        later = [r for r in session.records if r.epoch > 4]
        self.assertTrue(later)
        self.assertTrue(all(r.state in (engine.IGNORED, engine.CANCELLED) for r in later))
        self.assertTrue(all("忽略" in r.note or "未发起" in r.note for r in later))

    def test_tie_keeps_earliest_epoch(self):
        """最低损失并列时取最早轮：0.5,0.5,0.6,0.7 -> 最佳为第1轮。"""
        session = make_session(max_epochs=6)
        ds = session.dataset
        scripted = iter([0.5, 0.5, 0.6, 0.7, 0.4, 0.3])

        def fake_val_loss(w, X, y):
            if X is ds.X_val:
                return next(scripted)
            return 0.7

        with mock.patch.object(model, "bce_loss", side_effect=fake_val_loss), \
             mock.patch.object(
                 model, "train_step",
                 side_effect=lambda w, X, y, lr: (w.copy(), 0.7),
             ):
            async_run(engine.run_session(session, delay_fn=lambda e: 0.01))

        self.assertEqual(session.status, "early_stopped")
        self.assertEqual(session.best_epoch, 1)
        self.assertAlmostEqual(session.best_val_loss, 0.5)

    def test_completed_when_loss_keeps_falling(self):
        """损失一直刷新 -> 跑满最后一轮，正常 completed。"""
        session = make_session(max_epochs=5)
        ds = session.dataset
        scripted = iter([0.9, 0.8, 0.7, 0.6, 0.5])

        def fake_val_loss(w, X, y):
            if X is ds.X_val:
                return next(scripted)
            return 0.7

        with mock.patch.object(model, "bce_loss", side_effect=fake_val_loss), \
             mock.patch.object(
                 model, "train_step",
                 side_effect=lambda w, X, y, lr: (w.copy(), 0.7),
             ):
            async_run(engine.run_session(session, delay_fn=lambda e: 0.01))

        self.assertEqual(session.status, "completed")
        self.assertEqual(session.best_epoch, 5)
        self.assertTrue(
            all(r.state == engine.JUDGED for r in session.records)
        )

    def test_unjudged_window_structurally_bounded(self):
        """信号量按判定释放：任意时刻“已发起未判定”轮次数恒 <= 3（含 returned）。"""

        async def scenario(delay, jitter, max_epochs):
            session = make_session(
                max_epochs=max_epochs, val_delay=delay, jitter=jitter
            )
            task = asyncio.create_task(engine.run_session(session))
            max_window = 0
            observed_returned = False
            while not task.done():
                await asyncio.sleep(0.002)
                window = [
                    r.epoch
                    for r in session.records
                    if r.state in (
                        engine.TRAINING, engine.VALIDATING, engine.RETURNED
                    )
                ]
                max_window = max(max_window, len(window))
                if any(r.state == engine.RETURNED for r in session.records):
                    observed_returned = True
                self.assertLessEqual(len(window), engine.MAX_IN_FLIGHT)
            await task
            return session, max_window, observed_returned

        # 高延迟让轮次大量停留在 validating/returned，压力最大
        session, max_window, observed_returned = async_run(
            scenario(delay=0.5, jitter=False, max_epochs=12)
        )
        self.assertEqual(max_window, engine.MAX_IN_FLIGHT)
        self.assertIn(session.status, ("early_stopped", "completed"))
        # 抖动场景同样成立
        session2, max_window2, observed_returned2 = async_run(
            scenario(delay=0.3, jitter=True, max_epochs=10)
        )
        self.assertLessEqual(max_window2, engine.MAX_IN_FLIGHT)
        self.assertTrue(observed_returned or observed_returned2)
        # 结束后所有空位必须归还，无信号量泄漏
        self.assertEqual(session._semaphore._value, engine.MAX_IN_FLIGHT)
        self.assertEqual(session2._semaphore._value, engine.MAX_IN_FLIGHT)

    def test_fuzz_many_configs_never_breaks_invariants(self):
        """随机配置模糊测试：无挂起、无信号量泄漏、并发恒 <=3、状态机自洽。"""

        async def run_one(seed):
            rng = __import__("random").Random(seed)
            session = make_session(
                max_epochs=rng.randint(1, 25),
                val_delay=rng.choice([0.0, 0.01, 0.05, 0.2]),
                jitter=rng.choice([True, False]),
                lr=rng.choice([0.3, 1.1, 1.5, 3.0, 8.0, 20.0]),
                seed=seed,
            )
            task = asyncio.create_task(engine.run_session(session))
            while not task.done():
                await asyncio.sleep(0.002)
                window = sum(
                    r.state in (
                        engine.TRAINING, engine.VALIDATING, engine.RETURNED
                    )
                    for r in session.records
                )
                assert window <= engine.MAX_IN_FLIGHT
            await task
            assert session._semaphore._value == engine.MAX_IN_FLIGHT
            assert session.status in (
                "early_stopped", "completed"
            ), session.status
            # 终态只能是 judged / ignored / cancelled
            assert all(
                r.state in (
                    engine.JUDGED, engine.IGNORED, engine.CANCELLED
                )
                for r in session.records
            )
            if session.best_epoch is not None:
                judged = [r.epoch for r in session.records if r.state == engine.JUDGED]
                assert session.best_epoch in judged
                # 最佳轮确实是判定范围内损失最低（并列最早）
                judged_losses = {
                    r.epoch: r.val_loss
                    for r in session.records
                    if r.state == engine.JUDGED
                }
                earliest_min = min(judged_losses, key=lambda e: (judged_losses[e], e))
                assert earliest_min == session.best_epoch

        for seed in range(30):
            async_run(run_one(seed))

    def test_real_dataset_early_stop_demo_and_weights(self):
        """内置 UPS 样本 + 大学习率：应实际触发早停，且最佳权重可用、准确率合理。"""
        session = make_session(max_epochs=20, val_delay=0.0, jitter=False, lr=3.0)
        async_run(engine.run_session(session))
        self.assertEqual(session.status, "early_stopped")
        self.assertIsNotNone(session.best_weights)
        self.assertEqual(session.best_weights.shape, (7,))
        acc = model.accuracy(
            session.best_weights, session.dataset.X_val, session.dataset.y_val
        )
        self.assertGreater(acc, 0.8)
        # 最佳轮属于判定范围
        judged_epochs = [r.epoch for r in session.records if r.state == engine.JUDGED]
        self.assertIn(session.best_epoch, judged_epochs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
