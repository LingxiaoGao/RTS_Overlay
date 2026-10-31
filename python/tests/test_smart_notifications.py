import unittest

from aoe4.smart_advisor import Advice, AdvicePriority
from aoe4.smart_notifications import (
    Notification,
    NotificationConfig,
    NotificationManager,
)


def _advice(key, priority=AdvicePriority.MEDIUM, message="msg"):
    return Advice(key=key, priority=priority, message=message)


class NotificationManagerTest(unittest.TestCase):
    def test_priority_and_stable_order(self):
        manager = NotificationManager(NotificationConfig(max_per_batch=10))
        batch = [
            _advice("low1", AdvicePriority.LOW),
            _advice("high1", AdvicePriority.HIGH),
            _advice("med1", AdvicePriority.MEDIUM),
            _advice("high2", AdvicePriority.HIGH),
            _advice("med2", AdvicePriority.MEDIUM),
        ]
        result = manager.publish(batch, 0)
        self.assertEqual(
            [n.key for n in result],
            ["high1", "high2", "med1", "med2", "low1"],
        )

    def test_duplicate_key_keeps_highest_priority(self):
        manager = NotificationManager(NotificationConfig(max_per_batch=10))
        batch = [
            _advice("dup", AdvicePriority.LOW),
            _advice("dup", AdvicePriority.HIGH),
            _advice("dup", AdvicePriority.MEDIUM),
        ]
        result = manager.publish(batch, 0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].priority, AdvicePriority.HIGH)

    def test_cooldown_before_and_exact_boundary(self):
        manager = NotificationManager(NotificationConfig(cooldown_seconds=10.0))
        first = manager.publish([_advice("k")], 0)
        self.assertEqual(len(first), 1)

        during = manager.publish([_advice("k")], 5)
        self.assertEqual(during, ())

        boundary = manager.publish([_advice("k")], 10)
        self.assertEqual(len(boundary), 1)

    def test_independent_keys(self):
        manager = NotificationManager(NotificationConfig(cooldown_seconds=10.0))
        manager.publish([_advice("a")], 0)
        result = manager.publish([_advice("a"), _advice("b")], 1)
        self.assertEqual([n.key for n in result], ["b"])

    def test_batch_limit_leaves_item_eligible(self):
        manager = NotificationManager(
            NotificationConfig(cooldown_seconds=10.0, max_per_batch=1)
        )
        batch = [
            _advice("a", AdvicePriority.HIGH),
            _advice("b", AdvicePriority.MEDIUM),
        ]
        first = manager.publish(batch, 0)
        self.assertEqual([n.key for n in first], ["a"])

        second = manager.publish([_advice("b")], 1)
        self.assertEqual([n.key for n in second], ["b"])

    def test_reset_clears_cooldown(self):
        manager = NotificationManager(NotificationConfig(cooldown_seconds=10.0))
        manager.publish([_advice("k")], 0)
        self.assertEqual(manager.publish([_advice("k")], 1), ())
        manager.reset()
        self.assertEqual(len(manager.publish([_advice("k")], 1)), 1)

    def test_invalid_clock(self):
        manager = NotificationManager()
        for bad in (True, -1.0, float("inf"), float("nan")):
            with self.assertRaises((TypeError, ValueError)):
                manager.publish([], bad)
        with self.assertRaises(TypeError):
            manager.publish([], "0")  # type: ignore[arg-type]

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            NotificationConfig(cooldown_seconds=0.0)
        with self.assertRaises(ValueError):
            NotificationConfig(cooldown_seconds=float("inf"))
        with self.assertRaises(TypeError):
            NotificationConfig(cooldown_seconds=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            NotificationConfig(max_per_batch=0)
        with self.assertRaises(TypeError):
            NotificationConfig(max_per_batch=True)  # type: ignore[arg-type]

    def test_notification_frozen(self):
        notification = Notification(
            key="k",
            priority=AdvicePriority.HIGH,
            message="m",
            created_at=1.0,
        )
        with self.assertRaises(Exception):
            notification.key = "other"  # type: ignore[misc]

    def test_advice_subclass_accepted(self):
        class CustomAdvice(Advice):
            pass

        manager = NotificationManager()
        result = manager.publish(
            [CustomAdvice(key="sub", priority=AdvicePriority.HIGH, message="m")],
            0,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].key, "sub")

    def test_invalid_item_rejected(self):
        manager = NotificationManager()
        with self.assertRaises(TypeError):
            manager.publish(["not advice"], 0)  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
