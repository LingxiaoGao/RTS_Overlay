import unittest

from aoe4.smart_advisor import (
    Advice,
    AdvicePriority,
    AdvisorConfig,
    AdvisorEngine,
    EconomyTarget,
)
from aoe4.smart_runtime import FieldState, GameState, RecognitionStatus


def confirmed(value, status=RecognitionStatus.CONFIRMED):
    return FieldState(value=value, status=status, confidence=1.0)


def engine(**kwargs):
    return AdvisorEngine(AdvisorConfig(**kwargs))


class TargetParsingTests(unittest.TestCase):
    def test_from_build_order_step_reads_villager_resources(self):
        step = {
            "resources": {
                "food": 10,
                "wood": 4,
                "gold": 3,
                "stone": -1,
            }
        }
        target = EconomyTarget.from_build_order_step(step)
        self.assertEqual(
            dict(target.villagers), {"food": 10, "wood": 4, "gold": 3}
        )
        self.assertEqual(dict(target.stockpiles), {})

    def test_from_build_order_step_missing_resources_is_empty(self):
        target = EconomyTarget.from_build_order_step({})
        self.assertEqual(dict(target.villagers), {})
        self.assertEqual(dict(target.stockpiles), {})

    def test_from_build_order_step_reads_smart_resource_targets(self):
        step = {
            "smart_resource_targets": {"food": 500, "wood": 800},
        }
        target = EconomyTarget.from_build_order_step(step)
        self.assertEqual(
            dict(target.stockpiles), {"food": 500, "wood": 800}
        )
        self.assertEqual(dict(target.villagers), {})

    def test_from_build_order_step_rejects_non_mapping(self):
        with self.assertRaises(TypeError):
            EconomyTarget.from_build_order_step([("food", 1)])

    def test_from_build_order_step_rejects_unknown_smart_key(self):
        step = {"smart_resource_targets": {"iron": 100}}
        with self.assertRaises(ValueError):
            EconomyTarget.from_build_order_step(step)

    def test_from_build_order_step_rejects_bool_values(self):
        step = {"resources": {"food": True}}
        with self.assertRaises(TypeError):
            EconomyTarget.from_build_order_step(step)

    def test_from_build_order_step_rejects_negative_stockpile(self):
        step = {"smart_resource_targets": {"food": -5}}
        with self.assertRaises(ValueError):
            EconomyTarget.from_build_order_step(step)


class ConfigValidationTests(unittest.TestCase):
    def test_rejects_bool_threshold(self):
        with self.assertRaises(TypeError):
            AdvisorConfig(villager_difference_threshold=True)

    def test_rejects_non_positive_villager_threshold(self):
        with self.assertRaises(ValueError):
            AdvisorConfig(villager_difference_threshold=0)

    def test_rejects_negative_population_headroom(self):
        with self.assertRaises(ValueError):
            AdvisorConfig(population_headroom=-1)

    def test_rejects_non_int_stockpile_tolerance(self):
        with self.assertRaises(TypeError):
            AdvisorConfig(stockpile_tolerance=1.5)


class ConfirmedOnlyTests(unittest.TestCase):
    def test_ignores_non_confirmed_fields(self):
        state = GameState(
            idle_villagers=confirmed(3, status=RecognitionStatus.INFERRED),
            population_cap=confirmed(50, status=RecognitionStatus.STALE),
            population_current=confirmed(49),
            villagers_food=FieldState(
                value=20, status=RecognitionStatus.UNKNOWN
            ),
        )
        target = EconomyTarget(villagers={"food": 5}, stockpiles={})
        self.assertEqual(engine().evaluate(state, target), ())


class IdleAndPopulationTests(unittest.TestCase):
    def test_idle_villagers_high_priority(self):
        state = GameState(idle_villagers=confirmed(2))
        advice = engine().evaluate(state, EconomyTarget({}, {}))
        self.assertEqual(len(advice), 1)
        self.assertEqual(advice[0].key, "idle_villagers")
        self.assertIs(advice[0].priority, AdvicePriority.HIGH)
        self.assertIsInstance(advice[0], Advice)

    def test_no_idle_advice_when_zero(self):
        state = GameState(idle_villagers=confirmed(0))
        self.assertEqual(engine().evaluate(state, EconomyTarget({}, {})), ())

    def test_population_cap_near(self):
        state = GameState(
            population_current=confirmed(48),
            population_cap=confirmed(50),
        )
        advice = engine().evaluate(state, EconomyTarget({}, {}))
        self.assertEqual(len(advice), 1)
        self.assertEqual(advice[0].key, "population_cap")
        self.assertIs(advice[0].priority, AdvicePriority.HIGH)

    def test_population_cap_no_advice_with_headroom(self):
        state = GameState(
            population_current=confirmed(40),
            population_cap=confirmed(50),
        )
        self.assertEqual(engine().evaluate(state, EconomyTarget({}, {})), ())

    def test_population_cap_not_below_current(self):
        state = GameState(
            population_current=confirmed(60),
            population_cap=confirmed(50),
        )
        self.assertEqual(engine().evaluate(state, EconomyTarget({}, {})), ())


class VillagerTests(unittest.TestCase):
    def test_add_villagers_at_threshold(self):
        state = GameState(villagers_food=confirmed(3))
        target = EconomyTarget(villagers={"food": 5}, stockpiles={})
        advice = engine().evaluate(state, target)
        self.assertEqual(len(advice), 1)
        self.assertEqual(advice[0].key, "villagers_food")
        self.assertIs(advice[0].priority, AdvicePriority.MEDIUM)
        self.assertIn("2", advice[0].message)

    def test_remove_villagers_at_threshold(self):
        state = GameState(villagers_wood=confirmed(8))
        target = EconomyTarget(villagers={"wood": 6}, stockpiles={})
        advice = engine().evaluate(state, target)
        self.assertEqual(len(advice), 1)
        self.assertEqual(advice[0].key, "villagers_wood")
        self.assertIn("2", advice[0].message)

    def test_below_threshold_no_advice(self):
        state = GameState(villagers_gold=confirmed(5))
        target = EconomyTarget(villagers={"gold": 6}, stockpiles={})
        self.assertEqual(engine().evaluate(state, target), ())


class StockpileTests(unittest.TestCase):
    def test_stockpile_low_medium(self):
        state = GameState(food=confirmed(100))
        target = EconomyTarget(villagers={}, stockpiles={"food": 500})
        advice = engine().evaluate(state, target)
        self.assertEqual(len(advice), 1)
        self.assertEqual(advice[0].key, "stockpile_food_low")
        self.assertIs(advice[0].priority, AdvicePriority.MEDIUM)

    def test_stockpile_high_low_priority(self):
        state = GameState(stone=confirmed(1000))
        target = EconomyTarget(villagers={}, stockpiles={"stone": 500})
        advice = engine().evaluate(state, target)
        self.assertEqual(len(advice), 1)
        self.assertEqual(advice[0].key, "stockpile_stone_high")
        self.assertIs(advice[0].priority, AdvicePriority.LOW)

    def test_stockpile_within_tolerance(self):
        state = GameState(gold=confirmed(475))
        target = EconomyTarget(villagers={}, stockpiles={"gold": 500})
        self.assertEqual(engine().evaluate(state, target), ())

    def test_stockpile_exact_tolerance_edge(self):
        state = GameState(gold=confirmed(450))
        target = EconomyTarget(villagers={}, stockpiles={"gold": 500})
        self.assertEqual(engine().evaluate(state, target), ())


class OrderingAndImmutabilityTests(unittest.TestCase):
    def test_deterministic_order(self):
        state = GameState(
            idle_villagers=confirmed(1),
            population_current=confirmed(49),
            population_cap=confirmed(50),
            villagers_food=confirmed(0),
            villagers_wood=confirmed(0),
            villagers_gold=confirmed(0),
            villagers_stone=confirmed(0),
            food=confirmed(0),
            wood=confirmed(0),
            gold=confirmed(0),
            stone=confirmed(0),
        )
        target = EconomyTarget(
            villagers={"food": 5, "wood": 5, "gold": 5, "stone": 5},
            stockpiles={
                "food": 500,
                "wood": 500,
                "gold": 500,
                "stone": 500,
            },
        )
        keys = [a.key for a in engine().evaluate(state, target)]
        self.assertEqual(
            keys,
            [
                "idle_villagers",
                "population_cap",
                "villagers_food",
                "villagers_wood",
                "villagers_gold",
                "villagers_stone",
                "stockpile_food_low",
                "stockpile_wood_low",
                "stockpile_gold_low",
                "stockpile_stone_low",
            ],
        )

    def test_input_immutability(self):
        villagers = {"food": 5}
        stockpiles = {"food": 100}
        target = EconomyTarget(villagers=villagers, stockpiles=stockpiles)
        villagers["food"] = 99
        stockpiles["food"] = 99
        self.assertEqual(dict(target.villagers), {"food": 5})
        self.assertEqual(dict(target.stockpiles), {"food": 100})

        state = GameState(idle_villagers=confirmed(1))
        before = (state.idle_villagers.value, state.idle_villagers.status)
        engine().evaluate(state, target)
        self.assertEqual(
            (state.idle_villagers.value, state.idle_villagers.status), before
        )

    def test_mapping_read_only(self):
        target = EconomyTarget(villagers={"food": 5}, stockpiles={})
        with self.assertRaises(TypeError):
            target.villagers["food"] = 10  # type: ignore[index]

    def test_target_rejects_unknown_resource(self):
        with self.assertRaises(ValueError):
            EconomyTarget(villagers={"iron": 1}, stockpiles={})


if __name__ == "__main__":
    unittest.main()
