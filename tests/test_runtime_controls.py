import unittest

from runtime_controls import PARAM_SPECS, RuntimeControls


class RuntimeControlsTests(unittest.TestCase):
    def test_adjust_snaps_to_step_grid(self):
        controls = RuntimeControls()
        exaggeration = PARAM_SPECS[0]

        value = controls.adjust(exaggeration, 3)

        self.assertEqual(value, 0.65)
        self.assertEqual(controls.exaggeration, 0.65)

    def test_adjust_clamps_to_bounds(self):
        controls = RuntimeControls()
        volume = next(spec for spec in PARAM_SPECS if spec.attr == "volume")

        self.assertEqual(controls.adjust(volume, 1_000), volume.hi)
        self.assertEqual(controls.adjust(volume, -1_000), volume.lo)

    def test_snapshot_is_detached(self):
        controls = RuntimeControls(speed=1.1)

        snapshot = controls.snapshot()
        snapshot.speed = 2.0

        self.assertEqual(controls.speed, 1.1)
        self.assertEqual(snapshot.speed, 2.0)


if __name__ == "__main__":
    unittest.main()
