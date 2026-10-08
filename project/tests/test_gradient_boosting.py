import unittest

import numpy as np

from project.src.models import run_gradient_boosting_experiment


class TestGradientBoosting(unittest.TestCase):
    def test_gradient_boosting_extension_runs(self):
        rng = np.random.RandomState(0)
        y = np.repeat([1, 2, 3, 4, 5], 8)
        X = np.column_stack([y / 5.0 + rng.normal(0, 0.05, len(y)), rng.randint(0, 2, (len(y), 3))])

        row = run_gradient_boosting_experiment(X, y)

        self.assertEqual(row["model"], "gradient_boosting")
        for key in ["test_precision", "test_recall", "test_f1", "class_1_test_precision"]:
            self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
