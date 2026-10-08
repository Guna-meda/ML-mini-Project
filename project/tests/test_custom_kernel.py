import unittest

import numpy as np

from project.src.custom_kernel import category_probabilities, mixed_feature_kernel


class TestCustomKernel(unittest.TestCase):
    def test_category_probabilities_are_vectorized_and_kernel_runs(self):
        X = np.array(
            [
                [0.1, 0, 0, 0],
                [0.4, 1, 1, 1],
                [0.9, 0, 1, 0],
            ],
            dtype=float,
        )

        category_prob, material_prob, version_prob = category_probabilities(X)

        self.assertEqual(category_prob.shape, (2,))
        self.assertEqual(material_prob.shape, (2,))
        self.assertEqual(version_prob.shape, (2,))

        kernel = mixed_feature_kernel(X, X, category_prob, material_prob, version_prob)

        self.assertEqual(kernel.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
