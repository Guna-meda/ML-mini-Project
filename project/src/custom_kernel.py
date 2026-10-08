"""Custom mixed-feature kernel reproduced from the reference notebook."""

import math

import numpy as np


def category_probabilities(X_ordinal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    category_count = int(np.max(X_ordinal[:, 1])) + 1
    material_count = int(np.max(X_ordinal[:, 2])) + 1
    version_count = int(np.max(X_ordinal[:, 3])) + 1
    category_prob = np.zeros(category_count, dtype=float)
    material_prob = np.zeros(material_count, dtype=float)
    version_prob = np.zeros(version_count, dtype=float)
    for i in range(category_count):
        category_prob[i] = np.sum(X_ordinal[:, 1] == i) / X_ordinal.shape[0]
    for i in range(material_count):
        material_prob[i] = np.sum(X_ordinal[:, 2] == i) / X_ordinal.shape[0]
    for i in range(version_count):
        version_prob[i] = np.sum(X_ordinal[:, 3] == i) / X_ordinal.shape[0]
    return category_prob, material_prob, version_prob


def mixed_feature_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    category_prob: np.ndarray,
    material_prob: np.ndarray,
    version_prob: np.ndarray,
) -> np.ndarray:
    theta = 0.2
    alpha = 0.5
    feature_weights = [2, 1, 1, 3]
    k_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1_search_vol, x1_category, x1_material, x1_version = x1
            x2_search_vol, x2_category, x2_material, x2_version = x2
            search_similarity = math.exp(
                -((x1_search_vol - x2_search_vol) * (x1_search_vol - x2_search_vol))
                / (2 * theta * theta)
            )
            category_similarity = (
                0
                if x1_category != x2_category
                else math.pow(1 - math.pow(float(category_prob[int(x1_category)]), alpha), 1 / alpha)
            )
            material_similarity = (
                0
                if x1_material != x2_material
                else math.pow(1 - math.pow(float(material_prob[int(x1_material)]), alpha), 1 / alpha)
            )
            version_similarity = (
                0
                if x1_version != x2_version
                else math.pow(1 - math.pow(float(version_prob[int(x1_version)]), alpha), 1 / alpha)
            )
            k_matrix[i, j] = np.dot(
                [search_similarity, category_similarity, material_similarity, version_similarity],
                feature_weights,
            )
    return k_matrix
