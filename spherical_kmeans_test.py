#!usr/bin/env python3 #_spherical_kmeans_test.py

import spherical_kmeans as skmeans
import numpy as np
import numpy.testing as npt
import unittest

class SphericalKMeansTest(unittest.TestCase):

	def test_spherical_k_means(self):
		input_vector = [
				[0.7, 2],
				[0.8, 2],
				[0.9, 2],
				[1, 2],
				[1, 1.9],
				[1, 1.8],
				[1, 1.7],
				[1.7, 1],
				[1.8, 1],
				[1.9, 1],
				[2, 1],
				[2, 0.9],
				[2, 0.8],
				[2, 0.7]
				]
		feature_num = 2
		k = 2

		actual_results = skmeans.SphericalKMeans(input_vector, feature_num, k)
		actual_clusters = actual_results[0]
		actual_centers = actual_results[1]

		actual_centers = np.array(sorted(actual_centers, key=tuple))

		expected_centers = [np.array([1,2]), np.array([2,1])]
		expected_centers_norm = list()
		for center in expected_centers:
			center = center / np.linalg.norm(center)
			expected_centers_norm.append(center)

		npt.assert_almost_equal(actual_centers, expected_centers_norm, decimal=1)


if __name__ == '__main__':
	unittest.main()

