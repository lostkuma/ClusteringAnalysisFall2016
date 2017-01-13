#!usr/bin/env python _#spherical_kmeans

import numpy as np


def Normalization(datapoints):
	""" takes in a list of datapoints and normalize to unit vectors
		output a list of np array of unit vector
	"""
	normalized_data = list()
	for data in datapoints:
		if np.linalg.norm(data) != 0:
			normalized = data / np.linalg.norm(data)
		else:
			normalized = np.array(data)
		normalized_data.append(normalized)

	return normalized_data


def InitialCenters(m, k):
	""" takes number of features m, and predicted cluster number k
			generate k np array with m-dimentional unit vectors as initial centers
			output a list of arraies (centers)
	"""
	init_centers = list()
	for i in range(0, k):
		center = np.random.rand(m)
		normalized = Normalization(list([center]))
		init_centers.append(normalized[0])

	return init_centers


def SphericalDistance(array1, array2):
	""" calculate the spherical distances from one datapoint to the other 
			the distance is the defined by cosine of angle between two vecotrs
			consine on unit sphere of m*1 dimentional is define by dot product of two vectors
			return the distance
	"""
	distance = np.dot(array1, array2)

	return distance


def Grouping(datapoints, centers, k):
	""" group the input to argmax of center distances
			input datapoints is list of arraies, centers is list, k is int
			use the SphericalDistance function to calculate distance
			group the data with the center has max cosine distance between all centers
			assign the datapoints to different cluster
			keep a list of empty clusters
			keep a list of furthest point from center of each cluster
			clusters is a dictionary of lists, key=kth-index, value=list of datapoints
			furthest point is a dict, key=kth-index, value=(distance, point)
			empty cluster is a list of index
			return cluster
	""" 
	clusters = dict()
	furthest_points = dict()
	empty_clusters = list()

	for i in range(0, k):
		clusters[i] = list()
		furthest_points[i] = [1, 0] # (distance, point)

	for i in range(0, len(datapoints)):
		distance_list = list()
		for center in centers:
			distance = SphericalDistance(datapoints[i], center)
			distance_list.append(distance)

		index_cluster = np.argmax(np.array(distance_list))

		distance_to_center = distance_list[index_cluster]
		furthest = furthest_points[index_cluster][0]

		# keep track of the furthest point from center in the cluster
		# if current cosine value is smaller than update the furthest
		if furthest > distance_to_center:
			furthest_points[index_cluster][0] = distance_to_center
			furthest_points[index_cluster][1] = datapoints[i]

		clusters[index_cluster].append((datapoints[i], i))

	for i in range(0, k):
		if len(clusters[i]) == 0:
			empty_clusters.append(i)

#	print("empty clusters:\n")
#	print(empty_clusters)
#	print("\n")
#	print("furthest points:\n")
#	print(furthest_points)
#	print("\n")

	return clusters, furthest_points, empty_clusters


def ReCenter(clusters, k, furthest_points, empty_clusters, m):
	""" recalculate the k centers based on grouped data
		if empty cluster exists, assign furthest point from the furthest list to be new center
			points with smallest cosine value is furthest
		otherwise just recalculate centers based on mean
		return new calcualated centers, a list of arraies
	""" 
	new_centers = [0] * k

	if len(empty_clusters) != 0:
		sorted_points = sorted(furthest_points.items(), key=lambda x: x[1][0])
		for i in range(0, len(empty_clusters)):
			new_centers[empty_clusters[i]] = sorted_points[i][1][1]

	for i in range(0, k):
		if type(new_centers[i]) == int:
			sum_cluster = 0
			for vector, index in clusters[i]:
				sum_cluster += vector

			norm_sum = np.linalg.norm(sum_cluster)
			if norm_sum != 0:
				new = sum_cluster / norm_sum
			else:
				new = np.zeros(m)

			new_centers[i] = new

	return new_centers


def WasUpdated(current_centers, new_centers):
	""" compare and adjust the new center
			if current center = new recalculated center then no change
			if current center != new recalculate dcenter then update
			return a list of new updated centers or None
	"""
	if np.any(np.array(current_centers) != np.array(new_centers)):
		return True
	else: 
		return False


def Clustering(datapoints, k, m):
	""" the main process
			1. group datapoints to nearest centers
			2. recalculate centers based on data in group
			3. compare the new centers with current centers
			4. if new centers are different from current centers
						update new centers and do 1 - 3
				if new centers are no different than current centers
				 		return clusters
	"""
	old_centers = None
	new_centers = InitialCenters(m, k)
	
	while WasUpdated(old_centers, new_centers):
		old_centers = new_centers

		group_results = Grouping(datapoints, new_centers, k)
		clusters = group_results[0]
		furthest_points = group_results[1]
		empty_clusters = group_results[2]

		new_centers = ReCenter(clusters, k, furthest_points, empty_clusters, m)

	return clusters, new_centers


def SphericalKMeans(datapoints, feature_num, k):
	""" follows the spherical k-means algorithm
	"""
	normalized_data = Normalization(datapoints)
	cluster_result =  Clustering(normalized_data, k, feature_num)

	return cluster_result


# cluster_result format:
# (clusters, centers)
# where centers is a list of arraies
# and clusters is dictionary
# key = index of cluster
# value = [(data_array, review_index), (data_array, review_index)]
# exp: {0: [(data, review#), (data, #)], 1: [(data, #), (data, #)]}

""" reference of spherical k-means:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.111.8125&rep=rep1&type=pdf
"""