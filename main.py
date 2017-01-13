#!usr/bin/env python3 #_main.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from init import FilePath
from vectorize import Vectorization
import preprocess as pre
import text_extraction 
import select_features 
import spherical_kmeans as skmeans
import skmeans_visualize as v

from kmeans import Kmeans


def Process(mode, dirctory):
	# vectorize reviews
	vec_result = Vectorization(mode) 
	vectorized = vec_result[0]
	num_features = vec_result[1]

	# run skmeans cluster
	k = 35
	cluster_result = skmeans.SphericalKMeans(vectorized, num_features, k)
	clusters = cluster_result[0]
	centers = cluster_result[1]
	
	# run mds
	mds_results = v.Mds(k, num_features, clusters, centers)
	x_cor = mds_results[0]
	y_cor = mds_results[1]
	labels = mds_results[2]
	
	# plot overall skmeans
	v.PlotSKmeans(k, x_cor, y_cor, labels, dirctory)

	# plot within cluster
	v.PlotWithinCluster(k, x_cor, y_cor, labels, dirctory)

	# write weighted centers to csv file
	v.WriteToFile(k, centers, num_features, dirctory)

	# plot the weight centers
#	v.PlotCenters(k, centers, num_features)


def main():

	# preprocess
	# run if there is no dataset.json.uft8.txt file in data folder
#	pre.JsonFormat()

	# run if there is no selected_tags.json.utf8.txt file in data folder
#	pre.SelectTags()

	# run if there is no amod.csv file in output folder
#	pre.GetAmodPairs()

	# text extraction
	# run if there is no bow/tfidf related csv file in output folder
#	text_extraction.main()

	# select features 
	# run if there is no features.txt file in output folder
#	select_features.main()

	# vectorize based on tfidf features, running on skmeans
	Process(1, "skmeans_tfidf")

	# vectorize based on bow features, runnning on skmeans
	Process(0, "skmeans_bow")

	# run kmeans with tfidf vectors
#	Kmeans(1)

	# run kmeans with bow vectors
#	Kmeans(0)


if __name__ == '__main__':
	main()

