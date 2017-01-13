#!usr/bin/env python3 #_skmeans_visualize.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from init import FilePath
import spherical_kmeans as skmeans

from sklearn import manifold
from sklearn.metrics.pairwise import cosine_similarity


def Mds(k, num_features, clusters, centers):
	feature_array = np.array([np.zeros(num_features)])
	labels = np.array([])
	
	for i in range(k):
		for j in range(len(clusters[i])):
			feature_array = np.append(feature_array,[clusters[i][j][0]],axis=0)
			labels = np.append(labels,i)
	
	for i in range(k):
		labels = np.append(labels,-999)
	
	input = 1-cosine_similarity(np.append(feature_array[1:],np.array(centers),axis=0))
	mds = manifold.MDS(n_components=2, n_init=1, random_state=40,dissimilarity="precomputed")
	pos = mds.fit(input).embedding_

	x_cor = pos[:,0]
	y_cor = pos[:,1]

	return x_cor, y_cor, labels


def PlotSKmeans(k, x_cor, y_cor, labels, directory):
	for i in range(k):
		plt.plot(x_cor[labels == i],y_cor[labels == i],'.',color = cm.spectral(1.*(i+1)/(k+1)))
	for i in range(k):
		plt.plot(x_cor[-1*k+i],y_cor[-1*k+i],'D',color = cm.spectral(1.*(i+1)/(k+1)))
	#plt.show()

	if not os.path.exists(directory):
		os.makedirs(directory)

	plt_path = FilePath(directory, "total.png")
	plt.savefig(plt_path)


def PlotWithinCluster(k, x_cor, y_cor, labels, directory):
	for i in range(k):
		plt.clf()
		plt.plot(x_cor[labels == i],y_cor[labels == i],'.',color = cm.spectral(1.*(i+1)/(k+1)))
		plt.plot(x_cor[-1*k+i],y_cor[-1*k+i],'D',color = cm.spectral(1.*(i+1)/(k+1)))
		plt_path = FilePath(directory, "cluster_" + str(i+1) + ".png")
		plt.savefig(plt_path)


def WriteToFile(k, centers, num_features, directory):
	""" write the clusters centers to a csv file
		with the format of weights decsendingly
			cluster1:
				feature1, 1st highest weight
				feature2, 2nd highest weight
				...

			cluster2:
				feature1, 1st highest weight
				feature2, 2nd highest weight
				...
	"""
	# read in refined features
	file_path = FilePath("output", "refined_features.txt")
	features = list()
	with open(file_path, "r", encoding="utf-8") as textfile:
		temp = textfile.readline()
		while temp:
			features.append(temp.split()[0])
			temp = textfile.readline()

	# writing cluster centers to csv file
	info_path = FilePath(directory, "cluster_info.csv")

	with open(info_path, "w", encoding="utf-8") as csvfile:

		for i in range(0, k):
			index = sorted(range(num_features), key=lambda j: centers[i][j], reverse=True)
			ordered_features = [features[m] for m in index]
			ordered_weights = [centers[i][m] for m in index]
			
			csvfile.write("cluster " + str(i + 1) + ": \n")
			
			# write ordered feature and ordered weight to csvfile
			for j in range(0, num_features):
				csvfile.write("," + ordered_features[j]+ "," + str(ordered_weights[j]) + "\n")
			csvfile.write("\n")


def PlotCenters(k, centers, num_features):
	""" plot the all the cluster centers to see the features relations
	"""
	# output a plot with x=features, y=weight with all centers in		
	save_path = FilePath("output", "tfidf_centers.png")

	x_axis = np.arange(1, num_features + 1)

	plt.axis([0, num_features + 1, 0.0, 1.0])
	plt.xlabel("features")
	plt.ylabel("weights")
	plt.xticks(x_axis, features, rotation="vertical")

	cmap = plt.cm.get_cmap("hsv")
	for i in range(0, k):
		plt.plot(x_axis, centers[i], "-", color = cmap(i/k))
	plt.savefig(save_path)
#	plt.show()
