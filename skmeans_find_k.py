#!usr/bin/env python3 #_skmeans_find_k.py

import preprocess as pre
import text_extraction 
import select_features 
from vectorize import Vectorization
import spherical_kmeans as skmeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def FindK(mode):
	# vectorize reviews
	vec_result = Vectorization(mode)
	vectorized = vec_result[0]
	num_features = vec_result[1]

	plotdata = np.array([[0,0]])
	for k in range(2,70):
		# run spherical k-means 
		cluster_result = skmeans.SphericalKMeans(vectorized, num_features, k)
		cohesion = 0
		for t in range(k):
			for y in range(len(cluster_result[0][t])):
				cohesion += cosine_similarity([cluster_result[0][t][y][0]],[cluster_result[1][t]])
		plotdata = np.append(plotdata,[[k,cohesion]],axis=0)
	
	plt.plot(plotdata[1:,0],plotdata[1:,1])
	plt.xticks(plotdata[1:,0])
	plt.xlabel('k')
	plt.ylabel('cohesion')
	plt.show()


def main():

	# preprocess
	pre.JsonFormat()
	pre.SelectTags()
	pre.GetAmodPairs()

	# text extraction
	text_extraction.main()

	# select features 
	select_features.main()

	# plot for bow based features
#	FindK(0)

	# plot for tfidf based features
	FindK(1)
	

if __name__ == '__main__':
	main()