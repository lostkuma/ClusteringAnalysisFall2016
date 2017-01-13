#!usr/bin/env python3 #_kmeans_find_k.py

import preprocess as pre
import text_extraction 
import select_features 
from vectorize import Vectorization
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def FindK(mode):
	# vectorize reviews
	vec_result = Vectorization(mode) 
	vectorized = vec_result[0]
	num_features = vec_result[1]
	
	plotdata = np.array([[0,0]])
	for k in range(2,70):
		# run spherical k-means 
		kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(vectorized))
		sse = 0
		for t in range(k):
			for y in range(len(vectorized)):
				if kmeans.labels_[y] == t:
					sse += sum((vectorized[y]-kmeans.cluster_centers_[t])*(vectorized[y]-kmeans.cluster_centers_[t]))
		plotdata = np.append(plotdata,[[k,sse]],axis=0)
	
	plt.plot(plotdata[1:,0],plotdata[1:,1])
	plt.xticks(plotdata[1:,0])
	plt.xlabel('k')
	plt.ylabel('sse')
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

	# find k using tfidf vectorized features
#	FindK(1)

	# find k useing bow vectorized features
	FindK(0)


if __name__ == '__main__':
	main()