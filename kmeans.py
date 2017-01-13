#!usr/bin/env python3 #_kmeans.py

from init import FilePath
import preprocess as pre
import text_extraction 
import select_features 
from vectorize import Vectorization
from sklearn.cluster import KMeans
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics.pairwise import cosine_similarity
import os 


def Kmeans(mode):
	# preprocess
#	pre.JsonFormat()
#	pre.SelectTags()
#	pre.GetAmodPairs()

	# text extraction
#	text_extraction.main()

	# select features 
#	select_features.main()

	# vectorize reviews
	vec_result = Vectorization(mode)
	vectorized = vec_result[0]
	num_features = vec_result[1]

	k = 35
	
	kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(vectorized))
	
	labels = kmeans.labels_
	
	for i in range(k):
		labels = np.append(labels,-999)
	
	input = np.append(np.array(vectorized),kmeans.cluster_centers_,axis=0)
	mds = manifold.MDS(n_components=2, n_init=1, random_state=40,dissimilarity="euclidean")
	pos = mds.fit(input).embedding_


	x_cor = pos[:,0]
	y_cor = pos[:,1]
	
	for i in range(k):
		plt.plot(x_cor[labels == i],y_cor[labels == i],'.',color = cm.spectral(1.*(i+1)/(k+1)))
	for i in range(k):
		plt.plot(x_cor[-1*k+i],y_cor[-1*k+i],'D',color = cm.spectral(1.*(i+1)/(k+1)))

	#plt.show()
	if not os.path.exists('kmeans_tfidf'):
		os.makedirs('kmeans_tfidf')
	plt.savefig('kmeans_tfidf/kmeans_tfidf_total.png')
	
	file = open('kmeans_tfidf/cluster_information.txt','w')
	input = FilePath("output", "refined_features.txt")
	feature = []
	with open(input, "r", encoding="utf-8") as textfile:
		temp = textfile.readline()
		while temp:
			feature.append(temp.split()[0])
			temp = textfile.readline()
			
	for i in range(k):
		plt.clf()
		plt.plot(x_cor[labels == i],y_cor[labels == i],'.',color = cm.spectral(1.*(i+1)/(k+1)))
		plt.plot(x_cor[-1*k+i],y_cor[-1*k+i],'D',color = cm.spectral(1.*(i+1)/(k+1)))
		plt.savefig('kmeans_tfidf/cluster_'+str(i+1)+'.png')
		
		#cluster_result[1][i]
		index = sorted(range(len(kmeans.cluster_centers_[i])), key=lambda j: kmeans.cluster_centers_[i][j],reverse=True)
		temp = [feature[m] for m in index]
		file.write('cluster '+str(i+1)+' : ')
		for item in temp:
			file.write("%s " % item)
		file.write('\n')


def main():

	# using tfidf based vectors
#	Kmeans(1)

	# using bow based vectors
	Kmeans(0)

if __name__ == '__main__':
	main()