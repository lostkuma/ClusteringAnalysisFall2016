#!usr/bin/env python3 #_vectorize.py

import json

from init import FilePath
import numpy as np

def Vectorization(mode):
	""" vectorize all reviews based on features 
		load processed_reviews.json.utf8.txt file, and features.txt file
		return a dict with key=index, value=vectorized review

		manually designed network for extracted 53 features
		features with similar meanings
			"staff", "person"
			"price", "value", "rate"
		features with hierarchy that can join not too important lower features
			"distance": "block"
			"bathroom": "shower"
			"area": "access"
			"size": "standard"
			"stay": experience"
	"""
	print("mode using: " + str(mode) + "\n0 stand for bow\n1 stand for tfidf\n")
	features_map = {
			"person": "staff", "value": "price", "rate": "price",
			"block": "distance", "shower": "bathroom", "access": "area", 
			"standard": "size", "experience": "stay"
			}

	del_features = ["person", "value", "rate", "block", "shower",
			"access", "standard", "experience"]


	# load processed reviews
	processed_reviews = dict()
	metadata_file = FilePath("data", "processed_reviews.json.utf8.txt")
	with open(metadata_file, "r", encoding="utf-8") as textfile:
		data = json.load(textfile)
		for metadata_line in data:
			key = metadata_line["id"]
			value = metadata_line["review"]
			processed_reviews[key] = value

	# load features
	features = list()
	file_path = FilePath("output", "features.txt")
	with open(file_path, "r", encoding="utf-8") as textfile:
		for row in textfile:
			row = row.strip("\n")
			features.append(row)

	# take out features to be joint with others		
	kept_features = list(set(features) - set(del_features))

	# output a text file with refined features
	output = FilePath("output", "refined_features.txt")
	with open(output, "w", encoding="utf-8") as textfile:
		for word in kept_features:
			textfile.write(word + "\n")

	# get a bow for each review
	bow_each_review = dict()
	for index, review in processed_reviews.items():
		bow = dict()
		for word in review:
			if word not in bow.keys():
				bow[word] = 0
			bow[word] += 1
		bow_each_review[index] = bow

	num_features = len(kept_features)

	vectors_list = list()

	# get idf for each words
	idf = np.zeros(num_features)
	for i in range(num_features):
		for j in range(len(data)):
			if kept_features[i] in data[j]["review"]:
				idf[i]+=1
			else:
				if kept_features[i] in features_map.values():
					for rr in features_map.items():
						if rr[1] == kept_features[i] and rr[0] in data[j]["review"]:
							idf[i]+=1
							break
		idf[i] = np.log10(len(data)/idf[i])
	
	
	# keep track of vectors by adding both index and vector to dict
	for index, bow in bow_each_review.items():

		review_vector = [0] * num_features

		for i in range(0, num_features):
			if kept_features[i] in bow.keys():
				review_vector[i] = bow[kept_features[i]]

		for removed in del_features:
			if removed in bow.keys():
				mapped_feature = features_map[removed]
				index = kept_features.index(mapped_feature)
				review_vector[index] += bow[removed]
		
		if mode != 0:
			review_vector *= idf
			
		vectors_list.append(review_vector)

#	print("vectors are:\n")
#	print(vectors_list)
#	print("\n")
	print("number of refined features: ")
	print(num_features)
	print("\n")
	print("features joint other features that were removed: ")
	print(del_features)
	print("\n")

	return vectors_list, num_features


if __name__ == '__main__':
	Vectorization()


