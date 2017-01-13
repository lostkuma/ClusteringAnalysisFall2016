#!usr/bin/env python3 #_select_features.py

import csv

from math import ceil

from init import FilePath


def OpenCsvFile(filename):
	# load nouns list csv file
	metadata_file = FilePath("output", filename)

	nouns = dict()
	with open(metadata_file, newline="\r\n") as csvfile:
		metadata = csv.reader(csvfile, delimiter=",")
		for row in metadata:
			nouns[row[0]] = float(row[1])

	return nouns


def CutoffPercent(sorted_list, x):
	""" get top x percentage words/pair
			output a list of words
	"""
	selected = list()
	for i in range(0, ceil(len(sorted_list) * x)):
		selected.append(sorted_list[i][0])

	return selected


def SelectFeatures(bow_nouns, tfidf_nouns):
	""" select features to be used in clustering
			compare tfidf and bow result with top x%
			if both gave the same feature with top x% then select that x as cutoff point
	"""
	# load amod pair csv file
	metadata_file = FilePath("output", "amod.csv")
	amod_pairs = dict()
	with open(metadata_file, newline="") as csvfile:
		metadata = csv.reader(csvfile, delimiter=",")
		for row in metadata:
			if row[2].isdigit():
				pair = (row[0], row[1])
				amod_pairs[pair] = int(row[2])

	# sort decsendingly based on freq
	sorted_bow = sorted(bow_nouns.items(), key=lambda x: -x[1])
	sorted_tfidf = sorted(tfidf_nouns.items(), key=lambda x: -x[1])
	sorted_amod = sorted(amod_pairs.items(), key=lambda x: -x[1])

	# take first x% most frequent bow/tfidf/amod noun
	selected_bow_noun = CutoffPercent(sorted_bow, 0.02)
	selected_tfidf_noun = CutoffPercent(sorted_tfidf, 0.02)
	selected_amod_pair = CutoffPercent(sorted_amod, 0.02)

	# take out nouns selected in the most freq amod pairs
	selected_amod_noun = list()
	for pair in selected_amod_pair:
		if pair[1] not in selected_amod_noun:
			selected_amod_noun.append(pair[1])
	
	# take frequent bow/tfidf nouns that are also tagged as nouns 
	# in most frequent amod pairs as features
	features0 = list()
	for word in selected_bow_noun:
		if word in selected_amod_noun:
			features0.append(word)

	features1 = list()
	for word in selected_tfidf_noun:
		if word in selected_amod_noun:
			features1.append(word)

	num_features0 = len(features0)
	num_features1 = len(features1)

	print("selected bow features:")
	print(features0)
	print("count: " + str(num_features0))
	print("selected tfidf featurs:")
	print(features1)
	print("cnount: " + str(num_features1))

	if any(word not in features1 for word in features0) or any(word not in features0 for word in features1):
		print("they are different\n")
	else:
		print("they are the same\n")

	output_file = FilePath("output", "features.txt")
	with open(output_file, "w", encoding="utf-8") as textfile:
		for word in features0:
			textfile.write(word + "\n")


def main():
	bow_nouns = OpenCsvFile("bow_noun.csv")
	tfidf_nouns = OpenCsvFile("tfidf_noun.csv")
	SelectFeatures(bow_nouns, tfidf_nouns)

if __name__ == '__main__':
	main()

