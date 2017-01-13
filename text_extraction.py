#!usr/bin/env python3 #text_extraction.py

import csv
import json
import re

from copy import deepcopy
from inflection import singularize
from math import log

from init import FilePath


def OpenDataset(filename):
	""" import dataset and output a dict, key=id, value=review """
	metadata_file = FilePath("data", filename)

	dict_review = dict()
	with open(metadata_file, "r", encoding="utf-8") as textfile:
		data = json.load(textfile)
		for metadata_line in data:
			key = metadata_line["id"]
			value = metadata_line["original_text"]
			dict_review[key] = value

	return dict_review


def ProcessReview(dict_review, dict_stopwords, plural_list):
	""" process the review dataset and output a dictionary

	get rid of all punctuations, transfer all words to lower case
	split reviews by white spaces into words
	take out stopwords and words with numbers
	deal with plural using inflection library
	output a file with processed reivews
	output a dictionary of processed reviews, key=index, value=list of words in review 
	"""
	processed_reviews = dict()
	metadata = list()

	for index, review in dict_review.items():
		review = re.sub("-|'", "", review)
		review = re.sub("\W|_", " ", review)
		review = re.sub("\s+", " ", review)
		review = review.split()
		
		each_review = list()
		for word in review:
			if word == "" or any(char.isdigit() for char in word):
				continue
			word = word.lower()
			if word[0] in dict_stopwords.keys():
				if word in dict_stopwords[word[0]]:
					continue
			if word in plural_list:
				word = singularize(word)
			each_review.append(word)

		processed_reviews[index] = each_review

		metadata_dictionary = dict()
		metadata_dictionary["id"] = index
		metadata_dictionary["review"] = each_review
		metadata.append(metadata_dictionary)
	
	# output a file with processed_reviews
	metadata_file = FilePath("data", "processed_reviews.json.utf8.txt")
	with open(metadata_file, "w", encoding="utf-8") as jsonfile:
		json.dump(metadata, jsonfile, ensure_ascii=False, sort_keys=True, indent=4)

	return processed_reviews


def GetStopwords(filename):
	""" import stopwords and create a dict, key=initial, value=stopword """
	file_path = FilePath("data", filename)

	dict_stopwords = dict()
	with open(file_path, "r", encoding="utf-8") as textfile:
		for line in textfile:
			word = line.strip("\n")
			if word[0] not in dict_stopwords.keys():
				dict_stopwords[word[0]] = list()
			dict_stopwords[word[0]].append(word)

	return dict_stopwords


def GetTaggedResults(filename):
	""" import from Stanford parser results 
	output a list of list with [noun, adj, plural] 
	"""
	metadata_file = FilePath("data", filename)

	noun_list = list()
	adj_list = list()
	plural_list = list()
	
	with open(metadata_file, "r", encoding="utf-8") as textfile:
		data = json.load(textfile)
		for metadata_line in data:
			local_noun = metadata_line["noun"]
			local_adj = metadata_line["adj"]
			local_plural = metadata_line["plural"]

			noun_list += list(set(local_noun) - set(noun_list))
			adj_list += list(set(local_adj) - set(adj_list))
			plural_list += list(set(local_plural) - set(plural_list))

	tagged_results = [noun_list, adj_list, plural_list]

	return tagged_results


def BagOfWords(processed_reviews):
	""" calculate freq for each word
	output a dict of words, key=term, value=count 
	"""
	bow = dict()
	for listed_review in processed_reviews.values():
		for word in listed_review:
			if word not in bow.keys():
				bow[word] = 0
			bow[word] += 1

	return bow


def Tfidf(bow, processed_reviews):
	""" calculate tfidf value for each word

	tf - term freq = word freq
	idf - inversed document freq = log(times word appeard/total review #)
	tfidf = tf * idf
	return a dictionay, key=term, value=idf value 
	"""
	tfidf = dict()
	total_review = len(processed_reviews)

	# make a deep copy of BoW and set values to be 0 as initial idf
	idf = deepcopy(bow)
	for key in idf.keys():
		idf[key] = 0

	for listed_review in processed_reviews.values():
		terms_list = list()
		for word in listed_review:
			if word not in terms_list:
				terms_list.append(word)
				idf[word] += 1
	for key, value in idf.items():
		idf[key] = round(log(total_review / value), 4)
		tfidf[key] = bow[key] * idf[key]

	return tfidf


def SelectSpecificPos(input_dict, pos_list):
	""" select specific pos out from input dict results,
	output a dict, key=term, value=tfidf 
	"""
	specific_pos = dict()
	for term, value in input_dict.items():
		if term in pos_list:
			specific_pos[term] = value

	return specific_pos


def OutPutFile(filename, input_dict):
	""" output csv file of sorted word, freq with decsending freq 
	"""
	metadata_output = FilePath("output", filename)

	sorted_list = sorted(input_dict.items(), key=lambda x: -x[1])
	
	with open(metadata_output, "w") as csvfile:
		output = csv.writer(csvfile)
		for row in sorted_list:
			output.writerow(row)


def main():
	""" load hotel review dataset and calculate tfidf for terms in reviews"""

	# load json format dataset into a dictionary
	dict_review = OpenDataset("dataset.json.utf8.txt")

	# get a dictionary of stopwords from stopwords txt file
	dict_stopwords = GetStopwords("stopwords.txt")

	# get a dictionary of tagged results (include noun adj, and plural)
	tagged_results = GetTaggedResults("selected_tags.json.utf8.txt")

	# get word type lists
	noun_list = tagged_results[0]
	adj_list = tagged_results[1]
	plural_list = tagged_results[2]

	# clean and transform reviews
	processed_reviews = ProcessReview(dict_review, dict_stopwords, plural_list)

	# apply bag of words algorithm
	bow = BagOfWords(processed_reviews)

	# apply tf-idf algorithm
	tfidf = Tfidf(bow, processed_reviews)

	# select specific pos dict out
	noun_tfidf = SelectSpecificPos(tfidf, noun_list)
	adj_tfidf = SelectSpecificPos(tfidf, adj_list)
	noun_bow = SelectSpecificPos(bow, noun_list)
	adj_bow = SelectSpecificPos(bow, adj_list)

	# combine noun and adj tfidf, bow
	combined_tfidf = {**noun_tfidf, **adj_tfidf}
	combined_bow = {**noun_bow, **adj_bow}

	# output csv files
	OutPutFile("bow_all.csv", bow)
	OutPutFile("tfidf_all.csv", tfidf)
	OutPutFile("bow_noun.csv", noun_bow)
	OutPutFile("tfidf_noun.csv", noun_tfidf)


if __name__=="__main__":
	main()

