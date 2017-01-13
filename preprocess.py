#!usr/bin/env python3 #_preprocess

import json

from inflection import singularize

from init import FilePath


def JsonFormat():
    """ make original dataset a json format output dataset.json.utf8.txt"""
    metadata_text_file = FilePath("data", "final_data_1850.txt")

    metadata = []

    with open(metadata_text_file, "r", encoding="utf-8") as textfile:
        index = 1
        for metadata_text_line in textfile:
            metadata_dictionary = dict()
            metadata_dictionary["id"] = index
            metadata_dictionary["original_text"] = metadata_text_line.strip("\n ")
                        
            metadata.append(metadata_dictionary)
            index = index + 1

    metadata_json_file = FilePath("data", "dataset.json.utf8.txt")
    with open(metadata_json_file, "w", encoding="utf-8") as jsonfile:
        json.dump(metadata, jsonfile, ensure_ascii=False, sort_keys=True, indent=4)


def SelectTags():
    """ import result of Stanford parser, and selected nouns, adj, and amod pairs

    all words have been transformed to lower cases
    all the plurals were dealt with during selecting nouns process
    plurals within amod pairs were also dealt with

    the out put text file has json format: 
        with review id, nouns, adjs, and amod paris for each review

    """
    metadata_results = FilePath("data", "dataset_with_parser_results.json.utf8.txt")

    metadata = list()

    with open(metadata_results, "r", encoding="latin-1") as textfile:

        data = json.load(textfile)

        for dataline in data:
            index = dataline["id"] + 1

            local_noun = list()
            local_adjective = list()
            plurals = list()

            for tagged_word in dataline["tagged_text"]:
                pos = tagged_word["part_of_speech"]
                word = tagged_word["original_text"]
                word = word.lower()

                # select singluar noun and proper noun
                if pos == "NN" or pos == "NNP":
                    if word not in local_noun:
                        local_noun.append(word)
                        continue

                # select plural noun and proper noun
                if pos == "NNS" or pos == "NNPS":
                    if word not in plurals:
                        plurals.append(word)

                        # if a word is plural then singularize
                        word = singularize(word)
                        if word not in local_noun:
                            local_noun.append(word)
                            continue

                # select original, comparative, and superlative adjective
                if pos == "JJ" or pos == "JJR" or pos == "JJS":
                    if word not in local_adjective:
                        local_adjective.append(word)
                        continue

            # select amod pair (adjective modified noun)
            amod_pairs = list()
            for group in dataline["dependencies"]:
                for dep_pair in group:
                    if dep_pair["relation"] == "amod":
                        adj = dep_pair["dependent"]["original_text"]
                        if any(char.isalpha() == False for char in adj):
                            continue
                        adj = adj.lower()
                        noun = dep_pair["governor"]["original_text"]
                        if any(char.isalpha() == False for char in noun):
                            continue
                        noun = noun.lower()
                        if noun in plurals:
                            noun = singularize(noun)
                        amod_pairs.append((noun, adj))

            selected_tags = dict()
            selected_tags["id"] = index
            selected_tags["noun"] = local_noun
            selected_tags["adj"] = local_adjective
            selected_tags["amod"] = amod_pairs
            selected_tags["plural"] = plurals

            metadata.append(selected_tags)

    metadata_json_file = FilePath("data", "selected_tags.json.utf8.txt")
    with open(metadata_json_file, "w", encoding="utf-8") as jsonfile:
        json.dump(metadata, jsonfile, ensure_ascii=False, sort_keys=True, indent=4)


def GetAmodPairs():
    """ get amod pairs with freq
        output amod.csv file with amod pairs and freq
    """
    metadata_text_file = FilePath("data", "selected_tags.json.utf8.txt")

    amod_dict = dict()

    with open(metadata_text_file, "r", encoding="utf-8") as textfile:
        metadata = json.load(textfile)
        for metadata_line in metadata:
            amod_pairs = metadata_line["amod"]
            for pair in amod_pairs:
                tupled_pair = tuple(pair)
                if tupled_pair not in amod_dict.keys():
                    amod_dict[tupled_pair] = 0
                amod_dict[tupled_pair] += 1

    # sort based on aphabetical order of nouns in the pairs
    sorted_list = sorted(amod_dict.items(), key=lambda x: x[0][0])

    output_csv = FilePath("output", "amod.csv")

    with open(output_csv, "w") as csvfile:
        csvfile.write("adj, noun, count\n")
        for pair in sorted_list:
            row = pair[0][1] + "," + pair[0][0] + "," + str(pair[1]) + "\n"
            csvfile.write(row)


def main():
    JsonFormat()
    SelectTags()
    GetAmodPairs()


if __name__ == '__main__':
    main()
