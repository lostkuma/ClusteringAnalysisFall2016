# ClusteringAnalysisFall2016
Clustering analysis on TripAdvisor hotel reviews  

For Data file, go to "https://drive.google.com/open?id=0B2B_Q-zgRyWrR0ZOSmd4bE5jSnM"  

All credits reserved to Xiaonan Jing, Pai-ying Hsiao and Chien-yi Hsiang  
Please contect Xiaonan Jing for any problems  

**Work Distribution:** 
	
	Jing:  
		init.py  
		main.py  
		preprocess.py  
		select_features.py  
		spherical_kmeans.py  
		spherical_kmeans_test.py  
		text_extraction.py  
		vectorize.py  
		skmeans_visualize.py --PlotCenters, WriteToFile  
		refine_features_instruction.JPG  
		README.md  
	Hsiao:  
		kmeans_find_k.py  
		kmeans.py  
		skmeans_find_k.py  
		skmeans_visualize.py --Mds, PlotSKmeans, PlotWithCluster, WriteToFile  
	Hsiang:  
		LDA  

	data cleaning and selecting: Hsiao  
	organization of scripts and contents: Jing  


**Project Description:**  
	1. the project mainly concentrated on spherical k-means clustering on text data  
	2. the project also tried standard k-means approach and some other approaches  


**Brief Content Description:**  

	*To run spherical k-means*  
	
		Folder initialization:  
			init.py  
			main.py  
			prepocess.py  
			select_features.py  
			skmeans_find_k.py  
			skmeans_visualiza.py  
			spherical_k_means_test.py  
			spherical_k_means.py  
			text_extraction.py  
			vectorize.py  
			data/  
				dataset_with_parser_results.json.utf8.txt  
				final_data_1850.txt  
				stopwords.txt  
		Steps:  
			1. make a dir called output  
			2. run skmeans_find_k.py with preferred mode and decide k  
			3. change k in main.py  
			4. run through all necessary functions, and preferred mode in main.py  
			5. check results  
			(for better output graph, adjust the plotting functions)  


	*To run standard k-means (not recommanded)*  
	
		Folder initialization:  
			init.py  
			main.py  
			prepocess.py  
			select_features.py  
			kmeans_find_k.py  
			kmeans.py  
			text_extraction.py  
			vectorize.py  
			data/  
				dataset_with_parser_results.json.utf8.txt  
				final_data_1850.txt  
				stopwords.txt  
		Steps:  
			1. install needed modules  
			2. make a dir called output  
			3. run kmeans_find_k.py with preferred mode and decide k  
			4. run Kmeans(mode) with preferred mode  
			5. check results  
			(for better output graph, adjust the plotting functions)  


Special thanks to Jason Macnak who provided ApplyStanfordParser.java for using Stanford parser  
