// This setup.go module creates the Dataset struct as well as
// working in collaboration with the fm.go module (file manager module)
// to prepare, filter and separate the data.
package main

import (
	"io/ioutil"
	"regexp"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// Type Definition for Dataset.
// Data 1 refers to original data source /cleansed data.
// Data 2 (TBImplemented):  Refers to new modfied and updated data sources.


type Dataset struct {
	DATA1_Wordlist        	string			// Get the string representing the whole dataset.
	DATA1_WordlistSegments 	[]string		// Gets each individual phrase, updated when user inputs.

	DATA1_UniqueData   		[]string		// Individual Unique Words.
	DATA1_UniqueLabels 		[]string		// These are the Classes

	DATA1_OrderedData   	[]string		 
	DATA1_OrderedLabels 	[]string
	
	DATA2_CountMap  				map[string][]float64 // (ORDERED) Key:label, value : stored frequency of each unique word
	DATA2_BaseClassProbabilities	map[string]float64 	 // (ORDERED) Stores Base probability for each class 
	DATA2_AnalyzedSegment		    []string			 // Inputs the currently Analyzed String
	DATA2_Vocabulary 				[]float64			 // (ORDERED) // V := Vocabulary, defined as all the unique instances within the corpus, for a given class.
	DATA2_N_AnalyzedSegment			[]float64	 // (ORDERED) Stores TF for all words in the Analyzed String
	DATA2_Nk_AnalyzedSegment		map[string][]float64 // Stores Base TF for each word in each class 
	DATA2_ProbabilityMap			map[string][]float64 // Stores the probability data of each unique word
	DATA2_UserInput					string				 // Defines the Analyzed String 
	
	DATA2_BayesProbabilities		[]float64			 // Outputs the class according to the Bayes Classifier
	DATA2_CummulativeProbabilities	float64			 	 // Stores the cummulative Probabilities of all classes for an analyzed instance
	DATA2_BayesPrediction			string				 // Outputs the class according to the Bayes Classifier
	DATA2_NeuralPrediction			string				 // Outputs the class of the Analyzed String 
	DATA2_StoredWeights				*mat.VecDense        // Stored Vectors for the optimized Class. 

	DATA3_Iterations				int					 // If the model is better, it'll require less iterations to improve accuracy

}

// 1. GET_1WordList : 
// Returns the full text dataset to be cleansed.
// IMPORTANT: Must be in UTF8 Format, convert source if necessary.
func GET_1WordList(filename string, Dataframe Dataset) Dataset {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		ErrorHandler(err.Error(), "ERROR : IMPORTAR TEXTO : Funcion GET_1WordList(), token.go")
		return Dataframe
	}
	Dataframe.DATA1_Wordlist = strings.ToLower(string(b))
	buffer_list := strings.Split(Dataframe.DATA1_Wordlist, ")")
	for _, v := range buffer_list {
		v = strings.TrimSpace(v)
		Dataframe.DATA1_WordlistSegments = append(Dataframe.DATA1_WordlistSegments, v)
	}

	return Dataframe
}

// 2. GET_2Labels : Get the string of unique words.
// Returns a Dataset instance with updated label values.
func GET_2Labels(sourceDataset Dataset) Dataset {

	text := sourceDataset.DATA1_Wordlist // References to original wordlist
	var expression_labels = regexp.MustCompile(`\([^)]*\)`)
	var list_clean_Labels []string
	list_filtered_Labels := expression_labels.FindAllString(text, -1)

	for idx := range list_filtered_Labels {
		nosymbol_labels1 := strings.Replace(list_filtered_Labels[idx], "(", "", -1)
		nosymbol_labels2 := strings.Replace(nosymbol_labels1, ")", "", -1)
		list_clean_Labels = append(list_clean_Labels, nosymbol_labels2)
	}
	sourceDataset.DATA1_OrderedLabels = list_clean_Labels
	sourceDataset.DATA1_UniqueLabels = GET_3PrepareLabel(list_clean_Labels, nil)
	return sourceDataset
}

// 2. GET_2Data : Get the string of unique Data for the bag of words.
// Creates a list containing the Ordered and Individual Words(Unique data).
// Returns a Dataset instance with updated data values.
func GET_2Data(sourceDataset Dataset) Dataset {

	// References to original wordlist & Regexp
	text := sourceDataset.DATA1_Wordlist

	// Match labels and trailing whitespace to erase from original text
	// Eliminate the # , labels and preserve the data with double whitespace separator
	var expression_data = regexp.MustCompile(`\s+\([^)]*\)`)
	list_filtered_greetings := expression_data.FindAllString(text, -1)
	for _, val := range list_filtered_greetings {
		text = strings.ReplaceAll(text, "#", "")
		text = strings.TrimSpace(text)
		text = strings.ReplaceAll(text, val, "  ")
	}

	// Generate Ordered Data: Used for intputs
	// Assign a non double whitespaced / non leading # string list to Ordered Data
	list_buffer := strings.SplitAfterN(text, "  ", -1)

	sourceDataset.DATA1_OrderedData = LIST_FilterDataLists(list_buffer, nil)
	sourceDataset.DATA1_UniqueData = GET_3PrepareData(list_buffer)
	return sourceDataset
}

// Prepare Labels:
// Function: That eliminates duplicate strings in a list of strings, given an optional filter list.
// Else, it'll try to filter out any repeated items in the list.
// Returns: A list of unique strings.
// Worst Time: N
func GET_3PrepareLabel(data []string, s2_optionalList []string) []string {
	allItems := HELPER_JoinLists(data)
	duplicateList := LIST_SeparateTerms(allItems, "!")
	uniqueList := LIST_FilterDataLists(duplicateList, s2_optionalList)
	return uniqueList
}

// Prepare Data:
// Function: Prepares the Ordered data to create an unique data list.
// Generate Unique Data: Used for Bag Of Words.
// Returns a whitespace-trimmed unique entry list .
// Worst Time: N2
func GET_3PrepareData(data []string) []string {
	undefined_map := make(map[string]bool)
	var buffer_list []string
	for _, val := range data {
		buffer_list = strings.Split(val, " ")
		for _, entry := range buffer_list {
			entry = strings.TrimSpace(entry)
			if !undefined_map[entry] {
				undefined_map[entry] = true
			}
		}
	}
	var return_list []string
	for key := range undefined_map {
		if len(key) > 1 {
			return_list = append(return_list, key)
		}
	}
	return return_list
}

