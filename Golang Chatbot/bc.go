// BC stands for:
// Bayes Classifier
// 	As the base for my Bayes Neural Network

package main

import 	"strings"

// Main Function that Calls for the first map generation
// Returns : An Dataframe with an updated map of labels with their corresponding
// string distributions...
func NN_GenerateInitialMap(Dataframe Dataset) Dataset {
	Dataframe = NN_GenerateCountMap(Dataframe)
	return Dataframe
}

// Function: Generates the Base Class probability within the corpus
// Returns a Dataset instance with updated Base Class Probability fields 
func CalculateBaseClassProbability(Dataframe Dataset) Dataset {
	ClassInstances := Dataframe.DATA1_OrderedLabels // Documents
	UniqueClasses := Dataframe.DATA1_UniqueLabels   // Classes of Objects
	PROB_Label := make([]float64, len(UniqueClasses))
	totalSize := float64(len(ClassInstances))
	for idx, className := range UniqueClasses {
		instances := 0.0
		for _, analyzedLabel := range ClassInstances {
			if strings.Contains(analyzedLabel, className) {
				instances += 1.0
			}
		}
		PROB_Label[idx] = instances / totalSize
	}
	Dataframe.DATA2_BaseClassProbabilities = PROB_Label
	return Dataframe
}

// Function : Calculates the word term frequency of words in an Analyzed String within the Corpus 
// according to class.  
// Returns: A Dataset instance with updated TF fields for each word in the corpus.
func CalculateNkN(Dataframe Dataset) Dataset {
	
	Corpus := HELPER_CleanseTextCorpus(Dataframe.DATA1_Wordlist, Dataframe)
	CountMap := Dataframe.DATA1_CountMap // Term Frequency of the words in Corpus
	UniqueData := Dataframe.DATA1_UniqueData
	UniqueClasses := Dataframe.DATA1_UniqueLabels   // Classes of Objects
	AnalyzedWordList := HELPER_UsableString(Dataframe.DATA2_UserInput) // Generates Usable Wordlist from User Input
	
	// Cleanse the corpus (removes labels & )

	Nk := make(map[string][]float64)
	Nk = HELPER_ZeroMap(Nk , UniqueClasses , AnalyzedWordList)
	for idx , Word := range AnalyzedWordList {
		WordIndex := HELPER_IndexGenerator(Word,UniqueData)
		for _, Class := range UniqueClasses {
			if (strings.Contains(Corpus,Word)){
				Nk[Class][idx] = CountMap[Class][WordIndex] 
			}
		}
	}
	N := make(map[string]float64, len(UniqueClasses))
	for _, Class := range UniqueClasses {
		N[Class] = 0.0
	}
	for idx , v := range Nk {
		N[idx] = HELPER_Sum64(v)
	}

	Dataframe.DATA2_CurrentAnalyzedSegment = AnalyzedWordList
	Dataframe.DATA2_N_AnalyzedSegment = N
	Dataframe.DATA2_Nk_AnalyzedSegment = Nk
	return Dataframe
}

// Function: Calculates Class based on the probability distribution of the phrase.
// Uses the Base probability, word occurrence and word population for each class given the sentence
// Returns: A dataset instance with a Sentence Prediction
func CalculateBayes(Dataframe Dataset) Dataset {
	Dataframe = CalculateBaseClassProbability(Dataframe)
	Dataframe = CalculateNkN(Dataframe)

	BaseProbability := Dataframe.DATA2_BaseClassProbabilities
	Vocabulary := float64(len(Dataframe.DATA2_CurrentAnalyzedSegment))
	NK := Dataframe.DATA2_Nk_AnalyzedSegment
	N := Dataframe.DATA2_N_AnalyzedSegment
	// Mapa que da resultados dadas las clases
	result_Prediction := make(map[string]float64, len(N))
	idx := 0
	for Class := range N {
		result_Prediction[Class] = BaseProbability[idx]
		idx ++
		for _, nk := range NK[Class] {
			result_Prediction[Class] *= HELPER_GenerateProbabilityField( nk , N[Class], Vocabulary)
		}
	}

	var Prediction string = "No prediction was found :'( , words are unknown" 
	Classes := Dataframe.DATA1_UniqueLabels
	Probability  := HELPER_ExtractItems(result_Prediction) 

	buffer := 0.0
	for ClassKey , val := range Probability {
		if (val > buffer){
			buffer = val
			Prediction = Classes[ClassKey]
		}
	}
	
	Dataframe.DATA2_BayesProbabilities = Probability 
	Dataframe.DATA2_BayesPrediction = Prediction
 	return Dataframe
}

// 1. Function Generate Map of Probabilities:
// Go through (VALX ) every single Wordlist_Segment || Go through the specified wordlist segment
// 	Go through (IDX1 , VAL1)all the Unique Words
// 		Go through all the (IDX2 , VAL2)Unique Labels
func NN_GenerateCountMap(Dataframe Dataset) Dataset {
	Wordlist := HELPER_CleanseTextListCorpus(Dataframe.DATA1_WordlistSegments,Dataframe)
	UniqueWords := Dataframe.DATA1_UniqueData
	UniqueLabels := Dataframe.DATA1_UniqueLabels
	buffer_map := make(map[string][]float64, len(UniqueLabels))
	buffer_map = HELPER_ZeroMap(buffer_map, UniqueLabels, UniqueWords)

	for _, VAL1 := range UniqueLabels {
		for _, query := range Wordlist {
			if strings.Contains(query+" ", VAL1) {
				for IDX2, VAL2 := range UniqueWords {
					if strings.Contains(query, VAL2) {
						query = HELPER_CleanseTextCorpus(query,Dataframe)
						buffer_map[VAL1][IDX2] += float64(strings.Count(query, VAL2))
					}
				}
			}
		}
	}
	Dataframe.DATA1_CountMap = buffer_map
	return Dataframe
}

	// Modification on Helper Implementation (adding instead of chain multiplication due to Dataset type) 
	// Modification on Helper Definition (log filter, to eliminate low frequency noise) 
