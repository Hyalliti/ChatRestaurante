// BC stands for: Bayes Classifier
// 	As the base for my Bayes Neural Network

package main

import "strings"

// Function that Calls for the first map generation
// Returns : An Dataframe with an updated map of labels with their corresponding
// string distributions...
func NN_GenerateInitialMap(Dataframe Dataset) Dataset {
	Dataframe = GenerateCountMap(Dataframe)
	return Dataframe
}

// Function Generate Map of Term Frequencies:
// Returns: 
/*
 Go through (VALX ) every single Wordlist_Segment 
 OR
 Go through the specified wordlist segment
 	Go through (IDX1 , VAL1) all the Unique Words
		Go through all the (IDX2 , VAL2) Unique Labels
*/
func GenerateCountMap(Dataframe Dataset) Dataset {
	Wordlist := HELPER_CleanseTextListCorpus(Dataframe.DATA1_WordlistSegments, Dataframe)
	UniqueWords := Dataframe.DATA1_UniqueData
	UniqueLabels := Dataframe.DATA1_UniqueLabels
	buffer_map := make(map[string][]float64, len(UniqueLabels))
	buffer_map = HELPER_ZeroMap(buffer_map, UniqueLabels, UniqueWords)

	for _, VAL1 := range UniqueLabels {
		for _, query := range Wordlist {
			query_word := HELPER_LastElement(query)
			hasClass := query_word == VAL1
			if hasClass {
				for IDX2, VAL2 := range UniqueWords {
					if len(VAL2) <= 2 {
						buffer_map[VAL1][IDX2] += 0.0
					} else {
						if strings.Contains(query, VAL2) {
							cleansed_query := HELPER_CleanseTextCorpus(query, Dataframe)
							buffer_map[VAL1][IDX2] += float64(strings.Count(cleansed_query, VAL2))
						}
					}
				}
			}
		}
	}
	Dataframe.DATA2_CountMap = buffer_map
	return Dataframe
}

// Function: Generates the Base Class probability within the corpus.
// Returns a Dataset instance with updated Base Class Probability fields.
/* 
	Added a "!" tag, for labels with the same content but different leading values
	Such as liked , disliked, as there is no strings.Contains Exactly//Full Match function
*/
func CalculateBaseClassProbability(Dataframe Dataset) Dataset {
	ClassInstances := Dataframe.DATA1_OrderedLabels // Documents
	UniqueClasses := Dataframe.DATA1_UniqueLabels   // Classes of Objects
	PROB_Label := make(map[string]float64, len(UniqueClasses))
	totalSize := float64(len(ClassInstances))
	for _, className := range UniqueClasses {
		instances := 0.0
		for _, analyzedLabel := range ClassInstances {
			if strings.Contains("!"+analyzedLabel, "!"+className) {
				instances += 1.0
			}
		}
		PROB_Label[className] = instances / totalSize
	}
	Dataframe.DATA2_BaseClassProbabilities = PROB_Label
	return Dataframe
}

// Function : Calculates the word term frequency of words in an Analyzed String
// within the Corpus according to class.
// Returns: A Dataset instance with updated TF fields for each word in the corpus.
/*
	N := Frequency of All Words in the corpus related to an specific class
	Count c is already contained as the individual unique word data within the class. 
	Nk := Frequency of an analyzed word within the sentence , corresponding to an specific class
	Count w,c := How many times does word nk , occur in all Class Wordlist Segments 
		 Find word NK in Wordlist Segment for a given class... 
		 Add Nk tf to a map[classes][]float64 (containing each nk)
			Look within the Unique Word Data (which is copied on the Count Map)
			to extract the index corresponding to the term frequency of the item 
			The count map has a UniqueData copy for each class, within lies the term frequency 
			Once the AnalyzedWord index is extracted, it's then placed on the count map to 
			obtain nk's Term Frequency item.
	V := Vocabulary, defined as all the unique instances within the corpus, 
			for a given class.
* *  FIXED WITH : https://youtu.be/j1uBHvL6Yr0
*/
func CalculateNkN(Dataframe Dataset) Dataset {
	CountMap := Dataframe.DATA2_CountMap // Term Frequency of the words in Corpus
	UniqueData := Dataframe.DATA1_UniqueData
	UniqueClasses := Dataframe.DATA1_UniqueLabels                      // Classes of Objects
	
	// Generates Usable Wordlist from User Input & Stores the Wordlist Segments
	AnalyzedInput := HELPER_UsableString(Dataframe.DATA2_UserInput) 
	N := HELPER_GenerateOnesF64(len(UniqueClasses))
	NK := make(map[string][]float64, len(UniqueClasses)) 

	for index , ClassName := range UniqueClasses {
		N[index] = HELPER_Sum64(CountMap[ClassName])
		NK[ClassName] = HELPER_GenerateOnesF64(len(AnalyzedInput))
	}

	for ClassIndex  := range UniqueClasses {
		NK[UniqueClasses[ClassIndex]] = HELPER_GenerateOnesF64(len(AnalyzedInput))
	}
	for ClassIndex , ClassName := range UniqueClasses {
		for AnalyzedIndex, AnalyzedWord := range AnalyzedInput {
			IndexDictionary := HELPER_IndexGenerator(AnalyzedWord,UniqueData)
			NK[UniqueClasses[ClassIndex]][AnalyzedIndex] = CountMap[ClassName][IndexDictionary]
		}
	}
	Vocabulary := HELPER_GenerateOnesF64(len(UniqueClasses))
	var buffer []float64
	for ClassIndex  := range UniqueClasses {
		buffer = make([]float64,0)
		for _, val := range CountMap[UniqueClasses[ClassIndex]] {
			if val != 0 {
				val = 1
			}
			buffer = append(buffer, val)
		}
		Vocabulary[ClassIndex] = HELPER_Sum64(buffer)
	}

	Dataframe.DATA2_AnalyzedSegment = AnalyzedInput
	Dataframe.DATA2_Vocabulary = Vocabulary
	Dataframe.DATA2_N_AnalyzedSegment = N
	Dataframe.DATA2_Nk_AnalyzedSegment = NK

	return Dataframe
}

// Function: Calculates Class based on the probability distribution of the phrase.
// Returns: A dataset instance with a Sentence Prediction.
/* 
	Uses the Base probability, word occurrence and word population for each class given the sentence
	Redoing naive bayes with a better classifier... https://www.youtube.com/watch?v=EGKeC2S44R
*/ 
func CalculateBayes(Dataframe Dataset) Dataset {
	Dataframe = CalculateBaseClassProbability(Dataframe)
	Dataframe = CalculateNkN(Dataframe)
	
	BaseProbability := Dataframe.DATA2_BaseClassProbabilities
	Vocabulary := Dataframe.DATA2_Vocabulary
	NK := Dataframe.DATA2_Nk_AnalyzedSegment 
	N := Dataframe.DATA2_N_AnalyzedSegment	
	
	// Mapa que da resultados dadas las clases
	result_Prediction := make(map[string]float64, len(BaseProbability))
	Probability := HELPER_GenerateOnesF64(len(BaseProbability))
	Classes := Dataframe.DATA1_UniqueLabels
	Wordlist := Dataframe.DATA2_AnalyzedSegment

	for idx := range Classes {
		result_Prediction[Classes[idx]] = BaseProbability[Classes[idx]]
	}
	for ClassIndex , ClassName := range Classes {
		for AnalyzedSegmentIndex  := range Wordlist {
			result_Prediction[ClassName] *= HELPER_GenerateProbabilityField(NK[ClassName][AnalyzedSegmentIndex], N[ClassIndex], Vocabulary[ClassIndex])
		}
	}
	for ClassIndex , ClassName := range Classes {
		Probability[ClassIndex] = result_Prediction[ClassName]
	}
	
	// If probabilities are fixed to 0 "completely unknown word/words with no references",
	// Nor Bayes or the Neural Network can be used
	var Prediction string = "No he podido comprenderle, favor repita una vez mas. Si desea ordenar, favor escriba 'quiero ordenar ' seguido de lo que desea. "
	CummulativeProbabilities := HELPER_Sum64(Probability)
	if CummulativeProbabilities != 0 {
		buffer := 0.0
		for ClassKey, val := range Probability {
			if val > buffer {
				buffer = val
				Prediction = Classes[ClassKey]
			}
		}
	}

	Dataframe.DATA2_BayesProbabilities = Probability
	Dataframe.DATA2_CummulativeProbabilities = CummulativeProbabilities
	Dataframe.DATA2_BayesPrediction = Prediction

	return Dataframe
}
