package main

import (
	"math"
	"strings"
	"gonum.org/v1/gonum/mat"
)

// HELPER FUNCTIONS

// Helper Function: Calculates the Probability field for a given nk Term and Class
// The probability field is the corresponding probability for each element in its class.
// Returns a float64 with a probability value for a word in a sentence of Length Vocabulary
// nk (NK[Class]) , contains a single float64 value from the []float64 representing
// 		the ocurrence of the word in a  in the Analyzed Sentence for a given class
// N Contains the occurrences of the words for a given class in the corpus
// Vocabulary contains length in words of a sentence
func HELPER_GenerateProbabilityField(nk float64, n float64,Vocabulary float64 ) float64 {
	numerator := nk+1
	denominator := n + Vocabulary
	if denominator == 0 {
		return 0.0
	}
	return numerator/denominator
}

// Helper Function: Used in NN1 to Infuse a Dataset instance with a Query.
// Defines the query of the user, this function is called whenever the 
// Input bar content is changed within the UI.
// Results: Dataframe Instance with User Input updated
func HELPER_AssignQuery(Dataframe Dataset, Text string) Dataset{
	Dataframe.DATA2_UserInput = strings.ToLower(Text)
	return Dataframe
}

// Helper Function that Generates an Usable Unique Word String out of the 
// input string of the user. Returns: A list of individual words of the 
// input query.
func HELPER_UsableString(AnalyzedSegment string) []string {
	AnalyzedWordList := LIST_SeparateTerms(AnalyzedSegment,"!")
	UsableString := make([]string,0)
	for _ , unfiltered_word := range AnalyzedWordList {
		unfiltered_word = strings.ReplaceAll(unfiltered_word,"!","")
		unfiltered_word = strings.TrimSpace(unfiltered_word)
		if (len(unfiltered_word) > 2){
			UsableString = append(UsableString,unfiltered_word)
		}
	}
	return UsableString
}

// Helper Function that extracts the items from a string map and appends them to a float64 list 
// Returns an extracted items list
func HELPER_ExtractItems(result map[string]float64) []float64{
	buffer_results := make([]float64, len(result))
	idx := 0
	for  _ , item := range result {
		buffer_results[idx] = item
		idx++
	}
	return buffer_results
}

// Helper Function that Calculates the Softmax exponential for a given input.
// Parameters: An input vector []float64.
// Returns the Float Softmax []float64 parameter vector.
func HELPER_Softmax(elements []float64) []float64 {
	denominator := 0.0
	for _, denom := range elements {
		denominator += math.Exp(denom)
	}

	result := elements
	for idx, num := range elements {
		result[idx] = math.Exp(num) / denominator
	}
	return result
}
// Helper Function that sums the elements of a []float64
// Returns a float 64 with the Sum result of all term frequencies.
func HELPER_Sum64(list []float64) float64 {
	res := 0.0
	for _, n := range list {
		res += n
	}
	return res
}

// Helper function that returns the last element of a string which will be split
// Returns a string which holds the last element. 
func HELPER_LastElement(query string) string {
	if query == "" {
		return ""
	}
	query_LastElementSlice := strings.SplitAfterN(strings.TrimSpace(query)," ",len(query))
	lastElement := len(query_LastElementSlice)-1
	return strings.TrimSpace(query_LastElementSlice[lastElement])
}
// Helper function that cleanses a corpus of text and returns 
// a filtered and cleaned string
// Cleanse the corpus (removes labels & symbols)
func HELPER_CleanseTextCorpus(text string, Dataframe Dataset) string {
	UniqueClasses := Dataframe.DATA1_UniqueLabels   // Classes of Objects
	text = strings.ReplaceAll(text,"(","")
	text = strings.ReplaceAll(text,")","")
	text = strings.ReplaceAll(text,"#","")
	for _, v := range UniqueClasses {
		text = strings.ReplaceAll(text,v,"")
	}

	return  strings.TrimSpace(text)
}
 
// Helper function that cleanses a corpus of text and returns 
// a filtered and cleaned string
func HELPER_CleanseTextListCorpus(text []string, Dataframe Dataset) []string {
	for idx := range text {
		text[idx] = strings.ReplaceAll(text[idx],"(","")
		text[idx] = strings.ReplaceAll(text[idx],")","")
		text[idx] = strings.ReplaceAll(text[idx],"#","")
		text[idx] = strings.TrimSpace(text[idx])
	}

	return text
}
 
// Helper Function : Generates an Index out of the word scanned from the UniqueData set,
// Corresponding to its position on the Probability Map
func HELPER_IndexGenerator(word string , vocabulary []string) int {
	var return_value int 
	for idx, v := range vocabulary {
		if (strings.EqualFold(v,word)){
			return_value = idx
		}
	}
	return return_value 
}

// Helper Function: Generates a prediction out of the result value 
// Returns: Returns a string corresponding to the class prediction
func HELPER_GetPrediction(result map[string]float64) string {
	var buffer_Probability float64 = 0.0
	var buffer_Prediction string = "I don't know"

	for Class , Probability := range result {
		if (buffer_Probability < Probability){
			buffer_Probability = result[Class]
			buffer_Prediction = Class
		}  
	}

	return buffer_Prediction
}

// Helper Func: Initializes the map elements of a []float64 map to zero given an unique word size
// Returns a buffer map with all elements equal to zero.
func HELPER_ZeroMap(buffer_map map[string][]float64, size1_UniqueLabels []string, size2_UniqueWords []string) map[string][]float64 {
	for _, val := range size1_UniqueLabels {
		buffer_map[val] = make([]float64, len(size2_UniqueWords))
	}
	return buffer_map
}

// Function For Dot Product between previously known vectors
// Replaces/Fixes the broken Dot product (multnum) at vector.go "matnum"
// Returns a vector of dimensions P1, or P2. 
func HELPER_DotProduct(P1 *mat.VecDense, P2 *mat.VecDense) *mat.VecDense{
	var buffer_f64 []float64
	n := P1.Len()
	for i := 0; i < n; i++ {
		buffer_f64 = append(buffer_f64,P1.AtVec(i)*P2.AtVec(i))
	} 
	buffer_vector := mat.NewVecDense(P1.Len(), buffer_f64)
	return buffer_vector
}

// Function for Dot Substraction between previously known vectors
// Replaces/Fixes the broken Dot SubVec (multnum) at vector.go "matnum"
// Returns a vector of dimensions P1, or P2. 
func HELPER_DotSubstract(P1 *mat.VecDense, P2 *mat.VecDense) *mat.VecDense{
	var buffer_f64 []float64
	n := P1.Len()
	for i := 0; i < n; i++ {
		buffer_f64 = append(buffer_f64,P1.AtVec(i)-P2.AtVec(i))
	} 
	buffer_vector := mat.NewVecDense(P1.Len(), buffer_f64)
	return buffer_vector
}

// Helper Function.
// Function to extract float64 list out of a 2D Vector.
// Returns: Float 64 list with contents of P1.
func HELPER_VectorExtract(P1 *mat.VecDense) []float64 {
	return P1.RawVector().Data
}

// Helper Function: Generates an f64 vector of ones.
// Returns: Size (dimensions) of 1.0's in a []float64.
func HELPER_GenerateOnesF64(dimensions int) []float64 {
	result := make([]float64 , dimensions)
	for idx  := range result {
		result[idx] += 1 
	}
	return result
} 

// Helper Function: Generates the Class Label Index given the Class list.
// Returns: Int with corresponding class index to optimize.
func HELPER_GetLabelIndex(Classes []string, label string) int{
	idx := 0
	for ClassIndex , v := range Classes {
		if(strings.Contains(v, label)){
			idx = ClassIndex
		}
	} 
	return idx 
}
// Function: Outputs a string out of a string list: []string.
// Returns: A string made out of all the elements of the list with
// a single space per new element.
func HELPER_JoinLists(s1 []string) string {
	return strings.Join(s1, " ")
}

// Function: Separates terms in a string to return all the individual strings by the separator " ".
// Returns a list of strings containing each separate element, including repeated items.
func LIST_SeparateTerms(s1 string, separator string) []string {
	s1 = strings.ReplaceAll(s1, " ", separator)
	return strings.Split(s1, separator)
}

// Function: filters data on S1, using the S2 filter.
// Returns: A filtered string list S1 made out of S1 elements not present in S2.
func LIST_FilterDataLists(s1 []string, s2 []string) []string {
	check := make(map[string]bool)
	for _, val := range s1 {
		check[val] = true
	}

	for _, val := range s2 {
		if check[val] {
			check[val] = false
		}
	}

	var output []string
	for key, val := range check {
		if val {
			output = append(output, key)
		}
	}

	return output
}

