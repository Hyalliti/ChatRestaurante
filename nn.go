// NN Stands for:
//  Neural Network
package main

import (
	"math"
	"math/rand"
	"gonum.org/v1/gonum/mat"
)
/* Function That operates the Neural Network , generates weights or trains if required.
Returns: Dataset instance with a NeuralPrediction string, as well as Stored Weights for the next iterations.
*/ 
func NeuralNetwork(Dataframe Dataset, label string, FirstIteration bool , Training bool) Dataset {

	// Define variables
	Probabilities := Dataframe.DATA2_BayesProbabilities
	Classes := Dataframe.DATA1_UniqueLabels
	ClassIndex := HELPER_GetLabelIndex(Classes, label)
	Sample := len(Probabilities)
	InputVector := mat.NewVecDense(Sample, Probabilities)
	
	// Calculate Weights 
	// Create New Weights or Reference Old Weights or 
	var InputWeights *mat.VecDense
	var weightBuffer *mat.VecDense
	if FirstIteration {
		InputWeights = GenerateWeights(InputVector.Len())
	} else {
		InputWeights = Dataframe.DATA2_StoredWeights
	}
	weightBuffer = InputWeights
	
	// Assign the corresponding weights and operate the Neuron with bias 0:
	if Training {
		weightBuffer = NetworkExecute(Dataframe, InputVector, weightBuffer, 0, ClassIndex)
		
	}
		
	NeuralNetworkResult := mat.NewVecDense(Sample, HELPER_GenerateOnesF64(Sample))
	NeuralNetworkResult.MulElemVec(InputVector, weightBuffer)
	
	
	// Pass de Prediction Through the Activation Function
	Estimation := HELPER_Softmax(NeuralNetworkResult.RawVector().Data)
	buffer := 0.0
	for idx, val := range Estimation {
		if buffer < val {
			buffer = val
			Dataframe.DATA2_NeuralPrediction = Classes[idx]
		}
	}

	Dataframe.DATA2_StoredWeights = weightBuffer

	return Dataframe
}

/* Function to initialize Weights using Xavier/Glorot Initialization.
Returns a Vector with an Initialized Weight Vector of Randomly Distributed Weights.
*/ 
func GenerateWeights(samples int) *mat.VecDense {
	dimension := samples
	weightVector := make([]float64, 0)
	var weight *mat.VecDense

	GlorotInitVector := GlorotInit(dimension)
	
	for i := 0.0; i < float64(dimension); i++ {
		weightVector = append(weightVector, rand.Float64()/float64(dimension))
	}

	weight = mat.NewVecDense(dimension, weightVector)
	initializedWeights := HELPER_DotProduct(weight, GlorotInitVector)

	return initializedWeights
}

/* Function to generate a Glorot Type Initialization :
Returns a Glorot Initialization of Weights 
*/ 
func GlorotInit(dimension int) *mat.VecDense{
	GlorotInit := make([]float64, 0)
	ScalarComponent := math.Sqrt(1/float64(dimension))
	for i := 0.0; i < float64(dimension); i++ {
		GlorotInit = append(GlorotInit, ScalarComponent ) // Glorot init (reduce the vanishing gradient problem)
	}
	 
	return mat.NewVecDense(dimension, GlorotInit)
}

/* Function to Execute the Neuron Operations: Generate Neuron Input , Error Calculations / Adjustments.
1. Weighted sum for neuron input, the output is then used to obtain a buffer output and calculate the Error given a Learning Rate

2. Once the expected output is defined from a to-optimize class

* Returns : Vector with New Weights for an specific optimized class.
*/ 
func NetworkExecute(Dataframe Dataset, InputVector *mat.VecDense, InitialWeights *mat.VecDense, bias float64, classToOptimize int) *mat.VecDense {
	Probabilities := InputVector.RawVector().Data
	Dimension := InputVector.Len()

	NeuralNetworkResult := mat.NewVecDense(Dimension , HELPER_GenerateOnesF64(Dimension))
	NeuralNetworkResult.MulElemVec(InputVector, InitialWeights)
	NeuralOutput :=	HELPER_VectorExtract(NeuralNetworkResult)
	
	ExpectedOutput := NN_GETExpectedOutput(Probabilities, classToOptimize, 1.5)
	NewWeights := ErrorCalculations(Probabilities, ExpectedOutput, NeuralOutput, InitialWeights)

	return mat.NewVecDense(Dimension, NewWeights)
}

/* Function: Defines the Expected Output for a given Set of Data
Result: Generates an expected output with a modified multiplier to favor the weights during training
*/ 
func NN_GETExpectedOutput(probabilities []float64, class int, LearningRate float64) []float64 {
	probabilities[class] *= LearningRate
	return probabilities
}

/* Function to calculate the error related metrics and update the weights for each input
GUIDE:  https://www.youtube.com/watch?v=Wq3TC_2tDgQ

Result: Outputs a float64 of updated weights
*/ 
func ErrorCalculations(Input []float64, ExpectedOutput []float64, GeneratedOutput []float64, Weights *mat.VecDense) []float64 {
	Y := mat.NewVecDense(len(ExpectedOutput), ExpectedOutput)
	Y1 := mat.NewVecDense(len(GeneratedOutput), GeneratedOutput)
	Ones := mat.NewVecDense(len(GeneratedOutput), HELPER_GenerateOnesF64(len(GeneratedOutput)))

	errorVector := HELPER_DotSubstract(Y, Y1)
	oneminusY1 := HELPER_DotSubstract(Ones, Y1)
	adjustment := HELPER_DotProduct(errorVector, Y1)
	adjustment = HELPER_DotProduct(adjustment, oneminusY1)

	AdjustmentInfo := adjustment.RawVector().Data
	Wn := make([]float64, len(AdjustmentInfo))
	for idx := range Wn {
		Wn[idx] = Weights.AtVec(idx)
	}

	// New Weights
	for idx := range Wn {
		Wn[idx] += AdjustmentInfo[idx] + Input[idx]
	}

	return Wn
}


/* Function to Initialize the inputlayer from the Probability Results of The Bayes Classifier.

Returns : A Vector with the Probability Results of The Bayes Classifier
*/ 
func VectorConversion(Input []float64) *mat.VecDense {
	result := mat.NewVecDense(len(Input), Input)
	return result
}

/* Function to Execute a Trained Neural Network

Returns: A String format Classification for the given string.
*/ 
func TrainedNetworkExecution(Dataframe Dataset ) string {

	Input := VectorConversion(Dataframe.DATA2_BayesProbabilities)
	Weights := Dataframe.DATA2_StoredWeights
	Dimension := Input.RawVector().N
	Classes := Dataframe.DATA1_UniqueLabels

	NeuralNetworkResult := mat.NewVecDense(Dimension , HELPER_GenerateOnesF64(Dimension))
	NeuralNetworkResult.MulElemVec(Input, Weights)
	Prediction := NeuralNetworkResult.RawVector().Data

	buffer := 0.0
	var result string = ""
	for idx, val := range Prediction {
		if buffer < val {
			buffer = val
			result = Classes[idx]
		}
	}

	return result
}