// NN Stands for:
//  Neural Network
package main

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func NeuralNetwork(Dataframe Dataset, epochs int, label string, ShouldRetrain bool) Dataset {

	Probabilities := Dataframe.DATA2_BayesProbabilities
	Classes := Dataframe.DATA1_UniqueLabels
	Sample := len(Probabilities)
	InputVector := mat.NewVecDense(Sample, Probabilities)

	// Reference Old Weights or Create New Ones!
	var weightBuffer *mat.VecDense
	if Dataframe.DATA2_StoredWeights == nil || ShouldRetrain {
		InitialWeights := WeightSetup1(InputVector.Len(), nil)
		weightBuffer = InitialWeights
	} else {
		InitialWeights := WeightSetup1(InputVector.Len(), HELPER_VectorExtract(Dataframe.DATA2_StoredWeights))
		weightBuffer = InitialWeights
	}

	// "Pizza Order Optimizer" :v , might have to change the bias or class.
	// XavierBias is wayy too big, we're using 0 for now. Bias remains static.
	// If the class Index is 9999 it'll skip the probability skewering in the Expected result.
	// Else it'll sort the Input Vector, since during training of the classifier,
	// the middle value turned out to be the label almost every single time, i turn the middle value
	// into the tboptimized class. (Is it still learning?)
	// And will treat all inputs as the same. This is much better since it skewers depending
	// on the dataset and the original distribution, instead of prepared values, which may
	// vary with the dataset.
	// Input Vectors WILL CHANGE with different phrases, so it may be necessary to run inference
	// on every new phrase, even with saved weights to ensure maximum goodness.
	ClassIndex := 9999
	if label != "" {
		ClassIndex = HELPER_GetLabelIndex(Classes, label)
	}

	for i := 0; i < epochs; i++ {
		weightBuffer = NeuronExecute(InputVector, weightBuffer, 0, ClassIndex)
	}

	NeuralNetworkResult := mat.NewVecDense(Sample, HELPER_GenerateOnesF64(Sample))
	NeuralNetworkResult.MulElemVec(InputVector, weightBuffer)
	Prediction := NeuralNetworkResult.RawVector().Data

	buffer := 0.0
	for idx, val := range Prediction {
		if buffer < val {
			buffer = val
			Dataframe.DATA2_NeuralPrediction = Classes[idx]
		}
	}

	Dataframe.DATA2_StoredWeights = weightBuffer

	return Dataframe
}

// Function to initialize Weights using Xavier/Glorot Initialization
// Returns a matrix with an Initialized Weight Vector
func WeightSetup1(samples int, previous_weights []float64) *mat.VecDense {

	GlorotInit := make([]float64, 0)
	n := float64(samples)

	for i := 0.0; i < n; i++ {
		GlorotInit = append(GlorotInit, math.Sqrt(1/n)) // Glorot init (reduce the vanishing gradient problem)
	}

	weight := WeightSetup2(samples, previous_weights)
	GlorotInitVector := mat.NewVecDense(samples, GlorotInit)
	updated_weight := HELPER_DotProduct(weight, GlorotInitVector)

	return updated_weight
}

// Function to generate weights depending on if a new weight is available:
// Return a weight vector that can be updated
func WeightSetup2(dimension int, previous_weights []float64) *mat.VecDense {
	weightVector := make([]float64, 0)
	var weight *mat.VecDense
	if previous_weights != nil {
		return mat.NewVecDense(dimension, previous_weights)
	} else {
		for i := 0.0; i < float64(dimension); i++ {
			weightVector = append(weightVector, rand.Float64())
		}
		weight = mat.NewVecDense(dimension, weightVector)
	}

	return weight
}

// Function to Execute the calculations of a Neuron
// Executes the neuron to optimize weights towards an expected desired output
// Returns : Vector with Neuron Computated Weights for an specific optimized class.
func NeuronExecute(InputVector *mat.VecDense, InitialWeights *mat.VecDense, bias float64, classToOptimize int) *mat.VecDense {
	Probabilities := InputVector.RawVector().Data
	Sample := InputVector.Len()

	NeuronResult := WeightedSum(InputVector, InitialWeights, bias)
	GeneratedOutput := make([]float64, len(Probabilities))
	for idx := range GeneratedOutput {
		GeneratedOutput[idx] = NeuronResult
	}
	ExpectedOutput := NN_GETExpectedOutput(Probabilities, classToOptimize, 1292873980) // Pizza
	NewWeights := ErrorCalculations(Probabilities, ExpectedOutput, GeneratedOutput, InitialWeights)

	return mat.NewVecDense(Sample, NewWeights)
}

// Function to calculate the error related metrics and update the weights for each input
// oneminusY1: https://www.youtube.com/watch?v=Wq3TC_2tDgQ
// Result: Outputs a float64 of updated weights
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

// Defines the expected output for a given training set
// Softmax function applied to generate expected output
// Result: Generates an expected output with a modified multiplier to favor the weights during training
// If the class == 9999 , it'll skip the skewer. Else, it'll treat all probabilities the same.
func NN_GETExpectedOutput(probabilities []float64, class int, multiplier float64) []float64 {
	for idx := range probabilities {
		probabilities[idx] *= 1 / multiplier
	}
	if class != 9999 {
		probabilities[class] *= multiplier
	}
	return HELPER_Softmax(probabilities)
}

// Function to combine the weights with the input for the first prediction.
// SUM {i1*w1... In*Wx + bias}
// First Layer Neuron // , Bias *mat.VecDense.
// Returns: A float 64 with the value calculated by the neuron for a
// chosen bias & (Weight,Input) vectors .
func WeightedSum(Input *mat.VecDense, Weights *mat.VecDense, bias float64) float64 {
	WeightedInput := HELPER_DotProduct(Input, Weights).RawVector().Data
	NeuronValue := HELPER_Sum64(WeightedInput) + bias
	return NeuronValue
}

// Function to Initialize the inputlayer from the Probability Results of The Bayes Classifier
// Returns : A Vector with the Probability Results of The Bayes Classifier
func InputLayer(Input []float64) *mat.VecDense {
	result := mat.NewVecDense(len(Input), Input)
	return result
}
