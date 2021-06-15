// brain is the Chatbot's Brain, just a crop out of the main.go
package main 
import "gonum.org/v1/gonum/mat"


func BrainTrain(Dataframe Dataset, input string, label string, ShouldRetrain bool) Dataset{
	Dataframe = HELPER_AssignQuery(Dataframe, input)
	Dataframe = CalculateBayes(Dataframe)
	Dataframe = NeuralNetwork(Dataframe, 10, label, ShouldRetrain) 
	// Opcionales
	if(Dataframe.DATA2_NeuralPrediction == label){
		PrintText("Good Prediction",1)
		} else {
		PrintText("Bad Prediction",1)
		Dataframe.DATA2_StoredWeights = mat.NewVecDense(Dataframe.DATA2_StoredWeights.Len(),HELPER_GenerateOnesF64(Dataframe.DATA2_StoredWeights.Len()))
	}
	return Dataframe	
}	
