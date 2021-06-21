// Brain is the Chatbot's Brain, just a crop out of the main.go
// Easier to manage and with all the parameters required to train the network.
package main 


func BrainTrain(Dataframe Dataset, input string, label string, ShouldTrain bool, Training bool) Dataset{
	Dataframe = HELPER_AssignQuery(Dataframe, input)
	Dataframe = CalculateBayes(Dataframe)

	for {
		if label == "" {
			BP := Dataframe.DATA2_BayesPrediction
			Dataframe = NeuralNetwork(Dataframe, BP, ShouldTrain, Training) 
		} else {
			Dataframe = NeuralNetwork(Dataframe, label, ShouldTrain, Training) 
		}
		if (label == Dataframe.DATA2_NeuralPrediction) {
			break
		}
		Dataframe.DATA3_Iterations++
	}

	PrintNumber(Dataframe.DATA3_Iterations)
	return Dataframe	
}	

func ExecuteTrainedNetwork(Dataframe Dataset, msg string) string{
	Dataframe = HELPER_AssignQuery(Dataframe, msg)
	Dataframe = CalculateBayes(Dataframe)

	if Dataframe.DATA2_CummulativeProbabilities != 0 {
		Result := TrainedNetworkExecution(Dataframe)
		return Result
	}
	Result := Dataframe.DATA2_BayesPrediction
	return Result
}


