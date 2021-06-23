// Within the initialize function, all the calls to the Bayes Classifier and the Neural Network are managed. (This was the first main.go) 
package main

func initialize() Dataset {

	var Dataframe Dataset
	Dataframe = GET_1WordList("chats - Copy", Dataframe)
	Dataframe = GET_2Labels(Dataframe)
	Dataframe = GET_2Data(Dataframe)

	// Map a frequency of unique words 
	Dataframe = GenerateCountMap(Dataframe)
	Dataframe = BrainTrain(Dataframe, "Yo quiero ordenar una pizza", "food,order,pizza", true , false)
	Dataframe = BrainTrain(Dataframe, "Saludos hola que tal", "greeting", false, true)
	Dataframe = BrainTrain(Dataframe, "genial la comida", "liked", false, true)
	Dataframe = BrainTrain(Dataframe, "la comida dejo mucho que desear", "disliked", false, true)
	Dataframe = BrainTrain(Dataframe, "quiero un reembolso", "disliked", false, true)
	Dataframe = BrainTrain(Dataframe, "quiero ordenar una comida vegana", "food,order,salad", false, true)
	Dataframe = BrainTrain(Dataframe, "quiero refresco rojo", "food,order,soda",false, true)
	Dataframe = BrainTrain(Dataframe, "por favor quiero una gaseosa", "food,order,soda", false, true)
	Dataframe = BrainTrain(Dataframe, "quiero ordenar hamburguesa", "food,order,hamburger", false, true)
	Dataframe = BrainTrain(Dataframe, "estuvo buenísima la comida", "liked", false, true)
	Dataframe = BrainTrain(Dataframe, "quiero comer algo", "greeting", false, true)
	Dataframe = BrainTrain(Dataframe, "quiero ordenar un hamburguer", "food,order,hamburger", false, true)
	Dataframe = BrainTrain(Dataframe, "dame un hamburguer", "food,order,hamburger", false, true)
	Dataframe = BrainTrain(Dataframe, "quiero ordenar una ensalada a la calle pequeña", "food,order,salad", false, true)
	Dataframe = BrainTrain(Dataframe, "me encantó la comida", "liked", false, true)

	return Dataframe
}