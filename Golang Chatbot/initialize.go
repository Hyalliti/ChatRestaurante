// Within the initialize function, all the calls to the Bayes Classifier and the Neural Network are managed. (This was the first main.go) 
package main

func initialize() Dataset {
	var Dataframe Dataset
	Dataframe = GET_1WordList("chats - Copy", Dataframe)
	Dataframe = GET_2Labels(Dataframe)
	Dataframe = GET_2Data(Dataframe)

	// Map a frequency of unique words 
	Dataframe = NN_GenerateInitialMap(Dataframe)

	Dataframe = BrainTrain(Dataframe, "Yo quiero ordenar una pizza", "food,order,pizza", true)
	Dataframe = BrainTrain(Dataframe, "Saludos hola que tal", "greeting", false)
	Dataframe = BrainTrain(Dataframe, "genial la comida", "liked", false)
	Dataframe = BrainTrain(Dataframe, "la comida dejo mucho que desear", "disliked", true)
	Dataframe = BrainTrain(Dataframe, "quiero un reembolso", "disliked", false)
	Dataframe = BrainTrain(Dataframe, "quiero refresco rojo", "food,order,soda",true)
	Dataframe = BrainTrain(Dataframe, "por favor quiero una gaseosa", "food,order,soda", true)
	Dataframe = BrainTrain(Dataframe, "quiero ordenar hamburguesa", "food,order,hamburger", true)
	Dataframe = BrainTrain(Dataframe, "estuvo buenísima la comida", "liked", true)
	Dataframe = BrainTrain(Dataframe, "quiero comer algo", "greeting", false)
	Dataframe = BrainTrain(Dataframe, "quiero ordenar una comida vegan", "food,order,salad", false)
	Dataframe = BrainTrain(Dataframe, "quiero ordenar una ensalada a la calle pequeña", "food,order,salad", true)
	Dataframe = BrainTrain(Dataframe, "me encantó la comida", "liked", false)

	return Dataframe
}