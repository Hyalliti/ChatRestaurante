package main

import (
	"github.com/googollee/go-socket.io"
	"log"
	"net/http"
)

func main() {

	var Dataframe Dataset
	Dataframe = initialize()

	// Server Code
	server := socketio.NewServer(nil)

	//sockets
	// Une al usuario al chatroom al conectarse
	server.OnConnect("/", func(so socketio.Conn) error {
		so.SetContext("")
		so.Join("chat_room")
		PrintText("Usuario Conectado",1)
		return nil
	})

	// Aquí presentamos en un log el mensaje , aca lo podemos manipular con el resto del codigo conocido.
	// Emision del Servidor... 
	// Retransmite el msg a usuari@s dentro del chat_room
	server.OnEvent("/", "user input", func(so socketio.Conn, msg string){
		Dataframe = BrainTrain(Dataframe, msg , "" , false)
		Prediction := Dataframe.DATA2_NeuralPrediction
		log.Print(msg) 
		EmitTo(server, so, "chat_room", "user input", msg , Prediction)
	})	

	go server.Serve()
	defer server.Close()

	//Modulo Http
	http.Handle("/socket.io/", server)
	http.Handle("/", http.FileServer(http.Dir("./public")))
	log.Println("Server on Port 3000")
	log.Fatal(http.ListenAndServe(":3000", nil))
}

// Function used to Emit/Broadcast to a socket a message and the neural network output Processed Message
func EmitTo(server *socketio.Server,so socketio.Conn, room string, event_origin string, InitialMessage string, ProcessedMessage string ){
	so.Emit(event_origin,InitialMessage) 
	server.BroadcastToRoom(room, event_origin, InitialMessage) 
	so.Emit(event_origin,ProcessedMessage) 
	server.BroadcastToRoom(room, event_origin, ProcessedMessage) 
}