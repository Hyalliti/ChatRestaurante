package main

import (
	"log"
	"net/http"
	"strings"
	socketio "github.com/googollee/go-socket.io"
)

func main() {

	// Dataframe Dataset : Where the creation and implementation of the classifier + training of the Neural Network happens:
	Dataframe := initialize()

	// Server Code
	server := socketio.NewServer(nil)

	//sockets
	// Une al usuario al chatroom al conectarse
	server.OnConnect("/", func(so socketio.Conn) error {
		so.SetContext("")
		so.Join("chat_room")
		PrintText("Usuario Conectado", 1)
		return nil
	})

	// Aquí presentamos en un log el mensaje , aca lo podemos manipular con el resto del codigo conocido.
	// Emision del Servidor...
	// Retransmite el msg a usuari@s dentro del chat_room
	server.OnEvent("/", "user input", func(so socketio.Conn, msg string) {
		if len(strings.TrimSpace(msg)) >= 2 {
			Result := ExecuteTrainedNetwork(Dataframe, msg)
			EmitTo(server, so, "chat_room", "user input", msg, Result)
			log.Print(msg)
			log.Print(Result)
		}
	})

	go server.Serve()
	defer server.Close()

	//Modulo Http
	portNumber := "4000"
	http.Handle("/socket.io/", server)
	http.Handle("/", http.FileServer(http.Dir("./public")))
	log.Println("Server on Port " + portNumber)

	log.Fatal(http.ListenAndServe(":" + portNumber, nil))
}

// Function used to Emit/Broadcast to a socket a message and the neural network 
// output Processed Message.
// Returns void but it shows the message on the chat Room  ("chat_room").
func EmitTo(server *socketio.Server, so socketio.Conn, room string, event_origin string, InitialMessage string, ProcessedMessage string) {
	so.Emit(event_origin, "🧍: "+InitialMessage)
	server.BroadcastToRoom(room, event_origin, "🧍: "+InitialMessage)

	so.Emit(event_origin, "🤖: "+ ProcessedMessage)
	server.BroadcastToRoom(room, event_origin, "🤖: "+ ProcessedMessage)
}
