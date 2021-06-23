const socket = io();

new Vue({
    el: '#chat-app',
    data: {
        message: '',
        messages: [{
            text: "Hello, welcome to the chatbot: ",
            date: new Date(),
        }]
    },
    created() {
        const vm = this;
        socket.on('user input', function (msg) {
            vm.messages.push({
                text: msg,
            })
        })
    },
    methods: {
        // Envia el mensaje para un evento "user input" especifico
        sendMessage() {
            socket.emit('user input', this.message); 
            this.message = ''; // Limpia el mensaje luego de enviarlo
        }
    }
});