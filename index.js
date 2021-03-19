const http = require('http');
const express = require('express');
var bodyParser = require('body-parser');
var moment = require('moment');
const {spawn} = require('child_process');
const path = require('path');
const Server = require('socket.io');

//const hostname ='192.168.42.53'
const hostname = '0.0.0.0'
const port = 3000;

app = express();
server = http.createServer(app);
var io = Server();
io.listen(server);


var dir = path.join('./public');

app.use(express.static(dir));
//app.use('*/images',express.static(dir));


app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

app.get('/predict', function (req, res) {
	var dataToSend;
    const python = spawn('python3', ['predict.py']);
    python.stdout.on('data', function (data) {
        console.log('Pipe data from python script ...');
        dataToSend = data.toString();
    });
    python.on('close', (code) => {
    console.log(`child process close all stdio with code ${code}`);
        res.send(dataToSend)
    });
})

server.listen(port, hostname, () => {
	console.log(`AI compressive strength server started`);
	console.log(`server url : http://${hostname}:${port}/`);
	console.log("started at " + moment().format('YYYY-MM-DD HH:mm Z') + '\n\n');
});