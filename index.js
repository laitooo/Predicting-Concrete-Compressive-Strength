const http = require('http');
const express = require('express');
var bodyParser = require('body-parser');
var moment = require('moment');
const {spawn} = require('child_process');
const path = require('path');
const Server = require('socket.io');
let {PythonShell} = require('python-shell')

//const hostname ='192.168.42.53'
const hostname = '0.0.0.0'
const port = 3000;

app = express();
server = http.createServer(app);
var io = Server();
io.listen(server);


var dir = path.join(__dirname + '/public/');
console.log(dir);
//app.use(express.static(dir));
app.use(express.static(__dirname + '/public'));
//app.use('*/images',express.static(dir));


app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.get("/", function (req, res) {
  res.sendFile(path.join(__dirname, "/public/test.html"));
});

app.get('/predict', function (req, res) {
    // 360, 0.45, 160, 1880, 715, 1165
    let options = {
        mode: 'text',
        pythonOptions: ['-u'], // get print results in real-time
        scriptPath: '.',
        args: [360, 0.45, 160, 1880, 715, 1165]
      };

    PythonShell.run('predict.py', options, function (err, results) {
        if (err) throw err;
        // results is an array consisting of messages collected during execution
        res.send(results)
      });
})

app.post('/predict2', function (req, res) {    
    let options = {
        mode: 'text',
        pythonOptions: ['-u'], // get print results in real-time
        scriptPath: '.',
        args: [
          req.body.coarse_type,
          req.body.fine_type,
          req.body.max_size,
          req.body.passing,
          req.body.target_mean,
          req.body.grade,
          req.body.cement,
          req.body.w_c,
          req.body.water,
          req.body.additive,
          req.body.fine,
          req.body.coarse]
      };

    PythonShell.run('predict.py', options, function (err, results) {
        if (err) throw err;
        // results is an array consisting of messages collected during execution
        res.send(results)
      });
})

server.listen(port, hostname, () => {
	console.log(`AI compressive strength server started`);
	console.log(`server url : http://${hostname}:${port}/`);
	console.log("started at " + moment().format('YYYY-MM-DD HH:mm Z') + '\n\n');
});