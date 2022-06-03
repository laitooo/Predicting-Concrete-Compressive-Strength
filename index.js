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
//app.use(timeout(15000));

additivesTypes = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5]
additivesSGs = [0, 1.18, 1.18, 1.17, 1.18, 1.17, 1.17, 1.17, 1.2, 1.2, 1.19, 1.205, 1.2, 1.2, 1.13, 1.2, 1.21]

function fixArguement(a, b) {
  if (a == undefined) return b;
  return a.length == 0 ? b : a;
}

function getADsolid(add_index, add_dos) {
  if (add_index == 0 ) return 0;
  return (additivesSGs[add_index] - 1) * add_dos;
}

function getNewWaterContent(add_dos, water_content) {
  return parseInt(water_content) +  parseInt(1 * add_dos);
}

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
            fixArguement(req.body.coarse_type, 1),
            fixArguement(req.body.fine_type, 1),
            fixArguement(req.body.max_size, 20),
            fixArguement(req.body.passing, 26),
            fixArguement(req.body.cement, 350),
            fixArguement(req.body.w_c, 0.5),
            getNewWaterContent(fixArguement(req.body.additive_dosage, 0), fixArguement(req.body.water, 175)),
            additivesTypes[fixArguement(req.body.additive_index, 0)],
            fixArguement(req.body.additive_dosage, 0),
            getADsolid(fixArguement(req.body.additive_index, 0), fixArguement(req.body.additive_dosage, 0)),
            fixArguement(req.body.fine, 850),
            fixArguement(req.body.coarse, 1200)
        ]
      };
    
    console.log(options.args);

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