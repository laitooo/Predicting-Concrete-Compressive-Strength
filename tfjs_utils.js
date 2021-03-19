const tfjs = require('@tensorflow/tfjs-node');

const testSample = [2.668, -114.333, -1.908, 4.786, 25.707, -45.21, 78, 0];

exports.loadTfModel = async function() {
    model_dir = process.cwd() + '/tfjs';    
    return await tfjs.loadLayersModel(model_dir);
}

exports.testDataSample = async function () {
    model_dir = 'file:/' + process.cwd() + '/tfjs'; 
    console.log(model_dir);   
    const tfModel = await tfjs.loadLayersModel(model_dir);
    tfModel.then(function (res) {
        const p = res.predict((tfjs.tensor(testSample, [1,testSample.length])).arraySync());
        console.log('****************');
        console.log(p);
    }, function (err) {
        console.log('*****************');
        console.log(err);
    });
    //return tfModel.predict((tfjs.tensor(testSample, [1,testSample.length])).arraySync());
}