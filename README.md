# Predicting-Concrete-Compressive-Strength (Sudan)

Graduation project from the university of Khartoum faculty of civil engineering

An artificial neural network model was developed using keras (python) to predict the compressive strength of concrete mixture after 7 and 28 days. and also predict the slump of concrete.

Data used on this project is collected from university of khartoum - materials lab.
The output contains strength after 7 days , strength after 28 days and slump.

The input:
- type of coarse aggrergate
- type of fine aggrergate
- maximum coarse aggregate size
- % passing seive 0.6mm 
- amount of cement
- amount of water
- water to cement ratio
- concrete admixture class
- concrete admixture dosage
- amount of coarse aggregate
- amount of fine aggregate

The output:
- compressive strength after 7 days
- compressive strength after 28 days
- fresh concrete slump

For running the latest model, the data should be similar to [additives data](https://github.com/laitooo/Predicting-Concrete-Compressive-Strength/blob/additives-data/data_files/additives2.xlsx).

To run the model you can run the [google colab](https://github.com/laitooo/Predicting-Concrete-Compressive-Strength/blob/main/ANN_Concrete.ipynb).

A website for testing the model was developed using html, css and javascript.
The server was developed useing node.js with express.
