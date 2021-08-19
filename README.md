# Predicting-Concrete-Compressive-Strength (Sudan)
Using Machine learning to predict the compressive strength of a concrete mixture.
Data used on this project is collected from university of khartoum - materials lab.
The output contains strength after 7 days , strength after 28 days and slump.

used node.js as server

main.py is used for trainning the model and printing simple results
it contains :
- SAVE_GRAPH : if you want to save the graph set to true
- SAVE_WEIGHTS : if you want to save only the weights of the model set to true
- SAVE_MODEL : if you want to save the whole model set to true
- LOAD_WEIGHTS : if you want to load an already saved weights set to true
- NUM_EPOCHS : the number of trainning epochs

data.py is used for getting data from excel file and converted to numpy array then shuffeled 
and divided into input/output and trainning/test and perpare and save data into excel file

test.py is used for trainning and testing 2 models
- when passing seive 0.6% mm and max concrete size is available
- when they are missing
it saves results into excel file to make it easier when analysing the difference

index.js is the nodejs server it serves the python model to the public/index.html page 

data.xlsx contains simple complete data for testing the model
data2.xlsx contains the final data for the project

missing_data folder contains sample of data collected in 2019 also
it contains complete.py whice is used to complete the missing data using 3 method:
- Treatment by Replacement with mean
- Treatment by Replacement with mode (most appearing value)
- Treatment by Predictive Imputation (linear regression)
- Using algorithms that work with missing values (not done yet)

#note:
some data columns were deleted (incomplete/strings/output)

utils.py is used for:
- checking if the data has n/a
- plot the data 
- initialize the model
- convert keras model to tensorflow model

visualize.py is used for plotting every column of the data each per time

final.py is supposed to train the model with the final data 
it containes: 
- NUM_EPOCHS : the number of trainning epochs
- SAVE_INPUT : to save the trainning and test data before starting the trainning
- TRAIN_MODEL_1 : to decide if you want to train the first model which contains (slump and hardened density) as an outpu
- TRAIN_MODEL_2 : to decide if you want to train the second model which contains (compressive strength after 7 days) as an output
- TRAIN_MODEL_3 : to decide if you want to train the third model which contains (compressive strength after 28 days) as an output

