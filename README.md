# Predicting-Concrete-Compressive-Strength
Using Machine learning to predict the compressive strength of a concrete mixture

used node.js as server

main.py is used for trainning the model and printing simple results
it contains :
- SAVE_GRAPH : if you want to save the graph set to true
- SAVE_WEIGHTS : if you want to save only the weights of the model set to true
- SAVE_MODEL : if you want to save the whole model set to true
- LOAD_WEIGHTS : if you want to load an already saved weights set to true
- NUM_EPOCHS : the number of trainning epochs

data.py is used for getting data from excel file and converted to numpy array then shuffeled 
and divided into input/output and trainning/test

test.py is used for trainning and testing 2 models
- when passing seive 0.6% mm and max concrete size is available
- when they are missing
it saves results into excel file to make it easier when analysing the difference

index.js is the nodejs server it serves the python model to the public/index.html page 

data.xlsx contains simple complete data for testing the model

missing_data folder contains sample of data collected in 2019 also
it contains complete.py whice is used to complete the missing data using 3 method:
- Treatment by Replacement with mean
- Treatment by Replacement with mode (most appearing value)
- Treatment by Predictive Imputation (linear regression)
- Using algorithms that work with missing values (not done yet)

the result of the completed data will be found as excel file with the method name
#note:
some data columns were deleted (incomplete/strings/output)