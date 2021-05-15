# Neural Network Charity Analysis

## Overview
The purpose of this project was to create a neural network to help predict the success or failure of different charity projects that receive funding.

## Results
- _Data Preprocessing_
  - In the Dataset, the column/variable that was being predicted(targeted) was IS_SUCCESSFUL.
  - The columns EIN and NAME were removed as they had no bearing on the prediction.
  - Every other column/variable, aside from the ones removed above, are used as features. There are over 40, such as STATUS, ASK_AMT, with APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIATION, INCOME_AMT, and SPECIAL CONSIDERATIONS split into multiple columns with each  category receiving their own column for each possible value.
- _Compiling, Training, and Evaluating the Model_
  - 2 layers were used in this model at first with 80 neurons in the first hidden layer and 30 in the second hidden layer. The Relu activation function was used for the 2 hidden layers, with the Sigmoid activation function being used for the output layer. This is shown in the following picture.
  - ![Neural Network Model](https://raw.githubusercontent.com/jdwrhodes/Neural_Network_Charity_Analysis/main/Challege/resources/nn_model_shape.png 'Neural Network Model')
  - This first model was not successful in reaching the accuracy target of 75%.
  - 3 different attempts at optimizing were undertaken. 
     - First: An additional hidden layer was added with 50 neurons. This raised the accuracy percentage by 0.0001, or 0.01%.
     - Second: The bin size of Application Type's "Other" was changed from < 200 to < 50, thus increasing the number of total bins for Application Type. This also did not raise or lower the accuracy percentage.
     - Lastly: The bin size of Classification Type's "Other was changed from < 1800 to < 100, again increasing the number of total bins for Classification Type. This actually led to a decrease in accuracy, a loss of 0.0009, or 0.09%.  

  - ![Optimization Results](https://raw.githubusercontent.com/jdwrhodes/Neural_Network_Charity_Analysis/main/Challege/resources/optimization_results.png 'Optimizationi Results')

## Summary
Overall, the model, even with the optimization attempts, could not yield the 75% accuracy target, falling short at 73.07%. As there are quite a few negative values in the scaled dataset, one way to possibly improve the accuracy would be to use the Tanh activation function, rather than the Relu, to take into account the negative values.
