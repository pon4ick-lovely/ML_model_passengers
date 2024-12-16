Airline Passengers Prediction using LSTM
This project aims to predict the number of international airline passengers using Long Short-Term Memory (LSTM) neural networks. The dataset used in this project is the Airline Passengers dataset available at https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv.

Getting Started 

1.Clone the repository: git clone https://github.com/yourusername/ML_model_passengers.git

2.Install the required libraries: pip install numpy pandas matplotlib sklearn tensorflow requests 

3.Run the main.py script: python main.py


Project Structure
The project structure is as follows:
ML_model_passengers/
│   main.py
│   README.md
│
└───data/
    │   airline-passengers.csv

Key Features

1.Data Preprocessing: The script loads the dataset, normalizes the data, and creates training and testing datasets.

2.LSTM Model: The script builds an LSTM model with two layers and compiles it using the Adam optimizer and mean squared error loss function. 

3.Model Training: The model is trained for 100 epochs on the training dataset. 

4.Prediction and Evaluation: The script makes predictions on both the training and testing datasets, calculates the mean absolute error (MAE) and mean squared error (MSE) metrics, and displays the results. 

5.Visualization: The script generates a plot comparing the original data, training predictions, and testing predictions.


Results
The LSTM model achieved the following performance metrics on the test dataset:

Mean Absolute Error (MAE): 23.67
Mean Squared Error (MSE): 704.67


The visualization of the predictions is shown below:

Airline Passengers Prediction

Future Work
Experiment with different LSTM architectures, hyperparameters, and feature engineering techniques to improve the model's performance.
Incorporate additional external data sources to enhance the predictive capabilities of the model.
Deploy the trained model as a web application or API for real-time predictions.


Acknowledgments
This project was inspired by the work of Jason Brownlee, available at https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/.

