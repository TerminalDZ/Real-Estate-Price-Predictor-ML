# Real Estate Price Prediction

## Overview
This project is a simple machine learning application designed to predict real estate prices based on the area of the property. It leverages a linear regression model to make predictions and provides a graphical user interface (GUI) using tkinter for user interaction. The data is stored in a JSON file and is updated with new entries provided by the user.

## Features
- Load and save data from/to a JSON file.
- Train a linear regression model to predict prices.
- Evaluate the model's performance.
- Predict prices for new data.
- Provide a GUI for users to input new data and review predicted prices.
- Update the dataset with new user-provided data.
- Visualize actual and predicted prices with a scatter plot.

## Prerequisites
- Python 3.x
- Required Python packages: pandas, scikit-learn, matplotlib, tkinter

## Installation

### Clone the Repository
```
git clone https://github.com/TerminalDZ/Real-Estate-Price-Predictor-ML.git
cd Real-Estate-Price-Predictor-ML
```

### Install Required Packages

Using requirements.txt:

1. Save the required packages in a requirements.txt file:
```
pandas
scikit-learn
matplotlib
tkinter
```

2. Install the packages:
```
pip install -r requirements.txt
```

Alternatively, you can install the packages directly using pip:

```
pip install pandas scikit-learn matplotlib tkinter
```

## Running the Application
1. Save the provided script as `real_estate_price_prediction.py` if not already done.
2. Open a terminal or command prompt and navigate to the directory where the script is saved.
3. Run the script:
```
python real_estate_price_prediction.py
```

## Using the Application
1. When you run the script, a GUI will appear asking for the name of the home and its area in square meters.
2. After providing the inputs, the application will show a predicted price for the home.
3. If you are satisfied with the predicted price, you can accept it. If not, you can enter your own price.
4. The new data will be saved to the JSON file and used for future predictions.

## How It Works

### Loading Data:
- The application checks if a JSON file (data.json) exists.
- If it does, it loads the data from the file.
- If it does not, it initializes the dataset with some predefined values and saves it to the JSON file.

### Training the Model:
- The data is split into training and testing sets.
- A linear regression model is trained using the training set.

### Evaluating the Model:
- The model's performance is evaluated using the testing set.
- The R^2 score is printed to the console.

### Predicting Prices:
- When new data is input, the model predicts the price based on the area provided.
- The predicted price is shown to the user, who can accept it or provide their own price.

### Updating Data:
- The new data (either the predicted price or the user-provided price) is added to the dataset.
- The updated dataset is saved to the JSON file for future use.

### Plotting Results:
- The actual and predicted prices are plotted on a scatter plot for visualization.

## Benefits

### Hands-On Learning:
- This project provides a practical example of how to use machine learning to make predictions.
- Users can learn how to load and save data, train a model, evaluate its performance, and use it to make predictions.

### User Interaction:
- The use of a GUI makes the application more user-friendly and accessible.
- Users can easily input new data and review predictions without needing to interact with the terminal.

### Data Persistence:
- By saving data to a JSON file, the application ensures that new data is retained between sessions.

## Lessons Learned

### Machine Learning Basics:
- Understand how to use linear regression for simple predictions.
- Learn how to evaluate a model's performance using the R^2 score.

### Data Handling:
- Learn how to load and save data using JSON files.
- Understand how to manipulate data using pandas DataFrames.

### GUI Development:
- Gain experience in creating simple graphical user interfaces with tkinter.

### Model Deployment:
- See how a trained model can be used to make predictions on new data and update itself with new information.

This project is an excellent starting point for anyone interested in machine learning and real estate price prediction. It combines data handling, model training, and user interaction in a cohesive and practical application.
