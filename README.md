# Real Estate Price Predictor

## Overview
This project is an advanced machine learning application designed to predict real estate prices based on various features of the property, including area, number of rooms, bathrooms, location, and age. It uses a Random Forest Regressor model to make predictions and provides a graphical user interface (GUI) using tkinter for user interaction. The data is stored in a JSON file and is updated with new entries provided by the user.

## Features
- Load and save data from/to a JSON file
- Train a Random Forest Regressor model to predict prices
- Evaluate the model's performance using Mean Squared Error
- Predict prices for new data
- Provide a GUI for users to input new data and review predicted prices
- Update the dataset with new user-provided data
- Visualize actual and predicted prices with a scatter plot
- Display detailed information about existing properties
- Data validation to ensure data integrity
- Handle categorical data (location) using one-hot encoding

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
```

2. Install the packages:
```
pip install -r requirements.txt
```

Note: tkinter usually comes pre-installed with Python. If it's not available, you may need to install it separately depending on your operating system.

## Running the Application
1. Ensure the script is saved as `real_estate_price_prediction.py`.
2. Open a terminal or command prompt and navigate to the project directory.
3. Run the script:
```
python real_estate_price_prediction.py
```

## Using the Application
1. When you run the script, a GUI will appear with two tabs: "Data & Predictions".
2. In the "Data & Predictions" tab, you can view existing property data and add new properties.
3. To add a new property, click the "Add New Home" button and follow the prompts to enter property details (name, area, rooms, bathrooms, location, and age).
4. The application will show a predicted price for the property, which you can accept or modify.
5. New data will be validated, saved to the JSON file, and used for future predictions.
6. Select a property in the list to see its actual and predicted prices plotted on the graph.

## How It Works

### Loading Data:
- The application checks for a JSON file (data.json) and loads data if it exists.
- If the file doesn't exist, it initializes the dataset with predefined values.

### Data Preprocessing:
- Categorical data (location) is handled using one-hot encoding.

### Training the Model:
- The data is split into training and testing sets.
- A Random Forest Regressor model is trained using the training set.

### Evaluating the Model:
- The model's performance is evaluated using the Mean Squared Error on the testing set.
- The MSE is printed to the console.

### Predicting Prices:
- When new data is input, the model predicts the price based on the provided features.
- The predicted price is shown to the user, who can accept it or provide their own price.

### Data Validation:
- New entries are validated to ensure they meet specific criteria (e.g., area range, price range).

### Updating Data:
- New data (either with predicted or user-provided prices) is added to the dataset after validation.
- The updated dataset is saved to the JSON file for future use.

### Plotting Results:
- The actual and predicted prices are plotted on a scatter plot for visualization.

## Project Structure
- `real_estate_price_prediction.py`: Main script containing all the code
- `data.json`: JSON file storing the property data
- `README.md`: This file, containing project documentation

