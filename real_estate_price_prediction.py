import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import json
import os
import tkinter as tk
from tkinter import simpledialog, messagebox

DATA_FILE = 'data.json'

def load_data():
    """Load and prepare the dataset from a JSON file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as file:
            data = json.load(file)
    else:
        # Initial data if JSON file does not exist
        data = [
            {'name': 'home 1', 'area_m2': 34, 'price': 15000},
            {'name': 'home 2', 'area_m2': 50, 'price': 13000},
            {'name': 'home 3', 'area_m2': 67, 'price': 22000},
            {'name': 'home 4', 'area_m2': 32, 'price': 26000},
            {'name': 'home 5', 'area_m2': 43, 'price': 35000},
            {'name': 'home 6', 'area_m2': 80, 'price': 30000},
            {'name': 'home 7', 'area_m2': 100, 'price': 50000},
            {'name': 'home 8', 'area_m2': 120, 'price': 60000},
            {'name': 'home 9', 'area_m2': 150, 'price': 75000},
            {'name': 'home 10', 'area_m2': 200, 'price': 80000}
        ]
        save_data(data)
    df = pd.DataFrame(data)
    return df

def save_data(data):
    """Save the dataset to a JSON file."""
    with open(DATA_FILE, 'w') as file:
        json.dump(data, file, indent=4)

def split_data(data):
    """Split the data into training and testing sets."""
    X = data[['area_m2']]
    y = data['price']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """Train the Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    if len(X_test) > 1:
        score = model.score(X_test, y_test)
        print(f"Model Score: {score:.2f}")
    else:
        warnings.warn("Not enough samples in the test set to compute a reliable R^2 score.", UndefinedMetricWarning)

def display_model_info(model):
    """Display the model's coefficients and intercept."""
    print(f"Model Coefficients: {model.coef_}")
    print(f"Model Intercept: {model.intercept_}")

def predict_prices(model, new_data):
    """Predict prices for new data."""
    predicted_prices = model.predict(new_data)
    print(f"Predicted Prices: {predicted_prices} DZD")
    return predicted_prices

def plot_results(data, new_data, predicted_prices):
    """Plot the actual and predicted prices."""
    plt.scatter(data['area_m2'], data['price'], color='blue', label='Actual Prices')
    plt.scatter(new_data['area_m2'], predicted_prices, color='red', label='Predicted Prices')
    plt.xlabel('Area (m²)')
    plt.ylabel('Price (DZD)')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.show()

def input_new_data():
    """Input new data from the user and save it."""
    root = tk.Tk()
    root.withdraw()  

    name = simpledialog.askstring("Input", "Enter the name of the home:")
    area_m2 = simpledialog.askfloat("Input", "Enter the area in square meters (m²):")
    
    data = load_data().to_dict('records')
    
    new_data_df = pd.DataFrame([{'name': name, 'area_m2': area_m2}])
    predicted_prices = predict_prices(model, new_data_df[['area_m2']])
    predicted_price = predicted_prices[0]
    
    msg = f"Predicted Price for {name}: {predicted_price:.2f} DZD"
    user_price = simpledialog.askfloat("Input", msg + "\nIf you are satisfied with the predicted price, click Cancel.\nOtherwise, enter your own price (DZD):")
    
    if user_price is None:
        price = predicted_price
    else:
        price = user_price
    
    new_entry = {'name': name, 'area_m2': area_m2, 'price': price}
    data.append(new_entry)
    
    save_data(data)
    
    root.destroy()
    return pd.DataFrame([new_entry])

def main():
    global model  
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    display_model_info(model)
    
    new_data = input_new_data()
    predicted_prices = predict_prices(model, new_data[['area_m2']])
    plot_results(data, new_data[['area_m2']], predicted_prices)

if __name__ == "__main__":
    main()
