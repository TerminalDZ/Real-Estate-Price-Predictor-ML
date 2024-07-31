import pandas as pd
import json
import os
import warnings
from tkinter import Tk, Toplevel, ttk, simpledialog, messagebox
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import UndefinedMetricWarning

DATA_FILE = 'data.json'

# Load data
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as file:
            data = json.load(file)
    else:
        data = [
            {'name': 'home 1', 'area_m2': 34, 'price': 15000, 'rooms': 1, 'bathrooms': 1, 'location': 'A', 'age': 5},
            {'name': 'home 2', 'area_m2': 50, 'price': 13000, 'rooms': 2, 'bathrooms': 1, 'location': 'B', 'age': 10},
            {'name': 'home 3', 'area_m2': 67, 'price': 22000, 'rooms': 2, 'bathrooms': 1, 'location': 'A', 'age': 3},
            {'name': 'home 4', 'area_m2': 32, 'price': 26000, 'rooms': 3, 'bathrooms': 1, 'location': 'C', 'age': 15},
            {'name': 'home 5', 'area_m2': 43, 'price': 35000, 'rooms': 3, 'bathrooms': 2, 'location': 'B', 'age': 7},
            {'name': 'home 6', 'area_m2': 80, 'price': 30000, 'rooms': 3, 'bathrooms': 2, 'location': 'A', 'age': 20},
            {'name': 'home 7', 'area_m2': 100, 'price': 50000, 'rooms': 4, 'bathrooms': 2, 'location': 'C', 'age': 1},
            {'name': 'home 8', 'area_m2': 120, 'price': 60000, 'rooms': 4, 'bathrooms': 3, 'location': 'B', 'age': 3},
            {'name': 'home 9', 'area_m2': 150, 'price': 75000, 'rooms': 5, 'bathrooms': 3, 'location': 'A', 'age': 5},
            {'name': 'home 10', 'area_m2': 200, 'price': 80000, 'rooms': 5, 'bathrooms': 4, 'location': 'C', 'age': 2}

            # Add more initial data if needed
        ]
        save_data(data)
    df = pd.DataFrame(data)
    return df

# Save data
def save_data(data):
    with open(DATA_FILE, 'w') as file:
        json.dump(data, file, indent=4)

# Data validation
def validate_data(entry):
    try:
        if not entry['name']:
            raise ValueError("Name cannot be empty")
        if not (0 < entry['area_m2'] < 10000):
            raise ValueError("Area must be between 1 and 10000 m²")
        if not (0 <= entry['price'] < 10000000):
            raise ValueError("Price must be between 0 and 10,000,000 DZD")
        if not (0 <= entry['rooms'] < 100):
            raise ValueError("Rooms must be between 0 and 100")
        if not (0 <= entry['bathrooms'] < 100):
            raise ValueError("Bathrooms must be between 0 and 100")
        if not (0 <= entry['age'] < 200):
            raise ValueError("Age must be between 0 and 200 years")
        return True
    except ValueError as e:
        messagebox.showerror("Data Validation Error", str(e))
        return False

# Split data
def split_data(data):
    X = data[['area_m2', 'rooms', 'bathrooms', 'location', 'age']]
    X = pd.get_dummies(X, columns=['location'])
    y = data['price']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    if len(X_test) > 1:
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Mean Squared Error: {mse:.2f}")
    else:
        warnings.warn("Not enough samples in the test set to compute a reliable score.", UndefinedMetricWarning)

# Display model info
def display_model_info(model):
    print("Random Forest Model Trained")

# Predict prices
def predict_prices(model, new_data):
    new_data = pd.get_dummies(new_data, columns=['location'])
    missing_cols = set(X_train.columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0
    new_data = new_data[X_train.columns]
    predicted_prices = model.predict(new_data)
    return predicted_prices

# Plot results
def plot_results(data, new_data=None, predicted_prices=None):
    plt.clf()
    plt.scatter(data['area_m2'], data['price'], color='blue', label='Actual Prices')
    if new_data is not None and predicted_prices is not None:
        plt.scatter(new_data['area_m2'], predicted_prices, color='red', label='Predicted Prices')
    plt.xlabel('Area (m²)')
    plt.ylabel('Price (DZD)')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.draw()
    plt.pause(0.001)

# Add new home
def add_new_home():
    name = simpledialog.askstring("Input", "Enter the name of the home:")
    if name is None:
        return
    area_m2 = simpledialog.askfloat("Input", "Enter the area in square meters (m²):")
    if area_m2 is None:
        return
    rooms = simpledialog.askinteger("Input", "Enter the number of rooms:")
    if rooms is None:
        return
    bathrooms = simpledialog.askinteger("Input", "Enter the number of bathrooms:")
    if bathrooms is None:
        return
    location = simpledialog.askstring("Input", "Enter the location (A, B, C, etc.):")
    if location is None:
        return
    age = simpledialog.askinteger("Input", "Enter the age of the property:")
    if age is None:
        return

    new_data_df = pd.DataFrame([{'name': name, 'area_m2': area_m2, 'rooms': rooms, 'bathrooms': bathrooms, 'location': location, 'age': age}])
    predicted_prices = predict_prices(model, new_data_df[['area_m2', 'rooms', 'bathrooms', 'location', 'age']])
    predicted_price = predicted_prices[0]

    msg = f"Predicted Price for {name}: {predicted_price:.2f} DZD"
    user_price = simpledialog.askfloat("Input", msg + "\nIf you are satisfied with the predicted price, click Cancel.\nOtherwise, enter your own price (DZD):")

    if user_price is None:
        price = predicted_price
    else:
        price = user_price

    new_entry = {'name': name, 'area_m2': area_m2, 'rooms': rooms, 'bathrooms': bathrooms, 'price': price, 'location': location, 'age': age, 'predicted_price': predicted_price}
    if validate_data(new_entry):
        data.append(new_entry)
        save_data(data)
        refresh_ui()

# Refresh UI
def refresh_ui():
    global data, model, X_train

    data_df = pd.DataFrame(data)
    tree.delete(*tree.get_children())
    for entry in data:
        tree.insert('', 'end', values=(entry['name'], entry['area_m2'], entry['rooms'], entry['bathrooms'], entry['price'], entry.get('predicted_price', '')))

    X_train, X_test, y_train, y_test = split_data(data_df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    display_model_info(model)

    plot_results(data_df)

# Event handler for selection
def on_select(event):
    selected_item = tree.selection()
    if selected_item:
        selected_data = data[tree.index(selected_item[0])]
        selected_df = pd.DataFrame([selected_data])
        predicted_prices = predict_prices(model, selected_df[['area_m2', 'rooms', 'bathrooms', 'location', 'age']])
        plot_results(pd.DataFrame(data), selected_df, predicted_prices)

# Open settings
def open_settings():
    settings_window = Toplevel(root)
    settings_window.title("Settings")
    settings_window.geometry("300x200")
    ttk.Label(settings_window, text="Settings placeholder").pack()

# Main function
def main():
    global data, model, root, tree, X_train

    data = load_data().to_dict('records')

    root = Tk()
    root.title("Real Estate Price Prediction")
    root.geometry("1000x700")

    tab_control = ttk.Notebook(root)
    tab_control.pack(expand=1, fill="both")

    tab_data = ttk.Frame(tab_control)
    tab_control.add(tab_data, text='Data & Predictions')

    tree = ttk.Treeview(tab_data, columns=('Name', 'Area (m²)', 'Rooms', 'Bathrooms', 'Price', 'Predicted Price'), show='headings')
    tree.heading('Name', text='Name')
    tree.heading('Area (m²)', text='Area (m²)')
    tree.heading('Rooms', text='Rooms')
    tree.heading('Bathrooms', text='Bathrooms')
    tree.heading('Price', text='Price (DZD)')
    tree.heading('Predicted Price', text='Predicted Price (DZD)')
    tree.pack(fill="both", expand=True)
    tree.bind('<<TreeviewSelect>>', on_select)

    button_add = ttk.Button(tab_data, text="Add New Home", command=add_new_home)
    button_add.pack(pady=10)

    tab_settings = ttk.Frame(tab_control)
    tab_control.add(tab_settings, text='Settings')

    button_settings = ttk.Button(tab_settings, text="Open Settings", command=open_settings)
    button_settings.pack(pady=20)

    refresh_ui()
    root.mainloop()

if __name__ == "__main__":
    main()