import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import pickle

def preprocess_images(image_dir, image_size=(128, 128)):
    X = []
    y = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"): 
            img_path = os.path.join(image_dir, filename)
            
            img = Image.open(img_path)
            img = img.resize(image_size)
            
            img_array = np.array(img).flatten()
            
            X.append(img_array)
            
            percentage = float(filename.split('_')[1].replace('.png', ''))  
            y.append(percentage)

    return np.array(X), np.array(y)

def train():
    image_directory = "dataset"  
    X, y = preprocess_images(image_directory)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64),  
                    activation='relu',  
                    solver='adam',  
                    learning_rate='adaptive',  
                    max_iter=500,  
                    random_state=42)

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

    with open("model.pkl", "wb") as f:
        pickle.dump(mlp, f)
    
            
    figure, axis = plt.subplots(2, 2)

    # Predicted vs Actual Values
    axis[0, 0].scatter(y_test, y_pred, color='blue', alpha=0.6)
    axis[0, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Perfect prediction line
    axis[0, 0].set_title('Predicted vs Actual Values')
    axis[0, 0].set_xlabel('Actual Percentage')
    axis[0, 0].set_ylabel('Predicted Percentage')
    axis[0, 0].grid(True)

    # Residuals (Errors) Plot
    residuals = y_test - y_pred
    axis[1, 0].scatter(y_pred, residuals, color='purple', alpha=0.6)
    axis[1, 0].axhline(y=0, color='r', linestyle='--')
    axis[1, 0].set_title('Residuals Plot')
    axis[1, 0].set_xlabel('Predicted Percentage')
    axis[1, 0].set_ylabel('Residuals (Actual - Predicted)')
    axis[1, 0].grid(True)

    # Histogram of Errors (Residuals)
    axis[1, 1].hist(residuals, bins=20, color='green', edgecolor='black', alpha=0.7)
    axis[1, 1].set_title('Distribution of Residuals')
    axis[1, 1].set_xlabel('Residual (Error)')
    axis[1, 1].set_ylabel('Frequency')
    axis[1, 1].grid(True)

    plt.show()


def main():
    if not os.path.exists("model.pkl"):
        train()
    else:
        with open('model.pkl', 'rb') as f:
            mlp = pickle.load(f)
            img = Image.open("koulutesti.png")
            img = img.resize((128,128))
            scaler = StandardScaler()
            tester_image = np.array(img).flatten()
            predictions = mlp.predict(scaler.fit_transform(np.array([tester_image])))
            print(f"Predicted percentages: {predictions}")

  
if __name__ == '__main__':
    main()