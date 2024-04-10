import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the dataset
Data = pd.read_csv(r"C:\Users\SHIVA KUMAR\Desktop\model_deployment\Regrerssion_energy_production_data (1).csv")

# Assuming the data preprocessing steps are done similarly as in your previous code
# ...
df = Data.copy()
# Features (X) and target variable (y)
X = df.iloc[:, 0:4]
y = df["energy_production"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

import pickle
pickle_out = open("model.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()
