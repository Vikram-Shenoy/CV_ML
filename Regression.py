import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Example of another model you can import:
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# Example of another metric you can import:
# from sklearn.metrics import r2_score

# --- 1. Load the Data ---
# Load input features
try:
    X = pd.read_csv('inputs.csv')
except FileNotFoundError:
    print("Error: 'inputs.csv' not found. Please make sure the file is in the correct directory.")
    exit()

# Load target variables
try:
    y = pd.read_csv('targets.csv')
except FileNotFoundError:
    print("Error: 'targets.csv' not found. Please make sure the file is in the correct directory.")
    exit()
# Option 2: If you know the column POSITION (e.g., 0 for the first column)
# Uncomment the lines below if you want to use this method instead of Option 1.
# Make sure to comment out or remove Option 1 if you use Option 2.
serial_column_index = 0 # <--- !!! IMPORTANT: SET THIS if it's the first column !!!
if X.shape[1] > serial_column_index: # Check if the column index exists
    column_to_drop = X.columns[serial_column_index]
    X = X.drop(columns=[column_to_drop])
    # Alternatively, using iloc to drop by position:
    # X = X.iloc[:, serial_column_index+1:] # If it's the first column and you want all others
    # Or more generally to drop a column at a specific index:
    # X = X.drop(X.columns[serial_column_index], axis=1)
    print(f"\nDropped column at index {serial_column_index} (named '{column_to_drop}') from input features X.")
else:
    print(f"\nWarning: Column index {serial_column_index} is out of bounds for inputs.csv.")

# --- End of MODIFICATION ---



# Display the first few rows and info to verify (optional, but good practice)
print("--- Input Features (X) Head ---")
print(X.head())
print("\n--- Input Features (X) Info ---")
X.info()
print("\n--- Target Variables (y) Head ---")
print(y.head())
print("\n--- Target Variables (y) Info ---")
y.info()

# Basic check for data consistency (optional)
if len(X) != len(y):
    print(f"Error: The number of rows in inputs.csv ({len(X)}) and targets.csv ({len(y)}) do not match.")
    exit()

if X.isnull().values.any() or y.isnull().values.any():
    print("\nWarning: Missing values detected in your CSV files. Consider handling them (e.g., imputation or removal).")


# --- 2. Split the Data ---
# Split data into training and testing sets (70% train, 30% test)
# random_state is used for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Testing set size: X_test {X_test.shape}, y_test {y_test.shape}")

# --- 3. Initialize and Train the Model ---
# Initialize the model
# You can swap LinearRegression() with another model.
# For example, to use RandomForestRegressor:
# model = RandomForestRegressor(n_estimators=100, random_state=42)
model = LinearRegression()

# Train the model
# Scikit-learn's linear regression and many other models directly support multi-output regression
# if 'y' has multiple columns.
model.fit(X_train, y_train)

print(f"\nModel trained: {model}")

# --- 4. Make Predictions ---
y_pred = model.predict(X_test)

# --- 5. Evaluate the Model ---
# Calculate Mean Squared Error
# You can swap mean_squared_error with another metric.
# For example, to use R-squared score:
# r2 = r2_score(y_test, y_pred, multioutput='uniform_average') # or 'raw_values'
# print(f"R-squared Score: {r2}")

# For multi-output regression, mean_squared_error returns a single value by default (average of MSEs for each target)
# If you want MSE for each output variable, you can specify multioutput='raw_values'
mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # Default, averages MSEs
mse_raw = mean_squared_error(y_test, y_pred, multioutput='raw_values') # MSE for each output

print(f"\nMean Squared Error (average over outputs): {mse}")
print(f"Mean Squared Error (for each output {y.columns.to_list()}): {mse_raw}")

# You can also calculate MSE for each target variable individually if needed:
# print("\n--- MSE for each target variable ---")
# for i, target_name in enumerate(y.columns):
#     mse_target = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
#     print(f"MSE for {target_name}: {mse_target}")

print("\nScript finished.")