import pandas as pd

X = pd.read_csv('inputs.csv')



# (This part would be after loading X and potentially dropping the serial column)
print("\n--- Missing Values Report for Input Features (X) ---")
missing_values_count = X.isnull().sum()
print("Missing values per column:\n", missing_values_count[missing_values_count > 0])
total_missing = missing_values_count.sum()
total_cells = X.shape[0] * X.shape[1]
percentage_missing = (total_missing / total_cells) * 100
print(f"Total missing values in X: {total_missing}")
print(f"Percentage of missing data in X: {percentage_missing:.2f}%")