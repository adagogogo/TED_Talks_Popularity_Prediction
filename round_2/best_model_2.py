import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('../../data/ted_talks_cleaned.csv')

# Data Preprocessing
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

categorical_features = ['speaker', 'title', 'tag', 'year', 'month']
numeric_features = ['length']

# One-Hot Encoding Transformer
transformer = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Splitting Data into Train and Test
X = df.drop('popularity', axis=1)
y = df['popularity']
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the best model
best_model = joblib.load('SVR_best_model.joblib')

# Train the best model on the full training dataset
best_model.fit(X_train_full, y_train_full)

# Final Evaluation on Test Set
y_test_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

# Save the evaluation results to a text file with UTF-8 encoding
with open('evaluation_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Test - Mean Squared Error: {mse}\n")
    f.write(f"Test - R² Score: {r2}\n")

print(f"Test - Mean Squared Error: {mean_squared_error(y_test, y_test_pred)}")
print(f"Test - R² Score: {r2_score(y_test, y_test_pred)}")
print("Evaluation results saved to 'evaluation_results.txt'.")