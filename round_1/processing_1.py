import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
import joblib

# Load dataset
data = pd.read_csv('../../data/ted_talks_cleaned.csv')

df = pd.DataFrame(data)

# Data Preprocessing
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

# Model Hyperparameter Grids
model_params = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {'model__fit_intercept': [True, False]}
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10, 20]}
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1]}
    },
    'SVR': {
        'model': SVR(),
        'params': {'model__C': [0.1, 1, 10], 'model__gamma': ['scale', 'auto']}
    },
    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {'model__max_depth': [None, 10, 20]}
    }
}


# Model Selection, Hyperparameter Tuning, and Saving Models
best_models = {}
results = []
for model_name, mp in model_params.items():
    pipeline = Pipeline(steps=[('transformer', transformer), ('model', mp['model'])])
    grid_search = GridSearchCV(pipeline, mp['params'], cv=5, scoring='neg_mean_squared_error', verbose=3)
    print(f"Training and tuning {model_name}...")
    with tqdm(total=1) as pbar:
        grid_search.fit(X_train_full, y_train_full)
        pbar.update(1)
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    print(f"{model_name} Best Score: {grid_search.best_score_}")
    results.append({'Model': model_name, 'Best Score': grid_search.best_score_})
    # Save the model
    joblib.dump(best_model, f'{model_name}_best_model.joblib')

# Convert results to DataFrame for comparison
results_df = pd.DataFrame(results)
print(results_df)

