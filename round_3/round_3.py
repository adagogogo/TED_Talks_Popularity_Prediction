import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm
import joblib
import time

# Start time
start_time = time.time()

# Load dataset
data = pd.read_csv('../../data/ted_talks_cleaned.csv')
df = pd.DataFrame(data)

# Function to calculate sentiment polarity
def calculate_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply function to calculate sentiment for 'title'
df['title_sentiment'] = df['title'].apply(calculate_sentiment)

# Data Preprocessing
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
text_features = ['title', 'tag']  # Textual features to vectorize
categorical_features = ['speaker', 'year', 'month']  # Other categorical features
numeric_features = ['length', 'title_sentiment']  # Include sentiment

# Column Transformer with TF-IDF for text and OneHotEncoder for other categorical features
transformer = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('title_tfidf', TfidfVectorizer(), 'title'),
    ('tag_tfidf', TfidfVectorizer(), 'tag')
])


# Splitting Data into Train and Test
X = df.drop(['popularity', 'date'], axis=1)
y = df['popularity']
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Hyperparameter Grids with class_weight parameter for handling imbalanced dataset
model_params = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'params': {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10, 20]}
    },
    'Gradient Boosting': {
        # Note: Gradient Boosting does not have a class_weight parameter
        'model': GradientBoostingClassifier(random_state=42),
        'params': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1]}
    },
    'SVC': {
        'model': SVC(class_weight='balanced'),
        'params': {'model__C': [0.1, 1, 10], 'model__gamma': ['scale', 'auto']}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'params': {'model__max_depth': [None, 10, 20]}
    }
}

# Model Selection, Hyperparameter Tuning, and Saving Models
best_models = {}
results = []
for model_name, mp in model_params.items():
    pipeline = Pipeline(steps=[('transformer', transformer), ('model', mp['model'])])
    grid_search = GridSearchCV(pipeline, mp['params'], cv=5, scoring='f1_weighted', verbose=3)
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

# Save the results to a CSV file
results_df.to_csv('model_evaluation_results.csv', index=False)
print("Model evaluation results have been saved to 'model_evaluation_results.csv'")

# Identify the best-performing model based on F1 score
best_model_name = max(results, key=lambda x: x['Best Score'])['Model']
best_model = best_models[best_model_name]
print(f"Best Model: {best_model_name}")

# Train the best model on the full training dataset
best_model.fit(X_train_full, y_train_full)

# Predict on the test data
y_pred = best_model.predict(X_test)

# Evaluate the model
evaluation_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Classification Report:")
print(evaluation_report)
print("Accuracy Score:", accuracy)

# Save the evaluation results to a CSV file
evaluation_results = {
    'Actual': y_test,
    'Predicted': y_pred
}
evaluation_df = pd.DataFrame(evaluation_results)
evaluation_df.to_csv('prediction_results.csv', index=False)
print("Prediction results have been saved to 'prediction_results.csv'")

# End time
end_time = time.time()

# Calculate total running time
total_time = end_time - start_time
total_time_minutes = total_time / 60
print(f"Total running time: {total_time_minutes:.2f} minutes")
