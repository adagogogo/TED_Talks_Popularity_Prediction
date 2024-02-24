import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time

# Start time
start_time = time.time()

# Load dataset
data = pd.read_csv('../../data/ted_talks_cleaned.csv')
df = pd.DataFrame(data)

# Data Preprocessing
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Sentiment Analysis for 'title'
def calculate_sentiment(text):
    return TextBlob(text).sentiment

df['title_sentiment_polarity'] = df['title'].apply(lambda x: calculate_sentiment(x).polarity)
df['title_sentiment_subjectivity'] = df['title'].apply(lambda x: calculate_sentiment(x).subjectivity)

# TF-IDF Vectorization for 'title' and 'tag'
tfidf_vectorizer_title = TfidfVectorizer(max_features=100)  # Adjust max_features as needed
tfidf_vectorizer_tag = TfidfVectorizer(max_features=100)   # Adjust max_features as needed

# ColumnTransformer with TF-IDF for 'title' and 'tag'
transformer = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['length', 'title_sentiment_polarity', 'title_sentiment_subjectivity']),
    ('title_tfidf', tfidf_vectorizer_title, 'title'),
    ('tag_tfidf', tfidf_vectorizer_tag, 'tag'),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['speaker', 'year', 'month'])
])

# Splitting Data into Train and Test
X = df.drop('popularity', axis=1)
y = df['popularity']
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the best model (update the model name if necessary)
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

# End time
end_time = time.time()

# Calculate total running time
total_time = end_time - start_time
total_time_minutes = total_time / 60
print(f"Total running time: {total_time_minutes:.2f} minutes")
