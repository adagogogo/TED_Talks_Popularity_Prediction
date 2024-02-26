# TED_Talks_Popularity_Prediction

## Project Overview
This project is centered around predicting the popularity of TED Talks. Utilizing a range of machine learning techniques, the aim is to uncover the key drivers behind the viewership and engagement levels of these talks. The project encompasses comprehensive data preprocessing and feature engineering, with a particular focus on advanced Natural Language Processing (NLP) methods. It also involves the deployment of various machine learning models to effectively predict TED Talks' popularity.

## Features
- Extensive data preprocessing including OneHotEncoding and StandardScaler.
- Advanced NLP feature extraction, such as TF-IDF vectorization and sentiment analysis, to capture the essence of TED Talks.
- Application of diverse machine learning models: Random Forest, Support Vector Classifier(SVC), Gradient Boosting, and Decision Tree.
- Rigorous model evaluation and hyperparameter tuning using GridSearchCV to optimize performance.

## Getting Started

### Prerequisites
- Python 3.x
- Pandas
- NumPy
- scikit-learn
- TextBlob
- tqdm
- joblib

## Repository Structure
The repository is organized into separate rounds, each containing its own processing script and the best-performing model from that round:

- `round_1/`
  - `round_1.py`: Data preprocessing, modeling and predicting for Round 1 using basic One-Hot Encoding.

- `round_2/`
  - `round_2.py`: Data preprocessing, modeling and predicting for Round 2, incorporating TF-IDF vectorization for 'title' and 'tag'.

- `round_3/`
  - `round_3.py`: Data preprocessing, modeling and predicting for Round 3, adding sentiment analysis for 'title'.

Each round represents an iteration of feature engineering and model tuning, progressively incorporating more complex techniques to enhance prediction accuracy.

## Acknowledgments
- TED Talks for providing a rich dataset for analysis.
- Community contributors and collaborators who have offered valuable insights and suggestions.


