import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

# Load the dataset
df = pd.read_csv("airlines_reviews.csv")

# Preprocess the data
df['sentiment'] = df['Overall Rating'].apply(lambda x: 1 if x > 5 else 0)  # Binary sentiment: 1 for positive, 0 for negative
X = df['Reviews'].tolist()
y = df['sentiment'].tolist()

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the input texts
vectorizer = CountVectorizer(max_features=10000)  # Limit the vocabulary size for faster processing
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict on the validation set
y_pred = model.predict(X_val_vec)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
precision, recall, fscore, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {fscore}")

# Save the model
joblib.dump(model, "sentiment_analysis_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved successfully.")
