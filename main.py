import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import warnings 
warnings.filterwarnings('ignore') 
# --- 1. Load and Prepare the Data --- 
# Load the dataset 
# Ensure the 'spam.csv' file is in the same directory as this script. 
# The encoding 'latin-1' is often needed for this dataset. 
try: 
df = pd.read_csv('spam.csv', encoding='latin-1') 
except FileNotFoundError: 
print("Error: 'spam.csv' not found. Please download the dataset and place it in the correct 
directory.") 
exit() 
# We only need the first two columns, let's rename them for clarity 
df = df[['v1', 'v2']] 
df.columns = ['label', 'message'] 
# Map 'spam' and 'ham' to numerical values (1 for spam, 0 for ham) 
df['label'] = df['label'].map({'spam': 1, 'ham': 0}) 
# Separate features (X) and target (y) 
X = df['message'] 
y = df['label'] 
# --- 2. Split Data into Training and Testing Sets --- 
# Split the data into 80% for training and 20% for testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
print(f"Training data size: {len(X_train)}") 
print(f"Testing data size: {len(X_test)}") 
print("-" * 30) 
# --- 3. Feature Extraction (Convert Text to Numbers) --- 
# Create a CountVectorizer object 
# This will convert the text messages into a matrix of token counts (a "bag of words") 
vectorizer = CountVectorizer() 
# Fit the vectorizer on the training data and transform the training data 
X_train_vectors = vectorizer.fit_transform(X_train) 
# Only transform the test data using the already-fitted vectorizer 
X_test_vectors = vectorizer.transform(X_test) 
# --- 4. Train the Naive Bayes Classifier --- 
# Initialize the Multinomial Naive Bayes classifier 
model = MultinomialNB() 
# Train the model using the training data 
model.fit(X_train_vectors, y_train) 
print("Model trained successfully!") 
print("-" * 30) 
# --- 5. Evaluate the Model --- 
# Make predictions on the test data 
y_pred = model.predict(X_test_vectors) 
# Calculate accuracy 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy: {accuracy:.4f}") 
print("-" * 30) 
# Display the confusion matrix 
print("Confusion Matrix:") 
print(confusion_matrix(y_test, y_pred)) 
print("-" * 30) 
# Display the classification report (precision, recall, f1-score) 
print("Classification Report:") 
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])) 
print("-" * 30) 
# --- 6. Test with New Emails --- 
def classify_email(email_text): 
""" 
Classifies a single email text as spam or not spam. 
""" 
# Transform the new email text using the same vectorizer 
email_vector = vectorizer.transform([email_text]) 
# Make a prediction 
prediction = model.predict(email_vector) 
# Return the result 
return "Spam" if prediction[0] == 1 else "Not Spam (Ham)" 
# Example Usage 
email1 = "Congratulations! You've won a $1,000 Walmart gift card. Go to http://example.com to 
claim now." 
email2 = "Hey, are we still on for the meeting tomorrow at 2 PM? Let me know." 
email3 = "URGENT: Your account has been compromised. Please click here to verify your 
details immediately." 
print(f"Email: '{email1}'\nPrediction: {classify_email(email1)}\n") 
print(f"Email: '{email2}'\nPrediction: {classify_email(email2)}\n") 
print(f"Email: '{email3}'\nPrediction: {classify_email(email3)}\n")