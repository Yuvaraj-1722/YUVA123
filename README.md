import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection 
import train_test_split from sklearn.feature_extraction.text 
import TfidfVectorizer from sklearn.linear_model 
import PassiveAggressiveClassifier from sklearn.metrics 
import accuracy_score, confusion_matrix
📥 Load Dataset
try: df = pd.read_csv('news.csv') # Ensure 'news.csv' is in the same directory print("✅ Dataset loaded successfully.") except FileNotFoundError: print("❌ Error: 'news.csv' file not found.") exit()
print("🔢 Shape of dataset:", df.shape) print("📄 First few rows:\n", df.head())
🔍 Data Inspection
print("\n🧹 Missing values per column:") 
print(df.isnull().sum())
print("\n📊 Label distribution:") print(df['label'].value_counts())
🧼 Remove duplicates
df.drop_duplicates(inplace=True)
✅ Ensure no missing text data
df = df.dropna(subset=['text'])
🎯 Feature and Label Separation
X = df['text'] y = df['label']
🧪 Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
🔠 TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) 
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
tfidf_test = tfidf_vectorizer.transform(X_test)
🤖 Train PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50) pac.fit(tfidf_train, y_train)
📈 Predictions and Evaluation
y_pred = pac.predict(tfidf_test) score = accuracy_score(y_test, y_pred) print(f"\n✅ Model Accuracy: {round(score * 100, 2)}%")
🧾 Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']) print("\n🧾 Confusion Matrix:") print(conf_matrix)
📊 Visualize Confusion Matrix
plt.figure(figsize=(6,4)) 
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL']) 
plt.xlabel('Predicted') 
plt.ylabel('Actual') 
plt.title('📰 Fake News Detection - Confusion Matrix') 
plt.tight_layout() ]
plt.show()
