import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import string
import re

df = pd.read_csv('dataset.csv', encoding='latin1')
df.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'text']
X = df['text']
y = df['sentiment']

print(df.head())
print(f'\n{df.shape}')

print(f"\n\nX:\n\n{X.head()}\n\ny:\n\n{y.head()}\n")

X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=0.1)

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.fit_transform(X_test)
X_val_vec = vectorizer.fit_transform(X_val)

print(f"\n\nX_train_vec:\n\n{X_train_vec}\n\nX_test_vec:\n\n{X_test_vec}\n\nX_val_vec:\n\n{X_val_vec}\n")
print(vectorizer.fit_transform(X).shape)