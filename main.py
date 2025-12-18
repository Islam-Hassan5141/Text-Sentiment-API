import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from numpy import argmax

df = pd.read_csv('dataset.csv', encoding='latin1')
df.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'text']
X = df['text']
y = df['sentiment'] / 2

X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=0.1)

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
X_val_vec = vectorizer.transform(X_val)

print(X_train_vec.shape)

model = Sequential([
    Dense(128, activation='relu'),
    Dense(160, activation='relu'),
    Dense(160, activation='relu'),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_vec, y_train, epochs=10, batch_size=32, validation_data=(X_val_vec, y_val), verbose=1)

example = vectorizer.transform([input("Input prediction: ")])
prediction = model.predict(example)
labels = ["negative", "neutral", "positive"]

print(prediction)
print(labels[argmax(prediction)])