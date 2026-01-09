import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Path to CSV (same folder as this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "pima-indians-diabetes.csv")

# 1️⃣ Read CSV as raw (it is currently ONE column)
df = pd.read_csv(CSV_PATH, header=None)

print("Before split shape:", df.shape)
print(df.head())

# 2️⃣ FORCE split by comma into 9 columns
df = df[0].astype(str).str.strip().str.split(",", expand=True)

print("After split shape:", df.shape)
print(df.head())

# 3️⃣ Convert to numbers safely
df = df.apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)

dataset = df.to_numpy(dtype=float)

# 4️⃣ Split inputs and output
X = dataset[:, 0:8]
Y = dataset[:, 8]

# 5️⃣ Build model
model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# 6️⃣ Train
model.fit(X, Y, epochs=20, batch_size=10, verbose=1)

# 7️⃣ Evaluate
_, accuracy = model.evaluate(X, Y)
print(f"Accuracy: {accuracy * 100:.2f}%")