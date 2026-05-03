from noshow_iq.preprocess import load_and_clean, get_features
from noshow_iq import model as m
import os

DATA_PATH = os.getenv("DATA_PATH", "data/KaggleV2-May-2016.csv")

df = load_and_clean(DATA_PATH)
X, y = get_features(df)

print(f"Dataset shape: {X.shape}, No-show rate: {y.mean():.2%}")
metrics = m.train(X, y)
print("Training complete!")
print(metrics)