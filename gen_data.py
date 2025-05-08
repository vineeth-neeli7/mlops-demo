# gen_data.py
import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=4, random_state=42)
df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
df['target'] = y
df.to_csv('data/data.csv', index=False)
