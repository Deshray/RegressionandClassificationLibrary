import numpy as np
from python import LinearRegression, LogisticRegression

X = np.random.randn(100, 2)
y = 3 + X @ [1.5, -2.0] + np.random.randn(100)

model = LinearRegression(feature_names=['X1', 'X2'])
model.fit(X, y)
model.summary()