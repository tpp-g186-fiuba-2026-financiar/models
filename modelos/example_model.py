import modal
import numpy as np
from sklearn.linear_model import LinearRegression

REG = None

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "scikit-learn", "numpy")
app = modal.App("example-model")


@app.function()
def train_model(X: np.ndarray, y: np.ndarray):
    global REG
    if REG is not None:
        return
    REG = LinearRegression().fit(X, y)
    return REG

@app.function(image=image)
@modal.fastapi_endpoint()
def main(x: int):
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    REG = train_model.local(X, y)
    return {"prediction": REG.predict(np.array([[x]]))[0]}