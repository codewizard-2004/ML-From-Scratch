# Logistic Regression from Scratch using NumPy

This project implements **logistic regression** from scratch using only NumPy. It trains a binary classification model on the **Breast Cancer Wisconsin dataset** to predict whether a tumor is malignant or benign. The project includes manual gradient descent optimization, sigmoid activation, and evaluation using scikit-learn utilities.

---
## 📚 Table of Contents

- [🔍 Problem Overview](#-problem-overview)
- [📦 Libraries Used](#-libraries-used)
- [⚙️ Key Components](#️-key-components)
- [📊 Results](#-results)
- [📁 File Structure](#-file-structure)
- [🚀 Future Enhancements](#-future-enhancements)

---

## 🔍 Problem Overview

We use logistic regression to solve a **binary classification problem** where each input $\mathbf{x} \in \mathbb{R}^n$. is mapped to a class $y \in \{0, 1\}$. Logistic regression estimates the probability that a given input belongs to the positive class (i.e.,  y = 1 ) using the sigmoid function:


$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$ where $z = \mathbf{x}^T \boldsymbol{\theta}$.

where:
- $\mathbf{x}$ is the feature vector
- $\boldsymbol{\theta}$ includes both weights and bias (via augmented vector)
- $\sigma(z)$ is the sigmoid function mapping logits to probabilities

The model is trained by minimizing the **binary cross-entropy loss**:

where:
- $J(\boldsymbol{\theta})$ is the cost function for logistic regression (cross-entropy loss)
- $m$ is the number of training examples
- $y^{(i)}$ is the true label for the $i^{th}$ example
- $\hat{y}^{(i)} = \sigma(\mathbf{x}^{(i)T}\boldsymbol{\theta})$ is the predicted probability
- $\log$ is the natural logarithm (base $e$)

The gradient of the loss with respect to parameters $\boldsymbol{\theta} = (\theta_0, \theta_1, \dots, \theta_n)^T$ is used to update weights via **gradient descent**.

---

## 📦 Libraries Used

### ✅ `numpy`
- Used for efficient vector and matrix operations
- Core library for computing predictions, gradients, and optimization steps

### ✅ `sklearn.datasets.load_breast_cancer`
- Loads the **Breast Cancer Wisconsin** dataset, which contains real-world features (e.g., tumor size, texture, smoothness) and a binary label (malignant or benign)

### ✅ `sklearn.preprocessing.StandardScaler`
- Standardizes input features to have zero mean and unit variance
- Important to ensure faster and more stable convergence during gradient descent

### ✅ `sklearn.model_selection.train_test_split`
- Splits the dataset into training and test sets to evaluate generalization

### ✅ `sklearn.metrics.accuracy_score`
- Measures the model's prediction performance on both training and test sets

---

## ⚙️ Key Components

### `sigmoid(z)`
Applies the sigmoid function element-wise:
```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
```
### `gradient(theta, X , y)`
Computes the gradient of the loss function w.r.t. parameters:

```python
def gradient(theta , X , y):
    m = y.size
    return (X.T @ (sigmoid(X @ theta) - y)) / m
```
### `gradient_descent(...)`
Performs iterative gradient descent until convergence or a stopping threshold:
```python
def gradient_descent(X , y , lr = 0.1 , num_iters = 100 , tol=1e-7):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])
    for i in range(num_iters):
         grad = gradient(theta , X_b , y)
         theta -= lr * grad
         if np.linalg.norm(grad) < tol:
              break
    return theta
```
## 📊 Results
After training the model on 80% of the data and testing on the remaining 20%, accuracy metrics are printed:

* Train accuracy: 0.9736263736263736
* Test accuracy: 0.9736842105263158