# 🧠 Neural Network from Scratch — Just NumPy

A simple yet powerful 2-layer Neural Network for classifying handwritten digits from the MNIST dataset.  
This project is built entirely from scratch using only **NumPy** and **pure mathematics**.  
No high-level frameworks, just the raw fundamentals of deep learning. 😎

---

## 🔍 Why Build from Scratch?

Because **AI isn't magic — it's applied mathematics.**

If you've ever asked:

- ❓ What exactly is backpropagation?
- ❓ How do weights and biases actually get updated?
- ❓ Why doesn't my softmax function explode with large inputs?

This repository gives you answers — **not with black boxes**, but with:
- ✅ Clean code  
- ✅ Clear formulas  
- ✅ Deep intuition  

---

## 🚀 Key Features

- 🔢 **Pure NumPy**: No TensorFlow, no PyTorch. Just raw matrix operations.
- 🔄 **Full Forward & Backward Propagation**: Implemented from the ground up.
- 📏 **Stable Softmax**: Numerical stability fix included.
- 💡 **Clear Comments**: Every line of code is explained with *why*, not just *what*.

---

## 🧠 Neural Network Architecture

### ✅ Dataset: MNIST
- 28x28 grayscale images of handwritten digits (0–9)
- Pixel values normalized:  
X_norm = X_raw / 255.0

### 🧠 Layers

| Layer         | Description                              |
|---------------|------------------------------------------|
| Input Layer   | 784 nodes (28 × 28 pixels)               |
| Hidden Layer  | 256 neurons, ReLU activation             |
| Output Layer  | 10 neurons, Softmax activation (digits)  |

---

## 🔧 Forward Propagation

Forward propagation is how the network makes a prediction by passing data through the layers.

### 🔹 Hidden Layer
Z1 = X.dot(W1) + b1
A1 = np.maximum(0, Z1)  # ReLU

### 🔹 Output Layer
Z2 = A1.dot(W2) + b2
Z2_stable = Z2 - np.max(Z2, axis=1, keepdims=True)
A2 = np.exp(Z2_stable) / np.sum(np.exp(Z2_stable), axis=1, keepdims=True)  # Softmax


> 📌 **Softmax Stability Tip**

Z2 = A1.dot(W2) + b2
Z2_stable = Z2 - np.max(Z2, axis=1, keepdims=True)
A2 = np.exp(Z2_stable) / np.sum(np.exp(Z2_stable), axis=1, keepdims=True)  # Softmax

---

## ⬅️ Backpropagation

Backpropagation updates weights by calculating gradients and moving in the direction of lower error.

### 🔹 Gradients for Output Layer (Layer 2)
dZ2 = A2 - Y_true
dW2 = A1.T.dot(dZ2)
db2 = np.sum(dZ2, axis=0, keepdims=True)


### 🔹 Gradients for Hidden Layer (Layer 1)
dA1 = dZ2.dot(W2.T)
dZ1 = dA1 * (Z1 > 0)  # Derivative of ReLU
dW1 = X.T.dot(dZ1)
db1 = np.sum(dZ1, axis=0, keepdims=True)


---

## 🧮 Weight & Bias Updates

W1 -= learning_rate * dW1
b1 -= learning_rate * db1
W2 -= learning_rate * dW2
b2 -= learning_rate * db2

---

## 📈 Final Performance

- 🎯 **Test Accuracy**: ~87%
- 💪 Fully interpretable: every number, every weight, every step.

---

## 📚 Learn the Math, Understand the Magic

This project is perfect for those who want to:
- Truly understand how a neural network learns
- See the mechanics of backpropagation
- Write deep learning code without relying on abstraction

---
