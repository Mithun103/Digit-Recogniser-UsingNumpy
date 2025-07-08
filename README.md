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
  \[
  X_{\text{norm}} = \frac{X_{\text{raw}}}{255.0}
  \]

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
\[
Z^{[1]} = XW^{[1]} + b^{[1]}
\]
\[
A^{[1]} = \text{ReLU}(Z^{[1]})
\]

### 🔹 Output Layer
\[
Z^{[2]} = A^{[1]}W^{[2]} + b^{[2]}
\]
\[
A^{[2]} = \text{Softmax}(Z^{[2]})
\]

> 📌 **Softmax Stability Tip**  
> To prevent numerical overflow:
> \[
> \text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
> \]

---

## ⬅️ Backpropagation

Backpropagation updates weights by calculating gradients and moving in the direction of lower error.

### 🔹 Gradients for Output Layer (Layer 2)
\[
\frac{\partial L}{\partial W^{[2]}} = (A^{[1]})^T (A^{[2]} - Y_{\text{true}})
\]

### 🔹 Gradients for Hidden Layer (Layer 1)
\[
\frac{\partial L}{\partial W^{[1]}} = X^T \left( (A^{[2]} - Y_{\text{true}}) W^{[2]^T} \cdot \text{ReLU}'(Z^{[1]}) \right)
\]

---

## 🧮 Weight & Bias Updates

\[
W = W - \alpha \cdot \frac{\partial L}{\partial W}
\]
\[
b = b - \alpha \cdot \frac{\partial L}{\partial b}
\]

> where **α** is the learning rate (step size during training).

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
