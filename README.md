
# 🧠 Neural Network from Scratch — Just NumPy.

This project is a simple but powerful implementation of a **2-layer Neural Network** for handwritten digit classification using the **MNIST dataset** — built completely from scratch using **NumPy** and pure math. No frameworks. No high-level APIs. Just matrix multiplications, derivatives, and dopamine. 😎

---

## 🔍 Why This?

Because **AI is not magic — it's mathematics in motion**.

If you've ever asked:

> *Why do weights update?*  
> *What is backprop really doing?*  
> *How does softmax handle exploding values?*  

...then this repo answers it. Not with black boxes — but with formulas and **intuition**.

---

## 🚀 What’s Inside?

### ✅ Dataset:
- MNIST (handwritten digits 0–9)
- Balanced for underrepresented digits (1, 3, 9)
- Normalized pixel values:  
  \[
  X_{\text{norm}} = \frac{X}{255.0}
  \]

### 🧠 Neural Network Architecture:
- Input: 784 nodes (28×28 pixels)
- Hidden Layer: 256 neurons, ReLU activation
- Output: 10 neurons, Softmax activation
- Final Accuracy: **~87% on test data**

### 🔧 Math Behind the Code:

#### Forward Propagation:
\[
Z^{[1]} = XW^{[1]} + b^{[1]}, \quad A^{[1]} = \text{ReLU}(Z^{[1]})
\]
\[
Z^{[2]} = A^{[1]}W^{[2]} + b^{[2]}, \quad A^{[2]} = \text{Softmax}(Z^{[2]})
\]

#### Backpropagation (Gradient Descent):
\[
\frac{\partial L}{\partial W^{[2]}} = A^{[1]^T}(A^{[2]} - Y_{\text{true}})
\]
\[
\frac{\partial L}{\partial W^{[1]}} = X^T((A^{[2]} - Y)W^{[2]^T} \cdot \text{ReLU}'(Z^{[1]}))
\]
\[
W = W - \alpha \cdot \frac{\partial L}{\partial W}
\]

📌 *Softmax includes stability fix using max-subtraction before exponentiation.*

---
