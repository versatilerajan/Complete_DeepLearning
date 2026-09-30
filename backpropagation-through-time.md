# Backpropagation Through Time (BPTT) in RNNs

This note explains how a Recurrent Neural Network learns from sequences: the network is unrolled over its time steps, the loss is pushed backward through those steps, and the gradients of the shared weights are added up across time. It follows the CampusX lecture and your handwritten class notes (redrawn here as clean PNGs), and it continues from [recurrent-neural-networks-intro.md](recurrent-neural-networks-intro.md) (why sequences need memory) and [rnn-keras-sentiment-analysis.md](rnn-keras-sentiment-analysis.md) (building the model in Keras). The next topic in the series is the vanishing/exploding gradient problem, which comes directly out of the long chain products derived below.

## Table of Contents
1. [Setup: the Class-Notes Example](#1-setup-the-class-notes-example)
2. [Forward Propagation and Unrolling](#2-forward-propagation-and-unrolling)
3. [Loss and Gradient Descent](#3-loss-and-gradient-descent)
4. [Gradient for the Output Weights W_o](#4-gradient-for-the-output-weights-w_o)
5. [Gradient for the Input Weights W_i](#5-gradient-for-the-input-weights-w_i)
6. [Gradient for the Hidden Weights W_h](#6-gradient-for-the-hidden-weights-w_h)
7. [The BPTT Algorithm in One Picture](#7-the-bptt-algorithm-in-one-picture)
8. [Checking the Math in Code](#8-checking-the-math-in-code)
9. [Clean Copies of Your Three Class-Notes Pages](#9-clean-copies-of-your-three-class-notes-pages)
10. [Key Takeaways](#10-key-takeaways)
11. [Further Reading](#11-further-reading)

---

## 1. Setup: the Class-Notes Example

The example uses three-word sentences over a three-word vocabulary, with a binary label. Each word is a one-hot vector, so one sentence is a sequence of three 3-dimensional vectors, fed one per time step.

| Sentence | Label |
|---|---|
| cat mat rat | 1 |
| rat rat mat | 1 |
| mat mat cat | 0 |

One-hot vocabulary: cat = `[1 0 0]`, mat = `[0 1 0]`, rat = `[0 0 1]`.

![Dataset and encoding](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bptt_01_dataset_and_encoding.png)

*The three sentences, their labels, the one-hot vocabulary and the resulting input tensor X (3 samples x 3 time steps x 3 features), exactly as in your notes. Because the label is a single 0/1 value, the network ends in one sigmoid unit.*

![Network architecture](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bptt_02_network_architecture.png)

*The rolled-up network from your notes: a 3-dimensional input feeds 3 hidden units through `W_i` (3x3), the hidden layer loops back on itself through `W_h` (3x3), and the hidden layer feeds one output unit through `W_o` (3x1), followed by a sigmoid to give `y_hat`. There are 3 hidden biases and 1 output bias. In total: 9 + 9 + 3 + 3 + 1 = 25 parameters.*

---

## 2. Forward Propagation and Unrolling

Unrolling means drawing the same cell once per time step. The hidden state `O_t` (the "output" of the hidden layer at step `t`) starts from `O_0 = 0` and is updated at every step:

```
O_1 = f(x_i1 · W_i + O_0 · W_h)
O_2 = f(x_i2 · W_i + O_1 · W_h)
O_3 = f(x_i3 · W_i + O_2 · W_h)
y_hat = sigmoid(O_3 · W_o)
```

`f` is the hidden activation (tanh in the verification code below). Biases are left out of these equations, as in your notes.

![Unrolled RNN](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bptt_03_unrolled_rnn.png)

*The unrolled network: three copies of the same cell `f`, each receiving its own word (`x_i1`, `x_i2`, `x_i3`) through the same `W_i`, and passing the hidden state along through the same `W_h`. Only the last state `O_3` goes through `W_o` to produce `y_hat`. Weight sharing is the key point: three copies in the picture, but only one set of weights, which is why gradients from every step must be added together later.*

---

## 3. Loss and Gradient Descent

Binary cross-entropy loss for one sample:

```
L = -y · log(y_hat) - (1 - y) · log(1 - y_hat)
```

Gradient-descent updates, with learning rate `eta`:

```
W_i <- W_i - eta · dL/dW_i
W_h <- W_h - eta · dL/dW_h
W_o <- W_o - eta · dL/dW_o
```

![Forward equations, loss and updates](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bptt_04_forward_and_updates.png)

*Your page of equations gathered in one place: the three forward steps and the output on the left, the loss below them, and the three weight updates on the right. Learning means computing the three gradients in the update rules; the next sections derive them one at a time, starting with the easiest.*

---

## 4. Gradient for the Output Weights W_o

`W_o` is used once, at the very end, so its gradient is a plain two-factor chain rule (steps (1) and (2) in your notes):

```
dL/dW_o = dL/dy_hat · dy_hat/dW_o
```

Working it out for sigmoid plus binary cross-entropy (this goes one step beyond the class notes):

```
dL/dy_hat   = -y/y_hat + (1 - y)/(1 - y_hat)
dy_hat/dW_o = y_hat · (1 - y_hat) · O_3^T
dL/dW_o     = (y_hat - y) · O_3^T
```

![dL/dW_o](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bptt_05_dL_dWo.png)

*The two chain-rule steps from your notes, followed by the simplified result `(y_hat - y) · O_3^T`. The messy sigmoid and log terms cancel, leaving "prediction error times the last hidden state". No sum over time is needed because `W_o` only acts at the last step.*

---

## 5. Gradient for the Input Weights W_i

`W_i` is used at all three steps, so the loss depends on it through three different routes (the tangle of arrows on your first page). Each route is one term of the chain rule:

```
dL/dW_i =  dL/dy_hat · dy_hat/dO_3 · dO_3/dW_i
         + dL/dy_hat · dy_hat/dO_3 · dO_3/dO_2 · dO_2/dW_i
         + dL/dy_hat · dy_hat/dO_3 · dO_3/dO_2 · dO_2/dO_1 · dO_1/dW_i
```

or compactly:

```
dL/dW_i = sum_{j=1..3}  dL/dy_hat · dy_hat/dO_j · dO_j/dW_i
```

![Three paths to W_i](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bptt_06_dL_dWi_paths.png)

*A clean redraw of the arrow tangle from your notes. The red arrow is the short path through step 3 only, the blue arrow goes back one extra step, and the green arrow goes all the way back to step 1. Each colored expression under the title is the matching term of the sum. Longer paths contain more `dO_k/dO_(k-1)` factors, which is where the vanishing/exploding-gradient issue (next topic) will come from.*

One subtlety worth writing next to the formula: in the compact sum, `dy_hat/dO_j` is the **total** effect of `O_j` on the output (it already includes the later steps), while `dO_j/dW_i` is the **local** derivative at step `j`, treating `O_(j-1)` as a constant. Using both meanings consistently is what makes the three-term expansion and the compact sum equal.

---

## 6. Gradient for the Hidden Weights W_h

The same reasoning applies to `W_h`, which is also shared across all steps. With `n` = number of time steps:

```
dL/dW_h = sum_{j=1..n}  dL/dy_hat · dy_hat/dO_j · dO_j/dW_h
```

![Sum forms](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bptt_07_gradient_sums.png)

*The compact summation forms for `W_i` and `W_h` side by side, with reading notes underneath. `W_h` is the hardest of the three because each term's `dy_hat/dO_j` reaches back through every later hidden state. One detail visible in the code check below: the `j = 1` term for `W_h` is exactly zero here, because `O_0 = 0` and step 1 therefore has no `O_0 · W_h` contribution.*

---

## 7. The BPTT Algorithm in One Picture

In practice the sums are computed by one backward sweep instead of writing out every path:

1. Run the forward pass and store every `O_t`.
2. Compute `dL/dO_3` from the output layer.
3. Walk backward from step 3 to step 1. At each step, take the gradient arriving at `O_t`, multiply by the activation derivative (for tanh: `1 - O_t^2`), add the local contributions to the running totals for `W_i`, `W_h` and the bias, then pass the gradient on to `O_(t-1)` through `W_h^T`.
4. Update all weights with gradient descent.

![Backward flow](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bptt_08_backward_flow.png)

*The red arrows carry `dL/dO_t` from right to left through time; the blue arrows show each step depositing its own local `W_i` and `W_h` contribution into the shared gradient. The sum in the formulas is exactly this accumulation, so BPTT is ordinary backpropagation applied to the unrolled network, with the shared weights' gradients added up.*

---

## 8. Checking the Math in Code

I implemented the class-notes network from scratch in NumPy (tanh hidden activation, biases included, weights initialised with a seeded normal, sigma = 0.5) and ran it:

- **Gradient check:** the BPTT gradients for all five parameter sets (`W_i`, `W_h`, `b_h`, `W_o`, `b_o`) matched numerical central-difference gradients to within **1.0e-10**.
- **Sum of terms:** for sample 1, the per-step terms `j = 1, 2, 3` add up exactly to the total `dL/dW_i` and `dL/dW_h`, and `dL/dW_o` equals `(y_hat - y) · O_3^T`.
- **Keras cross-check:** loading the same weights into a Keras `SimpleRNN(3)` followed by `Dense(1, sigmoid)`, Keras produced the same forward outputs (difference about 2e-08), the same loss (0.632229 for both) and the same gradients (differences about 1e-08 or smaller).
- **Training:** full-batch gradient descent with `eta = 0.5` took the loss from **0.6322** (epoch 0) to **0.0086** (epoch 100) and **0.0023** (epoch 300). The final predictions for the three sentences were **0.998, 0.998 and 0.004** against the labels 1, 1, 0. This is only three training sentences, so it shows that the derivation works, not that the model generalises.

![Results](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bptt_09_verified_results.png)

*Left: the real training loss over 300 epochs on the three sentences. Right: the size of each time step's contribution to the `W_i` and `W_h` gradients for sample 1 at initial weights (Frobenius norms of the `j = 1, 2, 3` terms). The `W_h` bar for step 1 is zero because `O_0 = 0`. With one sample and small random weights, these bars only illustrate that every step contributes; they say nothing yet about long sequences.*

From-scratch BPTT for one sample (row-vector convention, as in the class notes):

```python
import numpy as np

sig = lambda z: 1 / (1 + np.exp(-z))

def forward(P, x):                       # x: (3 steps, 3 features)
    O = [np.zeros(3)]                    # O_0 = 0
    for t in range(3):
        O.append(np.tanh(x[t] @ P["Wi"] + O[-1] @ P["Wh"] + P["bh"]))
    return O, sig(O[3] @ P["Wo"] + P["bo"])[0]

def bptt(P, x, y):
    O, y_hat = forward(P, x)
    g = {k: np.zeros_like(v) for k, v in P.items()}
    dz = y_hat - y                                   # dL/dz for sigmoid + BCE
    g["Wo"] = np.outer(O[3], [dz]); g["bo"] = np.array([dz])
    dO = dz * P["Wo"][:, 0]                          # dL/dO_3
    for t in (3, 2, 1):                              # walk backward in time
        dpre = dO * (1 - O[t] ** 2)                  # through tanh
        g["Wi"] += np.outer(x[t - 1], dpre)          # local W_i term of step t
        g["Wh"] += np.outer(O[t - 1], dpre)          # local W_h term of step t
        g["bh"] += dpre
        dO = dpre @ P["Wh"].T                        # pass gradient to O_(t-1)
    return g
```

Keras equivalent of the whole network (Keras does the backward pass for you):

```python
from keras import Sequential, Input
from keras.layers import SimpleRNN, Dense

model = Sequential([
    Input(shape=(3, 3)),                 # 3 time steps, 3 one-hot features
    SimpleRNN(3, activation="tanh"),     # W_i (3x3), W_h (3x3), 3 biases
    Dense(1, activation="sigmoid"),      # W_o (3x1), 1 bias
])
model.compile(optimizer="sgd", loss="binary_crossentropy")
```

---

## 9. Clean Copies of Your Three Class-Notes Pages

These are typed, redrawn versions of your three handwritten pages, in reading order (setup, then equations, then the gradient for `W_i` and `W_h`).

![Class notes page 1](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bptt_class_notes_page1_clean.png)

*Page 1: the "Backpropagation in RNN" setup, with the three sentences, one-hot vocabulary, input tensor, rolled network with weight shapes, and the unrolled network.*

![Class notes page 2](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bptt_class_notes_page2_clean.png)

*Page 2: the forward equations, the loss, the gradient-descent updates, and the derivation of `dL/dW_o`.*

![Class notes page 3](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/bptt_class_notes_page3_clean.png)

*Page 3: the three-path picture for `dL/dW_i`, its expanded and compact forms, and the matching formula for `dL/dW_h`.*

---

## 10. Key Takeaways

- BPTT is ordinary backpropagation applied to the unrolled network; nothing new is added except that shared weights collect gradient from every time step.
- `W_o` is used once, so `dL/dW_o = (y_hat - y) · O_3^T` needs no sum over time (for sigmoid with binary cross-entropy).
- `W_i` and `W_h` are used at every step, so their gradients are sums over time steps: `dL/dW = sum_j dL/dy_hat · dy_hat/dO_j · dO_j/dW`.
- Terms from earlier steps contain longer products of `dO_k/dO_(k-1)`; repeated multiplication of such factors is the source of vanishing and exploding gradients, covered next.
- In the sum, `dy_hat/dO_j` is a total derivative and `dO_j/dW` is a local derivative; keeping that distinction clear avoids double counting.
- The derivation was checked against numerical gradients (agreement to about 1e-10) and against Keras on the class-notes example.

---

## 11. Further Reading

- Werbos, P. J. (1990). *Backpropagation Through Time: What It Does and How to Do It*. Proceedings of the IEEE, 78(10), 1550-1560.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning Representations by Back-Propagating Errors*. Nature, 323, 533-536.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the Difficulty of Training Recurrent Neural Networks*. ICML.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 10: Sequence Modeling: Recurrent and Recursive Nets. MIT Press.
