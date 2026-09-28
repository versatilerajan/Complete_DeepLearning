# Introduction to Recurrent Neural Networks (RNNs)

This note introduces **Recurrent Neural Networks (RNNs)**, a class of deep learning architectures purpose-built for **sequential data** — data where the order of elements carries meaning. It expands on Nitish Singh's lecture, which frames RNNs as the natural next step after Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs), motivated by a core limitation: standard ANNs have no way to represent order. Understanding *why* ANNs fail on sequences, and *how* RNNs fix that, is the foundation for everything that follows in this series (BPTT, vanishing/exploding gradients, LSTM, GRU).

## Table of Contents
1. [What Is Sequential Data?](#1-what-is-sequential-data)
2. [Why ANNs Struggle With Sequential Data](#2-why-anns-struggle-with-sequential-data)
3. [Worked Example: "Hi my name is Rajan" in an ANN vs an RNN](#3-worked-example-hi-my-name-is-rajan-in-an-ann-vs-an-rnn)
4. [RNN Architecture at a Glance](#4-rnn-architecture-at-a-glance)
5. [Core RNN Formulas](#5-core-rnn-formulas)
6. [Real-World Applications of RNNs](#6-real-world-applications-of-rnns)
7. [From-Scratch Implementation](#7-from-scratch-implementation)
8. [Roadmap: What Comes Next](#8-roadmap-what-comes-next)
9. [Key Takeaways](#9-key-takeaways)
10. [Further Reading](#10-further-reading)

---

## 1. What Is Sequential Data?

Sequential data is any data where **the order of elements changes its meaning** — reordering the elements gives you a different (or nonsensical) signal. Common examples:

- **Text** — the sentence "the dog bit the man" means something different from "the man bit the dog," even though both use the same five words.
- **Time-series data** — stock prices, sensor readings, weather measurements; each value depends on what came before it.
- **Audio waveforms** — amplitude over time; the order of samples defines the sound.
- **DNA sequences** — the order of bases (A, T, C, G) determines the gene/protein it encodes.

![Sequential data examples](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_sequential_data_examples.png)

*The four panels above show four different domains that all share one property: shuffle the elements and you destroy the information. A bag of words ("Hi", "my", "name", "is", "Rajan") is not a sentence; a shuffled time-series is not the same market history; a shuffled waveform is noise; a shuffled DNA sequence is a different (or invalid) gene. This shared property — order-dependence — is exactly what standard ANNs are not built to represent, and exactly what RNNs are designed to capture via their internal memory.*

---

## 2. Why ANNs Struggle With Sequential Data

Feeding sequential data into a standard ANN runs into three compounding problems:

1. **Fixed input size.** An ANN's input layer has a fixed number of neurons, decided when the network is built. But real sentences, audio clips, and time series come in different lengths.
2. **Zero-padding is wasteful and arbitrary.** One workaround is to pad every input up to some maximum length with zeros. This wastes computation on padding, and forces you to pick an arbitrary maximum length that may not fit future data.
3. **No memory of order (the fundamental problem).** Even if you solve the size problem, an ANN's input layer accepts *all* values simultaneously as an unordered flat vector. Nothing in the architecture encodes "this value came before that value" — the network sees a set of numbers, not a sequence.

That third point is the fundamental one — it's an architectural fact about feed-forward networks, not just an engineering inconvenience, and it's what section 3 illustrates directly.

![ANN vs RNN architecture](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_ann_vs_rnn_architecture.png)

*Left: an ANN's input layer has a fixed number of nodes (x1..x5) that all feed forward into hidden layers at once — there's no mechanism for "x1 happened before x2." Right: an RNN reuses the *same* cell at every time step, and passes a hidden state (h1 → h2 → h3 → h4, shown as the red arrows) from one step to the next. That hidden state is the network's memory — it lets the cell at time step 3 "know" something about what happened at time steps 1 and 2, and the number of time steps can grow or shrink with the input, so there's no fixed-length requirement.*

---

## 3. Worked Example: "Hi my name is Rajan" in an ANN vs an RNN

To make the abstract point above concrete, here's what happens to a specific sentence — **"Hi my name is Rajan"** — in each architecture.

### In an ANN

![ANN loses word order](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_ann_word_order_loss.png)

*Each word is mapped to a value and all five arrive at the input layer at the same time, forming one flat vector `[Hi, my, name, is, Rajan]`. The hidden layers process this vector as a set of numbers. Critically, the shuffled sentence "Rajan name is my Hi" would produce the exact same flat vector (just reordered inside it) and the ANN has no built-in way to tell that the meaning has changed — it wasn't designed to care about position within the vector beyond which fixed slot a value sits in, and it never relates slot 1 to slot 2 as "earlier" and "later" in a sequence.*

### In an RNN

![RNN keeps word order in memory](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_rnn_word_order_memory.png)

*The same sentence is now fed one word at a time. At each step, the RNN cell combines the current word with the hidden state carried over from the previous step, producing a new hidden state. By the time "Rajan" arrives, the hidden state h5 has been shaped by "Hi", then "my", then "name", then "is", in that order — the sequence itself is baked into the hidden state. Feed the words in a different order and you get an entirely different sequence of hidden states, so (unlike the ANN) the RNN's output is sensitive to word order.*

---

## 4. RNN Architecture at a Glance

An RNN can be thought of as the *same small neural network cell*, applied repeatedly — once per time step — while passing a hidden state forward:

- At time step `t`, the cell receives two things: the current input `x_t` and the previous hidden state `h_(t-1)`.
- It combines them to produce a new hidden state `h_t`.
- `h_t` is passed forward to time step `t+1`, and can also optionally be used to produce an output `y_t` at the current step.

Because the *same* weights are reused at every time step, an RNN can process sequences of any length without changing its parameter count — directly solving the fixed-input-size problem from Section 2.

---

## 5. Core RNN Formulas

At each time step `t`, a simple (vanilla) RNN computes:

**Hidden state update:**

```
h_t = tanh(W_hh · h_(t-1) + W_xh · x_t + b_h)
```

**Output (if produced at this step):**

```
y_t = W_hy · h_t + b_y
```

Where:
- `x_t` — input vector at time step `t`
- `h_t` — hidden state at time step `t` (the network's "memory")
- `h_(t-1)` — hidden state from the previous time step (`h_0` is typically initialized to zeros)
- `W_xh`, `W_hh`, `W_hy` — weight matrices (input-to-hidden, hidden-to-hidden, hidden-to-output) — **shared across all time steps**
- `b_h`, `b_y` — bias vectors
- `tanh` — the activation function commonly used in vanilla RNN cells (keeps hidden state values bounded between -1 and 1)

The key structural detail: `W_hh`, `W_xh`, `W_hy`, `b_h`, and `b_y` do **not** change from one time step to the next. This weight-sharing is what lets the same small set of parameters handle sequences of any length.

---

## 6. Real-World Applications of RNNs

![RNN applications](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_rnn_applications.png)

*Four applications where sequence-awareness is essential. **Sentiment analysis** reads a sequence of words and predicts an overall sentiment, where word order affects meaning ("not bad" vs "bad"). **Predictive text** (e.g. Gmail's Smart Compose) reads the words typed so far and predicts what's likely to come next. **Image caption generation** combines a vision component with a sequence model that generates a caption one word at a time, each new word conditioned on the words already generated. **Machine translation** consumes a sentence in one language as a sequence and produces a sentence in another language, where the output sequence's word order depends on the entire input sequence, not just isolated words.*

---

## 7. From-Scratch Implementation

Below is a minimal vanilla RNN cell implemented from scratch in NumPy — processing one sequence, one time step at a time — followed by the equivalent in PyTorch and Keras for comparison.

### NumPy (from scratch)

```python
import numpy as np

class SimpleRNNCell:
    def __init__(self, input_size, hidden_size, output_size, seed=0):
        rng = np.random.default_rng(seed)
        # Small random init keeps early hidden states from saturating tanh
        self.W_xh = rng.normal(0, 0.1, (hidden_size, input_size))
        self.W_hh = rng.normal(0, 0.1, (hidden_size, hidden_size))
        self.W_hy = rng.normal(0, 0.1, (output_size, hidden_size))
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        self.hidden_size = hidden_size

    def forward(self, inputs):
        """
        inputs: list of column vectors, one per time step, each shape (input_size, 1)
        Returns: (hidden_states, outputs) — one h_t and y_t per time step
        """
        h_t = np.zeros((self.hidden_size, 1))
        hidden_states, outputs = [], []
        for x_t in inputs:
            h_t = np.tanh(self.W_hh @ h_t + self.W_xh @ x_t + self.b_h)
            y_t = self.W_hy @ h_t + self.b_y
            hidden_states.append(h_t)
            outputs.append(y_t)
        return hidden_states, outputs


# Example: a toy "sequence" of 5 one-hot-ish word vectors (vocab size 8)
vocab_size, hidden_size, output_size = 8, 4, 8
rnn = SimpleRNNCell(vocab_size, hidden_size, output_size)

word_ids = [1, 2, 3, 4, 5]  # stand-ins for "Hi", "my", "name", "is", "Rajan"
one_hots = []
for wid in word_ids:
    v = np.zeros((vocab_size, 1))
    v[wid, 0] = 1.0
    one_hots.append(v)

hidden_states, outputs = rnn.forward(one_hots)
print("Number of hidden states:", len(hidden_states))
print("Shape of each hidden state:", hidden_states[0].shape)
```

### PyTorch (equivalent, one line for the recurrent layer)

```python
import torch
import torch.nn as nn

rnn_layer = nn.RNN(input_size=8, hidden_size=4, batch_first=True)
x = torch.eye(8)[[1, 2, 3, 4, 5]].unsqueeze(0)  # (batch=1, seq_len=5, input_size=8)
output, h_n = rnn_layer(x)  # output: all hidden states, h_n: final hidden state
```

### Keras (equivalent)

```python
from tensorflow.keras.layers import SimpleRNN
import numpy as np

rnn_layer = SimpleRNN(units=4, return_sequences=True)
x = np.eye(8)[[1, 2, 3, 4, 5]].reshape(1, 5, 8)  # (batch=1, seq_len=5, input_size=8)
output = rnn_layer(x)  # all hidden states across the sequence
```

All three versions implement the same recurrence: at each time step, combine the current input with the previous hidden state, pass the result through a nonlinearity, and carry the new hidden state forward.

---

## 8. Roadmap: What Comes Next

![Learning roadmap](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_learning_roadmap.png)

*This introduction sets up five upcoming topics in sequence. First, the exact forward-pass mechanics of a simple RNN cell. Then, how gradients are computed through time (Backpropagation Through Time, or BPTT) so the network can learn. That leads directly into the vanishing/exploding gradient problem, which is the central weakness of vanilla RNNs on long sequences. Finally, LSTM and GRU are introduced as architectures specifically designed to fix that weakness with gating mechanisms.*

---

## 9. Key Takeaways

- Sequential data (text, time series, audio, DNA) is defined by the fact that **reordering it changes or destroys its meaning**.
- Standard ANNs are a poor fit for sequences because of fixed input size, wasteful zero-padding, and — most fundamentally — **no mechanism for representing order**.
- RNNs solve this by reusing the same cell at every time step and passing a **hidden state** forward, so information about earlier elements influences the processing of later ones.
- The core recurrence is `h_t = tanh(W_hh · h_(t-1) + W_xh · x_t + b_h)`, with weights shared across all time steps.
- RNNs power sequence-dependent applications: sentiment analysis, predictive text, image captioning, and machine translation.
- Vanilla RNNs have known weaknesses (vanishing/exploding gradients over long sequences) that motivate LSTM and GRU, covered next in this series.

---

## 10. Further Reading

- Elman, J. L. (1990). *Finding Structure in Time*. Cognitive Science, 14(2), 179–211. — One of the foundational papers introducing simple recurrent networks.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning Representations by Back-Propagating Errors*. Nature, 323, 533–536. — Origin of the backpropagation algorithm that BPTT extends to sequences.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780. — Introduces LSTM, covered later in this series.
- Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. EMNLP. — Introduces the GRU architecture.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the Difficulty of Training Recurrent Neural Networks*. ICML. — Formal treatment of the vanishing/exploding gradient problem in RNNs, covered next in this series.
