# Next-Word Prediction with an LSTM: From N-grams to a Working Predictor

This note expands a CampusX lecture on building a next-word predictor (the "autocomplete" feature of mobile keyboards and email tools) with an LSTM in Keras. It covers the whole path: turning raw text into supervised N-gram pairs, tokenising and zero-padding, the Embedding → LSTM → Dense(softmax) architecture, what happens inside the LSTM cell, how training and generation work, and what the model can and cannot do on unseen text. Everything quantitative here was produced by a from-scratch NumPy implementation that I ran and verified (gradients checked numerically). The training data is a small 30-line FAQ corpus written for this note, **not** the video's dataset, so my absolute numbers differ from the video's; the mechanisms are the same. Keras and TensorFlow were not available where this was written, so the Keras snippets are **not executed**; they are labelled as such.

Related notes in this series (filenames assumed, adjust to your repo): `rnn.md`, `backpropagation.md`, `softmax-and-cross-entropy.md`, `word-embeddings.md`, `gradient-descent.md`.

## Table of Contents

1. [The task](#1-the-task)
2. [Data preparation](#2-data-preparation)
3. [Why LSTM and not a plain RNN](#3-why-lstm-and-not-a-plain-rnn)
4. [Inside the LSTM cell](#4-inside-the-lstm-cell)
5. [The model architecture](#5-the-model-architecture)
6. [Loss, training and a ceiling you can compute](#6-loss-training-and-a-ceiling-you-can-compute)
7. [Experiments](#7-experiments)
8. [Inference and what the gates do](#8-inference-and-what-the-gates-do)
9. [Complete code](#9-complete-code)
10. [Problems in the pasted Keras code](#10-problems-in-the-pasted-keras-code)
11. [Improving the model](#11-improving-the-model)
12. [Key Takeaways](#12-key-takeaways)
13. [Further Reading](#13-further-reading)
14. [Appendix: experiment code](#14-appendix-experiment-code)

---

## 1. The task

Next-word prediction is language modelling: given the words typed so far, output a probability for every word in the vocabulary being next. With a vocabulary $\mathcal{V}$ and a prefix $w_1,\dots,w_t$, the model estimates

$$P(w_{t+1}\mid w_1,\dots,w_t)\;=\;\mathrm{softmax}(z)_{w_{t+1}},\qquad z = W_o\,h_t + b_o$$

where $h_t$ is a fixed-size summary of the prefix produced by the LSTM. This turns an open-ended text problem into **multi-class classification with $|\mathcal{V}|$ classes**: the input is a (padded) sequence of word ids, the label is the id of the next word. The video's flow is: define the goal → prepare data → build the model → train and discuss improvements.

## 2. Data preparation

Three transformations turn text lines into a training matrix. Figure 1 shows them on the first line of my corpus.

1. **Tokenise.** A Keras-style `Tokenizer` lower-cases, strips punctuation and assigns each word an integer, most frequent first, starting at 1. Index 0 is reserved for padding, so the embedding table has `len(word_index) + 1` rows. That is why the video's vocabulary is 283 while its Embedding layer is `Embedding(283, ...)`. In my corpus: 30 lines, 81 ids including padding.
2. **N-gram pairs.** A line $[a,b,c,d]$ yields the pairs $([a],b)$, $([a,b],c)$, $([a,b,c],d)$. A line with $n$ words gives $n-1$ samples. My corpus gives 205 samples.
3. **Zero padding.** LSTM layers in Keras consume rectangular tensors, so every prefix is padded with zeros to the length of the longest prefix (`maxlen`). The video's `maxlen` is 56 (which, with the usual construction, means its longest line has about 57 words); mine is 11. `padding='pre'` puts zeros **before** the words so the real words sit next to the point where the LSTM's final state is read; Section 7.1 measures why that matters.

![Figure 1: N-gram pairs and pre-padding](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_pipeline_ngrams_padding.png)

*Figure 1. The first corpus line ("what is the fee for the data science course") shown as tokenizer ids and the 8 training pairs it produces. Each row is one sample: the input is the prefix left-padded with zeros (grey) to length 11, and the orange cell is the id of the word the model must predict. The ids shown are the real tokenizer output from my code. Note how the same early words appear in every row: the model is trained to predict from every prefix length, which is exactly the situation at typing time.*

**Targets.** The video one-hot encodes the target ($y\in\{0,1\}^{|\mathcal{V}|}$) and uses categorical cross-entropy. Using the integer id with sparse categorical cross-entropy computes the identical loss without allocating an $N\times|\mathcal{V}|$ matrix; my NumPy code uses integer ids.

| Quantity | Value (my corpus) |
|---|---|
| Lines | 30 |
| Vocabulary incl. padding index | 81 |
| Training samples (N-gram pairs) | 205 |
| `maxlen` (longest prefix) | 11 |
| Distinct padded prefixes | 154 |
| Prefixes followed by more than one different word | 12 |

## 3. Why LSTM and not a plain RNN

A plain RNN updates $h_t=\tanh(W[x_t,h_{t-1}]+b)$. Backpropagating a loss at the last step to an input $k$ steps earlier multiplies $k$ Jacobians:

$$\frac{\partial h_T}{\partial h_t}=\prod_{j=t+1}^{T}\operatorname{diag}\!\big(1-h_j^{2}\big)\,W_{hh}^{\top}$$

If these factors have norm below 1 the product shrinks exponentially (vanishing gradients); above 1 it explodes. Either way long-range dependencies are hard to learn (Bengio et al., 1994; Pascanu et al., 2013). Note $\tanh'\le$ 1 and $\sigma'\le$ 0.25 (computed numerically in Figure 2), so saturating activations only push factors down.

The LSTM adds a separate cell state $c_t$ with an additive update, so along the cell-state path $\partial c_t/\partial c_{t-1}=\operatorname{diag}(f_t)$: if the forget gate is near 1 the gradient passes through almost unchanged.

**Experiment (real, untrained networks).** I fed random inputs through a vanilla RNN and LSTMs (hidden size 64, input size 32, $T=40$), and measured $\lVert\partial L/\partial x_t\rVert$ for $L=\langle h_T,r\rangle$ with a random vector $r$, averaged over 10 random initialisations. "first/last" is the gradient reaching the oldest input divided by the gradient reaching the newest.

| Network (at initialisation) | first/last gradient ratio |
|---|---|
| Vanilla RNN (tanh) | 1.9e-07 |
| LSTM, forget bias 0 | 4.0e-09 |
| LSTM, forget bias 1 (Keras default) | 4.7e-04 |
| LSTM, forget bias 5 | 2.7e+00 |

![Figure 6: gradient flow, RNN vs LSTM](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_gradient_flow_rnn_vs_lstm.png)

*Figure 6. Gradient magnitude reaching each input position, normalised to the most recent step (log scale). The vanilla RNN loses about seven orders of magnitude over 40 steps. An LSTM is not automatically better: with forget bias 0 the forget gate starts near 0.5 (sigmoid of roughly 0) and the gradient decays even faster than in the RNN. With bias 1 (the usual initialisation) it decays much more slowly, and with bias 5 (forget gate about 0.99) it does not decay at all over 40 steps (first/last above 1). The lesson is that the LSTM provides a path for gradients and the forget gate value decides how open it is. These are untrained networks; training moves the gate values.*

So the honest summary of the experiment: the gradient highway exists in the architecture, but at default initialisation it only partially keeps the gradient alive over 40 steps. In this note's task the sequences are at most 11 steps, so this is not the limiting factor; it is the reason LSTMs scale to much longer text such as the video's 57-word lines.

## 4. Inside the LSTM cell

At each step the cell reads $x_t$ and $h_{t-1}$ and produces $c_t$ and $h_t$. Concatenating $[h_{t-1},x_t]$ and using one matrix for all four gates:

$$\begin{aligned}
f_t&=\sigma(W_f[h_{t-1},x_t]+b_f) &&\text{forget gate: how much old memory to keep}\\
i_t&=\sigma(W_i[h_{t-1},x_t]+b_i) &&\text{input gate: how much new content to write}\\
\tilde c_t&=\tanh(W_g[h_{t-1},x_t]+b_g) &&\text{candidate content}\\
c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t &&\text{cell state (long-term memory)}\\
o_t&=\sigma(W_o[h_{t-1},x_t]+b_o) &&\text{output gate: how much memory to expose}\\
h_t&=o_t\odot\tanh(c_t) &&\text{hidden state (short-term output)}
\end{aligned}$$

![Figure 3: LSTM cell](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_lstm_cell_diagram.png)

*Figure 3. The LSTM cell drawn from the equations above. The red horizontal line is the cell state: the old memory $c_{t-1}$ is scaled by the forget gate (left ×), new content (input gate × candidate, middle) is added (+), and the result $c_t$ leaves on the right. The blue output $h_t$ is the squashed cell state scaled by the output gate. All three gates and the candidate read the same concatenated input $[h_{t-1},x_t]$ shown as the bus at the bottom. Gates use sigmoid (0 to 1, acting as soft switches); the candidate and the cell squash use tanh (-1 to 1).*

![Figure 2: activation functions](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_activation_functions.png)

*Figure 2. Sigmoid and tanh (solid) with their derivatives (dashed). Sigmoid's range of 0 to 1 makes it a natural gate, and its derivative never exceeds 0.25; tanh's derivative peaks at 1 only at zero. Both derivatives approach 0 for large $\lvert z\rvert$ (saturation), which is where gradients vanish and "dead" gates come from. The maxima were computed numerically over a fine grid, not assumed.*

**Parameter count.** One LSTM layer with input size $d$ and $h$ units has four gate blocks, each with an $h\times d$ input matrix, an $h\times h$ recurrent matrix and a bias:

$$\#\text{params}_{\text{LSTM}}=4h\,(d+h+1)$$

## 5. The model architecture

The video's model is Embedding → LSTM(150) → Dense(softmax):

- **Embedding(V, d):** a lookup table of shape $V\times d$ that maps each id to a dense learnable vector. Output shape (batch, `maxlen`, $d$). Padding id 0 also has a (learned) vector unless masking is used.
- **LSTM(h):** processes the `maxlen` vectors in order and, with the default `return_sequences=False`, returns only the **last** hidden state $h_T$, shape (batch, $h$).
- **Dense(V, softmax):** $p=\mathrm{softmax}(W_oh_T+b_o)$, a probability for every vocabulary word.

![Figure 4: architecture and shapes](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_architecture_shapes.png)

*Figure 4. Tensor shapes through the network for the model I trained (top) and for the video's configuration (bottom, shapes and parameter counts computed from the formulas, not from a Keras `summary()` run). The LSTM collapses the time dimension: only the final hidden state is passed to the classifier, which is why the end of the padded sequence must contain the real words (Section 7.1).*

![Figure 5: parameter counts](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_param_counts_video_config.png)

*Figure 5. Parameters of the video's configuration (vocabulary 283, embedding 100, 150 LSTM units). The summary describes one LSTM layer, giving 221,633 parameters; the pasted code stacks two, giving 402,233. The LSTM layers dominate: layer 1 has 150,600 and layer 2 has 180,600, because every gate matrix is multiplied by four. The embedding table has 28,300 and the output layer 42,733.*

My trained model uses $V=$ 81, $d=32$, $h=64$, giving 32,689 parameters; the code asserts that the model's actual parameter count equals the formula.

## 6. Loss, training and a ceiling you can compute

**Loss.** With $p=\mathrm{softmax}(z)$ and true class $y$ the cross-entropy is $L=-\log p_y$ (identical to the one-hot form $-\sum_k y_k\log p_k$). The gradient with respect to the logits is the famously simple

$$\frac{\partial L}{\partial z}=p-\mathbf{1}_y$$

which then flows through the Dense layer into $h_T$ and backwards through time (backpropagation through time) into the LSTM weights and the embedding table. Optimiser: Adam (Kingma & Ba, 2015) with learning rate 0.01 and batch size 32 in my runs (Keras's default learning rate is 0.001; I raised it so 100 epochs suffice on this tiny corpus).

**Gradient check.** Hand-written BPTT is easy to get subtly wrong, so I compared every analytic gradient entry against central finite differences ($\varepsilon=10^{-5}$, float64, tiny model). The worst relative error across all parameters was 2.4e-06, i.e. the backward pass is correct.

**A ceiling for training accuracy.** If two different lines share the same prefix but continue with different words, no model can get both right. Here 12 of the 154 distinct prefixes are ambiguous (for example "what is the fee for the" is followed by *data*, *machine* and *deep* in different lines). Counting the majority continuation of each prefix gives a ceiling of 185/205 = 0.9024 training accuracy for **any** model, and the lowest achievable training cross-entropy is the conditional entropy $H(y\mid\text{prefix})=$ 0.1946 nats.

![Figure 7: training curves](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_training_curves.png)

*Figure 7. Training loss and accuracy of the reference model (seed 0) over 100 epochs. Loss is already 3.66 after the first epoch (chance level would be $\ln 81\approx4.39$) and falls to 0.2042; accuracy reaches 90% at epoch 30 and finishes at 0.9024, exactly the dashed ceiling of 0.9024. The final loss is within about 0.01 nats of the entropy floor of 0.1946. So the model has fit this training set essentially perfectly; it is memorising, which is why this number says nothing about new text (Section 7.2).*

This is the same story as the video's "high training accuracy after 100 epochs": useful as a sanity check that the pipeline works, uninformative about generalisation.

## 7. Experiments

### 7.1 Pre-padding vs post-padding

The LSTM reads left to right and only its **last** hidden state reaches the classifier. With `padding='pre'` the real words are the last things read; with `padding='post'` the network must carry information across several no-op steps of padding. I trained identical models (same hyperparameters, 5 seeds each, 100 epochs) with each option and no masking:

| | final train accuracy (mean ± std, min–max) | final loss (mean) | accuracy after 30 epochs (mean) |
|---|---|---|---|
| `pre` | 0.9015 ± 0.0020 (0.8976–0.9024) | 0.2060 | 0.892 |
| `post` | 0.8976 ± 0.0076 (0.8829–0.9024) | 0.2281 | 0.618 |

![Figure 8: pre vs post padding](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_padding_pre_vs_post.png)

*Figure 8. Training accuracy (left) and loss (right) for the two padding modes; lines are means over 5 seeds and the shaded bands are the min–max range. Both end at nearly the same accuracy on this small problem, but post-padding learns much more slowly (after 30 epochs 0.62 vs 0.89) and varies more between seeds. Pre-padding is the better default for a "read the sequence, classify from the last state" model, though this experiment shows slower and noisier optimisation rather than a different end point.*

The video's choice of `padding='pre'` is therefore sound. Masking (`mask_zero=True`) is the alternative way to make padding harmless; I did not test it.

### 7.2 Memorising vs generalising

The video notes the model should be improved "on unseen data". To measure it I held out **entire lines** (6 of the 30), trained on the other 24, and tested on the next-word pairs from the unseen lines. Five random splits, 100 epochs each. (The tokenizer was fitted on all lines, so every held-out word has an id, though a word that appears only in held-out lines has an untrained embedding.)

| Split | train pairs | held-out pairs | train acc | held-out acc | held-out loss at epoch 100 | best held-out loss (epoch) |
|---|---|---|---|---|---|---|
| 1 | 161 | 44 | 0.913 | 0.568 | 4.42 | 3.22 (14) |
| 2 | 169 | 36 | 0.917 | 0.361 | 5.48 | 3.59 (9) |
| 3 | 167 | 38 | 0.886 | 0.368 | 5.63 | 3.47 (10) |
| 4 | 165 | 40 | 0.909 | 0.475 | 4.01 | 2.98 (13) |
| 5 | 167 | 38 | 0.880 | 0.316 | 6.68 | 3.93 (8) |
| **mean** | | | **0.901** | **0.418** (std 0.092) | **5.25** | |

![Figure 9: generalisation to held-out lines](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_generalization_heldout_lines.png)

*Figure 9. Left: training accuracy (blue) is about 0.90 in every split, while accuracy on unseen lines (orange) is far lower and varies a lot between splits. Right: for split 1, training loss keeps falling while held-out loss reaches its minimum early and then climbs, the classic overfitting signature. The model learns the question templates ("how do i ...", "what is the ...") and the first words after them, which is why held-out accuracy is well above chance, but it cannot invent continuations it never saw.*

Held-out accuracy averages 0.42 versus training 0.90, and held-out loss at epoch 100 (mean 5.25) is above the chance level of $\ln 81=4.39$ in four of the five splits (the exception is split 4), because the model is confidently wrong on lines it has not seen. The held-out loss bottomed out between epochs 8 and 14 in these splits, so early stopping on a validation set would have stopped far earlier than 100 epochs. With 24 training lines this is a small-data effect; the video's suggestions (more data, tuning, bigger architectures) address exactly this.

## 8. Inference and what the gates do

**Generation loop.** Prediction repeats four steps: tokenise the text so far, pad to `maxlen`, run the model, take the highest-probability word ($\arg\max$, i.e. *greedy decoding*) and append it. Starting from "what is the fee", my trained model produced:

> what is the fee for the data science course sessions course sessions

The first part is a correct completion of a training question ("what is the fee for the data science course"). After "course" the model keeps going with "sessions course sessions" because the training lines have **no end-of-sentence token**: it has never seen what follows a line's last word, so the output is arbitrary. A production predictor would add an end marker, or simply stop at the end of a phrase.

![Figure 11: top-5 next-word probabilities per step](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11_next_word_distributions.png)

*Figure 11. The five most likely next words at each generation step (blue is the chosen word). Most steps are near-certain (probability above 0.99) because the prefix uniquely identifies the continuation in training. The third panel is the interesting one: after "what is the fee for the" the training data contains three different courses, and the model splits its probability roughly in proportion (0.46 data, 0.28 machine, 0.25 deep). Greedy decoding hides this uncertainty; showing the top-k words, as keyboards do, uses it.*

**Sampling with temperature.** Instead of always taking the argmax you can sample from $p_i\propto\exp(z_i/\tau)$: $\tau<1$ sharpens, $\tau>1$ flattens. (Not tested here.)

**Looking at the gates.** I ran the sentence "what is the fee for the course" through the trained network and recorded every gate for every hidden unit.

![Figure 10: gate activations](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10_gate_activations.png)

*Figure 10. Input, forget and output gate values (0 = closed, 1 = open) for each of the 64 hidden units (rows) at each time step (columns); the first four columns are padding and the rest are the words. Units behave very differently from each other: some forget gates stay near 1 for the whole sentence (long-term memory), others close at particular words. Averaged over units, the input gate is 0.53 at the first padding step and about 0.95 at "the", and the forget gate averages 0.80 on padding but 0.52 at the last word. This is one example sentence, so treat it as an illustration of the mechanism, not proof of what any individual unit "means".*

## 9. Complete code

### 9.1 From-scratch NumPy implementation (run and verified)

This is the exact file used for every number above (`python lstm_numpy.py` prints the headline results). It contains the tokenizer, N-gram builder, padding, the LSTM forward pass and hand-written backpropagation through time, Adam, and the generation loop.

```python
"""From-scratch LSTM next-word predictor (NumPy only).
Pipeline: Tokenizer -> N-gram pairs -> pre-padding -> Embedding -> LSTM -> Dense+softmax.
Run `python lstm_numpy.py` to train the reference model and print the headline numbers."""
import re
import numpy as np

CORPUS = """what is the fee for the data science course
what is the duration of the data science course
what is the fee for the machine learning course
how do i enroll in the course
how do i pay the fee for the course
how do i download the course videos
when does the next batch start
when does the course end
can i get a certificate after the course
can i pay the fee in installments
is there a refund policy for the course
is the course available in hindi
is there any placement support after the course
who are the instructors for the course
which tools will i learn in the course
will i get lifetime access to the recordings
do i need prior programming experience for the course
do i need a laptop for the live sessions
where can i find the course syllabus
where can i ask my doubts during the course
what is the fee for the deep learning course
what is the refund period for the course
how long is each live session
how many projects are there in the course
how can i contact the support team
what are the timings of the live sessions
what is the eligibility for the placement support
how do i get the course certificate
when will i receive the course certificate
can i switch to the next batch if i miss a session"""

# ---------------------------------------------------------------- data
class Tokenizer:
    """Mimics keras Tokenizer: lower-case, strip punctuation, index 1..V by
    descending frequency; index 0 is reserved for padding."""
    @staticmethod
    def _words(text):
        return re.findall(r"[a-z0-9']+", text.lower())

    def fit_on_texts(self, texts):
        counts = {}
        for t in texts:
            for w in self._words(t):
                counts[w] = counts.get(w, 0) + 1
        ordered = sorted(counts, key=lambda w: -counts[w])      # stable sort
        self.word_index = {w: i + 1 for i, w in enumerate(ordered)}
        self.index_word = {i: w for w, i in self.word_index.items()}
        return self

    def texts_to_sequences(self, texts):
        return [[self.word_index[w] for w in self._words(t) if w in self.word_index]
                for t in texts]

def make_ngram_pairs(sequences):
    """[a b c d] -> ([a],b) ([a b],c) ([a b c],d)"""
    X, y = [], []
    for seq in sequences:
        for i in range(1, len(seq)):
            X.append(seq[:i])
            y.append(seq[i])
    return X, np.array(y)

def pad_sequences(seqs, maxlen, padding="pre"):
    out = np.zeros((len(seqs), maxlen), dtype=int)
    for r, s in enumerate(seqs):
        s = s[-maxlen:]
        if padding == "pre":
            out[r, maxlen - len(s):] = s
        else:
            out[r, :len(s)] = s
    return out

def build_dataset(corpus=CORPUS, padding="pre"):
    lines = corpus.strip().split("\n")
    tok = Tokenizer().fit_on_texts(lines)
    seqs = tok.texts_to_sequences(lines)
    Xl, y = make_ngram_pairs(seqs)
    maxlen = max(len(s) for s in Xl)
    return tok, seqs, pad_sequences(Xl, maxlen, padding), y, maxlen

# ---------------------------------------------------------------- model
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def count_params(V, d, h, n_lstm=1):
    """Embedding + stacked LSTMs (4h(in+h+1) each) + Dense(h->V)."""
    total = V * d
    for k in range(n_lstm):
        total += 4 * h * ((d if k == 0 else h) + h + 1)
    return total + h * V + V

class LSTMLanguageModel:
    """Embedding(V,d) -> LSTM(h) -> Dense(V, softmax). Gate order: i, f, g, o."""
    def __init__(self, V, d, h, seed=0, forget_bias=1.0):
        rng = np.random.default_rng(seed)
        self.V, self.d, self.h = V, d, h
        lim = np.sqrt(6.0 / ((d + h) + 4 * h))                  # Glorot uniform
        b = np.zeros(4 * h)
        b[h:2 * h] = forget_bias
        self.p = {"E": rng.uniform(-0.05, 0.05, (V, d)),
                  "W": rng.uniform(-lim, lim, (d + h, 4 * h)),  # [x_t, h_{t-1}] -> gates
                  "b": b,
                  "Wo": rng.uniform(-np.sqrt(6 / (h + V)), np.sqrt(6 / (h + V)), (h, V)),
                  "bo": np.zeros(V)}

    def n_params(self):
        return sum(v.size for v in self.p.values())

    # --- forward over embeddings x: (B, T, d)
    def forward_x(self, x):
        p, h = self.p, self.h
        B, T, _ = x.shape
        H = np.zeros((B, T + 1, h)); C = np.zeros((B, T + 1, h)); G = np.zeros((B, T, 4 * h))
        for t in range(T):
            z = np.concatenate([x[:, t], H[:, t]], axis=1) @ p["W"] + p["b"]
            i, f = sigmoid(z[:, :h]), sigmoid(z[:, h:2 * h])
            g, o = np.tanh(z[:, 2 * h:3 * h]), sigmoid(z[:, 3 * h:])
            C[:, t + 1] = f * C[:, t] + i * g
            H[:, t + 1] = o * np.tanh(C[:, t + 1])
            G[:, t] = np.concatenate([i, f, g, o], axis=1)
        return H, C, G

    def forward(self, X):
        x = self.p["E"][X]
        H, C, G = self.forward_x(x)
        logits = H[:, -1] @ self.p["Wo"] + self.p["bo"]
        return logits, (x, H, C, G)

    # --- backprop through time, starting from dL/dh_T
    def backward_h(self, cache, dh):
        x, H, C, G = cache
        p, h, d = self.p, self.h, self.d
        B, T, _ = x.shape
        gW, gb = np.zeros_like(p["W"]), np.zeros_like(p["b"])
        dx = np.zeros_like(x)
        dc = np.zeros((B, h))
        for t in reversed(range(T)):
            i, f, g, o = (G[:, t, k * h:(k + 1) * h] for k in range(4))
            tc = np.tanh(C[:, t + 1])
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            dz = np.concatenate([(dc * g) * i * (1 - i),         # input gate
                                 (dc * C[:, t]) * f * (1 - f),   # forget gate
                                 (dc * i) * (1 - g ** 2),        # candidate
                                 do * o * (1 - o)], axis=1)      # output gate
            gW += np.concatenate([x[:, t], H[:, t]], axis=1).T @ dz
            gb += dz.sum(axis=0)
            dinp = dz @ p["W"].T
            dx[:, t], dh = dinp[:, :d], dinp[:, d:]
            dc = dc * f                                          # carry through the cell state
        return gW, gb, dx

    def loss_and_grads(self, X, y):
        logits, cache = self.forward(X)
        probs = softmax(logits)
        B = len(y)
        loss = -np.log(probs[np.arange(B), y] + 1e-12).mean()
        dlog = probs.copy(); dlog[np.arange(B), y] -= 1; dlog /= B
        H = cache[1]
        gW, gb, dx = self.backward_h(cache, dlog @ self.p["Wo"].T)
        gE = np.zeros_like(self.p["E"]); np.add.at(gE, X, dx)
        return loss, {"E": gE, "W": gW, "b": gb, "Wo": H[:, -1].T @ dlog, "bo": dlog.sum(0)}

class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-7):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)

def evaluate(model, X, y):
    probs = softmax(model.forward(X)[0])
    return (-np.log(probs[np.arange(len(y)), y] + 1e-12).mean(),
            (probs.argmax(1) == y).mean())

def fit(model, X, y, epochs=100, lr=1e-2, batch=32, seed=0, val=None):
    rng = np.random.default_rng(seed)
    opt = Adam(model.p, lr=lr)
    hist = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            idx = order[s:s + batch]
            _, grads = model.loss_and_grads(X[idx], y[idx])
            opt.step(model.p, grads)
        l, a = evaluate(model, X, y)
        hist["loss"].append(l); hist["acc"].append(a)
        if val is not None:
            vl, va = evaluate(model, *val)
            hist["val_loss"].append(vl); hist["val_acc"].append(va)
    return hist

# ---------------------------------------------------------------- inference
def next_word_probs(model, tok, text, maxlen, padding="pre"):
    X = pad_sequences(tok.texts_to_sequences([text]), maxlen, padding)
    p = softmax(model.forward(X)[0])[0].copy()
    p[0] = 0.0                                                   # index 0 is padding, never a word
    return p

def generate(model, tok, text, n_words, maxlen, padding="pre"):
    for _ in range(n_words):
        text += " " + tok.index_word[int(next_word_probs(model, tok, text, maxlen, padding).argmax())]
    return text

if __name__ == "__main__":
    tok, seqs, X, y, maxlen = build_dataset()
    V = len(tok.word_index) + 1
    print(f"lines={len(seqs)}  vocab(incl. pad)={V}  samples={len(y)}  maxlen={maxlen}")
    model = LSTMLanguageModel(V, d=32, h=64, seed=0)
    assert model.n_params() == count_params(V, 32, 64)
    hist = fit(model, X, y, epochs=100, lr=1e-2, seed=0)
    print(f"params={model.n_params()}  final loss={hist['loss'][-1]:.4f}  train acc={hist['acc'][-1]:.4f}")
    print(generate(model, tok, "what is the fee", 8, maxlen))
```

Output of running this file:

```
lines=30  vocab(incl. pad)=81  samples=205  maxlen=11
params=32689  final loss=0.2042  train acc=0.9024
what is the fee for the data science course sessions course sessions
```

### 9.2 Keras equivalent (not executed here)

Keras/TensorFlow were not installed where I wrote this, so this snippet was **not run**. It follows the summary's architecture (one LSTM layer) with the issues of Section 10 fixed. Compile settings are not given in the summary, so `adam` and categorical cross-entropy are the usual choice, not something taken from the video.

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# X: (N, 56) pre-padded ids, y: (N,) next-word ids, as built in Section 2
model = Sequential([
    Embedding(input_dim=283, output_dim=100),      # no input_length (see Section 10)
    LSTM(150),                                      # returns only the last hidden state
    Dense(283, activation="softmax"),
])
model.build(input_shape=(None, 56))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, to_categorical(y, num_classes=283), epochs=100)
# or: loss="sparse_categorical_crossentropy" and fit(X, y, ...) with no one-hot

# stacked version (what the pasted code intended): the FIRST LSTM must return sequences
# model = Sequential([Embedding(283, 100), LSTM(150, return_sequences=True), LSTM(150), Dense(283, activation="softmax")])

text = "what is the fee"
for _ in range(10):
    ids = tokenizer.texts_to_sequences([text])[0]
    ids = pad_sequences([ids], maxlen=56, padding="pre")
    pos = int(np.argmax(model.predict(ids, verbose=0)))
    text += " " + tokenizer.index_word[pos]         # direct lookup instead of scanning word_index
print(text)
```

## 10. Problems in the pasted Keras code

The code pasted with the summary has several issues worth knowing about. I could not execute Keras here, so items 1 and 2 are based on documented Keras behaviour, not on a run.

1. **Two LSTMs, but the first one lacks `return_sequences=True`.** `LSTM(150)` returns only $h_T$ with shape (batch, 150); a second LSTM expects a 3-D (batch, time, features) input, so Keras should reject this model when it is built. A stacked model needs `LSTM(150, return_sequences=True)` first. This also does not match the summary, which describes a single 150-unit LSTM.
2. **`input_length`.** Recent Keras 3 releases mark the Embedding argument `input_length` as deprecated; dropping it and giving the shape to `build()` or an `Input` layer is safer. Check against your installed version.
3. **Parameter count.** Because of item 1 the two descriptions differ substantially: 221,633 parameters for one LSTM vs 402,233 for two (Figure 5, computed from the formula).
4. **Generation loop.** `model.predict` inside a loop prints a progress bar each call (`verbose=0` silences it), and the inner `for word, index in tokenizer.word_index.items()` scans the whole vocabulary to find a word for an id; `tokenizer.index_word[pos]` is a direct lookup. The `time.sleep(2)` is only for the demo effect.
5. **No stopping rule.** As in Section 8, with no end token the loop will always produce the requested number of words, even when the sentence should have ended.

## 11. Improving the model

The video's last section lists improvements; here is how each relates to what the experiments showed.

- **More data.** The held-out experiment (Section 7.2) shows the gap is a data problem: with 24 training lines the model memorises. More lines mean more distinct prefixes and continuations.
- **Regularisation and early stopping.** Dropout (on the embedding output or inside the LSTM), weight decay, and stopping at the best validation loss. In Section 7.2 the best held-out loss came at epochs 8–14, not 100.
- **Hyperparameter tuning.** Embedding size, number of units, learning rate and batch size. For large vocabularies the output layer and its softmax grow expensive, which is a reason for subword vocabularies or sampled softmax.
- **Bidirectional LSTM.** It reads the known prefix in both directions and concatenates the two final states (Schuster & Paliwal, 1997). It does not leak the answer, because only the prefix is read; it roughly doubles the LSTM parameters.
- **Transformers.** Self-attention (Vaswani et al., 2017) connects all positions directly rather than through a recurrent path and is the basis of current language models. Training on a corpus at scale is what changes the quality, not just the architecture.
- **Better decoding.** Show the top-k words, sample with temperature, or use beam search, instead of greedy argmax.

## 12. Key Takeaways

1. Next-word prediction becomes supervised classification by turning each line into N-gram pairs: a line of $n$ words gives $n-1$ (prefix, next word) samples.
2. Index 0 is reserved for padding, so the Embedding input dimension is `len(word_index) + 1`; this is consistent with the video's 283.
3. Pre-padding keeps the real words adjacent to the position the LSTM reads last. In my 5-seed test post-padding reached similar final accuracy but learned much more slowly (0.62 vs 0.89 after 30 epochs).
4. An LSTM layer has $4h(d+h+1)$ parameters; with 150 units and a 100-dim embedding that is 150,600 for one layer.
5. The cell state's additive update is a gradient highway, but the forget gate sets how open it is: at initialisation, forget bias 0 was worse than a vanilla RNN, bias 1 much better, bias 5 no decay at all.
6. High training accuracy only shows the pipeline works. Here the model hit the exact data ceiling (0.9024) yet got only 0.42 on unseen lines; validation loss rose after epoch ~10.
7. Always verify hand-written gradients numerically (worst error here 2.4e-06); check the model definition too, since the pasted stacked-LSTM code is missing `return_sequences=True`.
8. Without an end-of-sentence token, greedy generation keeps producing words past the natural end of a phrase.

## 13. Further Reading

- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8).
- Gers, F. A., Schmidhuber, J. & Cummins, F. (2000). *Learning to Forget: Continual Prediction with LSTM.* Neural Computation, 12(10).
- Bengio, Y., Simard, P. & Frasconi, P. (1994). *Learning long-term dependencies with gradient descent is difficult.* IEEE Transactions on Neural Networks, 5(2).
- Pascanu, R., Mikolov, T. & Bengio, Y. (2013). *On the difficulty of training recurrent neural networks.* ICML.
- Jozefowicz, R., Zaremba, W. & Sutskever, I. (2015). *An Empirical Exploration of Recurrent Network Architectures.* ICML.
- Bengio, Y., Ducharme, R., Vincent, P. & Jauvin, C. (2003). *A Neural Probabilistic Language Model.* JMLR, 3.
- Mikolov, T., Karafiát, M., Burget, L., Černocký, J. & Khudanpur, S. (2010). *Recurrent neural network based language model.* Interspeech.
- Graves, A. (2013). *Generating Sequences With Recurrent Neural Networks.* arXiv:1308.0850.
- Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR.
- Schuster, M. & Paliwal, K. K. (1997). *Bidirectional recurrent neural networks.* IEEE Transactions on Signal Processing, 45(11).
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Greff, K. et al. (2017). *LSTM: A Search Space Odyssey.* IEEE Transactions on Neural Networks and Learning Systems, 28(10).

## 14. Appendix: experiment code

The script below produced every experiment number and the data behind every plot (it imports the module from Section 9.1). It takes about a minute on one CPU core.

```python
"""Runs every experiment quoted in the notes; writes results.json and arrays.npz."""
import json
import numpy as np
from lstm_numpy import *

R, A = {}, {}

# ---- A. dataset statistics and the best accuracy any model could reach on the training set
tok, seqs, X, y, maxlen = build_dataset()
V = len(tok.word_index) + 1
R.update(lines=len(seqs), vocab=V, samples=len(y), maxlen=maxlen)
groups = {}
for row, t in zip(map(tuple, X), y):
    groups.setdefault(row, []).append(int(t))
R["unique_prefixes"] = len(groups)
R["ceiling"] = sum(max(np.bincount(v)) for v in groups.values()) / len(y)
R["ambiguous_prefixes"] = sum(len(set(v)) > 1 for v in groups.values())
R["correct_at_ceiling"] = int(sum(max(np.bincount(v)) for v in groups.values()))
def _entropy(targets):                      # H(y | prefix) in nats, from empirical counts
    c = np.bincount(targets); c = c[c > 0] / len(targets)
    return float(-(c * np.log(c)).sum())
R["entropy_floor"] = sum(len(v) * _entropy(v) for v in groups.values()) / len(y)

# ---- B. reference model (identical to `python lstm_numpy.py`)
D, H = 32, 64
model = LSTMLanguageModel(V, D, H, seed=0)
hist = fit(model, X, y, epochs=100, lr=1e-2, seed=0)
R.update(params=model.n_params(), final_loss=hist["loss"][-1], final_acc=hist["acc"][-1],
         epoch_acc90=int(next(i + 1 for i, a in enumerate(hist["acc"]) if a >= 0.9)) if max(hist["acc"]) >= 0.9 else None)
A["hist_loss"], A["hist_acc"] = np.array(hist["loss"]), np.array(hist["acc"])
R["generated"] = generate(model, tok, "what is the fee", 8, maxlen)

# per-step distributions during generation
text, steps = "what is the fee", []
for _ in range(5):
    p = next_word_probs(model, tok, text, maxlen)
    top = np.argsort(-p)[:5]
    steps.append({"prefix": text, "words": [tok.index_word[int(i)] for i in top], "probs": [float(p[i]) for i in top]})
    text += " " + tok.index_word[int(p.argmax())]
R["steps"] = steps

# gate activations on one sentence
sent = "what is the fee for the course"
Xs = pad_sequences(tok.texts_to_sequences([sent]), maxlen)
_, (x, Hh, Cc, G) = model.forward(Xs)
A["gates_i"], A["gates_f"], A["gates_o"] = (G[0][:, k * H:(k + 1) * H].T for k in (0, 1, 3))
R["gate_tokens"] = ["<pad>"] * (maxlen - len(sent.split())) + sent.split()
R["gate_mean"] = {n: [float(A["gates_" + n][:, t].mean()) for t in range(maxlen)] for n in "ifo"}

# ---- C. parameter counts (video configuration: vocab 283, d=100, h=150, maxlen 56)
R["video_params_1lstm"] = count_params(283, 100, 150, 1)
R["video_params_2lstm"] = count_params(283, 100, 150, 2)
R["video_parts"] = {"embedding": 283 * 100, "lstm1": 4 * 150 * (100 + 150 + 1),
                    "lstm2": 4 * 150 * (150 + 150 + 1), "dense": 150 * 283 + 283}
assert sum(v for k, v in R["video_parts"].items() if k != "lstm2") == R["video_params_1lstm"]
assert sum(R["video_parts"].values()) == R["video_params_2lstm"]

# ---- D. activation facts
z = np.linspace(-8, 8, 100001)
R["sigmoid_dmax"] = float((sigmoid(z) * (1 - sigmoid(z))).max())
R["tanh_dmax"] = float((1 - np.tanh(z) ** 2).max())

# ---- E. gradient flow: dL/dx_t for L = <h_T, r>, T = 40, random inputs
def rnn_grad_norms(seed, T=40, d=32, h=64, B=32):
    rng = np.random.default_rng(seed)
    lim = np.sqrt(6 / ((d + h) + h))
    W, b = rng.uniform(-lim, lim, (d + h, h)), np.zeros(h)
    x, r = rng.normal(size=(B, T, d)), rng.normal(size=(B, h))
    hs = [np.zeros((B, h))]
    for t in range(T):
        hs.append(np.tanh(np.concatenate([x[:, t], hs[-1]], 1) @ W + b))
    dh, norms = r, np.zeros(T)
    for t in reversed(range(T)):
        dinp = (dh * (1 - hs[t + 1] ** 2)) @ W.T
        norms[t] = np.linalg.norm(dinp[:, :d], axis=1).mean()
        dh = dinp[:, d:]
    return norms

def lstm_grad_norms(seed, forget_bias, T=40, d=32, h=64, B=32):
    m = LSTMLanguageModel(10, d, h, seed=seed, forget_bias=forget_bias)
    rng = np.random.default_rng(seed + 1000)
    x, r = rng.normal(size=(B, T, d)), rng.normal(size=(B, h))
    Hh, Cc, G = m.forward_x(x)
    _, _, dx = m.backward_h((x, Hh, Cc, G), r)
    return np.linalg.norm(dx, axis=2).mean(axis=0)

configs = {"Vanilla RNN (tanh)": lambda s: rnn_grad_norms(s)}
for fb in (0.0, 1.0, 5.0):
    configs[f"LSTM, forget bias {fb:g}"] = (lambda s, fb=fb: lstm_grad_norms(s, fb))
R["gradflow"] = {}
for name, fn in configs.items():
    curves = np.array([fn(s) for s in range(10)])                    # 10 random initialisations
    A["gf_" + name] = curves.mean(0)
    R["gradflow"][name] = {"ratio_first_over_last": float(np.mean(curves[:, 0] / curves[:, -1])),
                           "last": float(curves[:, -1].mean()), "first": float(curves[:, 0].mean())}

# ---- F. pre vs post padding, 5 seeds each
R["padding"], A["pad_loss"], A["pad_acc"] = {}, {}, {}
for mode in ("pre", "post"):
    Xm = pad_sequences([list(r[r > 0]) for r in X], maxlen, mode)
    accs, losses = [], []
    for s in range(5):
        hh = fit(LSTMLanguageModel(V, D, H, seed=s), Xm, y, epochs=100, lr=1e-2, seed=s)
        accs.append(hh["acc"]); losses.append(hh["loss"])
    accs, losses = np.array(accs), np.array(losses)
    A["pad_acc_" + mode], A["pad_loss_" + mode] = accs, losses
    R["padding"][mode] = {"acc_mean": float(accs[:, -1].mean()), "acc_std": float(accs[:, -1].std()),
                          "acc_min": float(accs[:, -1].min()), "acc_max": float(accs[:, -1].max()),
                          "loss_mean": float(losses[:, -1].mean()),
                          "acc_at_epoch30": float(accs[:, 29].mean())}

# ---- G. held-out lines: 5 random splits, 24 train lines / 6 unseen lines
R["heldout"] = []
for split in range(5):
    rng = np.random.default_rng(100 + split)
    held = set(rng.choice(len(seqs), 6, replace=False).tolist())
    Xtr, ytr = make_ngram_pairs([s for i, s in enumerate(seqs) if i not in held])
    Xva, yva = make_ngram_pairs([s for i, s in enumerate(seqs) if i in held])
    Xtr, Xva = pad_sequences(Xtr, maxlen), pad_sequences(Xva, maxlen)
    hh = fit(LSTMLanguageModel(V, D, H, seed=split), Xtr, ytr, epochs=100, lr=1e-2, seed=split, val=(Xva, yva))
    R["heldout"].append({"train_acc": hh["acc"][-1], "val_acc": hh["val_acc"][-1], "val_loss": hh["val_loss"][-1],
                         "train_loss": hh["loss"][-1], "n_train": len(ytr), "n_val": len(yva),
                         "best_val_loss_epoch": int(np.argmin(hh["val_loss"]) + 1), "min_val_loss": float(min(hh["val_loss"]))})
    if split == 0:
        for k in hh: A["ho_" + k] = np.array(hh[k])
R["heldout_mean"] = {k: float(np.mean([r[k] for r in R["heldout"]])) for k in ("train_acc", "val_acc", "val_loss")}
R["heldout_std_val_acc"] = float(np.std([r["val_acc"] for r in R["heldout"]]))

# ---- H. gradient check (float64, tiny model, central differences)
rng = np.random.default_rng(1)
m = LSTMLanguageModel(V=8, d=3, h=4, seed=2, forget_bias=0.5)
m.p["b"] = m.p["b"] + rng.normal(0, 0.3, m.p["b"].shape); m.p["bo"] = m.p["bo"] + rng.normal(0, 0.3, m.p["bo"].shape)
Xg, yg = rng.integers(1, 8, (5, 4)), rng.integers(1, 8, 5)
_, ana = m.loss_and_grads(Xg, yg)
worst, eps = 0.0, 1e-5
for k, v in m.p.items():
    for idx in np.ndindex(v.shape):
        old = v[idx]
        v[idx] = old + eps; lp, _ = m.loss_and_grads(Xg, yg)
        v[idx] = old - eps; lm, _ = m.loss_and_grads(Xg, yg)
        v[idx] = old
        num, a = (lp - lm) / (2 * eps), ana[k][idx]
        if abs(a) + abs(num) > 1e-7:
            worst = max(worst, abs(a - num) / (abs(a) + abs(num)))
R["gradcheck_worst_rel_err"] = float(worst)

# token ids for the pipeline figure
first = seqs[0]
R["first_line"] = CORPUS.split("\n")[0]
R["first_ids"] = first
R["pairs_first"] = [(first[:i], first[i]) for i in range(1, len(first))]

json.dump(R, open("results.json", "w"), indent=1, default=float)
np.savez("arrays.npz", **A)
print(json.dumps({k: v for k, v in R.items() if k not in ("steps", "pairs_first", "gate_mean", "heldout")}, indent=1, default=float)[:3500])
```
