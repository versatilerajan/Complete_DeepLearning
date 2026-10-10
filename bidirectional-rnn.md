# Bidirectional RNNs: Reading a Sentence in Both Directions

An ordinary RNN reads a sentence left to right, so the state at word $t$ knows only words $1,\dots,t$. Many language tasks need the words that come *after* as well: in "Amazon the best website", the word "Amazon" alone could be a company or a river, and only the later word "website" settles it. A Bidirectional RNN runs two independent RNNs over the sentence, one forward and one backward, and concatenates their states at every position. These notes cover the idea, the three formulas from class (included in Section 3), and a from-scratch NumPy implementation of a Bidirectional SimpleRNN and a Bidirectional LSTM with 5 units per direction, with real experiments on what the second direction buys and what it costs. The Keras layers `Bidirectional(SimpleRNN(5))` and `Bidirectional(LSTM(5))` are shown in Section 9, but Keras was not installed where this was written, so that code is **not executed**. The training data is a small synthetic task I generated for this note, not data from the video.

Related notes in this series: `lstm-next-word-predictor.md` (the LSTM cell and gates used in Section 5) and `rnn.md` (assumed filename for your basic RNN note, adjust as needed).

## Table of Contents

1. [Why one direction is not enough](#1-why-one-direction-is-not-enough)
2. [The architecture](#2-the-architecture)
3. [The three formulas from class](#3-the-three-formulas-from-class)
4. [What each state can see](#4-what-each-state-can-see)
5. [Bidirectional LSTM](#5-bidirectional-lstm)
6. [Experiment: who can resolve "Amazon"?](#6-experiment-who-can-resolve-amazon)
7. [Real-time use: predictions that change as words arrive](#7-real-time-use-predictions-that-change-as-words-arrive)
8. [The cost: parameters and time](#8-the-cost-parameters-and-time)
9. [Complete code](#9-complete-code)
10. [Applications and limits](#10-applications-and-limits)
11. [Key Takeaways](#11-key-takeaways)
12. [Further Reading](#12-further-reading)
13. [Appendix: experiment code](#13-appendix-experiment-code)

---

## 1. Why one direction is not enough

A unidirectional RNN computes $h_t = f(x_t, h_{t-1})$, so $h_t$ is a function of $x_1,\dots,x_t$ only. For tagging tasks such as Named Entity Recognition (NER) the label of a word often depends on its right-hand context:

- "**Amazon** the best website ..." : Amazon is an organisation.
- "**Amazon** river basin ..." : Amazon is a place.

At the first word both sentences look identical to a left-to-right RNN. In your class screenshot the note on the right reads "website → org" for the first case, and there is a second label that looks like "loc" (the handwriting is hard to read, so that part is my interpretation). The idea of the lesson is that the information "website" must travel *backwards* to the position of "Amazon", and a forward RNN has no path for that. Section 6 measures exactly this.

## 2. The architecture

The model uses two completely separate RNNs over the same input:

1. **Forward RNN**: reads $x_1, x_2, \dots, x_T$ and produces $\overrightarrow{h}_1,\dots,\overrightarrow{h}_T$, starting from a zero state.
2. **Backward RNN**: reads $x_T, x_{T-1}, \dots, x_1$ and produces $\overleftarrow{h}_T,\dots,\overleftarrow{h}_1$, starting from its own zero state at the right end.
3. At every position $t$ the two states are **concatenated** into a vector of length $2u$ and passed to an output layer that produces $y_t$.

![Figure 1: unrolled bidirectional RNN](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/birnn_01_unrolled_bidirectional_rnn.png)

*Figure 1. The diagram from your class screenshot, redrawn for the words "Amazon the best website" at $t=1,\dots,4$. Blue boxes are the forward RNN (information flows left to right, starting from "zeros" on the left); green boxes are the backward RNN (information flows right to left, starting from "zeros" on the right). Each position concatenates its blue and green state (purple box) to produce $y_t$ (orange). The yellow band is the highlighted path in the screenshot: it is the only route by which the word "website" can influence the state at "Amazon", which is why the backward RNN is needed.*

Implementation detail: the backward RNN is an ordinary RNN applied to the reversed sequence, and its outputs are flipped back so position $t$ lines up with the forward state at position $t$. The NumPy code in Section 9 does exactly this (`x[:, ::-1]`).

## 3. The three formulas from class

With $u$ hidden units per direction and input size $d$:

$$\overrightarrow{h}_t=\tanh\!\left(\overrightarrow{W}\,\overrightarrow{h}_{t-1}+\overrightarrow{U}\,x_t+\overrightarrow{b}\right)\qquad\overrightarrow{h}_0=\mathbf 0$$

$$\overleftarrow{h}_t=\tanh\!\left(\overleftarrow{W}\,\overleftarrow{h}_{t+1}+\overleftarrow{U}\,x_t+\overleftarrow{b}\right)\qquad\overleftarrow{h}_{T+1}=\mathbf 0$$

$$y_t=\sigma\!\left(V\,[\overrightarrow{h}_t,\overleftarrow{h}_t]+b_y\right)$$

| Symbol | Shape | Meaning |
|---|---|---|
| $\overrightarrow{W},\overleftarrow{W}$ | $u\times u$ | recurrent weights (one set per direction, not shared) |
| $\overrightarrow{U},\overleftarrow{U}$ | $u\times d$ | input weights |
| $\overrightarrow{b},\overleftarrow{b}$ | $u$ | hidden biases |
| $V$ | $1\times 2u$ | output weights acting on the concatenated state |
| $b_y$ | $1$ | output bias (the screenshot writes it as plain $b$; it is a different parameter from the hidden biases) |

The screenshot's output uses a sigmoid, which suits a yes/no question per word ("is this word an organisation?"). For several tags you replace $\sigma$ by a softmax over tags and $V$ by a $K\times2u$ matrix. In Keras, `kernel` is $U^\top$ ($d\times u$), `recurrent_kernel` is $W^\top$ ($u\times u$) and `bias` is $b$.

![Figure 2: the formulas annotated](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/birnn_02_teacher_formulas_annotated.png)

*Figure 2. The three formulas from the screenshot with each term explained. The forward state looks one step back ($t-1$), the backward state looks one step forward ($t+1$), and the output layer sees both. The two RNNs have separate parameters, so a Bidirectional layer holds two complete parameter sets.*

![Figure 3: tanh and sigmoid](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/birnn_03_tanh_sigmoid_with_derivatives.png)

*Figure 3. The two nonlinearities in the formulas, each with its derivative (dashed). $\tanh$ keeps every hidden unit between -1 and 1 and has a maximum slope of 1 at zero; $\sigma$ squashes the output score into a probability and has a maximum slope of 0.25. Both slopes shrink towards 0 for large $\lvert z\rvert$, the saturation that makes gradients vanish over long sequences (see `lstm-next-word-predictor.md`).*

**Checking the formulas against the code.** I evaluated the three formulas with explicit loops for the screenshot's 4-word case (random weights, $u=4$, $d=3$, zeros at both ends) and compared with my layer implementation, which uses one concatenated weight matrix. The outputs were $y_1,\dots,y_4=$ 0.5885, 0.4196, 0.7790, 0.6751 from the formulas, and the largest difference to the layer code was 1.1e-16. The random weights are only for the check; these probabilities mean nothing linguistically.

## 4. What each state can see

To make "forward sees the past, backward sees the future" precise I measured, on a randomly initialised Bidirectional SimpleRNN ($u=5$, $d=4$, $T=4$), the size of the derivative $\lVert\partial h_t/\partial x_k\rVert$ of the state at position $t$ with respect to each input word $k$, by finite differences. A zero means that state is completely independent of that word.

![Figure 4: which inputs each state sees](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/birnn_04_which_inputs_each_state_sees.png)

*Figure 4. Rows are the position $t$ of the state, columns are the input words. The forward state (left) depends only on the current and earlier words, a lower-triangular pattern with 10 of 16 entries non-zero. The backward state (middle) is the mirror image with 10 of 16. The concatenation (right) depends on every word at every position, 16 of 16. The numbers are derivative norms; the zeros are exact zeros. In particular the state at "Amazon" ($t=1$) changes with "website" only through the backward half.*

## 5. Bidirectional LSTM

Nothing in the architecture requires the cell to be a SimpleRNN. A Bidirectional LSTM uses an LSTM cell in each direction (gates and cell state exactly as in `lstm-next-word-predictor.md`), and the concatenated hidden states $[\overrightarrow{h}_t,\overleftarrow{h}_t]$ feed the same output layer. An LSTM cell has four gate blocks, so it has four times the recurrent parameters of a SimpleRNN with the same number of units: $4u(d+u+1)$ versus $u(d+u+1)$. With $u=5$ and an 8-dimensional embedding that is 280 per direction (560 for the Bidirectional layer) versus 70 (140) for SimpleRNN. GRU works the same way. In Keras the wrapper is the same for all three: `Bidirectional(SimpleRNN(5))`, `Bidirectional(LSTM(5))`, `Bidirectional(GRU(5))`.

## 6. Experiment: who can resolve "Amazon"?

**Task.** Every training sentence has 6 words: filler words plus one ambiguous word (`amazon`, `apple` or `jordan`) and one cue word. Company cues are `website`, `stock`, `store`; place cues are `river`, `forest`, `border`. The label $y_t$ is 1 only at the ambiguous word when the cue is a company cue, and 0 everywhere else. In half of the sentences the cue comes **after** the ambiguous word (needs the future), in the other half **before** it (needs the past). The vocabulary has 20 ids including padding, the embedding has 8 dimensions, there are 3000 training and 1000 test sentences, and the output is the sigmoid of the screenshot with binary cross-entropy.

Since 0 is the right label for most tokens, plain accuracy is misleading: always answering 0 scores 0.920 over all tokens. The tables therefore report accuracy **on the ambiguous word only**, split by where the cue is.

I trained four model types with 5 units per direction (so the Bidirectional model has 5 forward + 5 backward) and a 10-unit forward-only model that has more parameters than the bidirectional one, for both cell types, 100 epochs, Adam (learning rate 0.01), batch 64, 5 random seeds each.

| Model | Params | Acc. when cue is **after** | Acc. when cue is **before** | Acc. over all tokens | Seeds (of 5) that learned everything their direction can see |
|---|---|---|---|---|---|
| SimpleRNN forward only (5) | 236 | 0.521 ± 0.011 | 1.000 | 0.958 | 5 |
| SimpleRNN forward only (10) | 361 | 0.524 ± 0.016 | 1.000 | 0.958 | 5 |
| SimpleRNN backward only (5) | 236 | 1.000 | 0.522 ± 0.006 | 0.963 | 5 |
| **SimpleRNN bidirectional (5+5)** | 311 | **1.000** | **1.000** | **1.000** | 5 |
| LSTM forward only (5) | 446 | 0.506 ± 0.019 | 1.000 | 0.956 | 5 |
| LSTM forward only (10) | 931 | 0.509 ± 0.010 | 1.000 | 0.956 | 5 |
| LSTM backward only (5) | 446 | 0.808 ± 0.235 | 0.523 ± 0.010 | 0.946 | 3 |
| **LSTM bidirectional (5+5)** | 731 | **1.000** | **1.000** | **1.000** | 5 |

(± is the standard deviation over the 5 seeds, shown in the two accuracy columns wherever it is not zero; the all-tokens column is a mean.)

![Figure 5: accuracy by direction](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/birnn_05_accuracy_needs_both_directions.png)

*Figure 5. Accuracy on the ambiguous word, for sentences whose cue comes after it (orange, needs the future) and before it (blue, needs the past); bars are means over 5 seeds and the black dots are the individual seeds. A forward-only model is perfect when the cue is earlier but is reduced to a coin flip (about 0.5) when the cue is later, because that information has not reached it yet. A backward-only model is the mirror image. Only the bidirectional model is perfect on both, for SimpleRNN and for LSTM. Note that doubling the forward-only model to 10 units (more parameters than the bidirectional one) does not help: the missing ingredient is information, not capacity.*

What the numbers say:

- **Direction, not size, is what matters.** Forward-only models stay at about 0.5 on "cue after" words whether they have 5 or 10 units, and the bidirectional models have fewer parameters than the 10-unit forward-only ones (311 vs 361 for SimpleRNN, 731 vs 931 for LSTM) yet reach 1.000 and 1.000.
- **The training loss shows the same ceiling.** Half of the ambiguous words have the cue after them, and for those the best a forward-only model can do is predict 0.5, which costs $\ln 2$ per such word. Averaged over all tokens in the training set that is a loss floor of $\ln 2\cdot(\text{fraction of cue-after sentences})/6=$ 0.0586, and the forward-only SimpleRNN models end at 0.059 (5 units) and 0.058 (10 units), i.e. at the floor. The bidirectional models go to 0.000 (SimpleRNN) and 0.001 (LSTM).
- **Optimisation can still fail.** The backward-only LSTM(5) learned its "cue after" case in only 3 of 5 seeds; the other two seeds ended at about 0.5 accuracy after 100 epochs (their individual accuracies are the orange dots above the backward-only LSTM bar in Figure 5). This is a training failure on one configuration, not an architectural limit, since the same architecture succeeds on other seeds. In an earlier exploratory run with 40 epochs the 5-unit LSTMs were also under-trained, which is why the final experiment uses 100 epochs.

![Figure 6: training loss](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/birnn_06_training_loss.png)

*Figure 6. Training loss (binary cross-entropy, log scale) averaged over 5 seeds with the min-max range shaded. The unidirectional models flatten at about 0.06, the loss floor computed above (the backward-only LSTM sits higher for the seeds that did not learn); the bidirectional models keep falling towards zero because they can see the cue word wherever it is. The wide band for the backward-only LSTM comes from those seeds.*

## 7. Real-time use: predictions that change as words arrive

The video points out that bidirectional models need the whole sequence, so they are a poor fit for streaming input such as live speech recognition. To see what that means in practice I fed the trained seed-0 models the sentence "amazon the best website our great" one word at a time (prefixes of length 1 to 6) and recorded the probability that "amazon" is a company. I repeated it with "river" instead of "website".

![Figure 7: streaming predictions](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/birnn_07_streaming_prediction_changes.png)

*Figure 7. Probability that "amazon" is a company as more words arrive. The forward-only model (grey dashed, about 0.59 for SimpleRNN and 0.58 for LSTM) never changes, because its output for the first word cannot depend on later words; it answers immediately but cannot use the cue. The bidirectional model's answer for the first word changes completely when the fourth word arrives: to 1.00 for "website" and 0.00 for "river" (SimpleRNN; LSTM values are 1.00 and 0.00). Before the cue arrives, the bidirectional values differ between the two cell types (about 0.28 for SimpleRNN and 0.94 for LSTM at the first word). Those are arbitrary guesses: every training sentence contained its cue word, so a prefix without one was never seen in training.*

So a bidirectional model cannot give a final answer for a word until the words after it exist, and its earlier answers are revised retroactively. For offline tasks (tagging a finished document) this is fine; for live speech it adds latency of at least the amount of right-hand context you need, and a common workaround is to run the model on short chunks with limited look-ahead (not tested here).

## 8. The cost: parameters and time

![Figure 8: parameters and time](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/birnn_08_parameters_and_time_cost.png)

*Figure 8. Left: trainable parameters of the models in Section 6, split into embedding, recurrent layer(s) and output layer. The recurrent part exactly doubles when the layer becomes bidirectional (70 to 140 for SimpleRNN(5), 280 to 560 for LSTM(5)), while the embedding is unchanged and the output layer grows because it now reads $2u$ inputs. Right: time for one training step (forward plus backward pass, batch 64, 50 words, median of 60 runs) in my NumPy implementation: the bidirectional layer took 1.79 times as long for SimpleRNN and 1.98 times for LSTM. The timing is specific to my single-core NumPy loops; it shows the extra work of a second sequential pass, not what you will see in Keras on a GPU.*

The video also says doubled parameters raise the risk of overfitting. That is plausible (more capacity), but I did not test it: on this task the bidirectional model had fewer parameters than the 10-unit forward-only baselines and was the only one that solved it, so more parameters were not the issue here.

## 9. Complete code

### 9.1 From-scratch NumPy implementation (run and verified)

This is the exact file behind every number above (except the experiment aggregation in the appendix). It contains the data generator, SimpleRNN and LSTM layers with hand-written backpropagation through time, the bidirectional wrapper (reverse, run, reverse back, concatenate), the sigmoid output of the screenshot, Adam, and a demo. I checked every combination of cell type (SimpleRNN, LSTM) and direction (forward, backward, bidirectional) against numerical gradients; the worst relative error over all parameters was 1.3e-05.

```python
"""Bidirectional RNN / LSTM sequence tagger in pure NumPy (hand-written BPTT).
Per-time-step output  y_t = sigmoid(V [h_fwd_t, h_bwd_t] + c)   (the teacher's formula).
Run `python bi_rnn_numpy.py` for a quick end-to-end demo."""
import numpy as np

# ------------------------------------------------------------------ synthetic NER-style data
FILLERS = ["the", "best", "new", "famous", "a", "visit", "our", "great", "online", "big"]
AMBIG = ["amazon", "apple", "jordan"]                  # could be a company (1) or a place (0)
ORG_CUES = ["website", "stock", "store"]               # context words that make it a company
LOC_CUES = ["river", "forest", "border"]               # context words that make it a place
WORDS = ["<pad>"] + FILLERS + AMBIG + ORG_CUES + LOC_CUES
W2I = {w: i for i, w in enumerate(WORDS)}
T_LEN = 6

def make_dataset(n, seed):
    """Each sentence holds ONE ambiguous word and ONE cue word. Half the sentences have the
    cue AFTER the ambiguous word (future context needed), half BEFORE it (past context).
    y[t] = 1 only at the ambiguous word when the cue says 'company', else 0."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, T_LEN), dtype=int)
    y = np.zeros((n, T_LEN))
    pos_a = np.zeros(n, dtype=int)
    cue_after = np.zeros(n, dtype=bool)
    for k in range(n):
        after, org = rng.random() < 0.5, rng.random() < 0.5
        if after:
            a = rng.integers(0, T_LEN - 1); c = rng.integers(a + 1, T_LEN)
        else:
            c = rng.integers(0, T_LEN - 1); a = rng.integers(c + 1, T_LEN)
        row = [FILLERS[i] for i in rng.integers(0, len(FILLERS), T_LEN)]
        row[a] = AMBIG[rng.integers(0, len(AMBIG))]
        row[c] = (ORG_CUES if org else LOC_CUES)[rng.integers(0, 3)]
        X[k] = [W2I[w] for w in row]
        y[k, a] = float(org); pos_a[k] = a; cue_after[k] = after
    return X, y, pos_a, cue_after

# ------------------------------------------------------------------ recurrent layers (hidden sequence out)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def rnn_forward(x, W, b, u):
    """h_t = tanh([x_t, h_{t-1}] W + b)  ==  tanh(U x_t + W_rec h_{t-1} + b); h_0 = 0."""
    B, T, _ = x.shape
    H = np.zeros((B, T + 1, u))
    for t in range(T):
        H[:, t + 1] = np.tanh(np.concatenate([x[:, t], H[:, t]], 1) @ W + b)
    return H[:, 1:], (x, H)

def rnn_backward(dHs, cache, W, u):
    x, H = cache
    B, T, d = x.shape
    gW, gb, dx, dh_next = np.zeros_like(W), np.zeros(W.shape[1]), np.zeros_like(x), np.zeros((B, u))
    for t in reversed(range(T)):
        dz = (dHs[:, t] + dh_next) * (1 - H[:, t + 1] ** 2)
        gW += np.concatenate([x[:, t], H[:, t]], 1).T @ dz
        gb += dz.sum(0)
        dinp = dz @ W.T
        dx[:, t], dh_next = dinp[:, :d], dinp[:, d:]
    return gW, gb, dx

def lstm_forward(x, W, b, u):
    B, T, _ = x.shape
    H, C, G = np.zeros((B, T + 1, u)), np.zeros((B, T + 1, u)), np.zeros((B, T, 4 * u))
    for t in range(T):
        z = np.concatenate([x[:, t], H[:, t]], 1) @ W + b
        i, f = sigmoid(z[:, :u]), sigmoid(z[:, u:2 * u])
        g, o = np.tanh(z[:, 2 * u:3 * u]), sigmoid(z[:, 3 * u:])
        C[:, t + 1] = f * C[:, t] + i * g
        H[:, t + 1] = o * np.tanh(C[:, t + 1])
        G[:, t] = np.concatenate([i, f, g, o], 1)
    return H[:, 1:], (x, H, C, G)

def lstm_backward(dHs, cache, W, u):
    x, H, C, G = cache
    B, T, d = x.shape
    gW, gb, dx = np.zeros_like(W), np.zeros(W.shape[1]), np.zeros_like(x)
    dh_next, dc = np.zeros((B, u)), np.zeros((B, u))
    for t in reversed(range(T)):
        i, f, g, o = (G[:, t, k * u:(k + 1) * u] for k in range(4))
        dh = dHs[:, t] + dh_next
        tc = np.tanh(C[:, t + 1])
        dc = dc + dh * o * (1 - tc ** 2)
        dz = np.concatenate([(dc * g) * i * (1 - i), (dc * C[:, t]) * f * (1 - f),
                             (dc * i) * (1 - g ** 2), (dh * tc) * o * (1 - o)], 1)
        gW += np.concatenate([x[:, t], H[:, t]], 1).T @ dz
        gb += dz.sum(0)
        dinp = dz @ W.T
        dx[:, t], dh_next = dinp[:, :d], dinp[:, d:]
        dc = dc * f
    return gW, gb, dx

LAYERS = {"rnn": (rnn_forward, rnn_backward, 1), "lstm": (lstm_forward, lstm_backward, 4)}

def run_dir(kind, x, W, b, u, reverse):
    """Run one direction. The backward RNN reads the reversed sequence, then its outputs are
    flipped back so that position t holds the state that has seen x_t ... x_T (h_bwd_t)."""
    fwd = LAYERS[kind][0]
    H, cache = fwd(x[:, ::-1] if reverse else x, W, b, u)
    return (H[:, ::-1] if reverse else H), cache

def back_dir(kind, dH, cache, W, u, reverse):
    gW, gb, dx = LAYERS[kind][1](dH[:, ::-1] if reverse else dH, cache, W, u)
    return gW, gb, (dx[:, ::-1] if reverse else dx)

# ------------------------------------------------------------------ the tagger
def count_params(kind, V, d, u, direction):
    gates = LAYERS[kind][2]
    n_dir = 2 if direction == "bi" else 1
    return V * d + n_dir * gates * u * (d + u + 1) + n_dir * u + 1

class Tagger:
    """Embedding(V,d) -> {forward | backward | Bidirectional} recurrent layer(u) -> Dense(1, sigmoid) per step."""
    def __init__(self, V, d, u, kind="rnn", direction="bi", seed=0):
        rng = np.random.default_rng(seed)
        self.kind, self.u, self.d, self.direction = kind, u, d, direction
        self.dirs = {"fwd": ["f"], "bwd": ["b"], "bi": ["f", "b"]}[direction]
        g, p = LAYERS[kind][2], {"E": rng.uniform(-0.05, 0.05, (V, d))}
        for s in self.dirs:
            lim = np.sqrt(6.0 / ((d + u) + g * u))                     # Glorot uniform
            p["W" + s] = rng.uniform(-lim, lim, (d + u, g * u))
            p["b" + s] = np.zeros(g * u)
            if kind == "lstm":
                p["b" + s][u:2 * u] = 1.0                              # unit forget bias, as in Keras
        n_out = u * len(self.dirs)
        lim = np.sqrt(6.0 / (n_out + 1))
        p["V"], p["c"] = rng.uniform(-lim, lim, (n_out, 1)), np.zeros(1)
        self.p = p

    def n_params(self):
        return sum(v.size for v in self.p.values())

    def forward(self, X):
        x = self.p["E"][X]
        feats, caches = [], {}
        for s in self.dirs:                                            # order: [forward, backward]
            H, caches[s] = run_dir(self.kind, x, self.p["W" + s], self.p["b" + s], self.u, s == "b")
            feats.append(H)
        F = np.concatenate(feats, axis=2)                              # [h_fwd_t , h_bwd_t]
        prob = sigmoid(F @ self.p["V"] + self.p["c"])[:, :, 0]
        return prob, (x, F, caches)

    def loss_and_grads(self, X, y):
        prob, (x, F, caches) = self.forward(X)
        B, T = y.shape
        loss = -(y * np.log(prob + 1e-12) + (1 - y) * np.log(1 - prob + 1e-12)).mean()
        dlogit = ((prob - y) / (B * T))[:, :, None]
        grads = {"V": np.einsum("btf,bto->fo", F, dlogit), "c": dlogit.sum((0, 1))}
        dF = dlogit @ self.p["V"].T
        dx = np.zeros_like(x)
        for k, s in enumerate(self.dirs):
            gW, gb, dxs = back_dir(self.kind, dF[:, :, k * self.u:(k + 1) * self.u], caches[s],
                                   self.p["W" + s], self.u, s == "b")
            grads["W" + s], grads["b" + s] = gW, gb
            dx += dxs
        gE = np.zeros_like(self.p["E"]); np.add.at(gE, X, dx); grads["E"] = gE
        return loss, grads

class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-7):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            params[k] -= self.lr * (self.m[k] / (1 - self.b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - self.b2 ** self.t)) + self.eps)

def fit(model, X, y, epochs=40, batch=64, lr=1e-2, seed=0):
    rng, opt, losses = np.random.default_rng(seed), Adam(model.p, lr), []
    for _ in range(epochs):
        order, tot = rng.permutation(len(y)), 0.0
        for s in range(0, len(y), batch):
            idx = order[s:s + batch]
            loss, grads = model.loss_and_grads(X[idx], y[idx])
            opt.step(model.p, grads); tot += loss * len(idx)
        losses.append(tot / len(y))
    return losses

def ambiguous_accuracy(model, X, y, pos_a, cue_after):
    """Accuracy on the ambiguous token only (the other tokens are trivially 0)."""
    pred = model.forward(X)[0] > 0.5
    ok = pred[np.arange(len(y)), pos_a] == (y[np.arange(len(y)), pos_a] > 0.5)
    return float(ok[cue_after].mean()), float(ok[~cue_after].mean())      # (cue after, cue before)

if __name__ == "__main__":
    Xtr, ytr, *_ = make_dataset(3000, seed=1)
    Xte, yte, pa, ca = make_dataset(1000, seed=2)
    for direction in ("fwd", "bi"):
        m = Tagger(len(WORDS), d=8, u=5, kind="rnn", direction=direction, seed=0)
        fit(m, Xtr, ytr, epochs=40, seed=0)
        after, before = ambiguous_accuracy(m, Xte, yte, pa, ca)
        print(f"{direction:>3}: params={m.n_params():3d}  acc(cue AFTER word)={after:.3f}  acc(cue BEFORE word)={before:.3f}")
```

Output of running this file (a quick demo: SimpleRNN, 40 epochs, one seed; the tables above use 100 epochs and 5 seeds):

```
fwd: params=236  acc(cue AFTER word)=0.511  acc(cue BEFORE word)=1.000
 bi: params=311  acc(cue AFTER word)=1.000  acc(cue BEFORE word)=1.000
```

### 9.2 Keras: `Bidirectional(SimpleRNN(5))` and `Bidirectional(LSTM(5))` (not executed)

Keras and TensorFlow were not installed in the environment where I wrote this, so none of this snippet was run. Keras has no layers literally named `BidirectionalRNN` or `BidirectionalLSTM`; you wrap a recurrent layer in the `Bidirectional` layer. The parameter counts in the comments are computed from the formulas in Section 5 and match the NumPy models above (311 and 731 in total).

![Figure 9: Keras layer shapes](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/birnn_09_keras_layer_shapes.png)

*Figure 9. Tensor shapes and parameter counts for the Keras models below, computed from the layer formulas (not from a `summary()` run, since Keras was not available). Row A keeps one output per word with `return_sequences=True`, which matches the screenshot's $y_t$ for every $t$; row B keeps only the final pair of states and produces one label per sentence. The Bidirectional layer outputs 10 features because it concatenates 5 forward and 5 backward units; swapping SimpleRNN for LSTM changes only the 140 into 560.*

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, SimpleRNN, LSTM, Dense

VOCAB, EMB, T = 20, 8, 6                      # as in Section 6

def build(cell, return_sequences=True):
    model = Sequential([
        Embedding(VOCAB, EMB),                                        # (B, T) -> (B, T, 8)    160 params
        Bidirectional(cell(5, return_sequences=return_sequences)),    # (B, T, 10) or (B, 10)  RNN 140 / LSTM 560
        Dense(1, activation="sigmoid"),                               # applied per word when 3-D   11 params
    ])
    model.build(input_shape=(None, T))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

bi_rnn  = build(SimpleRNN)     # the notes' "BidirectionalRNN(5)"   -> 311 params
bi_lstm = build(LSTM)          # the notes' "BidirectionalLSTM(5)"  -> 731 params
bi_rnn.summary(); bi_lstm.summary()

# Xtr: (N, 6) integer ids, ytr: (N, 6) 0/1 labels, from make_dataset in Section 9.1
bi_rnn.fit(Xtr, ytr[..., None], epochs=100, batch_size=64)

# One label per sentence instead of one per word (return_sequences=False):
# model = build(SimpleRNN, return_sequences=False); model.fit(Xtr, ytr.max(axis=1), ...)
```

Notes on the Keras behaviour (from its documentation, not verified here):

- `merge_mode="concat"` is the default, so the layer outputs $2u=10$ features per word. Other options are `"sum"`, `"mul"` and `"ave"`, which keep width $u$.
- With `return_sequences=True` the backward outputs are re-aligned so position $t$ holds $[\overrightarrow{h}_t,\overleftarrow{h}_t]$, as in my implementation. With `return_sequences=False` you get the forward state after the last word and the backward state after the first word, i.e. both "have read the whole sentence".
- The `Bidirectional` wrapper copies the layer you give it for the backward direction, so the two directions get independent weights.

## 10. Applications and limits

**Good fits** (the whole input is available before predicting): Named Entity Recognition, Part-of-Speech tagging, the encoder of machine translation models (Bahdanau et al., 2015), speech transcription of recorded audio, and time-series work on a complete window such as filling in missing values.

**Limits:**

1. **Twice the parameters and roughly twice the compute** (Section 8).
2. **Needs the whole sequence**, so it cannot give final answers in a live stream (Section 7).
3. **Not for generating the next word.** If you ran a Bidirectional RNN over a whole sentence and trained it to predict word $t+1$ at every position $t$, the backward half would see word $t+1$ and the model would learn to copy it. This is why the next-word predictor in `lstm-next-word-predictor.md` can use a Bidirectional LSTM only over the *prefix* it is given (which contains no future), and why the decoder of a translation model is unidirectional.

## 11. Key Takeaways

1. A Bidirectional RNN is two separate RNNs, one reading left to right and one right to left, with their states concatenated at each position: $y_t=\sigma(V[\overrightarrow{h}_t,\overleftarrow{h}_t]+b_y)$.
2. The forward state at $t$ depends only on words $1..t$, the backward state only on words $t..T$; measured: 10, 10 and 16 of 16 word-to-position links respectively.
3. On the ambiguous-word task a forward-only model scores about 0.52 when the clue is later in the sentence, a backward-only model about 0.52 when the clue is earlier, and the bidirectional model 1.00 in both cases, with fewer parameters than a 10-unit forward-only model.
4. The same wrapper works for SimpleRNN, LSTM and GRU; an LSTM(5) costs four times the recurrent parameters of a SimpleRNN(5) (280 vs 70 here), and the bidirectional version doubles each.
5. In Keras use `Bidirectional(SimpleRNN(5))` or `Bidirectional(LSTM(5))`; there are no classes called `BidirectionalRNN` or `BidirectionalLSTM`. (My Keras snippet is unexecuted.)
6. The costs are real: parameters in the recurrent layer exactly double, and in my NumPy implementation a step took 1.8 to 2.0 times longer.
7. Early outputs of a bidirectional model are revised when later words arrive (the probability for "amazon" jumped from 0.34 to 1.00 when "website" appeared), so it is unsuitable for live streaming and for next-word generation.
8. Training can still fail on a hard configuration: the backward-only LSTM(5) learned its case in only 3 of 5 seeds.

## 12. Further Reading

- Schuster, M. & Paliwal, K. K. (1997). *Bidirectional recurrent neural networks.* IEEE Transactions on Signal Processing, 45(11).
- Graves, A. & Schmidhuber, J. (2005). *Framewise phoneme classification with bidirectional LSTM and other neural network architectures.* Neural Networks, 18(5-6).
- Graves, A., Mohamed, A. & Hinton, G. (2013). *Speech recognition with deep recurrent neural networks.* ICASSP.
- Huang, Z., Xu, W. & Yu, K. (2015). *Bidirectional LSTM-CRF models for sequence tagging.* arXiv:1508.01991.
- Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K. & Dyer, C. (2016). *Neural architectures for named entity recognition.* NAACL.
- Bahdanau, D., Cho, K. & Bengio, Y. (2015). *Neural machine translation by jointly learning to align and translate.* ICLR.
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8).
- Cho, K. et al. (2014). *Learning phrase representations using RNN encoder-decoder for statistical machine translation.* EMNLP.
- Elman, J. L. (1990). *Finding structure in time.* Cognitive Science, 14(2).
- Pascanu, R., Mikolov, T. & Bengio, Y. (2013). *On the difficulty of training recurrent neural networks.* ICML.

## 13. Appendix: experiment code

The script below produced every experiment number and the data behind every plot (it imports the module from Section 9.1). It saves each model configuration as it finishes, so the whole run takes a few minutes on one CPU core and can be resumed.

```python
"""Runs every experiment quoted in the Bidirectional-RNN notes; writes results.json and arrays.npz."""
import json, time
import numpy as np
from bi_rnn_numpy import *

R, A = {}, {}
V, D, U = len(WORDS), 8, 5
Xtr, ytr, _, ca_tr = make_dataset(3000, seed=1)
Xte, yte, pa, ca = make_dataset(1000, seed=2)
R['loss_floor_uni'] = float(np.log(2) * ca_tr.mean() / T_LEN)   # best any forward-only model can do: coin flip on 'cue after' words
R.update(vocab=V, emb=D, units=U, n_train=len(ytr), n_test=len(yte), T=T_LEN,
         frac_after=float(ca.mean()), majority_baseline_all_tokens=float(1 - yte.mean()))

# ---- A. gradient check (float64, tiny model, central differences) for every direction x cell type
R["gradcheck"] = {}
for kind in ("rnn", "lstm"):
    for direction in ("fwd", "bwd", "bi"):
        m = Tagger(V, 3, 4, kind=kind, direction=direction, seed=3)
        rng = np.random.default_rng(5)
        for k in m.p:
            if k.startswith("b") or k == "c":
                m.p[k] = m.p[k] + rng.normal(0, 0.3, m.p[k].shape)
        X, y, *_ = make_dataset(6, seed=4)
        _, ana = m.loss_and_grads(X, y)
        worst, eps = 0.0, 1e-5
        for k, v in m.p.items():
            for idx in np.ndindex(v.shape):
                old = v[idx]
                v[idx] = old + eps; lp, _ = m.loss_and_grads(X, y)
                v[idx] = old - eps; lm, _ = m.loss_and_grads(X, y)
                v[idx] = old
                num, a = (lp - lm) / (2 * eps), ana[k][idx]
                if abs(a) + abs(num) > 1e-7:
                    worst = max(worst, abs(a - num) / (abs(a) + abs(num)))
        R["gradcheck"][f"{kind}-{direction}"] = float(worst)
R["gradcheck_worst"] = max(R["gradcheck"].values())

# ---- B. which inputs can each hidden state see? Jacobian norms ||dh_t / dx_k|| (T=4, like the screenshot)
rng = np.random.default_rng(0)
Tj, dj = 4, 4
x0 = rng.normal(size=(1, Tj, dj))
lim = np.sqrt(6 / (dj + U + U))
Wf, Wb = rng.uniform(-lim, lim, (dj + U, U)), rng.uniform(-lim, lim, (dj + U, U))
bz = np.zeros(U)
def states(x):
    hf, _ = run_dir("rnn", x, Wf, bz, U, False)
    hb, _ = run_dir("rnn", x, Wb, bz, U, True)
    return hf[0], hb[0]
def jac(part):
    J, eps = np.zeros((Tj, Tj)), 1e-6
    for k in range(Tj):
        sq = np.zeros(Tj)
        for i in range(dj):
            xp, xm = x0.copy(), x0.copy(); xp[0, k, i] += eps; xm[0, k, i] -= eps
            fp, bp = states(xp); fm, bm = states(xm)
            d = {"fwd": (fp - fm), "bwd": (bp - bm), "concat": np.concatenate([fp - fm, bp - bm], 1)}[part] / (2 * eps)
            sq += (d ** 2).sum(1)
        J[:, k] = np.sqrt(sq)
    return J
for part in ("fwd", "bwd", "concat"):
    A["jac_" + part] = jac(part)
    R["jac_nonzero_" + part] = int((A["jac_" + part] > 1e-12).sum())

# ---- C. parameter counts and (implementation-specific) timing
R["params"] = {}
for kind in ("rnn", "lstm"):
    for name, direction, u in (("fwd(5)", "fwd", 5), ("fwd(10)", "fwd", 10), ("bi(5)", "bi", 5)):
        m = Tagger(V, D, u, kind=kind, direction=direction)
        assert m.n_params() == count_params(kind, V, D, u, direction)
        R["params"][f"{kind} {name}"] = m.n_params()
R["layer_params"] = {f"{kind} {n}": (2 if n == "bi(5)" else 1) * LAYERS[kind][2] * 5 * (D + 5 + 1)
                     for kind in ("rnn", "lstm") for n in ("fwd(5)", "bi(5)")}
R["timing_ms"] = {}
Xt = np.random.default_rng(0).integers(1, V, (64, 50)); yt = np.zeros((64, 50))
for kind in ("rnn", "lstm"):
    for direction in ("fwd", "bi"):
        m = Tagger(V, D, 5, kind=kind, direction=direction)
        m.loss_and_grads(Xt, yt)
        ts = []
        for _ in range(60):
            t0 = time.perf_counter(); m.loss_and_grads(Xt, yt); ts.append(time.perf_counter() - t0)
        R["timing_ms"][f"{kind}-{direction}"] = float(np.median(ts) * 1000)
R["timing_ratio"] = {k: R["timing_ms"][f"{k}-bi"] / R["timing_ms"][f"{k}-fwd"] for k in ("rnn", "lstm")}

# ---- D. the main experiment: who can resolve "amazon" = company vs place?
configs = [("fwd(5)", "fwd", 5), ("fwd(10)", "fwd", 10), ("bwd(5)", "bwd", 5), ("bi(5)", "bi", 5)]
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 100
R['epochs'], R['seeds'] = EPOCHS, len(SEEDS)
R["tagging"] = {}
A["curves"] = {}
import os, pickle
os.makedirs("ckpt", exist_ok=True)
for kind in ("rnn", "lstm"):
    for name, direction, u in configs:
        ck = f"ckpt/{kind}_{name}.pkl"
        if os.path.exists(ck):
            R["tagging"][f"{kind} {name}"], A["curves"][f"{kind} {name}"], saved = pickle.load(open(ck, "rb"))
            for k2, v2 in saved.items(): A[k2] = v2
            continue
        res = []
        curves = []
        saved = {}
        for s in SEEDS:
            m = Tagger(V, D, u, kind=kind, direction=direction, seed=s)
            curves.append(fit(m, Xtr, ytr, epochs=EPOCHS, seed=s))
            after, before = ambiguous_accuracy(m, Xte, yte, pa, ca)
            pred = m.forward(Xte)[0] > 0.5
            res.append((after, before, float((pred == (yte > 0.5)).mean())))
            if s == 0 and name in ("fwd(5)", "bi(5)"):
                A[f"model_{kind}_{name}"] = {k: v for k, v in m.p.items()}; saved[f"model_{kind}_{name}"] = A[f"model_{kind}_{name}"]
        res = np.array(res)
        R["tagging"][f"{kind} {name}"] = {
            "params": R["params"].get(f"{kind} {name}", Tagger(V, D, u, kind=kind, direction=direction).n_params()),
            "after_mean": float(res[:, 0].mean()), "after_std": float(res[:, 0].std()),
            "before_mean": float(res[:, 1].mean()), "before_std": float(res[:, 1].std()),
            "all_tokens_mean": float(res[:, 2].mean()), "final_loss": float(np.mean([c[-1] for c in curves])),
            "per_seed_after": res[:, 0].tolist(), "per_seed_before": res[:, 1].tolist(),
            "solved": int(sum((r[0] >= 0.95 if direction != "fwd" else True) and (r[1] >= 0.95 if direction != "bwd" else True)
                              for r in res))}
        A["curves"][f"{kind} {name}"] = np.array(curves)
        pickle.dump((R["tagging"][f"{kind} {name}"], A["curves"][f"{kind} {name}"], saved), open(ck, "wb"))
        print("finished", kind, name, flush=True)

# ---- E. streaming: p(company) for "amazon" as the sentence arrives word by word
sent = ["amazon", "the", "best", "website", "our", "great"]
ids = np.array([[W2I[w] for w in sent]])
R["stream_sentence"] = sent
R["stream"] = {}
for kind in ("rnn", "lstm"):
    for name in ("fwd(5)", "bi(5)"):
        m = Tagger(V, D, 5, kind=kind, direction="bwd" if False else ("fwd" if name.startswith("fwd") else "bi"), seed=0)
        m.p = {k: v.copy() for k, v in A[f"model_{kind}_{name}"].items()}
        R["stream"][f"{kind} {name}"] = [float(m.forward(ids[:, :L])[0][0, 0]) for L in range(1, len(sent) + 1)]
# same sentence with a place cue for contrast
sent2 = ["amazon", "the", "best", "river", "our", "great"]
ids2 = np.array([[W2I[w] for w in sent2]])
R["stream2_sentence"] = sent2
R["stream2"] = {}
for kind in ("rnn", "lstm"):
    m = Tagger(V, D, 5, kind=kind, direction="bi", seed=0); m.p = {k: v.copy() for k, v in A[f"model_{kind}_bi(5)"].items()}
    R["stream2"][f"{kind} bi(5)"] = [float(m.forward(ids2[:, :L])[0][0, 0]) for L in range(1, 7)]

# ---- F. what the screenshot's three formulas compute, evaluated by hand-rolled loops vs the layer code
rng = np.random.default_rng(7)
dW, dU = 4, 3
x = rng.normal(size=(4, dU)); 
Wr_f, U_f, b_f = rng.normal(size=(dW, dW)) * .5, rng.normal(size=(dW, dU)) * .5, rng.normal(size=dW) * .1
Wr_b, U_b, b_b = rng.normal(size=(dW, dW)) * .5, rng.normal(size=(dW, dU)) * .5, rng.normal(size=dW) * .1
Vy, by = rng.normal(size=(1, 2 * dW)), 0.1
hf, hb = np.zeros((6, dW)), np.zeros((6, dW))                 # index 0 and 5 stay zero (the 'zeros' boxes)
for t in range(1, 5):
    hf[t] = np.tanh(Wr_f @ hf[t - 1] + U_f @ x[t - 1] + b_f)
for t in range(4, 0, -1):
    hb[t] = np.tanh(Wr_b @ hb[t + 1] + U_b @ x[t - 1] + b_b)
y_formula = [float(1 / (1 + np.exp(-(Vy @ np.concatenate([hf[t], hb[t]]) + by))[0])) for t in range(1, 5)]
Wj_f, Wj_b = np.concatenate([U_f.T, Wr_f.T], 0), np.concatenate([U_b.T, Wr_b.T], 0)    # joint matrix [x; h]
Hf, _ = run_dir("rnn", x[None], Wj_f, b_f, dW, False)
Hb, _ = run_dir("rnn", x[None], Wj_b, b_b, dW, True)
y_layer = (1 / (1 + np.exp(-(np.concatenate([Hf, Hb], 2) @ Vy.T + by)[0, :, 0]))).tolist()
R["formula_vs_layer_maxdiff"] = float(np.abs(np.array(y_formula) - np.array(y_layer)).max())
R["formula_y"] = y_formula
R["screenshot_h_fwd_1"] = hf[1].tolist(); R["screenshot_h_bwd_1"] = hb[1].tolist()

json.dump(R, open("results.json", "w"), indent=1, default=float)
np.savez("arrays.npz", **{k: v for k, v in A.items() if k != "curves" and not k.startswith("model_")},
         **{"curve_" + k.replace(" ", "_"): v for k, v in A["curves"].items()})
print("gradcheck worst", R["gradcheck_worst"])
for k, v in R["tagging"].items(): print(k, {a: round(v[a], 3) for a in ("after_mean", "before_mean", "all_tokens_mean", "final_loss")})
print("jac nonzeros", R["jac_nonzero_fwd"], R["jac_nonzero_bwd"], R["jac_nonzero_concat"])
print("timing", R["timing_ms"], R["timing_ratio"])
print("stream", {k: [round(x, 3) for x in v] for k, v in R["stream"].items()})
print("stream2", {k: [round(x, 3) for x in v] for k, v in R["stream2"].items()})
print("formula vs layer", R["formula_vs_layer_maxdiff"])
```
