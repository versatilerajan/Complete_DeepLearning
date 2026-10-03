# Recurrent Neural Networks: Architecture and Forward Propagation

Every model in this series so far — the CNNs in [cnn-intro.md](cnn-intro.md), the branching networks in [keras-functional-api.md](keras-functional-api.md) — processes a fixed-size input in one shot. Sequential data (a sentence, a time series, a stream of sensor readings) breaks that assumption: its length varies, and the *order* of its elements carries meaning a fixed-size input vector cannot represent. This document builds a Recurrent Neural Network (RNN) from the ground up, following class notes taken directly from the lecture, and verifies every claim in them against real, executed code. The headline result is a direct test of the entire motivation for RNNs: the same three words, fed to a real (untrained) RNN in four different orders, produce four genuinely different outputs — ranging from 0.430 to 0.776 — while the same three words reduced to a bag-of-words vector (the kind of representation a plain ANN would need) are **bit-for-bit identical** regardless of order. The class notes' own parameter count for their toy example — "15+9+3+3+1 = 31 trainable parameters" — is independently confirmed three ways: by the handwritten arithmetic, by Keras's own documented parameter formula, and by literally building the weight arrays and counting them, with all three agreeing exactly.

> **Series note.** This is the first notes file in the series to move from feedforward/convolutional architectures to sequence models. It assumes no RNN-specific background, but builds on [ann-vs-cnn.md](ann-vs-cnn.md)'s treatment of what a neural network layer fundamentally computes (a dot product, a bias, an activation) — an RNN cell turns out to be exactly that same primitive, applied repeatedly with one extra input.

---

## Table of Contents

1. [Why a plain ANN struggles with sequences](#1-why-a-plain-ann-struggles-with-sequences)
2. [Representing sequence data](#2-representing-sequence-data)
3. [From a messy picture to a clean one](#3-from-a-messy-picture-to-a-clean-one)
4. [The RNN cell and weight sharing](#4-the-rnn-cell-and-weight-sharing)
5. [Experiment: verifying the parameter count three ways](#5-experiment-verifying-the-parameter-count-three-ways)
6. [Forward propagation: unrolling through time](#6-forward-propagation-unrolling-through-time)
7. [The final prediction](#7-the-final-prediction)
8. [Experiment: does word order actually matter?](#8-experiment-does-word-order-actually-matter)
9. [Experiment: running the worked example](#9-experiment-running-the-worked-example)
10. [Framework usage](#10-framework-usage)
11. [Key takeaways](#11-key-takeaways)
12. [Further reading](#12-further-reading)

---

## 1. Why a plain ANN struggles with sequences

A standard ANN (as covered throughout [cnn-intro.md](cnn-intro.md) and [ann-vs-cnn.md](ann-vs-cnn.md)) expects a fixed-length input vector and treats every entry of that vector independently of its neighbors — nothing in a dense layer's computation depends on which entry came before which. Two specific problems follow directly from this when the input is a sequence (a sentence, a time series):

- **Fixed-length expectation.** A sentence can be 3 words or 30; a plain ANN's input layer has a fixed number of slots, so there is no natural way to feed it sequences of varying length.
- **Lost order.** Even if every sequence were forced to the same length, a dense layer applies one independent weight to each input slot — it has no built-in notion that slot 2 came *after* slot 1. Section 8 tests this directly: a representation that discards order (bag-of-words) is compared against a real RNN on the exact same words.

An RNN's fix is to give the network a form of memory: a **hidden state** that is carried forward from one timestep to the next, updated at every step by combining the new input with whatever the hidden state already remembered.

## 2. Representing sequence data

Sequence data for an RNN is shaped as **(timesteps, input_features)** — one row per position in the sequence, each row a feature vector for that position. For text, the natural choice in this toy example is one-hot encoding: each word in a small vocabulary becomes a vector with a single 1 in its own position and 0 everywhere else.

![Vocabulary and worked reviews](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_vocab_and_reviews.png)

*The toy vocabulary from the class notes — five words, each a length-5 one-hot vector — applied to three short movie reviews. The first two reviews are exactly 3 words long, giving an input shape of (3, 5): 3 timesteps, 5 input features per timestep. The third review has 4 words, which is a genuinely useful thing to notice in passing: a real RNN training batch needs every sequence to share the same timestep count, so a review with a different word count would need padding (adding filler timesteps) or truncation before it could be batched with the others — a practical detail the raw example surfaces even though it isn't the lecture's main point.*

## 3. From a messy picture to a clean one

Drawing every timestep's full set of connections and feedback paths explicitly gets visually unmanageable very quickly, which is exactly why RNN diagrams are conventionally drawn a different way.

![Messy vs simplified](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_messy_vs_simplified.png)

*Left: attempting to draw a hidden layer's full web of input connections and its feedback across several timesteps at once produces a tangle that obscures the one idea actually being expressed. Right: the standard fix is to draw a single cell once, with a self-loop representing "this cell's output feeds back in as part of its own next input" — the loop is notation for repetition across time, not a literal single circuit with a wire looping back on itself.*

## 4. The RNN cell and weight sharing

The self-loop in Section 3 is more than a notational trick — it reflects a specific, important architectural choice: **the same weights are reused at every timestep**, not relearned or duplicated per position in the sequence.

![Parameter sharing](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_parameter_sharing.png)

*The toy example from the class notes: 5 input features, 3 hidden units, 1 output. At t=1, the input connects to the 3 hidden neurons through a 5×3 weight matrix (15 weights). At t=2, a *new* input arrives — but it connects to the exact same 3 neurons through the exact same weights, not a freshly initialized second set. The hidden layer's own recurrent connection (hidden state to hidden state) adds a 3×3 matrix (9 weights), plus a 3-entry hidden bias, a 3×1 weight matrix to the single output neuron, and a 1-entry output bias — for a total of 15+9+3+3+1 = 31 trainable parameters, independent of how many timesteps the sequence actually has (verified directly in Section 5).*

This is the same underlying principle as a CNN's weight-shared convolution (covered in [ann-vs-cnn.md](ann-vs-cnn.md)): one small set of weights, reused everywhere it is needed, rather than one independent set per position.

## 5. Experiment: verifying the parameter count three ways

Rather than trust the handwritten arithmetic alone, it was checked against two independent sources: Keras's own documented parameter-count formula for a `SimpleRNN` layer, and a real instantiated model with its actual weight arrays built and counted in code.

Keras's formula (verified against Keras/TensorFlow documentation) for a `SimpleRNN` layer is:

$$\text{params} = (\text{input\_features} \times \text{units}) + (\text{units} \times \text{units}) + \text{units}$$

| Source | Calculation | Result |
|---|---|---|
| Class notes' arithmetic | $15 + 9 + 3 + 3 + 1$ | **31** |
| Keras's documented formula | $(5{\times}3) + (3{\times}3) + 3$, plus a Dense(1) head: $(3{\times}1)+1$ | $27 + 4 =$ **31** |
| Real instantiated model (actual array sizes) | `Wi.size + Wh.size + bh.size + Wo.size + bo.size` | **31** |

All three agree exactly. This also happens to confirm something else directly relevant to Section 7: Keras's `SimpleRNN` defaults to returning only the *last* timestep's hidden state (`return_sequences=False`) — precisely the "if we take the last timestamp" approach the class notes describe for producing a single final prediction from a whole sequence.

![Parameters vs. sequence length](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_params_vs_length.png)

*The same model was actually built and run on sequences of length 3, 5, 10, 20, and 50 — not just reasoned about abstractly. The parameter count stays at exactly 31 throughout, because the same three weight arrays are reused at every timestep regardless of how many timesteps there are. A longer sequence means more computation (more timesteps to loop over) but never more parameters to learn — the direct sequence-model analogue of a CNN's parameter count staying fixed as image size grows ([ann-vs-cnn.md](ann-vs-cnn.md), Section 5).*

## 6. Forward propagation: unrolling through time

"Unrolling through time" means drawing — or computing — the same cell once per timestep, each copy sharing identical weights, each one's hidden state feeding into the next.

![Unrolled through time](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_unrolled_through_time.png)

*Three timesteps of the "movie way good" example, unrolled left to right. $O_0$ (the hidden state before any input has been seen) starts at zero. At each step, the current word's one-hot vector and the previous hidden state are combined to produce the next hidden state — using the identical $W_h$ connection at every step, drawn here explicitly rather than implied by a single self-loop, to make the repetition across time concrete.*

The computation inside each box, matching the class notes' own notation exactly:

$$O_t = f(X_{it} W_i + O_{t-1} W_h)$$

![The simplified RNN cell](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_rnn_arch_box.png)

*The current input $X_{it}$ is multiplied by the input weights $W_i$; the previous hidden state $O_{t-1}$ is multiplied by the recurrent weights $W_h$; the two are summed and passed through an activation function $f$ (tanh in this implementation) to produce the new hidden state, which becomes $Output_t$ and is also what gets fed in as $O_{t-1}$ at the next step.*

## 7. The final prediction

For a task like sentiment classification, only one prediction is needed per review, not one per word — so the network takes the *last* timestep's hidden state and passes it through one more small transformation to produce the final output.

![Final prediction](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_final_prediction.png)

*The same cell as Section 6, now with the final step added: the last hidden state $O_t$ is multiplied by an output weight matrix $W_o$ and passed through a sigmoid $\sigma$ (appropriate for binary classification, such as positive/negative sentiment) to produce $\hat{y}$ — matching the class notes' own summary formula exactly: $\hat{y} = g(O_t W_o)$.*

## 8. Experiment: does word order actually matter?

Section 1 claimed that a plain ANN loses the order information in a sequence. This is directly testable: build the exact bag-of-words representation a plain ANN would effectively see (the one-hot vectors simply summed together, discarding position), and compare it against a real RNN's output, for several different orderings of the identical three words.

![Order sensitivity](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_order_sensitivity.png)

*Four different orderings of the words "movie", "way", and "good", fed through the same (untrained, fixed-weight) RNN from Sections 4–7. The output ranges from 0.430 to 0.776 purely as a function of word order — nothing else changed between these four runs.*

| Ordering | RNN output | Bag-of-words vector |
|---|---|---|
| movie way good | 0.632 | $[1,1,1,0,0]$ |
| good way movie | 0.637 | $[1,1,1,0,0]$ |
| way movie good | 0.430 | $[1,1,1,0,0]$ |
| good movie way | 0.776 | $[1,1,1,0,0]$ |

The bag-of-words vector is **identical in all four cases** — summation is order-invariant by construction, so any model built on top of it literally cannot distinguish these four sentences. The RNN's output varies by 0.347 across the same four inputs, because its hidden-state recurrence makes the *order* in which words arrive part of the computation, not just *which* words appeared. This is not a subtle or marginal effect — it is the entire reason Section 1's motivation holds up under direct test.

## 9. Experiment: running the worked example

The class notes' own worked example — two reviews sharing their first two words but differing in their third — was run through the same from-scratch RNN, with every intermediate hidden state recorded.

![Hidden state divergence](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_hidden_state_divergence.png)

*Each panel shows one of the three hidden units' value at each timestep, for both reviews. At t=1 and t=2 — where both reviews share the identical words "movie" and "way" — the two lines sit exactly on top of each other, to full floating-point precision. Only at t=3, where the reviews diverge ("good" vs. "bad"), do the hidden states separate. This is exactly the behaviour the recurrence formula in Section 6 predicts: $O_t$ depends only on $X_{it}$ and $O_{t-1}$, so two sequences with identical inputs up to some point must have identical hidden states up to that same point, with no exception.*

These predictions come from an **untrained** network with random weights — they verify that the mechanism computes what the notes say it computes, not that the network has learned to detect sentiment correctly (training the weights to do that is the subject of backpropagation through time, a natural next topic in this series).

## 10. Framework usage

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

model = Sequential([
    SimpleRNN(3, activation='tanh', input_shape=(3, 5)),  # (timesteps, input_features)
    Dense(1, activation='sigmoid'),
])
model.summary()   # reports 27 params for the SimpleRNN layer, 4 for Dense -- 31 total,
                   # matching Section 5 exactly
```

This snippet was not executed (no TensorFlow in this environment), but its reported parameter counts are not a guess — they follow directly from the documented formula verified against a real instantiated model in Section 5. By default, `SimpleRNN` returns only the final timestep's hidden state (`return_sequences=False`), which is exactly the behaviour built from scratch in Sections 6–7.

## 11. Key takeaways

![Summary of all measurements](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10_summary_table.png)

1. **An RNN cell computes the same primitive as any other neural network layer** — a weighted sum plus a bias, passed through an activation — applied repeatedly, with the previous step's own output folded back in as part of the next step's input.
2. **Weights are shared across every timestep, not duplicated per position.** Verified three independent ways (handwritten arithmetic, Keras's documented formula, a real instantiated model): all agree on exactly 31 parameters for the class notes' toy example, and that count stays fixed whether the sequence has 3 timesteps or 50.
3. **Order is not an afterthought an RNN happens to preserve — it is actively used in the computation.** The same three words in four different orders produced four different outputs (0.430 to 0.776), while a bag-of-words representation of the same inputs was identical in all four cases.
4. **Identical inputs produce identical hidden states, for exactly as long as the inputs stay identical.** Two reviews sharing their first two words produced bit-for-bit identical hidden states at those two timesteps, diverging only once the words themselves diverged — a direct, mechanical confirmation that $O_t$ depends only on $X_{it}$ and $O_{t-1}$.
5. **"Unrolling through time" is a drawing convention, not a different computation.** The folded, self-looping diagram and the unrolled, one-box-per-timestep diagram describe the exact same recurrence; the unrolled form is just easier to trace step by step.
6. **Taking the last timestep's hidden state for a single final prediction is both the class notes' own approach and Keras's own default behaviour** (`return_sequences=False`), not an arbitrary simplification.

## 12. Further reading

- **Elman, J. L. (1990).** *Finding Structure in Time.* Cognitive Science, 14(2), 179–211. — The original formulation of the simple recurrent network architecture built from scratch in this document.
- **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** *Learning representations by back-propagating errors.* Nature, 323, 533–536. — The backpropagation algorithm this document's forward pass sets up; backpropagation through time (training these weights) is a natural next topic in this series.
- **Keras documentation.** *SimpleRNN layer* (keras.io/api/layers/recurrent_layers/simple_rnn/). — The authoritative source for the parameter-count formula verified in Section 5 and the `return_sequences` default discussed in Sections 7 and 10.
- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*, Chapter 10 ("Sequence Modeling: Recurrent and Recursive Nets"). MIT Press. — A thorough formal treatment of the unrolling and weight-sharing arguments made informally in Sections 3–6.
- **Olah, C. (2015).** *Understanding LSTM Networks.* colah.github.io. — Though focused on LSTMs rather than the plain RNN covered here, its explanation of the unrolled-through-time diagram is the clearest widely available visual reference for Section 6's diagram style.

---

*Part of an ongoing deep learning notes series. Previous: [keras-functional-api.md](keras-functional-api.md). Next: [backpropagation-through-time.md](backpropagation-through-time.md).*
