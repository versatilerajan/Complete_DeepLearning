# Long Short-Term Memory (LSTM) Networks: Part 1 — The Core Idea

This is part one of a four-part series on LSTMs, mirroring the lecture it follows: this note covers the *concept* (why LSTMs exist and what their gates do), and later parts in the series will cover the full math and a coding project. LSTMs exist to fix a specific weakness in the plain RNNs covered in [recurrent-neural-networks-intro.md](recurrent-neural-networks-intro.md) and [backpropagation-through-time.md](backpropagation-through-time.md): the vanishing/exploding gradient problem that shows up once the chain-rule products from BPTT get long. This note explains the fix conceptually, with diagrams for every gate; the full gradient derivation is left for the next part in the series.

## Table of Contents
1. [The Problem LSTMs Solve](#1-the-problem-lstms-solve)
2. [The Core Idea: Two Memory Pathways](#2-the-core-idea-two-memory-pathways)
3. [Inside the Cell](#3-inside-the-cell)
4. [The Forget Gate](#4-the-forget-gate)
5. [The Input Gate](#5-the-input-gate)
6. [The Output Gate](#6-the-output-gate)
7. [All the Equations Together](#7-all-the-equations-together)
8. [The Computer Analogy](#8-the-computer-analogy)
9. [RNN vs LSTM, Side by Side](#9-rnn-vs-lstm-side-by-side)
10. [Key Takeaways](#10-key-takeaways)
11. [Further Reading](#11-further-reading)

---

## 1. The Problem LSTMs Solve

[backpropagation-through-time.md](backpropagation-through-time.md) showed that the gradient for a shared weight like `W_h` is a sum of terms, and that terms reaching further back in time contain longer chains of `dO_k/dO_(k-1)` factors. When those factors are consistently less than 1, the product shrinks toward zero (**vanishing gradients**); when they're consistently greater than 1, it grows without bound (**exploding gradients**). Either way, a vanilla RNN struggles to learn dependencies that span many time steps — by the time the gradient reaches an early step, it has been multiplied so many times that it carries almost no useful signal (or an unusably large one).

LSTMs were designed specifically to make long-range dependencies learnable, by changing *how* information is carried from one step to the next.

---

## 2. The Core Idea: Two Memory Pathways

A vanilla RNN has one hidden state, `h_t`, that is recomputed — effectively overwritten — at every step. An LSTM splits memory into two separate pathways that run alongside each other:

- **Cell state (`C_t`)** — the "long-term memory." It changes slowly and deliberately: information is only added to or removed from it through gates, never bluntly overwritten.
- **Hidden state (`h_t`)** — the "short-term memory," same role as a vanilla RNN's hidden state: it's recomputed every step and used for the current prediction.

![RNN vs LSTM pathways](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_rnn_vs_lstm_pathways.png)

*Left: a vanilla RNN has a single hidden-state pathway (red), so every step's update risks diluting older information. Right: an LSTM keeps two pathways — the cell state (red, top) acts as a highway that information can travel along largely unchanged, while the hidden state (blue, bottom) is derived from it fresh at each step. This separation is what gives LSTMs their name: long (cell state) and short (hidden state) term memory, together.*

---

## 3. Inside the Cell

Three gates control how information moves between the two pathways at every time step: a **forget gate**, an **input gate**, and an **output gate**. Each gate is simply a small neural-network layer (a sigmoid, usually) that outputs a number between 0 and 1 for each memory slot — acting like a dial that decides how much of something to let through.

![LSTM cell internals](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_lstm_cell_internals.png)

*The full picture: the cell-state highway runs along the top, the hidden-state highway along the bottom. The forget gate multiplies the incoming cell state (deciding what to erase); the input gate and candidate values combine to add new information; the output gate reads from the updated cell state to produce the new hidden state. Every gate takes the same input: the previous hidden state concatenated with the current input, `[h_(t-1), x_t]`. The next three sections zoom into each gate individually.*

---

## 4. The Forget Gate

The forget gate looks at `[h_(t-1), x_t]` and outputs a number between 0 and 1 for each value in the cell state — deciding how much of the old long-term memory to keep.

![Forget gate](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_forget_gate.png)

*The sigmoid output `f_t` is element-wise multiplied with the previous cell state `C_(t-1)`. A value near 0 erases that slot of memory; a value near 1 keeps it untouched. The illustrative numbers at the bottom show the mechanism with made-up values, not a measured example — in a trained network, `f_t` learns to stay close to 1 for information that's still relevant and drop toward 0 for information that should be forgotten (for example, forgetting the subject of a previous sentence once a new one begins).*

---

## 5. The Input Gate

The input gate decides what *new* information gets written into the cell state. It actually involves two parts working together: a sigmoid layer (`i_t`) that decides how much to let in, and a tanh layer (`C~_t`, the "candidate values") that proposes what the new information could look like.

![Input gate](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_input_gate.png)

*`i_t` is a gate (0 to 1, "how much"); `C~_t` is content (−1 to 1, "what"). Multiplying them element-wise gives the actual update, which is then added to the forget gate's result to form the new cell state: `C_t = f_t ⊙ C_(t-1) + i_t ⊙ C~_t`. This addition (rather than a full overwrite) is the key structural difference from a vanilla RNN, and it's why gradients can flow through the cell-state highway with much less shrinkage.*

---

## 6. The Output Gate

The output gate decides what part of the (now updated) cell state should be exposed as this step's output.

![Output gate](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_output_gate.png)

*The cell state is first squashed with `tanh` (to keep its values between −1 and 1), then multiplied element-wise by the output gate `o_t`. The result, `h_t`, serves two roles at once: it's used for this step's prediction and it's passed on as "short-term memory" input to the next time step — exactly like a vanilla RNN's hidden state, except now it's a filtered view of a much more carefully maintained cell state.*

---

## 7. All the Equations Together

![Full equations](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_full_equations.png)

*The six equations that define one LSTM step, color-matched to the gates above. Reading top to bottom: compute the three gates and the candidate values from `[h_(t-1), x_t]`, combine them into the new cell state, then use the output gate to produce the new hidden state. `W_f, W_i, W_C, W_o` and their biases are the learnable parameters — four sets instead of the two (`W_xh`, `W_hh`) a vanilla RNN has, which is part of why LSTMs have more parameters per cell.*

---

## 8. The Computer Analogy

The lecture frames the whole cell with an analogy: an LSTM behaves like a tiny computer that processes one instruction (time step) at a time.

![Computer analogy](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_lstm_as_computer.png)

*Input arrives (`x_t`), gets processed (the three gates decide what to keep, add, and output), long-term memory is updated (the cell state), and an output is produced (the hidden state) — which doubles as a signal carried into the next cycle. It's a loose analogy, but it captures the key shift from a vanilla RNN: instead of recomputing everything from scratch each step, an LSTM reads, selectively writes, and reads again.*

---

## 9. RNN vs LSTM, Side by Side

![RNN vs LSTM table](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_rnn_vs_lstm_table.png)

*A direct comparison. The core change is row 2: a vanilla RNN fully overwrites its one state every step, while an LSTM's forget-then-add mechanism lets old information pass through largely unchanged when the gates say so — which is the structural reason row 3 (gradient stability over long sequences) improves.*

---

## 10. Key Takeaways

- Vanilla RNNs struggle with long-range dependencies because of vanishing/exploding gradients, which come from the long products of derivatives in BPTT.
- LSTMs introduce a second memory pathway, the **cell state**, that acts as a highway information can traverse with only gated modifications, alongside the usual **hidden state**.
- The **forget gate** decides what to erase from the cell state; the **input gate** (paired with candidate values) decides what new information to add; the **output gate** decides what to expose as the hidden state.
- The cell-state update is additive (`C_t = f_t ⊙ C_(t-1) + i_t ⊙ C~_t`) rather than a full overwrite, which is the key structural reason gradients survive better over long sequences.
- An LSTM cell has four sets of weights (for the forget, input, candidate, and output computations) compared to a vanilla RNN's two.
- This is part 1 of a 4-part series; the next parts cover the full BPTT derivation for LSTMs and a coding project.

---

## 11. Further Reading

- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780. The original LSTM paper.
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). *Learning to Forget: Continual Prediction with LSTM*. Neural Computation, 12(10), 2451-2471. Introduces the forget gate used in modern LSTMs.
- Olah, C. (2015). *Understanding LSTM Networks*. colah's blog. A widely-used visual walkthrough of the same gates covered here.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the Difficulty of Training Recurrent Neural Networks*. ICML. The vanishing/exploding gradient problem LSTMs address.
