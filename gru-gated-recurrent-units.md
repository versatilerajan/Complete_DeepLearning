# Gated Recurrent Units (GRU)

GRUs are a simpler alternative to LSTMs: they solve the same vanishing-gradient problem from [backpropagation-through-time.md](backpropagation-through-time.md), using only **two gates** and a **single hidden state** instead of an LSTM's three gates and separate cell state (see [lstm-introduction.md](lstm-introduction.md)). Introduced in 2014, they're faster to train and often match LSTM performance, especially on smaller datasets. This note follows the worked example from class — a story about King Vikram and King Kaali, whose "moral" is tracked as a 4-dimensional vector [power, conflict, tragedy, revenge] — reconstructed as clean, consistent numbers so every step of the arithmetic can be checked by hand.

## Table of Contents
1. [Why GRUs Exist](#1-why-grus-exist)
2. [The Class Example: Encoding a Story's Moral as a Vector](#2-the-class-example-encoding-a-storys-moral-as-a-vector)
3. [Inside the Cell](#3-inside-the-cell)
4. [The Reset Gate](#4-the-reset-gate)
5. [The Candidate Hidden State](#5-the-candidate-hidden-state)
6. [The Update Gate and Final Blend](#6-the-update-gate-and-final-blend)
7. [All the Equations Together](#7-all-the-equations-together)
8. [Running the Whole Story Through the GRU](#8-running-the-whole-story-through-the-gru)
9. [LSTM vs GRU](#9-lstm-vs-gru)
10. [Key Takeaways](#10-key-takeaways)
11. [Further Reading](#11-further-reading)

---

## 1. Why GRUs Exist

LSTMs fix the vanishing-gradient problem well, but they're architecturally heavy: three gates (forget, input, output), a separate cell state alongside the hidden state, and four full sets of weight matrices. That's a lot of parameters to learn, which means slower training. GRUs, introduced in 2014, ask: can we get most of the benefit with less machinery?

![Why GRU exists](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_why_gru_exists.png)

*Side by side: an LSTM cell carries two separate highways (cell state and hidden state) through three gates. A GRU collapses this to one hidden-state highway through just two gates — a reset gate and an update gate — with three weight matrices instead of four. Fewer parameters means faster training and, on many tasks, comparable accuracy, which is why GRUs are often preferred for smaller datasets or quicker iteration.*

---

## 2. The Class Example: Encoding a Story's Moral as a Vector

The worked example from class uses a short story to show what a GRU's hidden state can represent. The task: read a story sentence by sentence and have the final hidden state represent the story's **moral**, encoded as a 4-dimensional vector over four themes: `[power, conflict, tragedy, revenge]`.

**The story:** There was a king, Vikram, and an enemy king, Kaali. Both had sons. Vikram's son grew up to become very strong, just like his father. Kaali also attacked, but was killed by Vikram. Vikram's son too had a son, called Vikram Jr, who grew up and, when the time came, fought Kaali's son — who had grown very powerful too. Kaali's son killed Kaali and his father, and took revenge.

Each word of the story is fed into the GRU as an input vector `x_t`, one word at a time, exactly like the earlier RNN and LSTM notes. The hidden state `h_t` is *also* 4-dimensional here — one value per moral theme — so you can watch which theme the story is leaning toward as each sentence is processed. `h_0 = [0, 0, 0, 0]` at the start, before any words have been read.

---

## 3. Inside the Cell

A GRU uses two gates to decide, at every time step, how to combine the old hidden state with new information from the current input.

![GRU cell internals](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_gru_cell_internals.png)

*Reading left to right along the hidden-state highway: the reset gate first decides how much of `h_(t-1)` to use when building the candidate hidden state. The candidate is a tanh-squashed proposal for the new state. Separately, the update gate decides the final mix between the old state and the candidate — `(1 - z_t)` keeps old information, `z_t` lets in the new candidate. The result lands back on the same single highway as `h_t`. Both gates read the same `[h_(t-1), x_t]`, exactly as in the class diagram.*

---

## 4. The Reset Gate

The reset gate decides how much of the previous hidden state to use when computing the new candidate — a value near 0 means "mostly ignore the past here," near 1 means "keep it."

![Reset gate](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_reset_gate.png)

*Using the class numbers: with `h_(t-1) = [0.6, 0.6, 0.7, 0.1]` (power, conflict, tragedy, revenge) and a reset gate output `r_t = [0.8, 0.2, 0.1, 0.9]`, the element-wise product `h_(t-1) ⊙ r_t = [0.48, 0.12, 0.07, 0.09]` keeps most of the "power" and "revenge" signal (gate values 0.8 and 0.9) while mostly discarding "conflict" and "tragedy" (gate values 0.2 and 0.1) before they feed into the candidate computation.*

---

## 5. The Candidate Hidden State

The candidate hidden state is a *proposal* for what the new hidden state could be, built from the current input and the reset-gated version of the previous state (not the raw previous state).

![Candidate hidden state](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_candidate_hidden_state.png)

*`h_(t-1) ⊙ r_t` from the reset gate step, together with the current input `x_t`, feeds a tanh layer to produce the candidate `h~_t = [0.7, 0.2, 0.1, 0.4]`. This is only a proposal — the update gate (next section) decides how much of it actually becomes the new hidden state.*

---

## 6. The Update Gate and Final Blend

The update gate decides the final balance between keeping the old hidden state and adopting the new candidate.

![Update gate and blend](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_update_gate_and_blend.png)

*With `z_t = [0.7, 0.7, 0.8, 0.2]`: the `(1 - z_t)` branch keeps 30%, 30%, 20% and 80% of the old state in each of the four themes respectively, giving `[0.18, 0.18, 0.14, 0.08]`. The `z_t` branch lets in 70%, 70%, 80% and 20% of the candidate, giving `[0.49, 0.14, 0.08, 0.08]`. Adding the two branches gives the new hidden state `h_t = [0.67, 0.32, 0.22, 0.16]` — "power" is now the clearly dominant theme, which matches how the class example was set up (the story opens on a powerful king).*

---

## 7. All the Equations Together

![Full equations](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_full_equations.png)

*The four equations that define one GRU step, color-matched to the diagrams above: the reset gate, the candidate hidden state (built from the reset-gated past), the update gate, and the final blended hidden state. Compare this to the six equations needed for an LSTM step in [lstm-introduction.md](lstm-introduction.md) — two fewer equations, and only three weight matrices (`W_r`, `W_c`, `W_z`) instead of an LSTM's four.*

---

## 8. Running the Whole Story Through the GRU

![Story timeline](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_story_timeline.png)

*Three sentences from the story, fed one at a time. `h_0` starts at all zeros. After sentence 1 (introducing a powerful king and an attacking enemy), the "power" and "conflict" components start rising. After sentence 2 (a death), "tragedy" picks up. By sentence 3 (revenge and a new king), the final hidden state in this worked example is `h_3 = [0.67, 0.32, 0.22, 0.16]` — reading off the four themes `[power, conflict, tragedy, revenge]`, "power" comes out as the dominant component of this story's moral, by this toy model's accounting.*

---

## 9. LSTM vs GRU

![LSTM vs GRU table](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_lstm_vs_gru_table.png)

*A direct comparison, following the lecture's conclusion. The structural difference — one state and two gates versus two states and three gates — is what drives every other row: fewer weight matrices means faster training, which is a big part of why GRUs are often the first thing to try on smaller datasets before reaching for the heavier LSTM.*

---

## 10. Key Takeaways

- GRUs (2014) simplify LSTMs (1997) by merging the cell state and hidden state into one, and reducing three gates to two.
- The **reset gate** `r_t` decides how much of the previous hidden state to use when building the candidate hidden state.
- The **candidate hidden state** `h~_t` is a tanh-based proposal, computed from the current input and the reset-gated previous state.
- The **update gate** `z_t` decides the final blend: `h_t = (1 - z_t) ⊙ h_(t-1) + z_t ⊙ h~_t`.
- A GRU has 3 weight matrices (`W_r`, `W_c`, `W_z`) versus an LSTM's 4, which is the direct source of its faster training.
- In the class's story example, representing the hidden state directly as a 4-dimensional "moral vector" over `[power, conflict, tragedy, revenge]` makes the gating mechanism easy to interpret by hand, step by step.
- GRUs often perform comparably to LSTMs and are commonly preferred for smaller datasets or when training speed matters more than squeezing out the last bit of accuracy on very long sequences.

---

## 11. Further Reading

- Cho, K., van Merrienboer, B., Gulcehre, C., et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. EMNLP. The original GRU paper.
- Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling*. arXiv:1412.3555. Direct empirical comparison of GRUs and LSTMs.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780. The LSTM architecture GRUs simplify.
- Olah, C. (2015). *Understanding LSTM Networks*. colah's blog. Also covers GRUs as a variant in its closing section.
