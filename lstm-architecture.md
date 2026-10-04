# LSTM Architecture: Cell State, Hidden State and the Three Gates

This document explains the **Long Short-Term Memory (LSTM)** cell, the recurrent unit designed to fix the two weaknesses of a vanilla RNN: forgetting information from early time steps and unstable (vanishing or exploding) gradients. An LSTM keeps **two** memories, a long-term **cell state** $C_t$ and a short-term **hidden state** $h_t$, and controls them with three small neural layers called **gates** (forget, input and output) that combine pointwise multiplications and additions. The note follows the class derivation step by step, using the class dimensions (hidden state of 3 numbers, input of 4 numbers, a concatenated vector of 7 numbers and weight matrices of shape 3×7), redraws every handwritten diagram cleanly, and then tests the claims with real code: a from-scratch NumPy LSTM with manual backpropagation (checked against finite differences and against PyTorch), measured gradient flow, a comparison with a vanilla RNN, and the real gate values of a trained network. Numbers taken from the class notes are labeled as such, and where the class numbers are only illustrative (or contain a slip) the text says so.

> **Prerequisites:** [rnn-architecture-and-forward-propagation.md](rnn-architecture-and-forward-propagation.md) (the recurrent cell), [problems-with-rnns.md](problems-with-rnns.md) (why vanilla RNNs struggle) and [encoder-decoder-architecture.md](encoder-decoder-architecture.md) (where LSTMs are used inside a translation model; that note links to this one as `lstm.md`, so rename the link or this file to match). The planned `attention.md` is an assumed filename.

---

## Table of Contents

1. [Why we need the LSTM](#1-why-we-need-the-lstm)
2. [The big picture: two memories and three gates](#2-the-big-picture-two-memories-and-three-gates)
3. [Notation and dimensions used in class](#3-notation-and-dimensions-used-in-class)
4. [The building blocks: pointwise operators and activations](#4-the-building-blocks-pointwise-operators-and-activations)
5. [Step 1: the forget gate](#5-step-1-the-forget-gate)
6. [Step 2: the input gate and the candidate cell state](#6-step-2-the-input-gate-and-the-candidate-cell-state)
7. [Step 3: updating the cell state](#7-step-3-updating-the-cell-state)
8. [Step 4: the output gate and the hidden state](#8-step-4-the-output-gate-and-the-hidden-state)
9. [One full step with numbers, and counting parameters](#9-one-full-step-with-numbers-and-counting-parameters)
10. [Why the LSTM avoids vanishing gradients](#10-why-the-lstm-avoids-vanishing-gradients)
11. [Experiments](#11-experiments)
    - [11.1 LSTM against a vanilla RNN](#111-lstm-against-a-vanilla-rnn)
    - [11.2 The forget-gate bias matters](#112-the-forget-gate-bias-matters)
    - [11.3 What a trained LSTM actually does](#113-what-a-trained-lstm-actually-does)
    - [11.4 Real gate values on a real sentence](#114-real-gate-values-on-a-real-sentence)
12. [From-scratch implementation (NumPy)](#12-from-scratch-implementation-numpy)
13. [Framework equivalents (PyTorch / Keras)](#13-framework-equivalents-pytorch--keras)
14. [Key Takeaways](#key-takeaways)
15. [Further Reading](#further-reading)

---

## 1. Why we need the LSTM

A vanilla RNN has one memory, the hidden state $h_t$, rewritten completely at every step: $h_t=\tanh(W[h_{t-1},x_t]+b)$. Two things go wrong ([problems-with-rnns.md](problems-with-rnns.md)):

- **Long-term dependency problem.** Information from early steps is overwritten and diluted by the time the end of a long sequence is reached (a "short-term memory" problem).
- **Unstable training.** Gradients flow back through the same matrix at every step, so they shrink (vanishing) or grow (exploding) exponentially with the number of steps.

The LSTM's answer is to add a **second, protected memory** that changes only by small, controlled edits, and to let the network *learn* when to erase, write and read it.

---

## 2. The big picture: two memories and three gates

| | Name | Role (class notes) | Symbol |
|---|---|---|---|
| Memory 1 | **Cell state** | long-term memory (LTM) | $C_t$ |
| Memory 2 | **Hidden state** | short-term memory (STM), also the output of the step | $h_t$ |

| Gate | Job (class notes) | Activation |
|---|---|---|
| **Forget gate** | remove something from the cell state | sigmoid |
| **Input gate** | add something to the cell state | sigmoid (gate) and tanh (candidate) |
| **Output gate** | calculate the hidden state $h_t$ | sigmoid |

![LSTM architecture, clean redraw](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_lstm_architecture_redraw.png)

*Figure 1 (clean redraw of the class diagram on the "LSTM Architecture" page). The purple line across the top is the cell state $C$ (long-term memory): it enters as $C_{t-1}$, passes through one multiplication (×, controlled by the forget gate $f_t$) and one addition (+, bringing in the filtered new content $i_t\otimes\tilde C_t$), and leaves as $C_t$. The teal line along the bottom is the hidden state $h$ (short-term memory): $h_{t-1}$ is joined with the current input $x_t$ and fed to all gates. Red: forget gate, which removes something from $C$. Green and orange: input gate and candidate cell state, which add something to $C$. Blue: output gate, which combines with $\tanh(C_t)$ to calculate the new hidden state $h_t$, which is also the output of this step. As in the class notes, at the very first step $C_{t-1}=[0,0,0]$.*

![Unrolled LSTM](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/12_unrolled_lstm_cell_state_highway.png)

*Figure 2 (schematic). Three consecutive LSTM steps. The cell state (thick purple arrows) travels from step to step and is touched by only one multiplication and one addition per step, so information can ride along it for a long time; the hidden state (teal) is recomputed from the cell state at every step and also leaves the cell upward as that step's output.*

---

## 3. Notation and dimensions used in class

The class derivation uses a small concrete size so every shape can be checked:

- hidden state $h_{t-1}$: **3-dimensional** (a 3×1 vector) ; cell state $C_{t-1}$: also 3×1,
- input $x_t$: **4-dimensional**, $x_t=[x_{t1},x_{t2},x_{t3},x_{t4}]$,
- the gates all look at the **concatenation** $[h_{t-1},x_t]$, which has $3+4=7$ numbers (a 7×1 vector),
- every gate is a layer of **3 neurons** (one per memory slot), so each weight matrix is **3×7** and each bias is **3×1**.

![Forget gate as a neural layer](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_forget_gate_network.png)

*Figure 3 (clean redraw of the neuron diagram on the "forget gate" page). The 3 numbers of $h_{t-1}$ (teal) and the 4 numbers of $x_t$ (gray) are stacked into one 7-number input. Each of the 3 sigmoid neurons is connected to all 7 inputs, so a gate has $3\times7=21$ weights plus 3 biases $b_{f1},b_{f2},b_{f3}$; the three outputs $f_{t1},f_{t2},f_{t3}$ together form the 3×1 vector $f_t$. The same picture, with $\tanh$ or $\sigma$ in the neurons, describes the other gates.*

The whole forget gate is the formula from the notes:

$$f_t=\sigma\big(W_f\,[h_{t-1},\,x_t]+b_f\big),\qquad \underbrace{W_f}_{3\times7}\;\underbrace{[h_{t-1},x_t]}_{7\times1}+\underbrace{b_f}_{3\times1}\;\Rightarrow\;3\times1\xrightarrow{\ \sigma\ }3\times1 .$$

![Shapes of the forget gate](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_forget_gate_shapes.png)

*Figure 4 (shapes exact, numbers illustrative: random weights with seed 2, $h_{t-1}=[0.5,-0.5,0.2]$, $x_t=[1,0,-1,0.5]$). The class computation $(3\times7)(7\times1)=(3\times1)$, then adding the 3×1 bias part, then applying $\sigma$ element by element. For these numbers the pre-activation is $z=[-2.68,\,0.84,\,0.66]$ and the gate is $f_t=[0.06,\,0.70,\,0.66]$, so unit 1 would almost completely forget while units 2 and 3 keep about two thirds of their memory.*

---

## 4. The building blocks: pointwise operators and activations

The class notes introduce the three symbols used inside the cell, calling them "bitwise" operators; the correct name is **pointwise** (element-wise): they act position by position on vectors of equal size.

![Pointwise operators](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_pointwise_operators.png)

*Figure 5 (the example at the bottom of the "LSTM Architecture" page, recomputed). With $C_{t-1}=[4,5,6]$ and $f_t=[1,2,3]$: pointwise multiply ⊗ gives $[4\cdot1,\,5\cdot2,\,6\cdot3]=[4,10,18]$, and pointwise add ⊕ gives $[5,7,9]$. Two corrections to the handwritten page: (1) the third line, tanh applied to $[4,5,6]$, is written as $[0.26,\,0.34,\,0.53]$, but tanh saturates: $\tanh(4)=0.9993$, $\tanh(5)=0.9999$ and $\tanh(6)=1.0000$ (computed). (2) The values 1, 2, 3 are only practice numbers; a real forget gate comes out of a sigmoid and is always between 0 and 1.*

Two activation functions appear, each for a reason:

![Sigmoid vs tanh](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_sigmoid_vs_tanh_roles.png)

*Figure 6. Sigmoid squashes to (0, 1), which makes it a natural valve: 0 blocks everything and 1 lets everything through, so it is used for the three gates. Tanh squashes to (−1, 1) and can be negative, so it is used where a memory entry must be able to move **down** as well as up: for the candidate cell state $\tilde C_t$ and for $\tanh(C_t)$ when the hidden state is computed.*

---

## 5. Step 1: the forget gate

**Goal (class notes):** (1) calculate $f_t$, (2) compute $C_{t-1}\otimes f_t$. This removes part of the old long-term memory: each of the 3 memory slots is multiplied by a number in (0, 1).

$$f_t=\sigma(W_f[h_{t-1},x_t]+b_f),\qquad C_{t-1}\;\to\;f_t\otimes C_{t-1}\quad(3\times1).$$

![Forget gate effect](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_forget_gate_effect.png)

*Figure 7 (the class example $C_{t-1}=[4,5,6]$ on the "forget gate" page, plus three more cases computed the same way). Leftmost: with $f_t=[\tfrac12,\tfrac12,\tfrac12]$ the product is $[2,\,2.5,\,3]$, exactly the class result: half of every memory slot is kept. Next: $f_t=[1,1,1]$ keeps everything ($[4,5,6]$). Next: $f_t=[0,0,0]$ erases everything ($[0,0,0]$). Right: a mixed gate $[1,\tfrac12,0]$ keeps slot 1, halves slot 2 and erases slot 3, giving $[4,\,2.5,\,0]$. As the class notes conclude, how much old information is carried forward depends on $f_t$.*

---

## 6. Step 2: the input gate and the candidate cell state

Adding new information takes three parts, all computed from the same 7-number input $[h_{t-1},x_t]$ (see Figure 3 for the layout).

1. **Candidate cell state** $\tilde C_t$ (written $\bar C_t$ in the class notes): what we *could* write, from a layer of 3 **tanh** neurons.
2. **Input gate** $i_t$: a filter, from a layer of 3 **sigmoid** neurons, that decides which information from the candidate is allowed into the cell state.
3. **Filtered candidate** $i_t\otimes\tilde C_t$.

$$\tilde C_t=\tanh\big(W_c[h_{t-1},x_t]+b_c\big),\qquad i_t=\sigma\big(W_i[h_{t-1},x_t]+b_i\big),$$

with $W_c$ and $W_i$ both 3×7, and $b_c$, $b_i$ both 3×1. Each result is a 3×1 vector.

![Input gate and candidate](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_input_gate_and_candidate.png)

*Figure 8 (clean redraw of the "input gate" pages). Left: two parallel layers of three neurons read the same 7×1 input; the tanh layer produces $\tilde C_t$ (values in (−1, 1)) and the sigmoid layer produces $i_t$ (values in (0, 1)), each of size 3×1. Right: the class example $i_t\otimes\tilde C_t=[0.5,0.5,0.5]\otimes[4,5,6]=[2,\,2.5,\,3]$, the "filtered candidate cell state". One caution the class numbers hide: because $\tilde C_t$ comes out of tanh, its real entries are between −1 and 1, so $[4,5,6]$ is only a convenient set of numbers for showing the multiplication.*

---

## 7. Step 3: updating the cell state

The new long-term memory is the kept old memory plus the filtered new content (the formula at the top of the class page "then, $C_t=f_t\otimes C_{t-1}\oplus i_t\otimes\tilde C_t$"):

$$\boxed{\,C_t\;=\;f_t\otimes C_{t-1}\;\oplus\;i_t\otimes\tilde C_t\,}$$

![Cell state update](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_cell_state_update.png)

*Figure 9 (computed; $f_t=i_t=[0.5,0.5,0.5]$ and $C_{t-1}=[4,5,6]$ from the class examples, with an illustrative realistic candidate $\tilde C_t=[0.8,-0.6,0.4]$). Left to right: the old cell state, the kept part $f_t\otimes C_{t-1}=[2,2.5,3]$, the added part $i_t\otimes\tilde C_t=[0.4,-0.3,0.2]$, and the new cell state $C_t=[2.4,\,2.2,\,3.2]$. Unit 2 shows why tanh matters: the candidate was negative, so the memory in that slot went down.*

Notice what the update is **not**: the old memory is not pushed through a weight matrix and a squashing function, only scaled by $f_t$ and added to. That additive structure is the reason for the gradient behaviour in Section 10.

---

## 8. Step 4: the output gate and the hidden state

The hidden state (short-term memory) is a filtered view of the cell state. The **output gate** $o_t$ decides how much of each cell-state slot to expose:

$$o_t=\sigma\big(W_o[h_{t-1},x_t]+b_o\big),\qquad \boxed{\,h_t=o_t\otimes\tanh(C_t)\,}$$

with $W_o$ of shape 3×7 and $b_o$ of shape 3×1. The $\tanh$ squashes the (unbounded) cell state into (−1, 1) first, which is why the diagram in the class notes has a tanh box after the cell-state line.

*Source note: the six photographed class pages cover the forget gate, the input gate, the cell-state update and the architecture overview; the output gate has no page among them, so this section follows the video summary ("processes the updated cell state to produce the new hidden state") and the standard LSTM equations.*

![Output gate](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_output_gate_hidden_state.png)

*Figure 10 (computed with illustrative numbers). Starting from $C_t=[2.4,2.2,3.2]$ (Figure 9): $\tanh(C_t)=[0.984,\,0.976,\,0.997]$, nearly saturated because the cell values are large; with an output gate $o_t=[0.9,0.1,0.5]$ the hidden state is $h_t=[0.885,\,0.098,\,0.498]$. Slot 2 is "remembered" in $C_t$ but almost hidden from the outside because its output gate is nearly closed: the cell state can hold information that $h_t$ does not currently show.*

The new pair $(C_t,h_t)$ is what the next time step receives as $(C_{t-1},h_{t-1})$, and $h_t$ is also the output of the step.

---

## 9. One full step with numbers, and counting parameters

Putting Sections 5 to 8 together, with the class sizes (hidden 3, input 4) and $C_{t-1}=[4,5,6]$, computed in NumPy:

![One-step numeric walkthrough](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10_one_step_numeric_walkthrough.png)

*Figure 11 (computed; random weights with seed 1, forget bias 0.5, $h_{t-1}=[0.5,-0.5,0.2]$, $x_t=[1,0,-1,0.5]$). Every vector the cell computes, in order, with its size. The forget gate keeps 26%, 75% and 53% of the three memory slots; the input gate lets in only 9%, 16% and 72% of the candidate; the new cell state is $C_t=[1.113,\,3.694,\,2.780]$ and the new hidden state $h_t=[0.568,\,0.549,\,0.667]$. The same numbers are reproduced by the from-scratch code (Section 12) and by PyTorch's `LSTMCell` (Section 13).*

**Parameter count.** Each of the four blocks ($f$, $i$, $\tilde C$, $o$) is a layer of $n$ neurons over $n+m$ inputs, so with hidden size $n$ and input size $m$:

$$\text{parameters}=4\,\big(n(n+m)+n\big)=4\,(3\cdot7+3)=4\times24=96 .$$

![Parameter counting](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11_parameter_counting.png)

*Figure 12 (counting; matches the NumPy model, whose weight array has shape (12, 7) and bias length 12). Each block has $3\times7=21$ weights and 3 biases, 24 parameters, and there are four blocks, 96 in total, which is four times the 24 parameters of a vanilla RNN layer of the same size. The price of the extra memory control is 4x the parameters and compute per layer.*

![Summary table](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/18_lstm_summary_table.png)

*Figure 13 (summary). Each part of the cell with its activation, formula, the shape arithmetic for hidden size 3 and input size 4, and its role.*

---

## 10. Why the LSTM avoids vanishing gradients

The video's conclusion is that the gated design prevents the vanishing-gradient problem. Here is the mechanism, and then a measurement.

Backpropagation through time needs $\partial C_t/\partial C_{t-1}$. Since $C_t=f_t\otimes C_{t-1}+i_t\otimes\tilde C_t$, the **direct** path from one cell state to the previous one is

$$\frac{\partial C_t}{\partial C_{t-1}}\Big|_{\text{direct}}=\mathrm{diag}(f_t),$$

a plain diagonal matrix with entries in (0, 1), instead of a dense weight matrix multiplied by a derivative $\varphi'<1$ at every step as in a vanilla RNN. (The gates also depend on $h_{t-1}$, which depends on $C_{t-1}$, so there are additional indirect terms; the direct one is the "highway".) Over $k$ steps the direct path contributes $\prod f_t$: if the network keeps $f_t$ near 1 for a memory slot, the gradient for that slot survives; the network can learn this.

![Gradient flow](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/13_gradient_flow_lstm_vs_rnn.png)

*Figure 14. (a) Exact arithmetic for the direct path with constant forget values: $0.5^{50}=8.9\times10^{-16}$, $0.9^{50}=5.2\times10^{-3}$, $0.99^{50}=0.61$, and $1.0^{n}=1$ forever. (b) Measured, not assumed: the full backpropagated gradient norm reaching each earlier step on the remember-the-first-bit task with $T=60$ (untrained networks, 32 units, averaged geometrically over 10 random nets, normalized by the value at the last step; for the LSTM the quantity is $\|\partial L/\partial C\|$, for the vanilla RNN $\|\partial L/\partial h\|$). At lag 59 the relative gradient is $1.5\times10^{-2}$ for an LSTM with forget-gate bias 1, against $1.7\times10^{-9}$ for an LSTM with forget bias 0 and $5.6\times10^{-10}$ for the vanilla tanh RNN: about seven orders of magnitude more gradient arrives at the start of the sequence. With bias 0 the forget gates start near 0.5, the product behaves like the $0.5^n$ curve in (a), and the LSTM is no better than the vanilla RNN, which is why initialization matters (Section 11.2).*

---

## 11. Experiments

All experiments use a **remember-the-first-bit** task: the first input $x_1=\pm1$ decides the label, the other $T-1$ inputs are Gaussian noise with standard deviation 1, and the label (1 if $x_1>0$) must be predicted from the final hidden state. Chance is 50%. Setup: 32 hidden units, Adam (lr 3e-3), gradient-norm clipping at 1, batch 64, 1,500 steps, 1,000 fresh test sequences, 3 seeds per setting. This is the same task as in [problems-with-rnns.md](problems-with-rnns.md).

### 11.1 LSTM against a vanilla RNN

![LSTM vs RNN](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/14_lstm_vs_rnn_memory_task.png)

*Figure 15 (measured, 3 seeds per point; dots are individual seeds, lines are means). The LSTM (forget-gate bias 1) reaches 100% test accuracy on all 3 seeds at $T=20$, 40 and 60, where the vanilla tanh RNN is at chance for every seed at $T=40$ and $T=60$ (48.4% to 51.3%) and succeeds on only 2 of 3 seeds at $T=20$ (51.0%, 100%, 100%). The LSTM extends the reachable length, but not without limit in this setup: at $T=80$ only 1 of 3 LSTM seeds succeeds (100%, 49.3%, 48.9%), and at $T=100$ and $T=150$ all three seeds are at chance (50.4%, 49.9%, 50.4% and 49.2%, 50.3%, 51.0%). Training longer did not rescue $T=100$: two further runs with 5,000 steps gave 50.2% and 52.1%. One vanilla-RNN seed reached 98.3% at $T=150$ (the other two: 51.5% and 53.7%), a lucky run that shows how seed-dependent these results are. So the honest summary is: gates help a lot (3/3 versus 0/3 at $T=40$ and 60), they are not magic, and results at long lengths depend on the seed, the width, the noise level and the training budget.*

### 11.2 The forget-gate bias matters

The forget gate pre-activation is $W_f[h_{t-1},x_t]+b_f$, so initializing $b_f$ to a positive number makes the gates start mostly open ($\sigma(1)=0.73$), which keeps the memory path alive at the beginning of training.

![Forget bias effect](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/15_forget_bias_effect.png)

*Figure 16 (measured, 3 seeds, black dots are seeds). The same LSTM at $T=60$ with forget-gate bias initialized to 0 stays at chance on all three seeds (49.4%, 50.8%, 49.9%), while bias 1 reaches 100% on all three. At $T=100$ both settings are at chance (bias 0: 49.7%, 49.2%, 50.0%; bias 1: 50.4%, 49.9%, 50.4%), consistent with Figure 15. A single, free initialization choice decides whether the LSTM works at all at $T=60$; the same recommendation appears in the literature (Gers et al., 2000; Jozefowicz et al., 2015).*

### 11.3 What a trained LSTM actually does

Textbook intuition says "the forget gate stays near 1 and the cell state stores the bit". I checked this on the trained $T=60$ network (seed 0, 100% accuracy) by feeding 300 noise sequences twice, once with first bit +1 and once with −1 and the same noise.

![Trained LSTM traces](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/16_trained_lstm_gate_traces.png)

*Figure 17 (measured on the trained network, averages over 300 sequences). (a) The cell state of the three units that best separate the two bits: solid lines for bit +1 and dashed for bit −1 follow clearly different trajectories for the whole sequence, and at the last step they sit about 6.3 to 6.6 apart (for example unit 20 ends near +3.0 for +1 and −3.6 for −1). (b) The forget gate is not pinned at 1: its mean is 0.76 at $t=2$, 0.77 at $t=30$ and 0.74 at $t=59$ over all units (0.85 at $t=30$ for the three units of (a)). (c) The input gate is about 0.55 at the first step and about 0.6 later. (d) The output gate rises from 0.56 at $t=1$ to 0.71 at $t=59$ (0.88 for the three units of (a)), so the network "opens the read-out" toward the end. The lesson is that this small trained LSTM does not freeze the bit in a flat memory with $f\approx1$; it uses gentle time-varying dynamics (the trajectories drift and even change sign) that still keep the two cases separable, and the gates are smooth functions rather than on/off switches. Gate values near 1 are what the architecture makes possible, not what training necessarily finds.*

### 11.4 Real gate values on a real sentence

The translation encoder from [encoder-decoder-architecture.md](encoder-decoder-architecture.md) is an LSTM with 64 units. Here are its actual gate values while it reads the English sentence "you read the blue book at home":

![Gates on a real sentence](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/17_gates_on_real_sentence.png)

*Figure 18 (real values of the trained model, one column per word, one row per hidden unit; bottom: mean and min-to-max band over the 64 units). The mean forget gate rises from 0.79 ("you") and 0.77 ("read") to about 0.92 to 0.97 for "the", "blue", "book", "at" and "home", so as the sentence unfolds the network increasingly keeps what it has accumulated. The mean input gate is 0.60 and 0.62 for the first two words and peaks at 0.90 for "blue", the colour word, then settles near 0.8; the mean output gate grows from 0.53 to 0.78. Each cell has many units with different behaviour (the bands span most of 0 to 1), and I do not attach a meaning to individual units. A reasonable hypothesis, not tested here, is that the colour word receives a large input gate because French must place it after the noun.*

---

## 12. From-scratch implementation (NumPy)

The code below contains everything used above: a batched LSTM with manual backpropagation through time (gate order $f,i,\tilde C,o$ and input $[h_{t-1},x_t]$, as in the class notes), the one-step computation written gate by gate with the class sizes, the pointwise-operator check, a finite-difference gradient check, and the training runs for the forget-bias comparison. It was run exactly as printed.

```python
import numpy as np
sigmoid = lambda z: 1.0/(1.0+np.exp(-np.clip(z, -60, 60)))

class LSTM:
    """Single-layer many-to-one LSTM.  Gate order in W: [f, i, c~, o];  z = W [h_{t-1}, x_t] + b  (same layout as the class notes)."""
    def __init__(self, n_in, n_h, forget_bias=1.0, seed=0, scale=None):
        rng = np.random.default_rng(seed); self.n_in, self.n_h = n_in, n_h; s = scale or 1/np.sqrt(n_h+n_in)
        self.W = rng.normal(0, s, (4*n_h, n_h+n_in)); self.b = np.zeros(4*n_h); self.b[:n_h] = forget_bias
        self.v = rng.normal(0, 1/np.sqrt(n_h), n_h); self.c0 = 0.0
    def params(self): return {'W': self.W, 'b': self.b, 'v': self.v, 'c0': np.array(self.c0)}
    def step(self, x, h, c):
        H = self.n_h; hx = np.concatenate([h, x], 1); z = hx @ self.W.T + self.b
        f = sigmoid(z[:, :H]); i = sigmoid(z[:, H:2*H]); g = np.tanh(z[:, 2*H:3*H]); o = sigmoid(z[:, 3*H:])
        c2 = f*c + i*g; tc = np.tanh(c2); h2 = o*tc
        return h2, c2, (hx, f, i, g, o, c, tc)
    def forward(self, X, keep=False):
        B, T, _ = X.shape; h = np.zeros((B, self.n_h)); c = np.zeros((B, self.n_h)); caches = []; tr = []
        for t in range(T):
            h, c, ca = self.step(X[:, t], h, c); caches.append(ca)
            if keep: tr.append((h.copy(), c.copy(), ca[1], ca[2], ca[3], ca[4]))
        return h, c, caches, tr
    def loss_grads(self, X, y, want_norms=False):
        B, T, _ = X.shape; H = self.n_h; h, c, caches, _ = self.forward(X)
        p = sigmoid(h @ self.v); pc = np.clip(p, 1e-12, 1-1e-12); loss = -np.mean(y*np.log(pc)+(1-y)*np.log(1-pc))
        dz = (p-y)/B; g = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b), 'v': h.T @ dz, 'c0': np.array(0.0)}
        dh = dz[:, None]*self.v[None, :]; dc = np.zeros((B, H)); cn = np.zeros(T+1); cn[T] = np.linalg.norm(dc, axis=1).mean()
        for t in range(T-1, -1, -1):
            hx, f, i, gg, o, cp, tc = caches[t]
            do = dh*tc; dc = dc + dh*o*(1-tc**2)
            dzt = np.concatenate([(dc*cp)*f*(1-f), (dc*gg)*i*(1-i), (dc*i)*(1-gg**2), do*o*(1-o)], 1)
            g['W'] += dzt.T @ hx; g['b'] += dzt.sum(0); dhx = dzt @ self.W; dh = dhx[:, :H]; dc = dc*f
            cn[t] = np.linalg.norm(dc, axis=1).mean()          # ||dL/dc_{t-1}||
        acc = np.mean((p > 0.5) == (y > 0.5)); return (loss, acc, g, cn) if want_norms else (loss, acc, g)

def gnorm(g): return np.sqrt(sum((v**2).sum() for v in g.values()))
class Adam:
    def __init__(self, m, lr=3e-3): self.m, self.lr, self.t = m, lr, 0; self.mm = {k: np.zeros_like(v) for k, v in m.params().items()}; self.vv = {k: np.zeros_like(v) for k, v in m.params().items()}
    def step(self, g, clip=1.0):
        s = min(1.0, clip/(gnorm(g)+1e-12)); self.t += 1
        for k in ['W', 'b', 'v', 'c0']:
            gk = g[k]*s; self.mm[k] = 0.9*self.mm[k]+0.1*gk; self.vv[k] = 0.999*self.vv[k]+0.001*gk**2
            upd = self.lr*(self.mm[k]/(1-0.9**self.t))/(np.sqrt(self.vv[k]/(1-0.999**self.t))+1e-8)
            if k == 'c0': self.m.c0 -= float(upd)
            else: getattr(self.m, k).__isub__(upd)

def train_lstm(T, seed=0, steps=1500, lr=3e-3, n_h=32, forget_bias=1.0, noise=1.0, clip=1.0, B=64):
    rng = np.random.default_rng(1000+seed); m = LSTM(1, n_h, forget_bias, seed); o = Adam(m, lr)
    for _ in range(steps):
        X, y = make_batch(rng, B, T, noise); _, _, g = m.loss_grads(X, y); o.step(g, clip)
    Xt, yt = make_batch(np.random.default_rng(99), 1000, T, noise); l, a, _ = m.loss_grads(Xt, yt); return m, l, a

def make_batch(rng, B, T, noise=1.0):
    """remember-the-first-bit task: x_1 = +-1 is the signal, x_2..x_T are Gaussian noise, label = 1 if x_1 > 0"""
    X = rng.normal(0, noise, (B, T, 1)); s = rng.integers(0, 2, B); X[:, 0, 0] = 2.0*s - 1.0
    return X, s.astype(float)

# ---------- one LSTM step written gate by gate, with the class dimensions (hidden = 3, input = 4) ----------
def lstm_step_by_gate(h_prev, x, C_prev, W, b):
    v = np.concatenate([h_prev, x])                      # [h_{t-1}, x_t]   -> 7 x 1
    f = sigmoid(W['f'] @ v + b['f'])                     # forget gate      (3x7)(7x1)+(3x1) -> 3x1
    i = sigmoid(W['i'] @ v + b['i'])                     # input gate
    c_tilde = np.tanh(W['c'] @ v + b['c'])               # candidate cell state
    o = sigmoid(W['o'] @ v + b['o'])                     # output gate
    C = f * C_prev + i * c_tilde                         # C_t = f_t (x) C_{t-1}  (+)  i_t (x) C~_t
    h = o * np.tanh(C)                                   # h_t = o_t (x) tanh(C_t)
    return f, i, c_tilde, o, C, h

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # 1) the class example: hidden size 3, input size 4, C_{t-1} = [4, 5, 6]
    rng = np.random.default_rng(1); W = {k: rng.normal(0, 0.7, (3, 7)).round(2) for k in 'fico'}; b = {k: np.zeros(3) for k in 'fico'}; b['f'] = np.full(3, 0.5)
    h_prev, x, C_prev = np.array([0.5, -0.5, 0.2]), np.array([1.0, 0.0, -1.0, 0.5]), np.array([4.0, 5.0, 6.0])
    f, i, c_tilde, o, C, h = lstm_step_by_gate(h_prev, x, C_prev, W, b)
    for name, val in [('f_t', f), ('i_t', i), ('C~_t', c_tilde), ('o_t', o), ('f_t * C_{t-1}', f*C_prev), ('i_t * C~_t', i*c_tilde), ('C_t', C), ('h_t', h)]: print('%-14s %s' % (name, val))
    print('parameters per gate: %d, LSTM layer (4 gates): %d' % (3*7+3, 4*(3*7+3)))

    # 2) pointwise operators from the class notes
    a, bb = np.array([4., 5., 6.]), np.array([1., 2., 3.]); print('multiply', a*bb, ' add', a+bb, ' tanh([4,5,6])', np.tanh(a))

    # 3) finite-difference gradient check of the batched LSTM
    rng = np.random.default_rng(0); worst = 0
    m = LSTM(2, 5, 0.0, seed=3); m.W += rng.normal(0, 0.3, m.W.shape); X = rng.normal(size=(4, 6, 2)); y = rng.integers(0, 2, 4).astype(float); _, _, g = m.loss_grads(X, y)
    for k in ['W', 'b', 'v']:
        P = getattr(m, k)
        for f_ in np.argsort(-np.abs(g[k]).ravel())[:15]:
            ix = np.unravel_index(f_, P.shape); old = P[ix]; P[ix] = old + 1e-5; lp = m.loss_grads(X, y)[0]; P[ix] = old - 1e-5; lm = m.loss_grads(X, y)[0]; P[ix] = old
            num = (lp - lm) / 2e-5; worst = max(worst, abs(num - g[k][ix]) / (abs(num) + abs(g[k][ix]) + 1e-12))
    print('gradient check: max relative error %.0e' % worst)

    # 4) remember the first bit across T = 60 noisy steps: forget-gate bias 1 vs 0 (same seed, same data)
    for fb in [1.0, 0.0]:
        model, loss, acc = train_lstm(60, seed=0, forget_bias=fb)
        print('forget bias %.0f: test loss %.3f, test accuracy %.3f' % (fb, loss, acc))
```

Output of running it as printed:

```text
f_t            [0.2623 0.7506 0.5277]
i_t            [0.0879 0.1559 0.715 ]
C~_t           [ 0.7264 -0.3791 -0.5399]
o_t            [0.7056 0.5501 0.6722]
f_t * C_{t-1}  [1.0492 3.7532 3.1663]
i_t * C~_t     [ 0.0638 -0.0591 -0.386 ]
C_t            [1.1131 3.6941 2.7803]
h_t            [0.5681 0.5494 0.667 ]
parameters per gate: 24, LSTM layer (4 gates): 96
multiply [ 4. 10. 18.]  add [5. 7. 9.]  tanh([4,5,6]) [0.9993 0.9999 1.    ]
gradient check: max relative error 6e-09
forget bias 1: test loss 0.000, test accuracy 1.000
forget bias 0: test loss 0.693, test accuracy 0.494
```

The first block reproduces Figure 11 (the class dimensions: forget gate 3×7 and so on) and the parameter count 24 per gate and 96 per layer; the pointwise line reproduces Figure 5 including the corrected tanh values; the gradient check agrees with finite differences to about $10^{-9}$; and the last two lines are seed 0 of Figures 15 and 16 (bias 1 solves $T=60$, bias 0 stays at chance).

---

## 13. Framework equivalents (PyTorch / Keras)

**PyTorch.** This was executed. `nn.LSTMCell` / `nn.LSTM` use gate order input, forget, cell, output (i, f, g, o) and split the weights into $W_{ih}$ (acting on $x_t$) and $W_{hh}$ (acting on $h_{t-1}$), plus two bias vectors, instead of one matrix over $[h_{t-1},x_t]$. Copying the class example's weights into an `LSTMCell(4, 3)` reproduces the NumPy result exactly: $C_t=[1.1131,3.6941,2.7803]$ and $h_t=[0.5681,0.5494,0.6670]$. Because of the second bias, `nn.LSTM(4, 3)` has 108 parameters instead of 96 (shapes (12,4), (12,3), (12,), (12,)).

```python
import torch, torch.nn as nn

lstm = nn.LSTM(input_size=4, hidden_size=3, batch_first=True)    # 108 parameters (two bias vectors)
x = torch.randn(1, 10, 4)                                          # batch of 1, 10 time steps, 4 features
out, (h_n, c_n) = lstm(x)                                          # out: (1, 10, 3) = h_t at every step
# h_n, c_n: final hidden state and cell state, shape (num_layers, batch, 3)

# forget-gate bias = 1 (Section 11.2): PyTorch gate order is (i, f, g, o), so the forget bias is the 2nd block of 3
with torch.no_grad():
    lstm.bias_ih_l0[3:6].fill_(1.0)

# gradient clipping (used in all experiments): after loss.backward(), before optimizer.step()
# torch.nn.utils.clip_grad_norm_(lstm.parameters(), max_norm=1.0)

cell = nn.LSTMCell(4, 3)                                           # one step at a time, as in the class derivation
h, c = cell(torch.randn(1, 4), (torch.zeros(1, 3), torch.zeros(1, 3)))   # c_{t-1} = [0, 0, 0] at the first step
```

**Keras / TensorFlow.** *This snippet was not executed* (TensorFlow was not available in the environment); check it against your installed version. Keras also uses the (i, f, c, o) order and a single bias per layer.

```python
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Input(shape=(None, 4)),                                 # any number of time steps, 4 features
    layers.LSTM(3, unit_forget_bias=True, return_sequences=False), # unit_forget_bias=True initializes the forget bias to 1
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(3e-3, clipnorm=1.0), loss="binary_crossentropy")
# return_sequences=True gives h_t at every step; return_state=True also returns (h, c)
```

---

## Key Takeaways

- An LSTM has **two memories**: the cell state $C_t$ (long-term) and the hidden state $h_t$ (short-term, also the output), controlled by **three gates** (forget, input, output), each a layer of sigmoid neurons over the concatenated $[h_{t-1},x_t]$ (7 numbers in the class example, so weights of shape 3×7).
- Forget gate: $f_t=\sigma(W_f[h_{t-1},x_t]+b_f)$ scales the old memory; with $f_t=[\tfrac12,\tfrac12,\tfrac12]$ and $C_{t-1}=[4,5,6]$ the result is $[2,2.5,3]$.
- Input gate and candidate: $i_t=\sigma(\cdot)$ filters $\tilde C_t=\tanh(\cdot)$; the update is $C_t=f_t\otimes C_{t-1}\oplus i_t\otimes\tilde C_t$ and the output is $h_t=o_t\otimes\tanh(C_t)$.
- Sigmoid is the valve (0 to 1); tanh lets memory move both up and down (−1 to 1). The class page's $\tanh([4,5,6])=[0.26,0.34,0.53]$ should read $[0.9993,\,0.9999,\,1.0000]$.
- One layer with hidden size 3 and input size 4 has $4(3\cdot7+3)=96$ parameters, four times a vanilla RNN of the same size (PyTorch counts 108 because it keeps two bias vectors).
- The additive cell-state update gives the gradient a direct path $\mathrm{diag}(f_t)$: measured over 59 steps, $1.5\times10^{-2}$ of the gradient survives with forget bias 1 against about $10^{-9}$ for a vanilla RNN or for forget bias 0 (Figure 14).
- On the remember-the-first-bit task the LSTM solved lengths 20, 40 and 60 on every seed where a vanilla RNN solved none of the seeds at 40 and 60, but it still failed at $T=100$ and beyond in this setup, and the **forget-gate bias** decided success at $T=60$ (bias 1: 3/3, bias 0: 0/3).
- A trained LSTM does not simply hold $f\approx1$: its gates were smooth (forget about 0.75), and the memory of the bit lived in time-varying cell-state trajectories. All experiments here are small, synthetic and use 3 seeds; they illustrate mechanisms and are not benchmarks.

---

## Further Reading

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780. The original LSTM (cell state with a constant error carousel; no forget gate yet).
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. *Neural Computation*, 12(10), 2451-2471. Introduced the forget gate and the practice of a positive forget bias.
- Gers, F. A., & Schmidhuber, J. (2000). Recurrent nets that time and count. *IJCNN*. Peephole connections.
- Hochreiter, S. (1991). *Untersuchungen zu dynamischen neuronalen Netzen* (Diploma thesis). Technische Universität München. The vanishing-gradient analysis that motivates the design.
- Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE Transactions on Neural Networks*, 5(2), 157-166.
- Greff, K., Srivastava, R. K., Koutnik, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A search space odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232. A large empirical study of which LSTM parts matter (the forget gate and output activation were found most important).
- Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An empirical exploration of recurrent network architectures. *ICML*. Recommends initializing the forget-gate bias to 1.
- Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*. Introduced the GRU, a simpler gated cell.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML*.
- Olah, C. (2015). *Understanding LSTM Networks* (blog post, colah.github.io). A widely read visual walkthrough of the same gates.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 10 (Section 10.10 on LSTMs and other gated RNNs). MIT Press.
