# Problems with RNNs: Long-Term Dependencies and Unstable Training

This document explains the two core weaknesses of a vanilla recurrent neural network (RNN): it struggles to carry information across long sequences (the **long-term dependency problem**), and its training is numerically fragile because gradients are multiplied by the same kind of factor at every time step, so they either shrink toward zero (**vanishing gradients**) or blow up (**exploding gradients**). It then walks through the standard remedies: gradient clipping, ReLU activations with identity or orthogonal initialization, and skip connections, which lead naturally to gated cells such as the LSTM. These problems are the reason plain RNNs are rarely used for long sequences, and understanding them makes the design of LSTMs and GRUs feel inevitable rather than arbitrary. Every number and plot below comes from code that was actually run (a from-scratch NumPy RNN with manual backpropagation through time, checked against finite differences); the few places that are schematic are labeled as such.

> **Prerequisite:** the RNN forward pass, weight sharing, and unrolling are covered in [rnn-architecture-and-forward-propagation.md](rnn-architecture-and-forward-propagation.md). The chain-rule mechanics of backpropagation are the same ones used in [backpropagation-in-cnn-part1.md](backpropagation-in-cnn-part1.md) and [backpropagation-in-cnn-part2.md](backpropagation-in-cnn-part2.md). The next note in the series (planned: `lstm.md`) builds the gated architecture that these problems motivate.

---

## Table of Contents

1. [Recap: what an RNN computes](#1-recap-what-an-rnn-computes)
2. [The two problems at a glance](#2-the-two-problems-at-a-glance)
3. [Backpropagation through time and the Jacobian product](#3-backpropagation-through-time-and-the-jacobian-product)
4. [Problem 1: the long-term dependency problem](#4-problem-1-the-long-term-dependency-problem)
5. [Problem 2a: vanishing gradients](#5-problem-2a-vanishing-gradients)
6. [Problem 2b: exploding gradients](#6-problem-2b-exploding-gradients)
7. [Solutions](#7-solutions)
   - [7.1 Gradient clipping](#71-gradient-clipping)
   - [7.2 ReLU activations and better initialization](#72-relu-activations-and-better-initialization)
   - [7.3 Skip connections, and a preview of gating](#73-skip-connections-and-a-preview-of-gating)
   - [7.4 Summary of the fixes](#74-summary-of-the-fixes)
8. [From-scratch implementation (NumPy)](#8-from-scratch-implementation-numpy)
9. [Framework equivalents (Keras / PyTorch)](#9-framework-equivalents-keras--pytorch)
10. [Key Takeaways](#key-takeaways)
11. [Further Reading](#further-reading)

---

## 1. Recap: what an RNN computes

A vanilla RNN reads a sequence $x_1, \dots, x_T$ one element at a time and keeps a hidden state $h_t$ that is meant to summarize everything seen so far:

$$a_t = W h_{t-1} + U x_t + b, \qquad h_t = \varphi(a_t), \qquad h_0 = 0$$

The same $W$, $U$, $b$ are reused at every step (weight sharing). For a many-to-one task such as classification, a readout sees only the last state, $\hat y = \sigma(v^\top h_T + c)$, and a loss $L$ is computed from it. Training means computing $\partial L / \partial W$, $\partial L / \partial U$, and so on, by backpropagating through the unrolled graph, an algorithm called **backpropagation through time (BPTT)**.

The important structural fact is that the *same matrix $W$* sits between every pair of consecutive time steps. That is what makes RNNs compact, and it is also the root of both problems below.

---

## 2. The two problems at a glance

![Overview of the two RNN problems](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_problem_overview.png)

*Figure 1 (schematic, not measured data). An RNN unrolled over six steps. Top (green): the share of information about the first input $x_1$ that survives in each later hidden state shrinks step by step, which is the long-term dependency problem, a forward-pass view of "short-term memory". Bottom (red): during training, the gradient of the loss at the last step travels backward through the same chain; each arrow is a multiplication by a Jacobian matrix, and when those factors are consistently smaller than one the gradient reaching early steps becomes vanishingly thin (thin arrows), while factors consistently larger than one make it huge. Both problems come from the same cause, repeated multiplication across time steps, which Sections 3 to 6 make precise and measure.*

| | Long-term dependency problem | Unstable training |
|---|---|---|
| Symptom | The model ignores early inputs, even when they decide the answer | Loss stalls (vanishing) or jumps / goes to `NaN` (exploding) |
| Where it shows up | Forward behavior and what is learnable | Backward pass (gradients) |
| Cause | Information and gradient must pass through many repeated transformations | Product of many Jacobians whose size is consistently below or above 1 |

---

## 3. Backpropagation through time and the Jacobian product

Because $h_t = \varphi(W h_{t-1} + U x_t + b)$, the derivative of one state with respect to the previous one is

$$J_t \;=\; \frac{\partial h_t}{\partial h_{t-1}} \;=\; \mathrm{diag}\big(\varphi'(a_t)\big)\, W .$$

Chaining these from step $k$ to the final step $T$ gives

$$\frac{\partial h_T}{\partial h_k} = J_T J_{T-1} \cdots J_{k+1}, \qquad
\frac{\partial L}{\partial h_k} = J_{k+1}^{\top} J_{k+2}^{\top} \cdots J_T^{\top}\, \frac{\partial L}{\partial h_T}.$$

The gradient with respect to the shared weights is a sum over time, and each term carries its own product of Jacobians:

$$\frac{\partial L}{\partial W} \;=\; \sum_{k=1}^{T} \Big(\frac{\partial L}{\partial h_k} \odot \varphi'(a_k)\Big)\, h_{k-1}^{\top}.$$

Terms with $k$ close to $T$ involve few Jacobians; terms with small $k$ (the far past) involve $T-k$ of them. Taking norms, with $\gamma = \max|\varphi'|$ ($\gamma = 1$ for tanh, $\gamma = \tfrac14$ for sigmoid) and $\sigma_{\max}(W)$ the largest singular value of $W$:

$$\Big\|\frac{\partial h_T}{\partial h_k}\Big\| \;\le\; \prod_{i=k+1}^{T} \|J_i\| \;\le\; \big(\gamma\,\sigma_{\max}(W)\big)^{T-k}.$$

Following Pascanu et al. (2013): if $\gamma\,\sigma_{\max}(W) < 1$ the long-range terms **must** vanish exponentially in $T-k$; and $\gamma\,\sigma_{\max}(W) > 1$ is a **necessary** (not sufficient) condition for them to explode.

![BPTT Jacobian chain](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_bptt_jacobian_chain.png)

*Figure 2. The formulas of this section drawn on the unrolled chain. Blue arrows are the forward transitions, each labeled with its Jacobian $J_t$; red arrows are the backward pass, which multiplies the loss gradient by the transposed Jacobians in reverse order. The two lines at the bottom state the norm bound and the resulting vanishing / exploding conditions. Everything that follows is a measurement of how these products behave in practice.*

### The simplest possible case

Take one hidden unit with no nonlinearity, $h_t = w\,h_{t-1}$. Then $\partial h_T/\partial h_k = w^{T-k}$ exactly.

![Powers of w](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_scalar_power_law.png)

*Figure 3. The curve $w^n$ for five values of $w$, on a linear scale (left) and a log scale (right). This is exact arithmetic, not an experiment. With $w = 0.9$ the factor is $0.349$ after 10 steps, $0.0052$ after 50 and $2.7\times10^{-5}$ after 100; with $w = 0.5$ it is $\approx 10^{-3}$ after 10 steps and $8.9\times10^{-16}$ after 50; with $w = 1.1$ it grows to $2.6$, $117$ and $1.4\times10^{4}$. Only $w$ almost exactly equal to 1 avoids both fates, and a trained network has no reason to sit on that knife edge. A matrix does the same thing along each of its directions, which is why the matrix case below behaves identically.*

---

## 4. Problem 1: the long-term dependency problem

To test the claim "RNNs forget early inputs" directly, I use a synthetic task where the answer depends **only** on the first element:

- $x_1 = \pm 1$ is a signal bit; $x_2, \dots, x_T$ are Gaussian noise with standard deviation $\sigma$.
- The label is $y = \mathbb{1}[x_1 > 0]$, predicted from $h_T$ alone.

So the network must carry one bit across $T-1$ steps of distraction. Chance accuracy is 50%.

![Memory task](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_memory_task_diagram.png)

*Figure 4. The "remember the first bit" task. Only $x_1$ carries information; everything else is noise the network has to learn to ignore. The dependency length is $T-1$ steps, so each extra time step adds one more Jacobian between the signal and the loss. This is a deliberately clean stand-in for real examples such as a sentence whose last word depends on its first.*

**Setup (all runs in this section):** vanilla tanh RNN, 32 hidden units, default Gaussian initialization ($W_{ij}\sim\mathcal N(0,1/n)$), batch size 64, Adam, gradient-norm clipping at 1.0 (so that exploding gradients do not confound the result), accuracy measured on 1000 fresh sequences.

![Accuracy vs sequence length](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_memory_accuracy_vs_T.png)

*Figure 5. Test accuracy of a vanilla RNN as the sequence length $T$ grows (each dot is one random seed, lines are means). With distractor noise $\sigma=0.5$ (3 seeds, Adam lr 3e-3, 1500 steps) accuracy is 100% up to $T=20$, 99.7% at $T=30$, 98.4% at $T=40$, then becomes unreliable: 81.8% at $T=50$, 67.5% at $T=60$ and 83.3% at $T=80$, with some seeds still succeeding and others sitting at chance, so the trend is noisy rather than a clean cliff. With stronger noise $\sigma=1.0$ (5 seeds, Adam lr 2e-3, 1200 steps) the picture is sharper: 100% at $T=10$, 80.2% at $T=20$ (two of five seeds at chance), and 49% to 51% (chance) at every length from $T=30$ to $T=80$. The architecture is expressive enough to solve the task at short lengths, so what fails at long lengths is not capacity but learnability, which the next sections trace to the gradient.*

A caveat: this is one synthetic task with a small network and few seeds. It demonstrates the mechanism; it is not a benchmark.

### Why learning fails: the gradient never reaches the signal

![Gradient reaching each time step](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_gradient_reaching_each_step.png)

*Figure 6. The real BPTT gradient $\|\partial L/\partial h_t\|$ at each time step, divided by its value at the last step, for untrained tanh RNNs on the $T=40$ memory task (geometric mean over 10 random networks, three recurrent-weight gains). The signal bit enters at $t=1$. With gain 0.7 the gradient that reaches $t=1$ is $2\times10^{-11}$ of the gradient at the output; with gain 1.0 (the standard scale) it is $1.3\times10^{-6}$; only with gain 1.5 does a usable $2\times10^{-2}$ survive. A learning signal a million times weaker than the one at the output cannot tell the network "$x_1$ matters". Weights that would let the network store the bit receive almost no push toward existing.*

Equivalently, from the sum in Section 3: the terms involving distant time steps vanish, so $\partial L/\partial W$ is dominated by short-range effects, and the network learns only short-range correlations. That is the long-term dependency problem, restated as an optimization problem: **the information path exists, but gradient descent cannot find it.**

---

## 5. Problem 2a: vanishing gradients

Section 3 predicts exponential decay when the per-step factor is below one. Two ingredients determine that factor: the activation derivative $\varphi'$ and the recurrent matrix $W$.

![Activation derivatives](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_activation_derivatives.png)

*Figure 7. Left: the tanh, sigmoid and ReLU activations. Right: their derivatives, which are the diagonal entries of $\mathrm{diag}(\varphi'(a_t))$ in each Jacobian. The sigmoid derivative never exceeds 0.25, so a sigmoid RNN shrinks gradients by at least 4x per step before $W$ even acts. The tanh derivative peaks at 1 but falls off quickly: it is 0.071 at $a=2$ and 0.0099 at $a=3$, so saturated units contribute almost nothing. ReLU's derivative is exactly 1 for active units and 0 for inactive ones, which is the property exploited by the fixes in Section 7.*

Now measure the actual product of Jacobians in a real tanh RNN (100 steps, 64 units), computing $\|\partial h_T/\partial h_k\|_2$ as the spectral norm of the explicit matrix product.

![Measured gradient vs lag](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_gradient_vs_lag_measured.png)

*Figure 8. Measured $\|\partial h_T/\partial h_k\|_2$ against the lag $T-k$ (log scale; dotted line = 1). (a) $W=\rho Q$ with $Q$ a random orthogonal matrix, so every singular value of $W$ equals exactly $\rho$. At $\rho=0.8$ the gradient is $0.10$ at lag 10, $1.1\times10^{-5}$ at lag 50 and $1.6\times10^{-10}$ at lag 99. At $\rho=0.95$ it is $1.7\times10^{-3}$ by lag 99. Even at $\rho=1.0$ it still decays, to $0.081$ at lag 99, because $\varphi'<1$ whenever a unit is not exactly at zero. Only $\rho=1.05$ hovers near 1 over this horizon (1.7 at lag 99), and $\rho=1.2$ grows to 51. (b) Standard Gaussian initialization $W_{ij}\sim\mathcal N(0,g^2/n)$. With gain $g=0.5$ the gradient is $10^{-42}$ at lag 99; at the usual $g=1.0$ it is $4.4\times10^{-15}$. The practical lesson is that a randomly initialized tanh RNN sits on the vanishing side unless its recurrent weights are deliberately large.*

---

## 6. Problem 2b: exploding gradients

If $\gamma\,\sigma_{\max}(W)$ is above one, the same product can grow instead. In Figure 8(b), gain $g=2.5$ gives a gradient of $8.8\times10^{3}$ at lag 50 and $3.8\times10^{7}$ at lag 99. (Growth is eventually limited in tanh networks because large activations saturate and $\varphi'\to0$; Figure 8(a) shows only a $51\times$ rise for $\rho=1.2$ for that reason.)

The consequence is unlike vanishing. Instead of slow learning you get occasional **enormous steps** that throw the parameters far from where they were. Pascanu et al. (2013) describe why with a one-unit RNN: its loss surface has steep "cliffs" next to flat plateaus.

![Cliff landscape](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_cliff_landscape.png)

*Figure 9. Loss (left) and gradient norm (right, $\log_{10}$) of a single-unit RNN $h_t=\tanh(w h_{t-1}+b)$ run for $T=50$ steps from $h_0=0$, with loss $(h_T-0.5)^2$, computed exactly with forward-mode derivatives on a 400 x 400 grid over $w\in[0,4]$, $b\in[-0.4,0.7]$. For $w>1$ the unit has two stable resting states (near $\pm1$ for larger $w$) and the sign of $b$ decides which one it falls into, so the loss changes abruptly across $b=0$. The gradient norm there forms a thin bright ridge: the median gradient norm on this grid is $0.085$, the 99.9th percentile is $11$, and the maximum is $1.4\times10^{3}$, roughly 16,000 times the median. A step sized for the plateau is far too large on the ridge.*

![Cliff trajectories](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10_cliff_trajectories.png)

*Figure 10. Gradient descent with learning rate 0.5 from $(w,b)=(2.5, 0.2)$ on the surface of Figure 9, with and without clipping the gradient norm at 1 (heat map covers only the plotted grid region; the unclipped path leaves it). Left: the unclipped path creeps along the plateau, then reaches the ridge and takes a single step of length $1.49$ (iteration 30, where the gradient norm is $2.98$), landing at roughly $w\approx0.1,\,b\approx1.0$ far outside the region it had been exploring; right: update lengths per iteration show that spike against steps of about $0.014$ at the start. The clipped path never takes a step longer than $0.5$ and settles near $(0.55, 0.31)$ with loss $0.002$. To be fair about the outcome: in this particular case both runs end at low loss (the unclipped one at $(-0.16, 0.63)$, loss $\approx0$); the difference is the wild detour, which in a real network can send weights somewhere they never recover from.*

### Does this happen in real training?

![Gradient spikes in training](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11_gradient_spikes_in_training.png)

*Figure 11. Real unclipped SGD training of a 32-unit tanh RNN on the memory task ($T=30$, recurrent gain 2.5, lr 0.05, 600 steps, 8 seeds). (a) Gradient norm per step for two seeds on a log axis: seed 7 has a median of $0.54$ but a peak of $1885$; seed 0 has a median of $0.02$ and a peak of $62$. (b) Ratio of maximum to median gradient norm per seed, from $51\times$ to $3481\times$. The gradient is not uniformly large; it is mostly modest with rare, giant spikes, which is exactly the shape that makes a single fixed learning rate dangerous. Honest note: in this run 6 of 8 unclipped seeds still learned the task (mean test accuracy 91.3%), so spikes are a hazard, not a guaranteed failure.*

---

## 7. Solutions

### 7.1 Gradient clipping

Clipping targets **exploding** gradients. Two common variants, with threshold $\theta$:

$$\text{clip by norm:}\quad g \leftarrow g\cdot\min\!\Big(1,\ \frac{\theta}{\|g\|}\Big) \qquad\qquad \text{clip by value:}\quad g_i \leftarrow \mathrm{clip}(g_i,\,-\theta,\,\theta).$$

![Clipping geometry](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/12_clipping_geometry.png)

*Figure 12. The two clipping rules applied to the gradient $g=(7, 1.5)$ with threshold 2 (exact arithmetic). Norm clipping scales the whole vector onto the dashed circle and gives $(1.96, 0.42)$, which keeps the gradient's direction. Value clipping truncates each coordinate to the dashed square and gives $(2.00, 1.50)$, which rotates the direction by $24.8^\circ$ relative to the original. Norm clipping is therefore the usual choice, since it only shortens the step and never changes where it points.*

Does it help on real training? A sweep over learning rate and clip threshold (tanh RNN, recurrent gain 2.0, plain SGD, $T=30$, 400 steps, 6 seeds each):

![Clipping sweep](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/13_clipping_sweep.png)

*Figure 13. Mean test accuracy (left, labels show how many of 6 seeds exceeded 90%) and the worst training-loss spike seen in any seed (right) for three learning rates and three clipping settings. The clearest effect is at the largest learning rate (1.0): unclipped training solves 5/6 seeds (mean accuracy 84.7%, worst loss 8.60), clipping at 2 solves 6/6 (mean 99.7%, worst loss 5.11), while clipping at 10 gives 5/6 (86.9%). At lr 0.1, clip 2 is slightly better (5/6 vs 4/6) and at lr 0.3 all three settings are within noise (5/6 each). So clipping helps most when the learning rate is aggressive, and the threshold matters. With only 6 seeds per cell, treat these as indicative rather than conclusive.*

A cautionary result from the earlier 8-seed run of Figure 11 (lr 0.05, gain 2.5): clipping at a threshold of **1.0** *hurt*, solving only 2 of 8 seeds (mean accuracy 66.0%) against 6 of 8 without clipping. In those clipped runs the pre-clip gradient norm had a median of 16 to 47, so a threshold of 1 shrank almost every update by more than 10x, which is consistent with the effective learning rate becoming too small for 600 steps. **Clipping is not free: the threshold should sit near the upper range of typical gradient norms, so it only fires on spikes.**

### 7.2 ReLU activations and better initialization

Clipping does nothing for vanishing gradients, since it only shrinks. For those, attack the per-step factor itself.

**ReLU:** $\varphi'$ is exactly 1 on active units, so the activation no longer contributes a shrinking factor. **Identity initialization** (Le, Jaitly & Hinton, 2015): set $W=I$, $b=0$. Then with ReLU and no input, $h_t = h_{t-1}$ for active units, the Jacobian is $J_t = \mathrm{diag}(\mathbb 1[a_t>0])\,I$, and the product of Jacobians neither shrinks nor grows along active units. **Orthogonal initialization** keeps every singular value of $W$ at 1, which is the $\rho=1$ curve of Figure 8(a).

The video lists these as separate options; the experiment below tests whether each ingredient suffices alone. Task: remember-the-first-bit, $T=40$, $\sigma=1.0$, 32 units, Adam lr 1e-3, 1500 steps, clip 1.0, 5 seeds per configuration.

| Configuration | Test accuracy per seed | Seeds solved |
|---|---|---|
| tanh + Gaussian init | 0.52, 0.49, 0.51, 0.50, 0.52 | 0 / 5 |
| tanh + orthogonal init | 0.51, 0.48, 0.50, 0.51, 0.49 | 0 / 5 |
| ReLU + Gaussian init | 0.51, 0.49, 0.50, 0.49, 0.51 | 0 / 5 |
| **ReLU + identity init** | **1.00, 1.00, 1.00, 1.00, 1.00** | **5 / 5** |

![Initialization comparison](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/14_initialization_comparison.png)

*Figure 14. Training-batch accuracy over time for the four configurations in the table (line = mean over 5 seeds, shaded band = min to max). The three configurations that change only one ingredient stay near chance (dashed line at 0.5; their mean curves never exceed 0.58) for the entire run, while ReLU with identity initialization separates from chance and reaches 100% on every seed. Neither ReLU alone nor an orthogonal $W$ alone was enough here; the combination, in which both the activation derivative and the recurrent matrix are 1 along the memory path, was. Caveat: one task, one length, one set of hyperparameters; orthogonal tanh in particular might do better with a larger learning rate or a gain slightly above 1, which I did not search.*

One more cost: ReLU is unbounded, so an identity-initialized ReLU RNN can have exploding gradients, which is why the winning runs above still used clipping.

### 7.3 Skip connections, and a preview of gating

A **skip connection** adds a shortcut for the hidden state through time:

$$h_t = h_{t-1} + \varphi(W h_{t-1} + U x_t + b) \quad\Longrightarrow\quad \frac{\partial h_t}{\partial h_{t-1}} = I + \mathrm{diag}(\varphi'(a_t))\,W.$$

The identity term $I$ gives gradients a path that does not shrink by construction.

![Skip connection diagram](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/16_skip_connection_diagram.png)

*Figure 15. Plain recurrence (top) versus a recurrence with an identity shortcut (bottom). In the plain version, the Jacobian is $\mathrm{diag}(\varphi')W$ and a long product of those can shrink or grow. With the shortcut, the state is added to the transformed state, so the per-step Jacobian is $I$ plus a correction, and the gradient has a direct highway back through time. The same idea underlies residual networks (He et al., 2016) in depth rather than in time.*

I measured how well gradients flow through five recurrences driven by the same random input sequence (32 units, 59 steps; the three RNN variants share the same $W$ and $U$, and the two LSTMs share their own weights and differ only in forget-gate bias; $\|\partial h_T/\partial \text{state}_k\|_2$ by central finite differences, so it applies equally to the LSTM, whose state is $(h,c)$):

![Skip and gated gradient flow](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/15_skip_and_gated_gradient_flow.png)

*Figure 16. Measured gradient norm versus lag. The vanilla tanh RNN decays to $0.10$ at lag 10, $9.3\times10^{-6}$ at lag 30 and $2.4\times10^{-9}$ at lag 59. The skip-connection RNN does not vanish, but **it grows**: $2.9$ at lag 10, $4.9$ at lag 30 and $4.5\times10^{3}$ at lag 59; scaling the branch by 0.1 tames it only partly ($2.1$, $8.8$, $92$). An LSTM with forget-gate bias 0 still decays, to $1.0\times10^{-4}$ at lag 30 and $5.1\times10^{-10}$ at lag 59, while the same LSTM with forget bias 1 decays far more slowly ($0.067$ at lag 30, $1.2\times10^{-3}$ at lag 59). Two lessons: a skip connection removes the guaranteed shrinkage but does not guarantee a stable norm, so it still needs clipping; and the LSTM's advantage comes from a learned, tunable per-unit gate rather than a fixed shortcut.*

The LSTM keeps a separate **cell state** updated additively, $c_t = f_t \odot c_{t-1} + i_t \odot g_t$, so the direct path from $c_{t-1}$ to $c_t$ has Jacobian $\mathrm{diag}(f_t)$, which the network can learn to hold near 1. That is the topic of the next note.

### 7.4 Summary of the fixes

![Summary table](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/17_solutions_summary_table.png)

*Figure 17. Each technique, which problem it targets, how it works and what this note measured. The table is the whole argument in one place: clipping handles the loud failure (exploding); ReLU with identity or orthogonal initialization and skip connections handle the quiet one (vanishing); none of the fixed schemes is a complete answer on its own, which is the opening that gated architectures use.*

---

## 8. From-scratch implementation (NumPy)

The code below is the complete implementation used for the experiments: a many-to-one vanilla RNN with manual BPTT, finite-difference gradient checking, norm clipping, Adam, and the memory task. It was run exactly as printed.

```python
import numpy as np

# ---------- data: "remember the first bit" ----------
def make_batch(rng, B, T, noise=1.0):
    X = rng.normal(0, noise, (B, T, 1))          # distractor noise
    s = rng.integers(0, 2, B)
    X[:, 0, 0] = 2.0 * s - 1.0                   # the signal bit (+1 / -1) at t = 1
    return X, s.astype(float)                    # label = 1 if x_1 > 0

# ---------- vanilla RNN with manual BPTT ----------
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
ACT = {'tanh': (np.tanh,                    lambda a, h: 1.0 - h**2),
       'relu': (lambda a: np.maximum(a, 0), lambda a, h: (a > 0).astype(float))}

class RNN:
    def __init__(self, n_in, n_h, act='tanh', W=None, seed=0):
        rng = np.random.default_rng(seed)
        self.f, self.fd = ACT[act]
        self.W = W.copy() if W is not None else rng.normal(0, 1/np.sqrt(n_h), (n_h, n_h))
        self.U = rng.normal(0, 1/np.sqrt(n_in), (n_h, n_in))
        self.b = np.zeros(n_h); self.v = rng.normal(0, 1/np.sqrt(n_h), n_h); self.c = np.zeros(())

    def loss_and_grads(self, X, y):
        B, T, _ = X.shape
        H = np.zeros((T+1, B, self.W.shape[0])); A = np.zeros((T, B, self.W.shape[0]))
        for t in range(T):                                    # forward: h_t = phi(W h_{t-1} + U x_t + b)
            A[t] = H[t] @ self.W.T + X[:, t] @ self.U.T + self.b
            H[t+1] = self.f(A[t])
        p = sigmoid(H[T] @ self.v + self.c)
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        loss = -np.mean(y*np.log(pc) + (1-y)*np.log(1-pc))
        dz = (p - y) / B
        g = {'W': np.zeros_like(self.W), 'U': np.zeros_like(self.U), 'b': np.zeros_like(self.b),
             'v': H[T].T @ dz, 'c': np.asarray(dz.sum())}
        dh = dz[:, None] * self.v[None, :]                    # dL/dh_T
        hnorm = np.zeros(T+1); hnorm[T] = np.linalg.norm(dh, axis=1).mean()
        for t in range(T-1, -1, -1):                          # backward through time
            da = dh * self.fd(A[t], H[t+1])                   # multiply by diag(phi'(a_{t+1}))
            g['W'] += da.T @ H[t]; g['U'] += da.T @ X[:, t]; g['b'] += da.sum(0)
            dh = da @ self.W                                  # multiply by W  ->  dL/dh_t
            hnorm[t] = np.linalg.norm(dh, axis=1).mean()
        acc = np.mean((p > 0.5) == (y > 0.5))
        return loss, acc, g, hnorm

# ---------- gradient clipping by global norm ----------
def clip_by_global_norm(g, threshold):
    norm = np.sqrt(sum(np.sum(v**2) for v in g.values()))
    scale = min(1.0, threshold / (norm + 1e-12))
    return {k: v * scale for k, v in g.items()}, norm

# ---------- Adam ----------
class Adam:
    def __init__(self, model, lr=1e-3, b1=0.9, b2=0.999):
        self.m, self.lr, self.b1, self.b2, self.t = model, lr, b1, b2, 0
        self.mm = {k: np.zeros_like(getattr(model, k)) for k in 'WUbvc'}
        self.vv = {k: np.zeros_like(getattr(model, k)) for k in 'WUbvc'}
    def step(self, g):
        self.t += 1
        for k in 'WUbvc':
            self.mm[k] = self.b1*self.mm[k] + (1-self.b1)*g[k]
            self.vv[k] = self.b2*self.vv[k] + (1-self.b2)*g[k]**2
            upd = self.lr * (self.mm[k]/(1-self.b1**self.t)) / (np.sqrt(self.vv[k]/(1-self.b2**self.t)) + 1e-8)
            setattr(self.m, k, getattr(self.m, k) - upd)

def train_and_test(T, act, W, seed=0, steps=1500, lr=1e-3, clip=1.0, n_h=32):
    rng = np.random.default_rng(1000 + seed)
    model = RNN(1, n_h, act, W=W, seed=seed); opt = Adam(model, lr)
    for _ in range(steps):
        X, y = make_batch(rng, 64, T)
        _, _, g, _ = model.loss_and_grads(X, y)
        g, _ = clip_by_global_norm(g, clip)
        opt.step(g)
    Xt, yt = make_batch(np.random.default_rng(99), 1000, T)
    loss, acc, _, _ = model.loss_and_grads(Xt, yt)
    return loss, acc

if __name__ == '__main__':
    # 1) finite-difference gradient check (tanh and relu)
    rng = np.random.default_rng(1)
    for act in ['tanh', 'relu']:
        m = RNN(2, 5, act, seed=3); X = rng.normal(size=(4, 6, 2)); y = rng.integers(0, 2, 4).astype(float)
        _, _, g, _ = m.loss_and_grads(X, y); worst = 0.0
        for k in 'WUbv':
            P = getattr(m, k)
            for idx in np.ndindex(*P.shape):
                old = P[idx]; P[idx] = old + 1e-6; lp = m.loss_and_grads(X, y)[0]
                P[idx] = old - 1e-6; lm = m.loss_and_grads(X, y)[0]; P[idx] = old
                num, ana = (lp - lm)/2e-6, g[k][idx]; worst = max(worst, abs(num-ana)/max(1e-8, abs(num)+abs(ana)))
        print('gradient check (%s): max relative error = %.1e' % (act, worst))

    # 2) how much gradient reaches early time steps?  (T = 40, W gain 1.0, untrained)
    T = 40; W = np.random.default_rng(50).normal(0, 1.0/np.sqrt(32), (32, 32))
    m = RNN(1, 32, 'tanh', W=W, seed=0); X, y = make_batch(np.random.default_rng(0), 256, T)
    hn = m.loss_and_grads(X, y)[3] / m.loss_and_grads(X, y)[3][T]
    print('||dL/dh_t|| relative to t=T:  t=T-10: %.1e   t=1: %.1e' % (hn[T-10], hn[1]))

    # 3) train on the T = 40 memory task: Gaussian-init tanh  vs  identity-init ReLU
    n = 32
    for name, act, W in [('tanh + Gaussian W ', 'tanh', None), ('ReLU + identity W ', 'relu', np.eye(n))]:
        loss, acc = train_and_test(40, act, W)
        print('%s -> test loss %.3f, test accuracy %.3f' % (name, loss, acc))
```

Output of running it as printed:

```text
gradient check (tanh): max relative error = 9.1e-09
gradient check (relu): max relative error = 4.9e-09
||dL/dh_t|| relative to t=T:  t=T-10: 2.6e-02   t=1: 2.9e-07
tanh + Gaussian W  -> test loss 0.694, test accuracy 0.501
ReLU + identity W  -> test loss 0.000, test accuracy 1.000
```

The gradient check confirms the manual BPTT matches finite differences to about $10^{-8}$ relative error for both tanh and ReLU. The third line reproduces Section 4 in miniature, since for an untrained network at the standard scale only $\approx3\times10^{-7}$ of the output gradient reaches $t=1$. The last two lines reproduce Section 7.2: with the same task, optimizer and clipping, only the ReLU + identity combination learns.

---

## 9. Framework equivalents (Keras / PyTorch)

These are the standard API equivalents of the three fixes. Unlike the NumPy code above, **these snippets were not executed while preparing this note** (no deep-learning framework was available in the environment), so check them against your installed version.

**Keras / TensorFlow**

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, activation="relu",
                              recurrent_initializer="identity",   # W = I
                              input_shape=(T, 1)),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)    # clip by norm
# clipvalue=1.0 would clip by value instead
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
```

**PyTorch**

```python
import torch, torch.nn as nn

rnn = nn.RNN(input_size=1, hidden_size=32, nonlinearity="relu", batch_first=True)
with torch.no_grad():
    rnn.weight_hh_l0.copy_(torch.eye(32))                  # W = I
head = nn.Linear(32, 1)
params = list(rnn.parameters()) + list(head.parameters())
opt = torch.optim.Adam(params, lr=1e-3)

# inside the training loop, after loss.backward() and before opt.step():
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)       # clip by global norm
# torch.nn.utils.clip_grad_value_(params, clip_value=1.0)  # clip by value
```

---

## Key Takeaways

- An RNN's gradient from step $T$ back to step $k$ is a product of $T-k$ Jacobians $\mathrm{diag}(\varphi')W$; shrinking or growing exponentially in $T-k$ is the default, not an accident.
- The **long-term dependency problem** is the learnability face of this: on a task needing one bit carried across noise, a vanilla RNN solved short lengths but fell to chance by $T=30$ with strong noise (Figure 5), because the gradient reaching the signal was a million times weaker than at the output (Figure 6).
- **Vanishing** is the quiet failure: a standard Gaussian-initialized tanh RNN sits on the vanishing side ($4.4\times10^{-15}$ at lag 99, Figure 8), and even $\rho=1$ decays because $\varphi'<1$.
- **Exploding** is the loud failure: gradients are usually modest with rare giant spikes (51x to 3481x the median, Figure 11), and steep loss "cliffs" make one fixed step size dangerous.
- **Gradient clipping** cures exploding but not vanishing, and the threshold matters: helpful at an aggressive learning rate (Figure 13), harmful when set far below typical gradient norms.
- **ReLU plus identity initialization** solved the $T=40$ task on 5/5 seeds, while ReLU alone, orthogonal tanh and default tanh each solved 0/5 (Figure 14); single changes were not enough.
- **Skip connections** remove guaranteed shrinkage but not instability (norm grew to $4.5\times10^{3}$, Figure 16); **gating** (LSTM) makes the shortcut learnable, which is why it is the standard answer and the subject of the next note.
- All experiments here are small, synthetic and use few seeds; they illustrate mechanisms rather than benchmark methods.

---

## Further Reading

- Hochreiter, S. (1991). *Untersuchungen zu dynamischen neuronalen Netzen* (Diploma thesis). Technische Universität München. The first analysis of vanishing gradients in recurrent nets.
- Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE Transactions on Neural Networks*, 5(2), 157-166.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. *Neural Computation*, 12(10), 2451-2471.
- Mikolov, T. (2012). *Statistical Language Models Based on Neural Networks* (PhD thesis). Brno University of Technology. Early use of gradient clipping.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *Proceedings of the 30th International Conference on Machine Learning (ICML)*. The source of the norm-bound analysis, the cliff picture and norm clipping.
- Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. *ICLR*. Motivation for orthogonal initialization.
- Le, Q. V., Jaitly, N., & Hinton, G. E. (2015). A simple way to initialize recurrent networks of rectified linear units. *arXiv:1504.00941*.
- Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An empirical exploration of recurrent network architectures. *ICML*. Includes the forget-gate bias of 1 recommendation.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Lin, T., Horne, B. G., Tino, P., & Giles, C. L. (1996). Learning long-term dependencies in NARX recurrent neural networks. *IEEE Transactions on Neural Networks*, 7(6), 1329-1338. Skip connections through time.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 10 (Sections 10.7 and 10.11 on long-term dependencies and optimization). MIT Press.
