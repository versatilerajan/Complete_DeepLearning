# Encoder-Decoder Architecture (Sequence-to-Sequence Learning)

This document explains the encoder-decoder architecture, the idea behind sequence-to-sequence (Seq2Seq) models such as machine translation: one network (the **encoder**) reads an input sequence of any length and squeezes it into a fixed-size **context vector**, and a second network (the **decoder**) unrolls that vector into an output sequence of a *different* length. It follows the lecture's path: why variable-length mappings are hard, the two-part architecture, how LSTMs sit inside both parts, training with teacher forcing, prediction, and the three improvements from Sutskever et al. (2014): embeddings, deep (stacked) LSTMs and reversing the input. Instead of a large dataset, one small English-to-French example runs through the whole note ("you read the blue book at home" becomes "tu lis le livre bleu à la maison"), so every diagram can be traced word by word. Every number and plot comes from code that was actually run (a from-scratch NumPy encoder-decoder with manual backpropagation, checked against finite differences); numbers taken from the original paper are labeled as quoted, and schematic diagrams are labeled as such.

> **Prerequisites:** [rnn-architecture-and-forward-propagation.md](rnn-architecture-and-forward-propagation.md) (unrolling and weight sharing) and [problems-with-rnns.md](problems-with-rnns.md) (vanishing gradients, which is why the cells below are LSTMs). The planned notes `lstm.md` (the cell in more depth) and `attention.md` (the fix for the bottleneck of Section 7) are assumed filenames; rename the links if yours differ.

---

## Table of Contents

1. [Why sequence-to-sequence is hard](#1-why-sequence-to-sequence-is-hard)
2. [High-level architecture](#2-high-level-architecture)
3. [Under the hood: LSTMs in the encoder and decoder](#3-under-the-hood-lstms-in-the-encoder-and-decoder)
4. [Training](#4-training)
   - [4.1 Words to numbers](#41-words-to-numbers)
   - [4.2 Teacher forcing and the loss](#42-teacher-forcing-and-the-loss)
   - [4.3 Does it learn? Does teacher forcing help?](#43-does-it-learn-does-teacher-forcing-help)
5. [Prediction (inference)](#5-prediction-inference)
6. [Improvements from the paper](#6-improvements-from-the-paper)
   - [6.1 Embeddings instead of one-hot vectors](#61-embeddings-instead-of-one-hot-vectors)
   - [6.2 Deep (stacked) LSTMs](#62-deep-stacked-lstms)
   - [6.3 Reversing the input](#63-reversing-the-input)
7. [The fixed-size bottleneck, and what comes next](#7-the-fixed-size-bottleneck-and-what-comes-next)
8. [Research context: Sutskever, Vinyals & Le (2014)](#8-research-context-sutskever-vinyals--le-2014)
9. [From-scratch implementation (NumPy)](#9-from-scratch-implementation-numpy)
10. [Framework equivalents (PyTorch / Keras)](#10-framework-equivalents-pytorch--keras)
11. [Key Takeaways](#key-takeaways)
12. [Further Reading](#further-reading)

---

## 1. Why sequence-to-sequence is hard

In a Seq2Seq task, the input is a sequence $x_1,\dots,x_n$ and the output is another sequence $y_1,\dots,y_m$, and in general $m\neq n$. Machine translation is the classic case, but the same shape appears in summarization, question answering, speech recognition and code generation.

### The running example

Everything in this note uses a deliberately tiny English-to-French grammar, so the whole system can be trained in seconds on a CPU. Four subjects (*I, you, we, they*), four verbs (*eat, read, want, watch*) with correct French conjugation, 18 objects (some with colors, where French reverses the order: *blue car* becomes *voiture bleue*), and an optional ending (*now*, *every day*, *at home*). That gives **336 sentence pairs**, with 4 to 7 English words and 5 to 9 French tokens (including the end marker). 269 pairs are used for training and **67 are held out** and never seen during training.

| English | French |
|---|---|
| i eat the apple | je mange la pomme |
| we want the red car | nous voulons la voiture rouge |
| you read the blue book at home | tu lis le livre bleu à la maison |
| they watch the cat every day | ils regardent le chat tous les jours |

![The seq2seq task](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_seq2seq_task.png)

*Figure 1 (schematic of real sentence pairs from the dataset). Four example pairs with their input and output lengths. The lengths differ both across examples (4 to 7 words in, 4 to 8 words out) and within an example: "at home" is 2 English words but "à la maison" is 3 French words, and "the red car" keeps three words while swapping the order of two of them. A model for this task has to cope with variable input length, variable output length, and a non-trivial relation between them.*

### Why a plain ANN or CNN is not enough

![Why ANN and CNN fail](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_why_ann_cnn_fail.png)

*Figure 2 (schematic). Left: a standard feed-forward network or CNN has a fixed number of input units and a fixed number of output units, so a sentence of 4 words and a sentence of 7 words cannot both be fed in without padding, and the network has no way to decide how many output words to produce. Right: what translation actually requires. Each output word depends on the entire input and on the words already written, which a fixed-size mapping from inputs to outputs does not model. Padding to a maximum length wastes capacity and still cannot say "stop here" for the output.*

An RNN (see [rnn-architecture-and-forward-propagation.md](rnn-architecture-and-forward-propagation.md)) handles variable input length because the same cell is reused at every step. The remaining problem is the output: the encoder-decoder design solves it by using a **second** recurrent network that *generates* the output one word at a time until it decides to stop.

---

## 2. High-level architecture

The model has two parts connected by one vector:

- **Encoder:** reads $x_1,\dots,x_n$ one word at a time. It produces no output of its own; what matters is its final state $v$, the **context vector** (also called the thought vector or sentence embedding).
- **Decoder:** a second recurrent network that starts from $v$, reads a start marker `<sos>`, and produces $y_1$, then (feeding in $y_1$) $y_2$, and so on, until it emits an end marker `<eos>`. The `<eos>` token is how the model signals the output length.

Probabilistically, the model factorizes the probability of a translation as

$$p(y_1,\dots,y_m \mid x_1,\dots,x_n)\;=\;\prod_{t=1}^{m} p\big(y_t \mid v,\; y_1,\dots,y_{t-1}\big),\qquad v=\text{encoder}(x_1,\dots,x_n).$$

Training maximizes this probability for the correct translations; at prediction time the model finds a high-probability $y$.

![Encoder-decoder overview](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_encoder_decoder_overview.png)

*Figure 3 (schematic, real example). The encoder (blue) reads the 7 words of "you read the blue book at home", one per step. Its final state, the red context vector, has a size fixed by the architecture (here $2\times64$ numbers) no matter how long the sentence was. The decoder (green) receives it and writes the 8 French words and `<eos>`. Everything the decoder knows about the English sentence has to pass through this one vector, which is both the elegance and the main weakness of the design (Section 7).*

![Unrolled encoder-decoder](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_unrolled_encoder_decoder.png)

*Figure 4 (schematic, real example). The same model unrolled over time. Bottom row: what enters each LSTM (English words for the encoder; `<sos>` followed by the French words for the decoder). Top row: what the decoder outputs at each step ("tu lis le livre bleu à la maison `<eos>`"). The red arrow is the only connection between the two networks: the encoder's final state $(h,c)$ becomes the decoder's initial state. The encoder emits nothing; the decoder's inputs are the previous words of the target, which is exactly what teacher forcing (Section 4.2) supplies during training.*

---

## 3. Under the hood: LSTMs in the encoder and decoder

Both networks use **LSTM** cells rather than vanilla RNN cells, because translation needs information to survive across many steps, which is exactly what [problems-with-rnns.md](problems-with-rnns.md) showed vanilla RNNs cannot do. An LSTM keeps two vectors: a hidden state $h_t$ and a **cell state** $c_t$ that is updated additively:

$$\begin{aligned}
f_t&=\sigma(W_f[x_t,h_{t-1}]+b_f), &\quad i_t&=\sigma(W_i[x_t,h_{t-1}]+b_i),\\
\tilde c_t&=\tanh(W_g[x_t,h_{t-1}]+b_g), &\quad o_t&=\sigma(W_o[x_t,h_{t-1}]+b_o),\\
c_t&=f_t\odot c_{t-1}+i_t\odot \tilde c_t, &\quad h_t&=o_t\odot\tanh(c_t).
\end{aligned}$$

![LSTM cell](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_lstm_cell.png)

*Figure 5 (diagram of the equations above). The purple horizontal line is the cell state, the long-term memory. The forget gate $f_t$ (red) scales what is kept of $c_{t-1}$, the input gate $i_t$ (green) scales how much of the new candidate $\tilde c_t$ (orange) is written, and the output gate $o_t$ (blue) decides how much of $\tanh(c_t)$ is exposed as $h_t$. Because $c_t$ is built by an addition rather than by repeatedly multiplying through a squashing function, gradients have a short path back through time, which is what makes LSTMs trainable on long sequences.*

**Wiring.** The encoder LSTM starts at $h_0=c_0=0$ and reads the source words. Its final $(h_n,c_n)$ is the context vector. The decoder LSTM is initialized with exactly that pair, then at each step takes the embedding of the previous word, produces $h_t$, and a linear layer plus softmax turns $h_t$ into a probability distribution over the French vocabulary:

$$p(y_t=w\mid\cdot)=\operatorname{softmax}(W_o h_t+b_o)_w=\frac{\exp\big((W_o h_t+b_o)_w\big)}{\sum_{w'}\exp\big((W_o h_t+b_o)_{w'}\big)}.$$

**Parameter count (model of this note: embedding size 16, 64 LSTM units, one layer).** An LSTM layer with input size $d$ and $H$ units has $4\,(H(d+H)+H)$ parameters (four gates, each with a weight matrix over $[x_t,h_{t-1}]$ and a bias). Counted directly from the arrays in the code:

| Block | Shape(s) | Parameters |
|---|---|---|
| source embedding | 27 × 16 | 432 |
| target embedding | 42 × 16 | 672 |
| encoder LSTM | $W$: 256 × 80, $b$: 256 | 20,736 |
| decoder LSTM | $W$: 256 × 80, $b$: 256 | 20,736 |
| output layer | 64 × 42 + 42 | 2,730 |
| **Total** | | **45,306** |

(PyTorch's `nn.LSTM` stores two bias vectors per layer instead of one, so the same model has 512 more parameters there: 45,818, as measured in Section 10.)

---

## 4. Training

### 4.1 Words to numbers

Each word is mapped to an integer id through a vocabulary (27 source words and 42 target tokens in this dataset; the target side also has `<pad>`, `<sos>` and `<eos>`). The id is then turned into a vector, either one-hot or a learned embedding (Section 6.1).

![Tokens, one-hot, embedding](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_tokens_onehot_embedding.png)

*Figure 6 (real vocabulary; the 16 embedding numbers drawn are illustrative placeholders, not learned values). Left: the actual integer ids assigned to the words of the running example. Right: two ways to represent the word "blue". A one-hot vector has 27 positions with a single 1, so it is sparse and says nothing about which words are similar. An embedding is a dense row of 16 numbers looked up from a table $E$ learned together with the rest of the network; multiplying a one-hot vector by $E$ is exactly the same as picking row $E[\text{id}]$. The model of this note has 1,104 embedding parameters in total.*

### 4.2 Teacher forcing and the loss

During training we know the correct French sentence. At step $t$ the decoder needs a previous word as input. There are two choices:

- **Free running:** feed the word the model itself predicted at step $t-1$. Early in training these predictions are mostly wrong, so one early mistake pushes every later step into a context it has never seen.
- **Teacher forcing:** feed the **true** previous word $y_{t-1}$, whatever the model predicted. The inputs are simply the target sentence shifted right by one position, with `<sos>` in front.

The loss is the average cross-entropy over the decoder steps:

$$L=-\frac1m\sum_{t=1}^{m}\log p_\theta\big(y_t\mid y_{<t},\,v\big),$$

and its gradient is backpropagated through the decoder, through the context vector, and on into the encoder, so both networks are trained jointly end to end.

![Teacher forcing diagram](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_teacher_forcing_diagram.png)

*Figure 7 (schematic, real example). Teacher forcing on "tu lis le livre bleu à la maison `<eos>`". Bottom: the decoder inputs, which are the true words shifted right with `<sos>` first. Middle: LSTM and softmax at every step. Top: the true target word at each step, each producing its own loss term $L_t$. Because the inputs come from the ground truth, all steps can be computed without waiting for the model's own predictions, and a wrong prediction at step $t$ cannot damage step $t+1$ during training. The cost is a mismatch with prediction time, where the model must consume its own outputs (Section 5).*

### 4.3 Does it learn? Does teacher forcing help?

**Setup (used in all runs of this section unless stated):** embedding size 16, one LSTM layer of 64 units in each part, Adam with learning rate 3e-3, batch size 32, gradient-norm clipping at 5, 2,500 steps, forget-gate bias initialized to 1. Success is measured on the 67 held-out sentences: **exact match** means the whole decoded French sentence equals the reference word for word (this is a strict measure, much harsher than BLEU). Batches are drawn from sentences of identical lengths, so no padding or masking is needed.

![Baseline training curves](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/08_baseline_training_curves.png)

*Figure 8 (measured, 3 seeds). Left: the per-token loss on the held-out sentences falls from 3.74 at initialization (the uniform guess over 42 tokens would give $\ln 42\approx3.74$) to 0.0113, 0.0207 and 0.0045 for the three seeds. Right: held-out exact-match accuracy is about 0 for the first 500 steps, then rises steeply; at step 2,500 it is 98.5%, 94.0% and 100% for the three seeds. Since these sentences were never seen during training, the model is combining known parts (a subject, a verb form, an object, an ending) in new ways, which is a small version of the generalization that translation needs. It is not evidence about real-world translation quality.*

![Teacher forcing, short sentences](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/12_teacher_forcing_short.png)

*Figure 9 (measured, 3 seeds, band = min to max). Teacher forcing against free running (the decoder is fed its own greedy predictions during training) on the short sentences. The two curves are close. The held-out loss first drops below 0.1 at steps 1300, 1500 and 1300 with teacher forcing and at 1600, 1300 and 1200 without it, so there is no consistent advantage, and free running even ends at 100%, 100% and 97.0% exact match against 98.5%, 94.0% and 100%. For sentences of this length the problem is easy enough that the early-mistake problem does not bite.*

The lecture states that teacher forcing speeds up convergence; the short-sentence result above does not show that, so I tested a harder setting where errors can accumulate. **Longer sentences:** 1 to 3 clauses joined by "and" / "et" (up to 23 English words and 26 French words plus `<eos>`), 8,000 training sentences, 196 held-out sentences whose exact word sequence never appeared in training (they all have 2 or 3 clauses, since every 1-clause combination also occurs in the training sample), 128 LSTM units, 3,000 steps, 3 seeds.

![Teacher forcing, long sentences](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/13_teacher_forcing_long.png)

*Figure 10 (measured, 3 seeds, band = min to max). On long sentences teacher forcing is decisive. Held-out loss (always computed with teacher forcing, so the two curves are comparable) is 0.794 at step 500 with teacher forcing against 2.484 without, 0.602 against 1.713 at step 1,500, and 0.339 against 0.763 at step 3,000. Exact match at step 3,000 is 1.5%, 6.6% and 7.7% with teacher forcing, but only 1.0%, 0.0% and 0.5% without. So teacher forcing speeds convergence when sequences are long and early errors compound, and does not matter when they are short. Both settings are still far from solved after 3,000 steps; Section 6.3 shows what fixes that.*

---

## 5. Prediction (inference)

After training the weights are frozen. There is no ground truth, so the decoder must feed on its own output:

1. Encode the source sentence to get $v$ and initialize the decoder with it.
2. Feed `<sos>`; take the distribution over the French vocabulary; choose the most probable word (**greedy decoding**).
3. Feed that word back in as the next input; repeat.
4. Stop when `<eos>` is produced (or a maximum length is reached).

A common refinement, used in the paper, is **beam search**, which keeps the best few partial translations instead of only one; this note uses greedy decoding only and does not measure beam search.

![Greedy decoding](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/09_inference_greedy_decoding.png)

*Figure 11 (real model output). One held-out sentence, "you read the blue book at home", translated by the trained baseline model (seed 0). For each step the three most probable words are shown with their probabilities; the winner (orange, top) becomes the next input (red, bottom row). The result "tu lis le livre bleu à la maison" matches the reference exactly. The least certain step is "bleu" (0.981, with "vert" at 0.015), which is exactly where the model had to commit to a color adjective; all other steps are at 0.996 or higher, and `<eos>` is predicted with 0.999.*

### What the encoder and the context vector contain

![Hidden-state heatmaps](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/10_hidden_state_heatmaps.png)

*Figure 12 (real activations of the trained model). Left: the encoder's hidden state (the 20 units with the largest activations) after reading each word of the source sentence, one column per word. Right: the decoder's hidden state at each of its steps. The patterns change from column to column as the sentence is read or written, and the encoder's last column is what the decoder starts from. Individual units are not labeled or interpreted here; the figure only shows that the state is a changing, distributed code rather than a stored copy of the words.*

![Context vector PCA](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/11_context_vector_pca.png)

*Figure 13 (real context vectors, projected to 2D with PCA). Each dot is the context vector $(h,c)$ (128 numbers) of one of the 336 sentences, in three colorings. The first two principal components explain 33% and 27% of the variance. In the middle panel the vertical axis visibly separates "i" (top), "you" (middle) and a mixed bottom band of "they" and "we"; the left panel shows the verbs grouped along the horizontal axis (for example "want" on the left, "eat" and "read" on the right). As a quantitative check independent of the 2D picture, a linear classifier trained on the 128 numbers predicts the sentence's verb, subject and ending with 100% cross-validated accuracy (chance 25%) and the object with 93.8% (chance about 6%). Caveat: 269 of these 336 sentences were also used to train the encoder, so this shows what the vector encodes, not how it generalizes. The paper makes the same point at scale by projecting sentence vectors of real translations.*

---

## 6. Improvements from the paper

### 6.1 Embeddings instead of one-hot vectors

A one-hot input has one dimension per vocabulary word. With 160,000 source words (the paper) that is enormous and treats "cat" and "dog" as equally unrelated to each other as to "carburetor". A learned embedding gives every word a short dense vector (the paper used 1,000 dimensions) that can place similar words near each other.

I compared three input representations on the short dataset, everything else identical (3 seeds each):

| Input representation | Parameters | Steps until held-out loss < 0.1 | Held-out exact match at step 2,500 (3 seeds) |
|---|---|---|---|
| embedding, 16 dims (baseline) | 45,306 | 1300, 1500, 1300 | 98.5%, 94.0%, 100% |
| one-hot | 53,674 | 900, 900, 800 | 100%, 100%, 100% |
| embedding, 4 dims | 38,334 | 2000, 2300, 2300 | 100%, 62.7%, 92.5% |

![Embedding experiment](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/14_embedding_vs_onehot.png)

*Figure 14 (measured, 3 seeds, band = min to max). Held-out loss (a), exact match (b) and parameter count (c) for the three representations. An honest reading: on this toy problem **one-hot was the fastest learner** (exact match 35% at step 500 against 4% for the 16-dim embedding, and 100% against 84% at step 1,500) and also had more parameters (53,674 against 45,306). That does not contradict the lecture, because the advantages of embeddings, namely smaller models and shared structure between similar words, only matter when the vocabulary is large. With 27 and 42 words one-hot is cheap and carries no cost, and a too-small 4-dim embedding clearly hurts (one seed reached only 62.7%). The practical rule is that the embedding size must be large enough to hold what the task needs.*

### 6.2 Deep (stacked) LSTMs

Instead of one LSTM layer, several layers are stacked: the hidden states of layer 1 are the inputs of layer 2, and so on. Each layer of the decoder starts from the final $(h,c)$ of the same layer of the encoder, so the context vector is the states of all layers together. The paper used 4 layers of 1,000 cells.

![Deep LSTM](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/15_deep_lstm_diagram.png)

*Figure 15 (schematic, 2 layers shown). Two stacked LSTM layers in the encoder (blue) and in the decoder (green). Red arrows carry each layer's final state across. Only the top decoder layer feeds the softmax. Stacking adds capacity for more abstract representations (lower layers closer to the words, upper layers closer to the meaning), at the cost of more parameters and a longer gradient path.*

![Depth experiment](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/16_depth_experiment.png)

*Figure 16 (measured, 3 seeds, band = min to max). One layer (45,306 parameters) against two layers (111,354 parameters) on the short dataset. Both reach 94% to 100% exact match by step 2,500 (the 2-layer model scores 100% on all three seeds against 98.5%, 94.0% and 100%), but the deeper model is not faster: held-out loss first drops below 0.1 at steps 1200, 1500 and 1700 against 1300, 1500 and 1300, and exact match at step 1,500 is 59% against 84%. So on a task this small, depth adds parameters without helping much, which is unsurprising since one layer already solves it. The paper's finding, that each extra layer cut perplexity by nearly 10%, was for a 12-million-sentence translation task where capacity matters; this experiment cannot confirm or refute it.*

### 6.3 Reversing the input

The paper's most surprising trick: feed the source sentence **backwards** ("home at book blue the read you") while keeping the target in normal order. Nothing about the architecture changes.

The reasoning is about the distance gradients must travel. Define the **lag** between a source word and its translation as the number of LSTM steps between reading the source word and writing its translation.

![Reversal lag diagram](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/17_reversal_lag_diagram.png)

*Figure 17 (exact calculation on the running example, not an experiment). Each arc joins a source word to its French translation and is labeled with its lag. The function word "la" has no direct English counterpart ("at home" becomes "à la maison") and is left out. Top, normal order: every lag is between 6 and 8, so even the first words are far from the first output. Bottom, reversed: "you" is read last, right before "tu" is written (lag 1), "read" and "lis" are 3 apart, but "home" and "maison" are now 14 apart. The **mean** lag is 7.14 in both cases, exactly as the paper notes; what changes is the **minimum** lag, from 6 down to 1. Short lags give backpropagation a short, easy path to start from, so the network can first learn the early words and then build on them.*

To measure whether this matters, I used the long sentences (1 to 3 clauses) from Section 4.3, with teacher forcing, 128 units, 3,000 steps and 3 seeds, once with the source in normal order and once reversed:

![Reversal experiment](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/18_reversed_input_experiment.png)

*Figure 18 (measured, 3 seeds, band = min to max, 196 unseen test sentences). Reversing the source makes learning clearly faster and better. Held-out loss at step 3,000 is 0.136 (seeds: 0.154, 0.113, 0.141) with reversed input against 0.339 (0.394, 0.304, 0.318) with normal input, and held-out exact match is 38.8%, 52.0% and 44.9% (mean 45.2%) against 1.5%, 6.6% and 7.7% (mean 5.3%). The loss gap is already visible at step 1,000 (0.543 against 0.735, means over the 3 seeds). This reproduces the direction of the paper's result (test BLEU 25.9 to 30.6 and perplexity 5.8 to 4.7 on real data) on a controlled toy problem. Caveat: this is a single dataset and architecture, and the reversal is helpful here partly because my translations are almost order-preserving, as is largely true of English-French; for language pairs with very different word order the benefit would be expected to differ.*

---

## 7. The fixed-size bottleneck, and what comes next

Whatever the length of the input, everything passes through one context vector of the same size. The longer the sentence, the more must be squeezed in.

![Accuracy by length](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/19_accuracy_by_length.png)

*Figure 19 (measured, 3 seeds, dots are individual seeds, same runs as Figure 18). Exact-match accuracy on unseen sentences split by the number of clauses. With reversed input the model gets 85.4% (75%, 94%, 87%) of the 2-clause sentences (up to 15 words) right but only 12.5% (9%, 17%, 12%) of the 3-clause ones (up to 23 words); with normal input it gets 11.5% and 0.3%. The sharp fall with length is what a limited-capacity context vector and a harder optimization problem look like, although with these runs alone I cannot separate the two causes, and a larger model or more training steps would shift both numbers.*

The standard remedy, developed right after this paper, is **attention**: instead of one fixed vector, the decoder looks back at all the encoder states at every step and chooses which source words to focus on (see the planned `attention.md`). Reversing the input is a cheap trick that helps optimization; attention removes the bottleneck itself.

---

## 8. Research context: Sutskever, Vinyals & Le (2014)

The paper, *Sequence to Sequence Learning with Neural Networks* (NIPS 2014, Google), introduced this architecture for English-to-French translation on the WMT'14 dataset. Its main points:

- A multilayer LSTM reads the input into a fixed-size vector and another deep LSTM decodes the output from it.
- A deep model (4 layers), word embeddings and reversed source sentences were the key design choices.
- Despite its small vocabulary and no explicit alignment, the LSTM translated on par with a strong phrase-based statistical machine translation (SMT) system, and re-ranking the SMT system's output with it set a new high.

![Paper setup and results](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/20_paper_setup_and_results.png)

*Figure 20 (numbers quoted from the paper and its summaries; none of these were reproduced here). Setup: 12M sentence pairs, 4-layer LSTMs with 1,000 cells and 1,000-dimensional embeddings, 160,000-word source and 80,000-word target vocabularies, a context vector of 8,000 numbers (4 layers x 1,000 cells x $h$ and $c$), 384M parameters per model and gradient-norm clipping at 5. Results: reversing the source improved a single model's test perplexity from 5.8 to 4.7 and BLEU from 25.9 to 30.6; an ensemble of 5 reversed LSTMs with beam search reached BLEU 34.81 (about 34.50 with beam size 2), against 33.30 for the phrase-based baseline; using the LSTM to re-rank the SMT system's 1,000-best lists gave 36.5, close to the best published 37.0. The 34.81 was penalized for out-of-vocabulary words. These figures use real data at a scale this note's toy experiments cannot match; the toy experiments only illustrate the mechanisms.*

---

## 9. From-scratch implementation (NumPy)

The code below is the complete implementation used for the experiments of Sections 4 to 6 (the sentence generator, an LSTM encoder-decoder with any number of layers, one-hot or embedding inputs, teacher forcing or free running, manual backpropagation through time, Adam with gradient clipping, and greedy decoding). It was run exactly as printed.

```python
import numpy as np

# ======================= data: English -> French toy translation =======================

SUBJ = [('i', 'je'), ('you', 'tu'), ('we', 'nous'), ('they', 'ils')]
# French present-tense forms for je / tu / nous / ils
VERBS = {'eat':   ['mange', 'manges', 'mangeons', 'mangent'],
         'read':  ['lis', 'lis', 'lisons', 'lisent'],
         'want':  ['veux', 'veux', 'voulons', 'veulent'],
         'watch': ['regarde', 'regardes', 'regardons', 'regardent']}
OBJ = {'eat':   [('the apple', 'la pomme'), ('the cake', 'le gâteau'), ('the bread', 'le pain')],
       'read':  [('the book', 'le livre'), ('the red book', 'le livre rouge'), ('the blue book', 'le livre bleu'), ('the green book', 'le livre vert')],
       'want':  [('the car', 'la voiture'), ('the red car', 'la voiture rouge'), ('the blue car', 'la voiture bleue'), ('the green car', 'la voiture verte'),
                 ('the house', 'la maison'), ('the red house', 'la maison rouge'), ('the blue house', 'la maison bleue'), ('the green house', 'la maison verte')],
       'watch': [('the movie', 'le film'), ('the cat', 'le chat'), ('the dog', 'le chien'),
                 ('the red car', 'la voiture rouge'), ('the blue car', 'la voiture bleue'), ('the green car', 'la voiture verte')]}
ADV = [('', ''), ('now', 'maintenant'), ('every day', 'tous les jours'), ('at home', 'à la maison')]

def clause_pairs():
    out = []
    for si, (se, sf) in enumerate(SUBJ):
        for v, forms in VERBS.items():
            for oe, of in OBJ[v]:
                for ae, af in ADV:
                    en = ' '.join(x for x in [se, v, oe, ae] if x)
                    fr = ' '.join(x for x in [sf, forms[si], of, af] if x)
                    out.append((en, fr, dict(subj=se, verb=v, obj=oe, adv=ae)))
    return out

def build_vocab(pairs):
    src = sorted({w for en, fr in pairs for w in en.split()}); tgt = sorted({w for en, fr in pairs for w in fr.split()})
    s2i = {'<pad>': 0}; [s2i.setdefault(w, len(s2i)) for w in src]
    t2i = {'<pad>': 0, '<sos>': 1, '<eos>': 2}; [t2i.setdefault(w, len(t2i)) for w in tgt]
    return s2i, t2i

def encode(pairs, s2i, t2i, reverse=False):
    data = []
    for en, fr in pairs:
        s = [s2i[w] for w in en.split()]
        if reverse: s = s[::-1]
        t = [t2i[w] for w in fr.split()] + [t2i['<eos>']]
        data.append((np.array(s), np.array(t)))
    return data

def split(items, frac_test=0.2, seed=0):
    idx = np.random.default_rng(seed).permutation(len(items)); k = int(len(items)*frac_test)
    te = [items[i] for i in idx[:k]]; tr = [items[i] for i in idx[k:]]
    return tr, te

def long_pairs(rng, n, kmin=1, kmax=4, pool=None):
    """concatenate k clauses with 'and' / 'et' (order-preserving translation)"""
    out = []
    for _ in range(n):
        k = rng.integers(kmin, kmax+1); cl = [pool[i] for i in rng.integers(0, len(pool), k)]
        out.append((' and '.join(c[0] for c in cl), ' et '.join(c[1] for c in cl), k))
    return out

# ======================= model: LSTM encoder-decoder with manual BPTT =======================
sigmoid = lambda z: 1.0/(1.0+np.exp(-np.clip(z, -40, 40)))

class Seq2Seq:
    """Encoder-decoder with stacked LSTMs. Encoder final (h,c) of every layer initialises the decoder."""
    def __init__(self, Vs, Vt, E=16, H=64, L=1, onehot=False, seed=0, sos=1, eos=2):
        rng = np.random.default_rng(seed); self.Vs, self.Vt, self.H, self.L, self.onehot = Vs, Vt, H, L, onehot
        self.sos, self.eos = sos, eos
        self.Es = None if onehot else rng.uniform(-0.1, 0.1, (Vs, E)); self.Et = None if onehot else rng.uniform(-0.1, 0.1, (Vt, E))
        self.ins, self.int_ = (Vs, Vt) if onehot else (E, E)
        def mk(nin):
            W = rng.uniform(-0.08, 0.08, (4*H, nin+H)); b = np.zeros(4*H); b[H:2*H] = 1.0   # forget-gate bias 1
            return W, b
        self.eW, self.eb, self.dW, self.db = [], [], [], []
        for l in range(L):
            W, b = mk(self.ins if l == 0 else H); self.eW.append(W); self.eb.append(b)
            W, b = mk(self.int_ if l == 0 else H); self.dW.append(W); self.db.append(b)
        self.Wo = rng.uniform(-0.08, 0.08, (H, Vt)); self.bo = np.zeros(Vt)
    def names(self):
        n = [('eW', l) for l in range(self.L)] + [('eb', l) for l in range(self.L)] + [('dW', l) for l in range(self.L)] + [('db', l) for l in range(self.L)] + [('Wo', None), ('bo', None)]
        if not self.onehot: n = [('Es', None), ('Et', None)] + n
        return n
    def get(self, n): a = getattr(self, n[0]); return a if n[1] is None else a[n[1]]
    def n_params(self): return sum(self.get(n).size for n in self.names())
    # ---- one LSTM layer step
    def _step(self, W, b, x, h, c):
        H = self.H; z = np.concatenate([x, h], 1) @ W.T + b
        i = sigmoid(z[:, :H]); f = sigmoid(z[:, H:2*H]); g = np.tanh(z[:, 2*H:3*H]); o = sigmoid(z[:, 3*H:])
        c2 = f*c + i*g; tc = np.tanh(c2); h2 = o*tc
        return h2, c2, (np.concatenate([x, h], 1), i, f, g, o, c, tc)
    def _bstep(self, W, cache, dh, dc):
        xh, i, f, g, o, c_prev, tc = cache; H = self.H
        do = dh*tc; dc = dc + dh*o*(1-tc**2)
        di = dc*g; dg = dc*i; df = dc*c_prev; dcp = dc*f
        dz = np.concatenate([di*i*(1-i), df*f*(1-f), dg*(1-g**2), do*o*(1-o)], 1)
        dxh = dz @ W; dW = dz.T @ xh; db = dz.sum(0); nx = xh.shape[1]-H
        return dxh[:, :nx], dxh[:, nx:], dcp, dW, db
    def _emb(self, E, tok, V):
        if self.onehot: x = np.zeros((len(tok), V)); x[np.arange(len(tok)), tok] = 1.0; return x
        return E[tok]
    def encode(self, src):                                 # src: (B, Ts)
        B, Ts = src.shape; hs = [np.zeros((B, self.H)) for _ in range(self.L)]; cs = [np.zeros((B, self.H)) for _ in range(self.L)]
        caches = []; top = []
        for t in range(Ts):
            x = self._emb(self.Es, src[:, t], self.Vs); cl = []
            for l in range(self.L):
                hs[l], cs[l], ca = self._step(self.eW[l], self.eb[l], x, hs[l], cs[l]); cl.append(ca); x = hs[l]
            caches.append(cl); top.append([h.copy() for h in hs])
        return hs, cs, caches, top
    def decode_train(self, hs, cs, tgt, teacher_forcing=True):
        B, m = tgt.shape; hs = [h.copy() for h in hs]; cs = [c.copy() for c in cs]
        prev = np.full(B, self.sos); caches = []; logits_all = []; toks = []
        for t in range(m):
            x = self._emb(self.Et, prev, self.Vt); cl = []
            for l in range(self.L):
                hs[l], cs[l], ca = self._step(self.dW[l], self.db[l], x, hs[l], cs[l]); cl.append(ca); x = hs[l]
            lg = hs[-1] @ self.Wo + self.bo; caches.append((cl, prev, hs[-1].copy())); logits_all.append(lg)
            prev = tgt[:, t] if teacher_forcing else lg.argmax(1)
        return np.stack(logits_all), caches
    def loss_grads(self, src, tgt, teacher_forcing=True):
        B, m = tgt.shape
        hs, cs, ecaches, _ = self.encode(src)
        logits, dcaches = self.decode_train(hs, cs, tgt, teacher_forcing)
        lg = logits - logits.max(2, keepdims=True); p = np.exp(lg); p /= p.sum(2, keepdims=True)
        tg = tgt.T                                         # (m, B)
        loss = -np.mean(np.log(p[np.arange(m)[:, None], np.arange(B)[None, :], tg] + 1e-12))
        dl = p.copy(); dl[np.arange(m)[:, None], np.arange(B)[None, :], tg] -= 1; dl /= (B*m)
        g = {k: np.zeros_like(self.get(k)) for k in self.names()}
        dh = [np.zeros((B, self.H)) for _ in range(self.L)]; dc = [np.zeros((B, self.H)) for _ in range(self.L)]
        for t in range(m-1, -1, -1):
            cl, prev, htop = dcaches[t]; g[('Wo', None)] += htop.T @ dl[t]; g[('bo', None)] += dl[t].sum(0)
            dh[-1] = dh[-1] + dl[t] @ self.Wo.T
            for l in range(self.L-1, -1, -1):
                dx, dhp, dcp, dW, db = self._bstep(self.dW[l], cl[l], dh[l], dc[l])
                g[('dW', l)] += dW; g[('db', l)] += db; dh[l] = dhp; dc[l] = dcp
                if l > 0: dh[l-1] = dh[l-1] + dx
                elif not self.onehot: np.add.at(g[('Et', None)], prev, dx)
        dh_e, dc_e = dh, dc                                # gradient arriving at encoder final states
        dh = [d.copy() for d in dh_e]; dc = [d.copy() for d in dc_e]
        for t in range(src.shape[1]-1, -1, -1):
            cl = ecaches[t]
            for l in range(self.L-1, -1, -1):
                dx, dhp, dcp, dW, db = self._bstep(self.eW[l], cl[l], dh[l], dc[l])
                g[('eW', l)] += dW; g[('eb', l)] += db; dh[l] = dhp; dc[l] = dcp
                if l > 0: dh[l-1] = dh[l-1] + dx
                elif not self.onehot: np.add.at(g[('Es', None)], src[:, t], dx)
        return loss, g
    def greedy(self, src, maxlen=30):
        hs, cs, _, _ = self.encode(src); B = src.shape[0]; prev = np.full(B, self.sos); out = []; probs = []
        for t in range(maxlen):
            x = self._emb(self.Et, prev, self.Vt)
            for l in range(self.L): hs[l], cs[l], _ = self._step(self.dW[l], self.db[l], x, hs[l], cs[l]); x = hs[l]
            lg = hs[-1] @ self.Wo + self.bo; e = np.exp(lg-lg.max(1, keepdims=True)); p = e/e.sum(1, keepdims=True)
            prev = p.argmax(1); out.append(prev); probs.append(p)
        return np.stack(out, 1), np.stack(probs, 1)

def global_norm(g): return np.sqrt(sum((v**2).sum() for v in g.values()))
class Adam:
    def __init__(self, m, lr=2e-3):
        self.m, self.lr, self.t = m, lr, 0; self.mm = {k: np.zeros_like(m.get(k)) for k in m.names()}; self.vv = {k: np.zeros_like(m.get(k)) for k in m.names()}
    def step(self, g, clip=5.0):
        n = global_norm(g); s = min(1.0, clip/(n+1e-12)); self.t += 1
        for k in self.m.names():
            gk = g[k]*s; self.mm[k] = 0.9*self.mm[k]+0.1*gk; self.vv[k] = 0.999*self.vv[k]+0.001*gk**2
            self.m.get(k).__isub__(self.lr*(self.mm[k]/(1-0.9**self.t))/(np.sqrt(self.vv[k]/(1-0.999**self.t))+1e-8))
        return n

# ======================= training / evaluation helpers =======================

def buckets(data):
    b = {}
    for s, t in data: b.setdefault((len(s), len(t)), []).append((s, t))
    return {k: (np.stack([x[0] for x in v]), np.stack([x[1] for x in v])) for k, v in b.items()}

def tf_loss(model, bk):
    tot = 0; n = 0
    for (s, t) in bk.values():
        l, _ = model.loss_grads(s, t, True); tot += l*t.size; n += t.size
    return tot/n

def exact_match(model, bk, eos=2):
    ok = 0; n = 0
    for (s, t) in bk.values():
        out, _ = model.greedy(s, maxlen=t.shape[1]+3)
        for o, ref in zip(out, t):
            e = np.where(o == eos)[0]; pred = o[:e[0]+1] if len(e) else o
            ok += int(len(pred) == len(ref) and (pred == ref).all()); n += 1
    return ok/n

def run(train_data, test_data, Vs, Vt, steps=1500, bs=32, lr=3e-3, seed=0, tf=True, eval_every=100, **kw):
    model = Seq2Seq(Vs, Vt, seed=seed, **kw); opt = Adam(model, lr)
    trb, teb = buckets(train_data), buckets(test_data); keys = list(trb); rng = np.random.default_rng(seed+77)
    sizes = np.array([len(trb[k][0]) for k in keys], float); pk = sizes/sizes.sum()
    hist = []; tl = []
    for it in range(steps+1):
        if it % eval_every == 0:
            hist.append((it, tf_loss(model, teb), exact_match(model, teb), float(np.mean(tl[-eval_every:])) if tl else np.nan))
        k = keys[rng.choice(len(keys), p=pk)]; S, T = trb[k]; ix = rng.integers(0, len(S), min(bs, len(S)))
        loss, g = model.loss_grads(S[ix], T[ix], tf); opt.step(g); tl.append(loss)
    return model, hist, trb, teb

# ======================= run =======================
if __name__ == '__main__':
    # 1) gradient check on the largest-magnitude gradient entries (finite differences)
    rng = np.random.default_rng(0)
    for onehot, L in [(False, 1), (False, 2), (True, 2)]:
        m = Seq2Seq(7, 8, E=5, H=6, L=L, onehot=onehot, seed=1)
        for k in m.names(): a = m.get(k); a += rng.normal(0, 0.3, a.shape)
        src = rng.integers(1, 7, (3, 4)); tgt = rng.integers(3, 8, (3, 5)); _, g = m.loss_grads(src, tgt, True); worst = 0
        for k in m.names():
            P = m.get(k)
            for f in np.argsort(-np.abs(g[k]).ravel())[:10]:
                ix = np.unravel_index(f, P.shape); old = P[ix]; e = 1e-5
                P[ix] = old + e; lp = m.loss_grads(src, tgt)[0]; P[ix] = old - e; lm = m.loss_grads(src, tgt)[0]; P[ix] = old
                num = (lp - lm) / (2*e); worst = max(worst, abs(num - g[k][ix]) / (abs(num) + abs(g[k][ix]) + 1e-12))
        print('gradient check (onehot=%s, layers=%d): max relative error %.0e' % (onehot, L, worst))

    # 2) data: 336 sentence pairs, 80% train / 20% unseen test
    cp = clause_pairs(); pairs = [(a, b) for a, b, _ in cp]; s2i, t2i = build_vocab(pairs); i2t = {v: k for k, v in t2i.items()}
    tr, te = split(pairs, 0.2, 0); trd, ted = encode(tr, s2i, t2i), encode(te, s2i, t2i)
    print('pairs: %d train, %d test | source vocab %d, target vocab %d' % (len(tr), len(te), len(s2i), len(t2i)))

    # 3) train the baseline encoder-decoder (embedding 16, one 64-unit LSTM layer, teacher forcing)
    model, hist, _, teb = run(trd, ted, len(s2i), len(t2i), steps=2500, H=64, E=16, L=1, seed=0, lr=3e-3, tf=True, eval_every=500)
    print('parameters: %d' % model.n_params())
    for step, test_loss, em, train_loss in hist: print('step %4d | held-out loss %.4f | held-out exact match %.3f' % (step, test_loss, em))

    # 4) translate unseen sentences
    for en, fr in te[:6]:
        out, _ = model.greedy(np.array([[s2i[w] for w in en.split()]]), 12); w = [i2t[i] for i in out[0]]
        pred = ' '.join(w[:w.index('<eos>')]); print('%-34s -> %-36s %s' % (en, pred, 'OK' if pred == fr else 'WRONG (ref: %s)' % fr))
```

Output of running it as printed:

```text
gradient check (onehot=False, layers=1): max relative error 1e-08
gradient check (onehot=False, layers=2): max relative error 7e-09
gradient check (onehot=True, layers=2): max relative error 1e-08
pairs: 269 train, 67 test | source vocab 27, target vocab 42
parameters: 45306
step    0 | held-out loss 3.7375 | held-out exact match 0.000
step  500 | held-out loss 0.5024 | held-out exact match 0.030
step 1000 | held-out loss 0.1485 | held-out exact match 0.418
step 1500 | held-out loss 0.0777 | held-out exact match 0.806
step 2000 | held-out loss 0.0160 | held-out exact match 0.985
step 2500 | held-out loss 0.0113 | held-out exact match 0.985
they read the red book now         -> ils lisent le livre rouge maintenant OK
i want the car at home             -> je veux la voiture à la maison       OK
they watch the blue car now        -> ils regardent la voiture bleue maintenant OK
they watch the red car every day   -> ils regardent la voiture rouge tous les jours OK
we read the green book at home     -> nous lisons le livre vert à la maison OK
they want the green car at home    -> ils veulent la voiture verte à la maison OK
```

The three gradient checks cover 1 layer with embeddings, 2 layers with embeddings and 2 layers with one-hot inputs, comparing the manual BPTT to finite differences on the ten largest-magnitude gradient entries of every parameter tensor (relative error about $10^{-8}$). The training lines are the seed-0 baseline of Figure 8 (held-out loss 0.0113 and exact match 0.985 at step 2,500, identical to the numbers there). The six translations show unseen sentences decoded correctly. To reproduce the other sections, change `tf=` (teacher forcing), `onehot=`, `E=`, `L=` or reverse the source with `encode(..., reverse=True)`.

---

## 10. Framework equivalents (PyTorch / Keras)

**PyTorch.** This version was executed on the same dataset (padded batches with `pack_padded_sequence`, teacher forcing, gradient clipping at 5, Adam 3e-3, batch 32): it has 45,818 parameters (512 more than the NumPy model, because PyTorch keeps two bias vectors per LSTM layer) and reached 100% held-out exact match at step 500, 1,000 and 1,500, a little faster than the NumPy runs, which I attribute to different initialization and batching rather than anything about the architecture.

```python
import torch, torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class Seq2Seq(nn.Module):
    def __init__(self, Vs, Vt, E=16, H=64, layers=1):
        super().__init__()
        self.src_emb = nn.Embedding(Vs, E, padding_idx=0); self.tgt_emb = nn.Embedding(Vt, E, padding_idx=0)
        self.encoder = nn.LSTM(E, H, layers, batch_first=True); self.decoder = nn.LSTM(E, H, layers, batch_first=True)
        self.out = nn.Linear(H, Vt)
    def forward(self, src, src_len, tgt_in):                       # teacher forcing: tgt_in = <sos> + true words
        packed = pack_padded_sequence(self.src_emb(src), src_len.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed)                             # state = (h_n, c_n) = the context vector
        dec_out, _ = self.decoder(self.tgt_emb(tgt_in), state)      # decoder starts from the encoder's final state
        return self.out(dec_out)                                    # (B, m, V_target) logits
    @torch.no_grad()
    def translate(self, src, src_len, sos=1, eos=2, max_len=12):
        packed = pack_padded_sequence(self.src_emb(src), src_len.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed); tok = torch.full((src.size(0), 1), sos); out = []
        for _ in range(max_len):                                    # feed the model's own previous output
            o, state = self.decoder(self.tgt_emb(tok), state); tok = self.out(o).argmax(-1); out.append(tok)
        return torch.cat(out, 1)

# training step: loss = nn.CrossEntropyLoss(ignore_index=0)(logits.reshape(-1, Vt), targets.reshape(-1))
#                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
# reversing the input (Section 6.3): flip each source sentence before building `src`, e.g. words[::-1]
# deep LSTM (Section 6.2): Seq2Seq(Vs, Vt, layers=2)  (nn.LSTM stacks layers and returns (h_n, c_n) for all of them)
```

**Keras / TensorFlow.** *This snippet was not executed* (TensorFlow was not available in the environment), so check it against your installed version. It is the standard functional-API formulation, with the encoder's final states passed to the decoder.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

E, H = 16, 64
enc_in = layers.Input(shape=(None,), dtype="int32")
enc_emb = layers.Embedding(Vs, E, mask_zero=True)(enc_in)
_, h, c = layers.LSTM(H, return_state=True)(enc_emb)               # context vector = [h, c]

dec_in = layers.Input(shape=(None,), dtype="int32")                # <sos> + true target words (teacher forcing)
dec_emb = layers.Embedding(Vt, E, mask_zero=True)(dec_in)
dec_seq = layers.LSTM(H, return_sequences=True)(dec_emb, initial_state=[h, c])
probs = layers.Dense(Vt, activation="softmax")(dec_seq)

model = Model([enc_in, dec_in], probs)
model.compile(optimizer=tf.keras.optimizers.Adam(3e-3, clipnorm=5.0),
              loss="sparse_categorical_crossentropy")
# model.fit([src_ids, tgt_in_ids], tgt_out_ids, batch_size=32, epochs=...)
# At prediction time the decoder must be run one step at a time, feeding back its own output (Section 5).
```

---

## Key Takeaways

- Seq2Seq tasks have variable-length inputs and outputs of different lengths, which fixed-size ANNs and CNNs cannot handle. The encoder-decoder design reads the input into a fixed-size context vector $v$ and generates the output one word at a time from $v$, stopping at `<eos>`.
- Both parts are LSTMs, and the decoder is initialized with the encoder's final $(h,c)$. In the model here (embedding 16, one 64-unit layer) that is 45,306 parameters, and it translated 98.5%, 94% and 100% of unseen sentences exactly on three seeds (Figure 8).
- Training uses **teacher forcing** (the true previous word is the decoder input) with a per-token cross-entropy loss. It made no difference on short sentences (Figure 9) but was decisive on long ones: held-out loss 0.339 against 0.763, exact match about 5% against 0.5% (Figure 10).
- At prediction time weights are frozen and the decoder feeds on its own greedy outputs until `<eos>` (Figure 11). The context vector provably carries the sentence's content: a linear probe recovers the verb, subject and ending with 100% accuracy (Figure 13).
- **Embeddings** replace huge one-hot inputs with short dense vectors; on this tiny vocabulary one-hot was actually faster (Figure 14), so the benefit shows at large vocabularies, and too small an embedding (4 dims) hurts.
- **Deep LSTMs** add capacity; here two layers did not converge faster than one (Figure 16), since the task is small, so the paper's benefit at scale is not confirmed or refuted by this experiment.
- **Reversing the source** leaves the mean word-to-word lag unchanged (7.14 in the example) but cuts the minimum lag from 6 to 1 (Figure 17). On long sentences it raised exact match from 5.3% to 45.2% and cut held-out loss from 0.339 to 0.136 (Figure 18).
- Accuracy still falls steeply with sentence length (Figure 19), which is the fixed-size bottleneck; attention is the standard fix.
- The paper (Sutskever, Vinyals & Le, 2014) reached BLEU 34.81 with an ensemble of 5 reversed deep LSTMs against 33.30 for phrase-based SMT (Figure 20). All experiments in this note are small, synthetic and use 3 seeds, and illustrate mechanisms rather than benchmark methods.

---

## Further Reading

- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. *Advances in Neural Information Processing Systems 27 (NIPS)*, 3104-3112. arXiv:1409.3215.
- Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*. arXiv:1406.1078.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR*. arXiv:1409.0473. The attention mechanism that removes the fixed-size bottleneck.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
- Williams, R. J., & Zipser, D. (1989). A learning algorithm for continually running fully recurrent neural networks. *Neural Computation*, 1(2), 270-280. The origin of teacher forcing.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *NIPS*. Word embeddings.
- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. *Journal of Machine Learning Research*, 3, 1137-1155.
- Luong, M.-T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. *EMNLP*. arXiv:1508.04025.
- Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: a method for automatic evaluation of machine translation. *ACL*. The metric quoted for the paper's results.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 10 (Section 10.4 on encoder-decoder sequence-to-sequence architectures). MIT Press.
