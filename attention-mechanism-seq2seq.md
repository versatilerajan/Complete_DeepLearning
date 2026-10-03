# The Attention Mechanism in Seq2Seq (Encoder-Decoder) Networks

This note explains why plain encoder-decoder (seq2seq) models break down on long sentences, and how the **attention mechanism** fixes it by letting the decoder dynamically focus on different parts of the input at every output step, instead of relying on one static summary. It builds directly on the variable-length many-to-many / encoder-decoder architecture introduced in [rnn-architectures-overview.md](rnn-architectures-overview.md), and on the LSTM/RNN cell mechanics from the earlier notes — attention is best understood as an upgrade bolted onto that same encoder-decoder skeleton.

## Table of Contents
1. [The Why: the Fixed-Vector Bottleneck](#1-the-why-the-fixed-vector-bottleneck)
2. [The Solution: Attend to Relevant Parts](#2-the-solution-attend-to-relevant-parts)
3. [What Changes: a Third Input to the Decoder](#3-what-changes-a-third-input-to-the-decoder)
4. [Computing Alignment Scores](#4-computing-alignment-scores)
5. [From Scores to Weights to Context](#5-from-scores-to-weights-to-context)
6. [Visualizing Attention Weights](#6-visualizing-attention-weights)
7. [Benefit: Stable Quality on Long Sentences](#7-benefit-stable-quality-on-long-sentences)
8. [Without Attention vs With Attention](#8-without-attention-vs-with-attention)
9. [Key Takeaways](#9-key-takeaways)
10. [Further Reading](#10-further-reading)

---

## 1. The Why: the Fixed-Vector Bottleneck

In the plain encoder-decoder architecture from [rnn-architectures-overview.md](rnn-architectures-overview.md), the encoder reads the entire input sequence and compresses it into a single, fixed-size **context vector** — nothing more than the encoder's last hidden state. The decoder then generates the whole output sequence from that one vector.

![The bottleneck problem](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_bottleneck_problem.png)

*An 8-word sentence has to be squeezed into one vector of fixed size, no matter how long the sentence is. Two problems follow directly from this. First, once sentences get long (the lecture cites roughly 25 words as the point where quality starts to suffer), a fixed-size vector simply doesn't have room to retain everything. Second, every decoder step receives the exact same vector — even though generating the 1st output word and the 8th output word usually depend on different parts of the input.*

---

## 2. The Solution: Attend to Relevant Parts

The attention mechanism's core idea, as the lecture frames it, mimics how a person translates a sentence: you don't memorize the whole sentence perfectly and then write the translation from memory — you glance back at different source words as you produce each target word. Attention gives the decoder that same ability: instead of one static summary, it computes a **new, custom-weighted context vector at every decoder step**, built from all of the encoder's hidden states, weighted by relevance to what's being generated right now.

---

## 3. What Changes: a Third Input to the Decoder

In a plain decoder, each step takes two inputs: the previous output word and the previous decoder hidden state. With attention, a **third input** is added: the context vector `c_t`, which is now a weighted sum over *all* encoder hidden states, recomputed fresh at every step `t`.

![Attention weighted context](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_attention_weighted_context.png)

*The encoder (top) keeps every hidden state `h_1 .. h_n` instead of discarding all but the last one. For decoder step `t`, each encoder state is scaled by its own attention weight `alpha_(t,j)` (thicker red line = larger weight) and summed into `c_t`. In this illustration the decoder mostly attends to `h_2` (weight 0.6) — a different step would likely produce a different weighting.*

---

## 4. Computing Alignment Scores

Before the weights `alpha_(t,j)` can be computed, the model needs a raw **alignment score** `e_(t,j)` for every encoder position `j`, measuring how relevant encoder state `h_j` is to the current decoder step `t`. This score comes from a small, separately learned feed-forward network that takes the previous decoder state and an encoder hidden state as input.

![Alignment score network](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_alignment_score_network.png)

*The same small network is reused for every encoder position: it's fed `s_(t-1)` (the previous decoder state) together with `h_j`, and outputs one number, `e_(t,j)`. Run once per encoder position, this produces a full set of scores `e_(t,1) .. e_(t,n)` for the current decoder step.*

---

## 5. From Scores to Weights to Context

Raw alignment scores can be any real number, so they're passed through a **softmax** to turn them into proper attention weights (positive, summing to 1), which are then used to build the context vector.

![Softmax and context](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_softmax_and_context.png)

*Top: the raw scores for all `n` encoder positions. Middle: softmax converts them into weights `alpha_(t,j)` that behave like a probability distribution over source words. Bottom: the context vector `c_t` is the weighted average of all encoder hidden states using those weights. Because softmax normalizes the scores, `c_t` is always a well-behaved blend of the encoder states, regardless of how large or small the raw scores were.*

---

## 6. Visualizing Attention Weights

Because the attention weights `alpha_(t,j)` are just numbers between 0 and 1, they can be laid out in a grid — one row per output word, one column per input word — and visualized as a heatmap. The lecture uses exactly this kind of grid to show which source words the model is focusing on for each generated word.

![Attention heatmap](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_attention_heatmap_illustrative.png)

*This grid is **illustrative**: the numbers are hand-built to show the pattern a trained English-to-French translation model typically produces, not measured output from an actual model. Darker cells are larger attention weights. Notice the near-diagonal pattern: "Le" attends mostly to "The", "chat" to "cat", and so on, with "sur" correctly attending to "on" even though French word order shifts things slightly ("s'est assis" for "sat" spans two words). This kind of grid is exactly what lets you sanity-check whether a trained model's attention aligns with what a human translator would expect.*

---

## 7. Benefit: Stable Quality on Long Sentences

The lecture shows BLEU-score graphs (a standard machine-translation quality metric) comparing models with and without attention as sentence length increases.

![BLEU vs length](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_bleu_vs_length_illustrative.png)

*This chart is **illustrative only** — drawn to show the general shape of the result the lecture describes, not digitized from the lecture's actual graph or any specific paper. The described pattern is that a model without attention tends to score well on short sentences but degrades as length increases past roughly 25 words (the fixed context vector runs out of capacity), while a model with attention holds up much better because it isn't relying on a single vector to hold the whole sentence.*

---

## 8. Without Attention vs With Attention

![Comparison table](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_comparison_table.png)

*A side-by-side summary. The "Interpretability" row is worth calling out: a plain seq2seq model's single context vector is an opaque blend with no easy way to inspect what it "focused on," whereas the attention-weight grid from Section 6 gives a direct, human-readable window into the model's behavior.*

---

## 9. Key Takeaways

- A plain seq2seq encoder-decoder compresses the entire input into one fixed-size context vector, which becomes a bottleneck for sentences longer than roughly 25 words, and which is reused unchanged at every decoder step.
- Attention replaces that single static vector with a fresh, custom-weighted context vector `c_t` computed at every decoder step from *all* encoder hidden states.
- Attention weights `alpha_(t,j)` come from raw alignment scores `e_(t,j)`, produced by a small learnable network taking the previous decoder state and an encoder hidden state as input, then normalized with softmax.
- The context vector is a weighted sum: `c_t = sum_j alpha_(t,j) * h_j`.
- Attention weights can be visualized as a grid (output words x input words), which is a direct, human-readable way to check what the model is attending to for each generated word.
- Models with attention keep translation quality much more stable as sentence length grows, compared to models relying on a single fixed context vector.

---

## 10. Further Reading

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR. The original attention mechanism for seq2seq models, including the alignment-score network and attention-weight visualizations described here.
- Luong, M.-T., Pham, H., & Manning, C. D. (2015). *Effective Approaches to Attention-based Neural Machine Translation*. EMNLP. Introduces simplified ("global" and "local") attention score functions as alternatives to Bahdanau's.
- Papineni, K., et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL. Defines the BLEU score referenced in Section 7.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks*. NeurIPS. The original fixed-context-vector seq2seq model that attention was designed to improve on.
