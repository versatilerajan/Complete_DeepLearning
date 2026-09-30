# RNN Architectures: One-to-One, One-to-Many, Many-to-One, Many-to-Many

Not every sequence problem has the same shape: sometimes a whole sequence collapses to one prediction, sometimes one input expands into a sequence, and sometimes both the input and the output are sequences. This note catalogues the five input/output patterns an RNN can be arranged into, with an example task for each. It follows Nitish's lecture, whose stated purpose is to lay this out *before* deriving backpropagation for each case — the shape of the architecture is exactly what changes in the BPTT derivation in [backpropagation-through-time.md](backpropagation-through-time.md) (a plain many-to-one setup there), so this note is a map of which case applies to which real task.

## Table of Contents
1. [Why Architecture Shape Matters](#1-why-architecture-shape-matters)
2. [Many-to-One](#2-many-to-one)
3. [One-to-Many](#3-one-to-many)
4. [Many-to-Many: Same Length](#4-many-to-many-same-length)
5. [Many-to-Many: Variable Length (Encoder-Decoder)](#5-many-to-many-variable-length-encoder-decoder)
6. [One-to-One](#6-one-to-one)
7. [Summary Table](#7-summary-table)
8. [Key Takeaways](#8-key-takeaways)
9. [Further Reading](#9-further-reading)

---

## 1. Why Architecture Shape Matters

An RNN cell can be wired up in different ways depending on **how many time steps receive a real input** and **how many time steps produce a real output**. The four earlier notes in this series used a many-to-one shape (a sequence of words in, a single sentiment prediction out) without naming it as such — this note names the full set of options.

![Five architectures overview](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/01_overview_five_architectures.png)

*All five shapes side by side. In each mini-diagram, a gray upward arrow marks a step that receives a real input, a blue upward arrow marks a step that produces a real output, and the red arrows are the hidden state moving forward through time, exactly as in the earlier notes' unrolled diagrams. Notice one-to-one has no red arrow at all — no recurrence is needed.*

---

## 2. Many-to-One

**Shape:** a sequence goes in, one value comes out. Only the final hidden state is used for prediction.

![Many-to-one](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/02_many_to_one_sentiment.png)

*Every word of "I loved this movie" updates the hidden state, but only the hidden state after the last word ("movie") is passed to a Dense + sigmoid layer to produce a single probability. This is exactly the shape used for the IMDB sentiment model and the BPTT derivation in the earlier notes: `dL/dW_o` needed no time-summation precisely because the output only happens once, at the end.*

**Typical uses:** sentiment analysis, star-rating prediction, document/topic classification, spam detection — any task where a whole sequence needs to collapse into one judgment.

---

## 3. One-to-Many

**Shape:** one non-sequential input goes in; a sequence comes out, generated one step at a time.

![One-to-many](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/03_one_to_many_captioning.png)

*A fixed-size image-feature vector (from a CNN) seeds the RNN, and the network then generates a caption word by word: `<start>`, `a`, `dog`, `running`, ... In practice, each generated word is often fed back in as the next step's input (not drawn here for simplicity), so the model conditions each new word on everything generated so far, not just the original image.*

**Typical uses:** image captioning, music generation from a seed note or style vector, and any task where a single starting point needs to be expanded into a sequence.

---

## 4. Many-to-Many: Same Length

**Shape:** a sequence goes in, a sequence of the *same length* comes out — one prediction per input step, produced immediately at that step.

![Many-to-many same length](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/04_many_to_many_same_length.png)

*Each word of "The dog runs fast" gets its own tag at the same time step: DET, NOUN, VERB, ADV. Unlike many-to-one, every cell here has both an input arrow and an output arrow — there's no waiting until the sequence ends.*

**Typical uses:** part-of-speech (POS) tagging, named entity recognition, per-frame video labeling — any task that assigns one label per input element, aligned in time.

---

## 5. Many-to-Many: Variable Length (Encoder-Decoder)

**Shape:** a sequence goes in, a sequence comes out, but the two can have **different lengths**. This needs a different structure than the same-length case: an **encoder** RNN reads the whole input into a summary (context vector), and a separate **decoder** RNN generates the output sequence from that summary.

![Encoder-decoder](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/05_many_to_many_encoder_decoder.png)

*The encoder (blue) reads "Hello how are" — 3 words — one step at a time, with no output at each step; only its final hidden state becomes the context vector. The decoder (purple) then starts from that context vector and generates "Comment allez vous ?" — 4 words — one at a time. Because the encoder finishes reading before the decoder starts writing, the input and output lengths are decoupled.*

**Typical uses:** machine translation (the input and output are rarely the same number of words), text summarization, speech-to-text — any sequence-to-sequence task where lengths genuinely differ. This encoder-decoder idea is also called a **sequence-to-sequence (seq2seq)** model.

---

## 6. One-to-One

**Shape:** one input, one output, no time steps at all.

![One-to-one](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/06_one_to_one.png)

*A single input produces a single output in one shot. There's no hidden state carried across steps because there's only one step — which is exactly what a standard ANN or CNN already does. This case is included in the taxonomy for completeness, but it is not really a recurrent network at all: nothing about it needs the machinery from the earlier notes in this series (unrolling, BPTT, shared weights across time).*

**Typical uses:** ordinary image classification, tabular-data prediction — any task with no sequence on either side.

---

## 7. Summary Table

![Summary table](https://raw.githubusercontent.com/versatilerajan/deepcontent/main/images/07_comparison_table.png)

*All five architectures side by side by input shape, output shape, and a representative task. Reading down the "Output" column shows the real dividing line in this taxonomy: single output (rows 1-2) needs no time-alignment machinery, sequence output with a known, matching length (row 4) can produce results as it goes, and sequence output with an unknown or different length (rows 3, 5) generally needs to either seed a generator (one-to-many) or separate reading from writing entirely (encoder-decoder).*

---

## 8. Key Takeaways

- The same RNN cell can be arranged into five different input/output shapes; the shape is chosen to match the task, not the other way around.
- **Many-to-one** collapses a sequence to one prediction (sentiment analysis) by using only the final hidden state.
- **One-to-many** expands a single input into a generated sequence (image captioning), typically seeding the first step and feeding generated outputs back in.
- **Many-to-many, same length** produces one output per input step in lock-step (POS tagging) — no separate encoder/decoder needed.
- **Many-to-many, variable length** needs an encoder-decoder (seq2seq) structure, because the input and output lengths can differ (machine translation).
- **One-to-one** has no sequence and no recurrence — it's really just a standard ANN/CNN, included in the taxonomy mainly for contrast.
- The BPTT derivation from [backpropagation-through-time.md](backpropagation-through-time.md) was worked out for the many-to-one case; the other shapes change *where* the loss enters the unrolled network, which changes how the gradient sums are built, but not the underlying chain-rule mechanics.

---

## 9. Further Reading

- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks*. NeurIPS. Introduces the encoder-decoder seq2seq framework.
- Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). *Show and Tell: A Neural Image Caption Generator*. CVPR. A one-to-many architecture for image captioning.
- Graves, A., Mohamed, A., & Hinton, G. (2013). *Speech Recognition with Deep Recurrent Neural Networks*. ICASSP. Discusses same-length sequence labeling with RNNs.
- Karpathy, A. (2015). *The Unreasonable Effectiveness of Recurrent Neural Networks*. Blog post covering many of these architecture patterns with worked examples.
