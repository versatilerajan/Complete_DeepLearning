# 🧠 Introduction to Deep Learning
### MIT 6.7960 — Extended Study Notes

> *"Deep learning is essentially: learning representations from data using differentiable neural networks optimized with gradient descent at scale."*

---

## 📚 Table of Contents

- [What is Deep Learning?](#what-is-deep-learning)
- [Why is it called "Deep"?](#why-is-it-called-deep)
- [Neural Networks](#neural-networks)
- [Activation Functions](#activation-functions)
- [Depth vs Width](#depth-vs-width)
- [Universal Approximation Theorem](#universal-approximation-theorem)
- [Differentiable Programming](#differentiable-programming)
- [Gradient Descent](#gradient-descent)
- [Loss Functions](#loss-functions)
- [Backpropagation](#backpropagation)
- [Common Training Problems](#common-training-problems)
- [Overfitting, Underfitting & Generalization](#overfitting-underfitting--generalization)
- [Training Terminology](#training-terminology)
- [Optimizers](#optimizers)
- [Representation Learning](#representation-learning)
- [Why Deep Learning Exploded After 2012](#why-deep-learning-exploded-after-2012)
- [Scaling Laws](#scaling-laws)
- [Major Architectures](#major-architectures)
- [LLM-Specific Concepts](#llm-specific-concepts)
- [Hardware for Deep Learning](#hardware-for-deep-learning)
- [Mathematical Foundations](#mathematical-foundations)
- [Learning Path for Beginners](#learning-path-for-beginners)

---

## What is Deep Learning?

Deep Learning is a **subfield of Machine Learning** where models automatically learn patterns, representations, and decision boundaries directly from raw data — using **multi-layer neural networks** trained via **gradient-based optimization**.

Unlike traditional machine learning, where humans manually engineer features (e.g., extracting edges from images or n-grams from text), deep learning allows the model to discover these features on its own through training.

At its core, deep learning is built on five pillars:

| Pillar | Description |
|---|---|
| **Neural Networks** | Multi-layered computational graphs loosely inspired by the brain |
| **Differentiable Programming** | Every operation must be differentiable so gradients can flow |
| **Large-scale Data** | More data generally means better-trained, more generalizable models |
| **Optimization Algorithms** | Techniques like SGD and Adam that iteratively improve parameters |
| **Compute Power** | GPUs/TPUs enable fast matrix operations needed for training |

### The Core Mathematical Framing

A deep learning model learns a **parameterized function**:

$$y = f(x; \theta)$$

Where:
- `x` = input data (image, text, audio, etc.)
- `y` = model prediction (label, next token, bounding box, etc.)
- `θ` (theta) = all **learnable parameters** (weights and biases)

The training process finds the **optimal θ** that minimizes prediction error on the training data, measured by a **loss function**.

---

## Why is it called "Deep"?

The word **"deep"** refers to the **number of layers** stacked in the neural network — not to any philosophical notion of intelligence.

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Hidden Layer 3 → Output Layer
```

A network with **one hidden layer** is considered *shallow*. A network with **many hidden layers** (typically 3+) is called *deep*.

**Why does depth matter?**

Each layer learns to represent the data at a **different level of abstraction**. In a deep image classifier, for example:

- **Early layers** detect low-level patterns: edges, corners, color gradients
- **Middle layers** combine those into mid-level structures: textures, shapes, curves
- **Deep layers** recognize high-level concepts: faces, objects, scenes

This hierarchical, compositional learning is one of the most powerful properties of deep networks.

---

## Neural Networks

A neural network is a collection of **neurons** organized into **layers**, where each neuron performs a simple mathematical computation.

### Key Components

| Component | Role |
|---|---|
| **Neurons** | Basic computational units; each applies a linear transformation + nonlinearity |
| **Layers** | Groups of neurons; information flows from one layer to the next |
| **Weights (W)** | Learned parameters that scale the inputs — determine feature importance |
| **Biases (b)** | Learned offsets that shift activation thresholds |
| **Activation Functions** | Nonlinear transformations applied after the linear step |

### What Each Neuron Computes

**Step 1 — Linear transformation:**

$$z = Wx + b$$

This is a **weighted sum** of the inputs plus a bias. It's essentially a dot product, which is why GPUs (built for matrix multiplication) are so valuable.

**Step 2 — Nonlinear activation:**

$$a = \sigma(z)$$

Where σ is an activation function (e.g., ReLU, sigmoid). This output `a` becomes the input to the next layer.

### Why Nonlinearity Matters

This is one of the most important conceptual points in all of deep learning.

**Without nonlinearities**, stacking multiple linear layers collapses to a single linear transformation:

```
Layer1: W₁x + b₁
Layer2: W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)
                            = W_combined · x + b_combined
```

No matter how many layers you stack, the result is still just `Wx + b`. You gain nothing from depth.

**With nonlinearities**, each layer creates complex, non-linear decision boundaries that cannot be reduced to a single layer. This allows networks to model:
- Curved boundaries in space
- Natural language meaning
- Visual object recognition
- Complex reasoning chains

---

## Activation Functions

Activation functions introduce nonlinearity. Here are the most important ones:

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- **Output range:** (0, 1)
- **Use case:** Binary classification output layer
- **Problems:**
  - **Vanishing gradients:** For very large or very small inputs, the gradient approaches zero — early layers receive almost no learning signal
  - Outputs are not zero-centered, which can slow down optimization
  - Computationally more expensive than ReLU

### Tanh (Hyperbolic Tangent)

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- **Output range:** (-1, 1)
- **Improvement over sigmoid:** Zero-centered, so gradients are more balanced
- **Still suffers:** Vanishing gradient problem for extreme input values

### ReLU — Rectified Linear Unit ⭐ Most Important

$$f(x) = \max(0, x)$$

- **Output range:** [0, ∞)
- **Advantages:**
  - **Simple and fast** to compute — just a threshold operation
  - **Sparse activations** — negative inputs become 0, creating sparsity that acts as implicit regularization
  - **Reduced vanishing gradient** — gradient is either 0 or 1, so it doesn't shrink as it backpropagates through layers
  - Dominates modern deep learning in hidden layers

- **Known issue:** "Dying ReLU" — neurons whose inputs are always negative permanently output 0 and stop learning. Addressed by variants like **Leaky ReLU** and **GELU**.

### GELU (Gaussian Error Linear Unit)

Used in transformers (BERT, GPT). Smoother approximation of ReLU that allows small negative values through, which improves gradient flow.

---

## Depth vs Width

Neural networks have two primary dimensions of capacity: **depth** (number of layers) and **width** (number of neurons per layer).

### Depth

```
Input
  ↓
[Layer 1: 64 neurons]
  ↓
[Layer 2: 64 neurons]
  ↓
[Layer 3: 64 neurons]
  ↓
Output
```

**Advantages of more depth:**
- Enables **hierarchical representation learning** — each layer builds on the previous
- Better for **complex reasoning** and abstract tasks
- More **parameter-efficient** — achieves more with fewer total parameters
- Deep networks better capture the compositional structure of the real world

### Width

```
Input
  ↓
[Layer 1: 4096 neurons]
  ↓
Output
```

**Advantages of more width:**
- More **feature capacity** per layer
- Can represent a larger variety of patterns simultaneously
- Wider layers are easier to parallelize
- Better **memorization** capacity

### Deep vs Wide in Practice

```
Deep Narrow Network          Wide Shallow Network
─────────────────            ────────────────────
Input                        Input
  ↓                            ↓
[64]                         [4096]
  ↓                            ↓
[64]                         Output
  ↓
[64]
  ↓
Output
```

Modern state-of-the-art models (GPT-4, Gemini, Claude) **scale both** — they are simultaneously very deep and very wide. The optimal ratio of depth to width is an active research area.

---

## Universal Approximation Theorem

> *A feedforward neural network with at least one hidden layer and a sufficient number of neurons can approximate any continuous function to arbitrary precision.*

This is a foundational theoretical result. It tells us that neural networks are **universal function approximators** — in principle, there's no function they can't learn if given enough capacity.

**Important caveats:**

1. The theorem guarantees *existence* of such a network, but says nothing about how to *find* the right weights
2. **Shallow networks** may require an impractically enormous number of neurons to approximate complex functions
3. **Deep networks** can represent the same functions far more efficiently — fewer total parameters, faster convergence
4. This is the theoretical justification for why depth matters beyond just empirical observation

---

## Differentiable Programming

Deep learning works because **every operation in the network is differentiable** with respect to the parameters. This allows the computation of **gradients** — the mathematical direction in which to adjust each weight to reduce error.

This is not a small detail. The entire training pipeline depends on it:
- Loss functions must be differentiable
- Activation functions must be differentiable (or have subgradients)
- Even operations like pooling or normalization are designed to be differentiable

Modern deep learning frameworks (PyTorch, JAX) automatically compute these gradients via **automatic differentiation (autograd)** — you write the forward computation, and the framework computes all gradients for free.

---

## Gradient Descent

Gradient descent is the **core optimization algorithm** used to train neural networks. It iteratively adjusts the model's parameters in the direction that reduces the loss.

### The Update Rule

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta)$$

Where:
- `θ` = current parameters
- `η` (eta) = **learning rate** — controls the step size
- `∇_θ L(θ)` = gradient of the loss with respect to parameters

### Intuition

Imagine the loss function as a hilly landscape. The gradient points **uphill** (direction of steepest increase). We move in the **negative gradient direction** — downhill — to find a valley (minimum loss).

### Variants

| Variant | Description | Use Case |
|---|---|---|
| **Batch GD** | Uses the entire dataset per update | Rarely used — too slow |
| **Stochastic GD (SGD)** | Uses one sample per update | Noisy but fast |
| **Mini-batch GD** | Uses a small batch (e.g., 32–512 samples) | Standard in practice |

### Learning Rate — The Most Critical Hyperparameter

| Learning Rate | Effect |
|---|---|
| **Too high** | Overshoots minima; training becomes unstable or diverges |
| **Too low** | Painfully slow convergence; may get stuck in local minima |
| **Just right** | Steady, stable descent toward a good minimum |

**Learning rate schedules** (cosine annealing, warmup, step decay) dynamically adjust the learning rate during training for better results.

---

## Loss Functions

A **loss function** (also called a cost function or objective function) quantifies **how wrong the model's predictions are**. Training aims to minimize this value.

### Mean Squared Error (MSE) — For Regression

$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- Measures average squared difference between prediction and true value
- Heavily penalizes large errors (due to squaring)
- **Used for:** House price prediction, forecasting, any continuous output

### Cross-Entropy Loss — For Classification

$$L = -\sum_{i} y_i \log(\hat{y}_i)$$

- Measures how different the predicted probability distribution is from the true distribution
- Penalizes confident wrong predictions very heavily
- **Used for:** Image classification, language modeling, NLP tasks

### Intuition

Think of the loss as a **report card** for the model. A loss of 0 would mean perfect predictions. The network's entire job during training is to minimize this score by adjusting its weights.

---

## Backpropagation

**Backpropagation** is the algorithm that computes the gradient of the loss with respect to every weight in the network, efficiently using the **chain rule of calculus**.

Without backpropagation, training deep networks would be computationally infeasible. It is the fundamental engine that makes deep learning work.

### The Full Training Loop

```
1. FORWARD PASS
   Input x → through all layers → prediction ŷ

2. COMPUTE LOSS
   Compare ŷ with true label y using loss function L

3. BACKWARD PASS (Backpropagation)
   Compute ∂L/∂θ for every parameter θ using chain rule
   Gradients flow backward: Output → ... → Hidden Layers → Input

4. WEIGHT UPDATE
   θ = θ - η · ∂L/∂θ  (gradient descent step)

5. REPEAT for all batches and epochs
```

### Chain Rule in Action

For a composed function `L = f(g(h(x)))`:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

Backprop applies this efficiently across the entire computation graph, reusing intermediate computations.

---

## Common Training Problems

### Vanishing Gradient Problem

In deep networks, gradients can become **exponentially small** as they propagate backward through many layers. Early layers (closer to input) receive near-zero gradient signals and effectively **stop learning**.

- **Most common with:** sigmoid and tanh activations (their gradients are always < 1)
- **Effect:** The first few layers never improve; training stagnates

**Solutions:**
- Use **ReLU** activations (gradient is 1 for positive inputs)
- **Residual connections** (skip connections in ResNets) — provide gradient "highways"
- **Batch Normalization** — normalizes activations to prevent saturation
- **Better weight initialization** (Xavier, He initialization)

### Exploding Gradient Problem

The opposite problem — gradients become **exponentially large**, causing wild parameter updates that destabilize training. You'll often see `NaN` loss values.

- **Most common in:** Recurrent networks (RNNs) with long sequences

**Solutions:**
- **Gradient clipping** — cap gradient magnitude at a threshold (e.g., 1.0)
- **Normalization layers**
- **Careful weight initialization**

---

## Overfitting, Underfitting & Generalization

These are among the most important concepts in all of machine learning.

### Overfitting

The model learns the **training data too well** — it memorizes noise and specific examples rather than underlying patterns.

```
Training loss:    0.001  ✓ (very low)
Validation loss:  2.340  ✗ (very high)
```

**Causes:** Model too complex, too many parameters, too few training examples

**Solutions:**
- **Dropout** — randomly zero out neurons during training
- **L1/L2 Regularization** — penalize large weights
- **Data Augmentation** — artificially expand training set
- **Early Stopping** — stop training when validation loss stops improving
- **More training data**

### Underfitting

The model is **too simple** to capture the underlying patterns, performing poorly on both training and validation data.

```
Training loss:    1.89  ✗
Validation loss:  1.91  ✗
```

**Causes:** Model too small, insufficient training time, too much regularization

**Solutions:** Use a larger model, train longer, reduce regularization

### Generalization

Generalization is the model's ability to **perform well on unseen data** — data it was never trained on.

This is the *true goal* of machine learning. A model with perfect training accuracy but poor generalization is useless in the real world.

Interestingly, modern large models (with billions of parameters) generalize remarkably well — far better than classical theory would predict. Understanding *why* is one of the deepest open problems in ML theory.

---

## Training Terminology

| Term | Definition |
|---|---|
| **Epoch** | One complete pass through the entire training dataset |
| **Batch** | A small subset of training data processed together in one forward/backward pass |
| **Iteration** | One parameter update step (processes one batch) |
| **Batch Size** | Number of samples in each batch (e.g., 32, 128, 256) |

**Relationship:**

```
Total Iterations per Epoch = Dataset Size / Batch Size

Example:
  Dataset: 50,000 images
  Batch Size: 100
  → 500 iterations per epoch
```

Larger batch sizes give more stable gradient estimates but require more memory. Smaller batches add noise that can actually help escape local minima.

---

## Optimizers

Optimizers are algorithms that implement the actual parameter update step, building on top of basic gradient descent.

### SGD (Stochastic Gradient Descent)

The simplest optimizer — just gradient descent with mini-batches.

$$\theta = \theta - \eta \cdot \nabla_\theta L$$

Can be enhanced with **momentum** to accelerate convergence and smooth out oscillations.

### Adam (Adaptive Moment Estimation) ⭐ Most Popular

Adam combines two key ideas:

1. **Momentum** — keeps a running average of past gradients (where have we been going?)
2. **Adaptive Learning Rates** — each parameter gets its own learning rate, scaled by the magnitude of recent gradients

This makes Adam robust and fast — it works well out-of-the-box for most tasks without careful learning rate tuning.

```
Adam Update (simplified):
  m = β₁ · m + (1 - β₁) · g        ← gradient mean (momentum)
  v = β₂ · v + (1 - β₂) · g²       ← gradient variance
  θ = θ - η · m / (√v + ε)          ← adaptive update
```

**Other notable optimizers:** AdamW (Adam + weight decay, used in most LLMs), RMSProp, Adagrad

---

## Representation Learning

One of the most transformative aspects of deep learning is its ability to **automatically learn useful representations** from raw data — no human feature engineering required.

### Traditional ML vs Deep Learning

| Approach | Feature Engineering | Example |
|---|---|---|
| **Traditional ML** | Manual, human-designed | HOG features for pedestrian detection |
| **Deep Learning** | Automatic, learned from data | CNN learns its own edge detectors |

### Feature Hierarchy in Vision

A trained CNN builds a hierarchy of representations:

| Layer | What It Learns |
|---|---|
| Layer 1 | Edges, color gradients, oriented lines |
| Layer 2 | Textures, simple shapes, corners |
| Layer 3 | Object parts (eyes, wheels, etc.) |
| Layer 4+ | Full objects, semantic concepts |

This hierarchical feature learning is why deep networks transfer so well across tasks — the early layers learn generally useful representations (edges, textures) that are useful for almost any vision task.

---

## Why Deep Learning Exploded After 2012

### AlexNet — The Moment Everything Changed

In 2012, AlexNet achieved a top-5 error rate of 15.3% on ImageNet — far below the 26.2% of the next best competitor. This wasn't a small improvement; it was a paradigm shift.

**What made it work:**

1. **GPUs** — AlexNet was trained on two NVIDIA GTX 580 GPUs, making large-scale training feasible
2. **ReLU activations** — replaced sigmoid/tanh, dramatically speeding up convergence
3. **Large labeled datasets** — ImageNet provided 1.2 million labeled images
4. **Dropout regularization** — reduced overfitting in the fully connected layers
5. **Data augmentation** — artificially multiplied training examples

AlexNet started the **modern AI boom**. Every major AI breakthrough since — from ResNets to transformers to LLMs — traces its lineage to the ideas validated in 2012.

### The Three Pillars of Modern AI Progress

```
Better Models + More Data + More Compute = Better AI
```

These three ingredients have been consistently scaled up over the past decade, producing increasingly capable systems.

---

## Scaling Laws

**Scaling laws** describe the empirically observed, **predictable relationship** between model performance and three quantities:

1. **Number of parameters** (model size)
2. **Amount of training data**
3. **Compute budget** (FLOPs)

As each of these increases, loss decreases in a smooth, power-law relationship. This means AI progress is — to a surprising degree — **predictable and systematic**.

### Implications

- You can predict how good a model will be before training it
- Larger models trained on more data consistently outperform smaller ones
- This has justified the enormous investment in scaling models like GPT, Gemini, Claude, and Llama

**Key insight from Chinchilla scaling laws (2022):** For a given compute budget, most models have been undertrained relative to their size. The optimal strategy is to train a **smaller model on more data** rather than a giant model on less data.

---

## Major Architectures

### CNNs — Convolutional Neural Networks

CNNs are specifically designed for **spatial data** like images. They exploit two key properties:

- **Local connectivity** — neurons only connect to a small patch of the input (their receptive field)
- **Weight sharing** — the same filter (kernel) is applied across the entire image, dramatically reducing parameters

```
Input Image → [Conv → ReLU → Pool] × N → Flatten → Fully Connected → Output
```

**Applications:**
- Image classification (ResNet, EfficientNet, VGG)
- Object detection (YOLO, Faster R-CNN)
- Medical imaging (tumor detection, radiology)
- Self-driving car perception

### RNNs — Recurrent Neural Networks

RNNs process **sequential data** by maintaining a hidden state that is updated at each time step, allowing information to persist across a sequence.

```
Input: [w₁, w₂, w₃, ...]
         ↓    ↓    ↓
  h₀ → [h₁] → [h₂] → [h₃] → Output
```

**Historical use cases:** Machine translation, text generation, speech recognition, time series

**Key problems:**
- **Long-term dependency problem** — information from early in a sequence "fades" as the sequence grows
- Difficult to parallelize (each step depends on the previous)

RNNs have been **largely replaced by Transformers** for most sequence modeling tasks.

### Transformers ⭐ Most Important Modern Architecture

Transformers, introduced in the landmark paper *"Attention Is All You Need"* (Vaswani et al., 2017), have become the **dominant architecture** for nearly all modern AI.

**Core innovation:** Replace recurrence with **self-attention** — letting every position in the sequence directly attend to every other position.

```
Input Tokens → Embeddings → [Attention + FFN] × N → Output
```

**Why Transformers win:**

| Property | Benefit |
|---|---|
| **Parallel computation** | All tokens processed simultaneously (vs. sequentially in RNNs) |
| **Long-range dependencies** | Any token can attend to any other, regardless of distance |
| **Scalability** | Performance improves smoothly with more parameters and data |
| **Transfer learning** | Pretrained transformers adapt to almost any task |

**Powered by:** GPT series, Gemini, Claude, Llama, BERT, T5, Whisper, DALL-E, Stable Diffusion, and essentially every frontier model today.

---

## LLM-Specific Concepts

### Attention Mechanism

Attention allows the model to **dynamically weight** how much focus to place on each part of the input when producing each output.

**Core idea:** Not all words or features are equally relevant to a given prediction. Attention learns to selectively focus.

For the sentence: *"The trophy didn't fit in the suitcase because it was too big"* — what does "it" refer to? Attention learns to link "it" back to "trophy" rather than "suitcase."

**Scaled Dot-Product Attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where Q (query), K (key), and V (value) are linear projections of the input.

### Embeddings

Embeddings are **dense vector representations** of discrete tokens (words, subwords, image patches, etc.) in a continuous vector space.

```
"king"  → [0.25, -1.43, 0.82, 3.17, ...]   (e.g., 768-dimensional vector)
"queen" → [0.27, -1.40, 0.85, 3.10, ...]   (similar direction!)
"apple" → [-2.1,  0.33, 1.24, -0.87, ...]  (very different direction)
```

Well-trained embeddings capture **semantic relationships** mathematically:
```
king - man + woman ≈ queen
```

### Tokens

LLMs don't process characters or full words — they process **tokens**, which are subword units produced by a tokenizer (e.g., BPE — Byte Pair Encoding).

```
"unbelievable" → ["un", "believ", "able"]   (3 tokens)
"hello"        → ["hello"]                  (1 token)
```

This lets the model handle any word, including rare words and neologisms, by decomposing them into known subword pieces.

### Context Window

The context window is the **maximum number of tokens** a model can "see" at once — its working memory for a single forward pass.

| Model Generation | Typical Context Window |
|---|---|
| Early GPT models | 512–2,048 tokens |
| GPT-4 | 8K–128K tokens |
| Modern frontier models | 1M+ tokens |

Larger context windows allow models to process entire books, codebases, or long conversations.

### Emergent Abilities

As models scale to hundreds of billions of parameters, they unexpectedly develop capabilities that were **not present in smaller models** and were **not explicitly trained**:

- Multi-step mathematical reasoning
- Code generation and debugging
- Translation between languages never seen together
- In-context learning (learning from examples in the prompt)
- Chain-of-thought reasoning

These emergent abilities are not well understood theoretically and are one of the most fascinating and surprising aspects of modern LLMs.

### Transfer Learning & Fine-Tuning

**Transfer Learning:** Knowledge learned on one task or dataset transfers to help with another task.

A model pretrained on massive text data learns grammar, facts, reasoning patterns, and world knowledge. This can then be applied to specialized tasks.

**Fine-Tuning:** Taking a pretrained model and continuing training on a **smaller, task-specific dataset** to adapt it to a particular use case.

```
Pretrained LLM
      ↓
  Fine-tune on:
  - Medical Q&A
  - Legal documents
  - Customer service transcripts
      ↓
Specialized Assistant
```

### Self-Supervised Learning

Modern LLMs are trained without manually labeled data using **self-supervised learning** — the labels are derived from the data itself.

**Next-token prediction (Causal LM):**
```
Input:  "The cat sat on the"
Target: "mat"
```

The model learns to predict the next token, and by doing this billions of times across trillions of tokens, it implicitly learns grammar, facts, reasoning, and world knowledge.

### RAG (Retrieval-Augmented Generation)

RAG enhances LLMs by retrieving **relevant documents** from an external knowledge base at inference time, allowing models to:
- Access up-to-date information beyond their training cutoff
- Cite sources
- Handle domain-specific knowledge without full fine-tuning

---

## Hardware for Deep Learning

### GPUs (Graphics Processing Units)

GPUs are the **workhorse of deep learning**. Originally designed for rendering graphics (which requires massive parallel computation), they turned out to be ideal for the matrix multiplications that dominate neural network computation.

- NVIDIA's A100 and H100 are the dominant training GPUs
- Thousands of CUDA cores operate in parallel
- High memory bandwidth allows fast movement of large tensors

### TPUs (Tensor Processing Units)

Google's custom AI accelerators, designed specifically for the operations common in deep learning:
- Outperform GPUs on certain transformer workloads
- Used internally by Google for training Gemini and other models
- Available via Google Cloud

### Why PyTorch?

PyTorch has become the **dominant deep learning framework**, especially in research.

**Advantages:**
- **Intuitive Pythonic API** — feels like writing normal Python
- **Dynamic computation graphs** — the graph is built at runtime, making debugging easy
- **Huge ecosystem** — Hugging Face, Lightning, TIMM, and thousands of libraries
- **Research-friendly** — easy to implement new ideas quickly

Used by: OpenAI, Meta AI, most major research labs, and the majority of academic researchers.

---

## Mathematical Foundations

To truly master deep learning, you need proficiency in these mathematical areas:

| Subject | Key Topics |
|---|---|
| **Linear Algebra** | Matrix multiplication, eigenvectors, SVD, vector spaces |
| **Calculus** | Partial derivatives, gradients, Jacobians, the chain rule |
| **Probability** | Distributions, Bayes' theorem, expectations, MLE |
| **Optimization** | Convexity, saddle points, convergence theory |

You don't need to be a mathematician, but a solid working understanding of these topics will make you a significantly better deep learning practitioner.

---

## Learning Path for Beginners

Follow this structured path to build a solid deep learning foundation:

```
1. Python
   └─ NumPy (numerical computing)
        └─ PyTorch (deep learning framework)
             ├─ Tensors, autograd, nn.Module
             ├─ Training loops, dataloaders
             └─ GPU acceleration

2. Neural Networks
   └─ Fully connected networks
        └─ Training, regularization, debugging

3. CNNs
   └─ Image classification, object detection

4. Transformers
   └─ Self-attention, positional encoding
        └─ Fine-tuning pretrained models (Hugging Face)

5. LLMs & Modern AI
   └─ RAG, Agents, Multimodal AI
        └─ Quantization, LoRA, RLHF
```

### Beginner Advice

✅ **Do start with:**
- PyTorch basics and tensor operations
- Training a small network on MNIST or CIFAR-10
- Understanding the full training loop
- Reading original papers (starting with simpler ones)

❌ **Do NOT start with:**
- Building AGI
- Training 70B parameter models from scratch
- Complex research topics before mastering fundamentals
- Jumping to fine-tuning before understanding what you're fine-tuning

---

## Topics to Explore Next

After mastering the fundamentals above, dive into these advanced topics:

- **Architecture:** Residual connections, batch normalization, layer normalization, positional encoding
- **Training Techniques:** Gradient clipping, learning rate warmup, mixed precision (fp16/bf16)
- **Efficient AI:** Quantization (INT8, INT4), LoRA fine-tuning, knowledge distillation
- **Generative Models:** Diffusion models (Stable Diffusion, DALL-E)
- **Alignment:** Reinforcement Learning from Human Feedback (RLHF), Constitutional AI
- **Scalable Architectures:** Mixture of Experts (MoE), sparse attention
- **Tokenization:** BPE, SentencePiece, tiktoken

---

## Final Big Picture

> Deep learning is **learning representations from data** using **differentiable neural networks** optimized with **gradient descent** at **scale**.

Every major breakthrough in modern AI — from AlexNet to GPT-4 to AlphaFold — is a variation on this central theme. The field continues to evolve rapidly, but the fundamentals you've learned here will remain relevant for a long time to come.

---

*Based on MIT 6.7960 Deep Learning course materials. Notes expanded for comprehensive self-study.*
