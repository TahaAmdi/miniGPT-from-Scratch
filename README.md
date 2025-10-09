# 🧠 GPT From Scratch — Educational Implementation

This repository walks through the **complete pipeline of building a GPT-style Transformer language model from scratch**,using **PyTorch** and **tiktoken**.  
It’s designed for educational and research purposes — demonstrating how modern LLMs like GPT-2 are structured and trained, step by step.

---

## 📁 Repository Structure

```
├── Book/                                # Text corpus used for training (e.g., Alice in Wonderland, etc.)
├── img/                                 # Images used for visual explanations in the notebooks
├── .gitignore
├── 1.Tokenization.ipynb                 # Text preprocessing, tokenization, and dataset creation
├── 2.Attention.ipynb                    # Core attention mechanisms (Self, Causal, Multi-Head)
├── GPT_Second_Try_Scratch_with_more_explanation.ipynb   # Full GPT model with extensive explanations
├── GPT_Third_Try_Scratch.ipynb          # Same GPT model with minimal markdowns (clean version)
├── README.md                            # Project documentation (this file)
```

> **Note:**  
> `0.examples.ipynb` and `gpt_download.py` are auxiliary or exploratory and not part of the main educational flow.

---

## 🚀 Project Overview

This project demonstrates, from the ground up, how a **decoder-only Transformer** (GPT-like model) processes text — from **raw text files** to **token embeddings**, **attention mechanisms**, and **text generation**.

It’s structured as a **progressive learning journey** across three main notebooks:

---

### 1️⃣ `1.Tokenization.ipynb` — Data Preparation & Tokenization

- Introduces **tokenization concepts** (word-based, regex-based, and Byte-Pair Encoding via GPT-2 tokenizer).
- Builds a **custom tokenizer class (`SimpleTokenizerV1`)**.
- Uses **tiktoken** for GPT-2-compatible tokenization.
- Prepares **(input, target)** next-token prediction pairs using a **1-step shift**.
- Demonstrates **sliding windows** for fixed-length context chunks.
- Implements a **PyTorch Dataset (`GPTDatasetV1`)** and **DataLoader**.
- Constructs **token & positional embeddings** for transformer input.
  
📘 *Foundation for all later notebooks — builds the data pipeline GPTs rely on.*

---

### 2️⃣ `2.Attention.ipynb` — Building the Attention Mechanism

- Explains the **mathematical foundations** of attention: Queries (Q), Keys (K), and Values (V).
- Implements:
  - **Dot-product attention**
  - **Self-attention**
  - **Causal (masked) attention**
  - **Multi-head attention** (both modular and fused QKV forms)
- Introduces **lower-triangular causal masks** (to prevent looking ahead).
- Explains **PyTorch tensor shapes**, transpositions, and `contiguous()` use.
- Adds **scaling by √dₖ**, dropout, and output projections.

📘 *Hands-on understanding of how attention allows GPTs to “focus” on relevant tokens.*

---

### 3️⃣ `GPT_Second_Try_Scratch_with_more_explanation.ipynb` — Full GPT Model (Detailed)

- Full **decoder-only GPT architecture** implemented from scratch:
  - Token & positional embeddings
  - Multi-head causal self-attention
  - Feedforward layers (with GELU activation)
  - Pre-Layer Normalization (Pre-LN)
  - Residual connections & dropout
  - Output projection head
- Includes:
  - **Parameter counting**
  - **Forward pass sanity checks**
  - **Greedy text generation loop (`generate_text_simple`)**
  - **Visualizations:** GELU vs ReLU, causal masks, and parameter structure.
- Uses **tiktoken GPT-2 BPE tokenizer** for text encoding/decoding.

📘 *This notebook represents a full, trainable GPT implementation with thorough explanations.*

---

### 4️⃣ `GPT_Third_Try_Scratch.ipynb` — Full GPT Model (Concise)

- Identical to the second notebook in logic and code.
- Markdown explanations and visualizations have been trimmed.
- Ideal for **fast experimentation and model fine-tuning**.

📘 *A cleaner version for training or extending your own GPT model.*

---

## 🧩 Learning Flow

1. **Start** with `1.Tokenization.ipynb` to understand tokenization and data batching.
2. **Study** `2.Attention.ipynb` to grasp the mechanics of attention.
3. **Build & Test** your own GPT in `GPT_Second_Try_Scratch_with_more_explanation.ipynb`.
4. **Experiment & Train** using `GPT_Third_Try_Scratch.ipynb` and your text data in `/Book/`.

---

## ⚙️ Technologies Used

- **Python 3.x**
- **PyTorch** — model implementation
- **tiktoken** — GPT-2 tokenizer
- **NumPy & Matplotlib** — visualizations
- **Jupyter Notebook** — interactive exploration

---

## 🧠 Concepts Demonstrated

- Byte-Pair Encoding (BPE) tokenization  
- Embedding representations (token & positional)  
- Scaled dot-product attention  
- Multi-head self-attention  
- Causal masking  
- Residual & normalization layers  
- Feedforward networks  
- Greedy text generation  
- Transformer architecture design principles  

---

## 🧾 Example Output

Once the model is trained and run with:
```python
pipe = pipeline("text-generation", model=model_q, tokenizer=tok, device=-1)
print(pipe("CHAPTER I. ", max_new_tokens=120)[0]["generated_text"])
```
It generates coherent, creative continuations of text, learned from your `/Book/` corpus.

---

## 📚 Folder “Book/”

This folder contains the **text datasets** used for fine-tuning the custom GPT.  
Examples: *Alice in Wonderland*, *Sherlock Holmes*, or any other literary text.

---

## 💡 Future Extensions

- Add **training loop** with AdamW and LR scheduling.
- Implement **top-k / nucleus sampling** for more creative text.
- Explore **LoRA fine-tuning** on new text corpora.
- Integrate **evaluation metrics** (perplexity, cross-entropy).
- Visualize **attention maps** during generation.

---

## 🧑‍💻 Authors

**David AhmadiShahraki**

*MSc Artificial Intelligence, University of Hull*  
Exploring the internals of modern LLMs through from-scratch implementation.

**Alireza Rashidi**

*MSc Artificial Intelligence, University of Hull*  
Exploring the internals of modern LLMs through from-scratch implementation.



---

## 📜 License

This project is open for educational and research use.  
Attribution is appreciated if used in tutorials, projects, or papers.

