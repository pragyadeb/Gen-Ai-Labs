# 🤖 Generative Chatbot using Attention (Lab 12)

## 📌 Objective
To implement a text generation model using an Attention mechanism to improve contextual understanding in chatbot responses.

---

## 🧠 Project Description
This project builds a simple chatbot using:
- LSTM (Encoder)
- Attention Mechanism
- Embedding Layer

The attention mechanism helps the model focus on important words in the input sentence and generate better responses.

---

## 📂 Dataset
A small sample dataset inspired by the Cornell Movie Dialog Dataset is used.

Example:
- Input: "hello"
- Output: "hi how are you"

---

## ⚙️ Technologies Used
- Python
- PyTorch
- Google Colab

---

## 🏗️ Model Architecture
Embedding → LSTM Encoder → Attention Layer → Fully Connected Layer

---

## 🔍 Key Concept: Attention
Attention assigns weights to input words so the model focuses on important parts of the sentence.

Example:  
Input: how are you  
Output: i am fine  
(Model focuses more on "you")

---

## 🚀 How to Run
1. Open Google Colab
2. Upload the notebook (`.ipynb`)
3. Run all cells step by step

---

## 📊 Output
Example:

Input: how are you  
Output: fine  

---

## ❗ Challenges Faced
- Tensor shape mismatch error
- Fixing CrossEntropyLoss input issue

---

## 📌 Conclusion
The Attention mechanism improves chatbot performance by focusing on relevant words in the input sequence.

---


