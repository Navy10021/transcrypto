<div align="center">

# 🔐 TransCrypto: Transformer-based Neural Network Encryption with RSA
</div>

## 📑 Project Overview

**TransCrypto** is a project that integrates ***Transformer-based neural network models*** with ***RSA encryption*** to ensure secure text data processing. This model encrypts text data using a Transformer network and adds an extra layer of security through RSA encryption.

## 🤖 What is a Transformer Model?

A **Transformer model** is a type of deep learning model introduced in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). It is widely used in natural language processing(NLP) tasks due to its ability to handle long-range dependencies and parallelize training. Transformers use self-attention mechanisms to weigh the importance of different parts of the input data dynamically.

## 🔐 What is RSA Encryption?

**RSA (Rivest–Shamir–Adleman) encryption** is a public-key cryptosystem that is widely used for secure data transmission. It involves two keys: a public key, which can be distributed widely, and a private key, which is kept secret. RSA is based on the mathematical difficulty of factoring the product of two large prime numbers.


## 📋 Features

- **Transformer-based Encryption**: Uses Transformer neural networks for encoding and decoding text data.
- **RSA Encryption**: Provides an additional security layer using RSA encryption.
- **Training and Evaluation**: Includes functions for training and evaluating the model on custom datasets.
- **Tokenization and Padding**: Tokenizes and pads text data for model input.

## 🚀 Installation

### Clone the repository

   ```bash
   git clone https://github.com/Navy10021/transcrypto.git
   cd transcrypto
   ```

## 🛠️ Usage
### 1. Preparing the Data

Define your text data and prepare it using the provided functions. Ensure that your texts are tokenized and padded appropriately.

### 2. Training the Model
Train the model using your prepared dataset. The train function handles the training process, including early stopping and learning rate scheduling.
```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Defining text data for training transformer-based encryption models
texts = ["hello world", "transformer model for encryption", "test text data", "1 1 1", "1 3 3 7"]
data = tokenize_and_pad(texts, vocab, tokenizer, maxlen)

# Create dataset and dataloader
dataset = TensorDataset(data)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

# Initialize the model and train
model = TransCryptoRSA(input_dim, embed_dim, num_heads, ff_dim, num_layers, maxlen, public_key, private_key).to(device)
train(model, train_loader, val_loader, epochs=50, device=device)
```

### 3. Test the Model
Test the encryption and decryption capabilities of the trained model on new text data.
```python
# Test texts
test_texts = ["hello world", "new test text", "1 3 3 7"]
test_data = tokenize_and_pad(test_texts, vocab, tokenizer, maxlen).to(device)

# Encrypt and decrypt the data
model.eval()
with torch.no_grad():
    encrypted_data = model(test_data)
    encrypted_rsa_data = model.encrypt(encrypted_data.argmax(dim=-1))
    decrypted_rsa_data = model.decrypt(encrypted_rsa_data)
    decrypted_data = model(decrypted_rsa_data)

# Convert decrypted data back to text
decrypted_texts = tensor_to_text(decrypted_data.argmax(dim=-1), vocab)

print(">> Encrypted Data:", encrypted_data, end = "\n")
print(">> Enhanced Encrypted Data:", encrypted_rsa_data, end = "\n")
print(">> Decrypted Texts:", decrypted_texts, end = "\n")
```

### 4. Running the Scripts
When you run main_0.py, it executes the basic Transformer-based encryption and decryption without RSA. Running main.py performs enhanced encryption and decryption using the Transformer-based neural network model integrated with RSA encryption.
   ```bash
   python main.py
   python main_0.py
   ```

## 📦 Model Architecture
- **Embedding Layer**: Converts input tokens into dense vectors.
- **Transformer Encoders**: Multiple layers of Transformer encoders for capturing long-range dependencies.
- **Transformer Decoders**: Multiple layers of Transformer decoders for generating output sequences.
- **RSA Encryption**: RSA public and private keys for encrypting and decrypting model outputs.

  
## 👨‍💻 Contributors
- **Seoul National University Graduate School of Data Science (SNU GSDS)**
- Under the guidance of ***Navy Lee***

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
