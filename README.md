<div align="center">

# 🔐TransCrypto🔐: Transformer-based Encryption with RSA
</div>

![TransCrypto Logo](https://yourimageurl.com/logo.png)
## 📑 Project Overview
This project implements a Transformer-based neural network model integrated with RSA encryption for secure text data handling. The model encrypts text data using a Transformer network and further secures it with RSA encryption.

## Features

- **Transformer-based Encryption**: Utilizes a Transformer neural network for encoding and decoding text data.
- **RSA Encryption**: Adds an additional layer of security using RSA encryption.
- **Training and Evaluation**: Includes functionality for training and evaluating the model on custom datasets.
- **Tokenization and Padding**: Prepares text data using tokenization and padding for input into the model.

## 🚀Installation
### Clone the repository
   ```bash
   git clone https://github.com/yourusername/transcrypto.git
   cd transcrypto
   ```

## 🛠️Usage
### 1. Preparing the Data

Define your text data and prepare it using the provided functions. Ensure that your texts are tokenized and padded appropriately.

### 2. Training the Model
Train the model using your prepared dataset. The train function handles the training process, including early stopping and learning rate scheduling.
```python
from torch.utils.data import DataLoader, TensorDataset
import torch

# Define your texts and preprocess them
texts = ["hello world", "transformer model for encryption", "test text data"]
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
test_texts = ["hello world", "new test text"]
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
print("Decrypted Texts:", decrypted_texts)
```

## Model Architecture
- **Embedding Layer**: Converts input tokens into dense vectors.
- **Transformer Encoders**: Multiple layers of Transformer encoders for capturing long-range dependencies.
- **Transformer Decoders**: Multiple layers of Transformer decoders for generating output sequences.
- **RSA Encryption**: RSA public and private keys for encrypting and decrypting model outputs.


## ✂️License
This project is licensed under the MIT License - see the LICENSE file for details.
