# TransCrypto: A Transformer-based Cryptography Model

![TransCrypto Logo](https://yourimageurl.com/logo.png)

TransCrypto is a PyTorch implementation of a Transformer-based model designed for cryptography tasks. This project includes implementations of both the Transformer encoder and decoder, along with a custom training loop that incorporates early stopping and learning rate scheduling.

## Features
- 🚀 Transformer Encoder and Decoder
- 🛠️ Custom Training Loop with Early Stopping
- 📉 Learning Rate Scheduler
- ✂️ Tokenization and Padding of Input Texts

## Requirements
- Python 3.6+
- PyTorch
- torchtext
- numpy

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/transcrypto.git
   cd transcrypto
   ```
## Usage

### Example Usage
Here's a step-by-step guide to using TransCrypto with sample data:

1. **Sample Data:**
   ```python
   texts = ["hello world", "this is a test", "transformer model example"]
   ```
2. **Tokenization and Vocabulary Building:**
   ```python
   from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
   ```
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(map(tokenizer, texts), specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"])
   ```
3. **Tokenize and Pad Texts:**
   ```python
   maxlen = 10  # Define maximum length for padding
padded_tokens = tokenize_and_pad(texts, vocab, tokenizer, maxlen)
   ```
4. **Create DataLoader:**
   ```python
   from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(padded_tokens)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(dataset, batch_size=2)
   ```
6. **Initialize and Train the Model:**
   ```python
   input_dim = len(vocab)
embed_dim = 32
num_heads = 4
ff_dim = 64
num_layers = 2
epochs = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransCrypto(input_dim, embed_dim, num_heads, ff_dim, num_layers, maxlen).to(device)
train(model, train_loader, val_loader, epochs, device)
   ```
