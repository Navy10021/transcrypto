# 🔐TransCrypto🔐: A Transformer-based Cryptography Model

![TransCrypto Logo](https://yourimageurl.com/logo.png)

TransCrypto is a PyTorch implementation of a Transformer-based model designed for cryptography tasks. This project includes implementations of the Transformer encoder and decoder, along with a custom training loop incorporating early stopping and learning rate scheduling.

## 🚀Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/transcrypto.git
   cd transcrypto
   ```
## 🛠️Usage

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
## 📉Code Overview

### Model Architecture
The TransCrypto model consists of a Transformer encoder and decoder. Here are the key components:

- **TransformerEncoder**: Implements a single layer of the Transformer encoder, which includes multi-head attention and feedforward neural network layers.
  
- **TransformerDecoder**: Implements a single layer of the Transformer decoder, incorporating both self-attention and encoder-decoder attention mechanisms.
  
- **TransCrypto**: Combines multiple encoder and decoder layers to form the complete Transformer-based cryptography model. It also includes an embedding layer and an output layer.

### Training Function
The `train` function handles the training process, including:

- **Forward Pass**: Computes the model predictions for the input data.
  
- **Loss Computation**: Calculates the loss using Binary Cross-Entropy Loss (BCELoss).
  
- **Backward Pass and Optimization**: Performs backpropagation and updates the model parameters using the Adam optimizer.
  
- **Learning Rate Scheduling**: Adjusts the learning rate based on the validation loss using `ReduceLROnPlateau`.
  
- **Early Stopping**: Stops training if the validation loss does not improve for a specified number of epochs (patience).

### Tokenization and Padding
The `tokenize_and_pad` function processes input texts by:

- **Tokenization**: Converts input texts into sequences of tokens using a specified tokenizer.
  
- **Padding**: Pads the token sequences to a specified maximum length to ensure uniform input dimensions for batch processing.

## ✂️License
This project is licensed under the MIT License - see the LICENSE file for details.
