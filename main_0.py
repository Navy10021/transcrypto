from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from model.transcrypto import * 
from train.train import *

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define some sample text data
    texts = ["hello world", "transformer model for encryption", "test text data"]

    # Build the vocabulary and tokenizer
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, texts), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])

    # Convert texts to tensor format
    maxlen = 32
    data = tokenize_and_pad(texts, vocab, tokenizer, maxlen)

    # Create dataset and dataloader
    dataset = TensorDataset(data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

    # Define model parameters
    input_dim = len(vocab)
    embed_dim = 64
    num_heads = 4
    ff_dim = 128
    num_layers = 2

    # Create and train the transformer-based crypto model
    transformer_crypto = TransCrypto(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers, maxlen=maxlen).to(device)
    train(transformer_crypto, train_loader, val_loader, epochs=50, device=device)

    # Load the best model
    transformer_crypto.load_state_dict(torch.load('best_model.pth'))

    # Test the encryption and decryption
    test_texts = ["hello world", "new test text"]
    test_data = tokenize_and_pad(test_texts, vocab, tokenizer, maxlen).to(device)
    transformer_crypto.eval()
    with torch.no_grad():
        encrypted_data = transformer_crypto(test_data)
        encrypted_data_int = torch.round(encrypted_data).argmax(dim=-1).to(torch.long).to(device)
        decrypted_data = transformer_crypto(encrypted_data_int)

    # Convert decrypted data back to text
    def tensor_to_text(tensor, vocab):
        reverse_vocab = {idx: token for token, idx in vocab.get_stoi().items()}
        return [' '.join([reverse_vocab[idx.item()] for idx in seq if idx != vocab['<pad>']]) for seq in tensor]

    decrypted_texts = tensor_to_text(decrypted_data.argmax(dim=-1), vocab)

    print(">> Original Texts:", test_texts, end = "\n\n")
    print(">> Encrypted Data:", encrypted_data, end = "\n\n")
    print(">> Decrypted Texts:", decrypted_texts, end = "\n\n")