import torch
from model.img2tex import Encoder, DecoderWithAttention

encoder = Encoder()
decoder = DecoderWithAttention(attention_dim=128, embed_dim=30, decoder_dim=128, vocab_size=10)

if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 64, 64)
    output_tensor = encoder.forward(input_tensor)
    print(output_tensor.shape)
    print(decoder.forward(output_tensor, torch.tensor([3, 2]), torch.tensor([1])))
