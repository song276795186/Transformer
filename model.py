from Transformer.Encoder import *
from Transformer.Decoder import *
from Transformer.Attention_layer import *
from Transformer.EncoderDecoder import *

"""
make a full model.
"""
def make_model(src_vocab, tgt_vocab, N=6, d_model=512,
               d_ff=2048, h=8, dropout=0.1):
    """
    Helper: Construct a mode from hyperparameters.
    :param src_vocab: vocab that need to be embedded
    :param tgt_vocab: vocab that need to be embedded
    :param N: number of encoder in Encoder
    :param d_model: length of the word vector
    :param d_ff: Position-wise feed-forward
    :param h: number of attention head
    :param dropout: dropout ratio
    :return: return a Transformer model
    """

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model
