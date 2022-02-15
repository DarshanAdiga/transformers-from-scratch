import torch
from torch import nn
import torch.nn.functional as F

from attention import MultiHeadSelfAttention

import logging
logging.basicConfig(level=logging.DEBUG)

class TransformerBlock(nn.Module):
    """A singal transformer block containing:
        - a multi-head self-attention layer
            - a residual connection to normalization layer by skipping the self-attention layer
        - layer normalization (at the embedding dimension)
        - a feed-forward layer (a singal MLP applied on each of the vectors)
            - a residual connection to normalization layer by skipping the fee-forward layer
        - layer normalization (at the embedding dimension)
    
    Normalization & Residual connections, in general, help the deep neural networks to train faster 
    and more accurately.

    Layer normalization (vs Batch normalization):
        Normalization: Calculate mean & variance, subtract the "activations" by mean and then devide 
            by standard deviation.
        
        Layer normalization normalizes input across the features instead of normalizing input features 
        across the batch dimension in batch normalization.
    """

    def __init__(self, emb_size, heads, ff_layer_factor=4) -> None:
        """Create a complete transformer block with a multi-head self-attention layer.

        Args:
            emb_size (int): Embedding vector size
            heads (int): Number of self-attention heads to be used
            ff_layer_factor (int, optional): Factor to be used while setting the number of hidden \
                units of MLP layer. Defaults to 4.
        """
        super().__init__()

        # A multi-head self-attention layer 
        self.multihead_attention = MultiHeadSelfAttention(emb_size, heads=heads)

        # First layer normalization which takes an input vector of size emb_size and produces 
        #   output vector of size emb_size
        self.norm_layer1 = nn.LayerNorm(emb_size)

        # An MLP layer
        self.feed_forward = nn.Sequential(
            # Create a layer with (ff_layer_factor * emb_size) hidden unites
            nn.Linear(emb_size, ff_layer_factor * emb_size),
            nn.ReLU(),
            # Downsize the dimension from (ff_layer_factor * emb_size) to a vector of size emb_size
            nn.Linear(ff_layer_factor * emb_size, emb_size)
        )

        # A second layer normalization which takes an input vector of size emb_size and produces 
        #   output vector of size emb_size
        self.norm_layer2 = nn.LayerNorm(emb_size)


    def forward(self, x_batch_input):
        """Apply the tranformer block on the input batch data tensor and return the tensor of same size  

        Args:
            x_batch_input (3-D tensor): A tensor of shape (batch_size, seq_size, embedding_size)

        Returns:
            tensor: Output activations of a transformer block, will be of size (batch_size, seq_size, embedding_size)
        """
        # Attention layer
        attention = self.multihead_attention(x_batch_input)

        # First layer-normalization with residual connection
        x_normalized_1 = self.norm_layer1(attention + x_batch_input)
        # Note Above: The addition of input to the attention activations is the residual connection!

        # FeedForward Layer
        feedforward = self.feed_forward(x_normalized_1)

        # Second layer-normalization with residual connection
        x_normalized_2 = self.norm_layer2(feedforward + x_normalized_1)
        # Note Above: The addition of x_normalized_1 to the feedforward activations is the residual connection!

        # Return the activations of the multi-head self-attention transformer block
        return x_normalized_2

