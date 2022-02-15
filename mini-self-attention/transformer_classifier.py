import torch
from torch import nn
import torch.nn.functional as F

from transformer import TransformerBlock

import logging
logging.basicConfig(level=logging.DEBUG)

class TransformerClassifier(nn.Module):
    """A simple classifier made up of a series of transformer blocks.
    The token embeddings are loaded using an embedding layer.
    There are two ways of implementing Position embeddings: 
        Simple approach: use some arbitrary vector of size emb_size per each position \
            index <0,1,2...seq_len>.
        Advanced approach: map the position index into a real-valued vector which has some meaning, \
            just like token-embedding vectors.
    In the both the approaches, a given position will always have the exact same position embedding \
        vector values. For eg, position embedding of 0'th position will always be [0.323, 0.123, 0.674,..0.128]

    The simple approach is used in this implementation.
    """
    def __init__(self, emb_size, heads, num_of_blocks, seq_len, vocab_size, num_classes) -> None:
        """Creates a transformer-based classifier model

        Args:
            emb_size (int): Size of the embedding vector (same for both token and positional)
            heads (int): Number of self-attention heads
            num_of_blocks (int): Number of transformer blcoks
            seq_len (int): Number of tokens in an input sequence 
            vocab_size (int): Size of the vocabulary of possible tokens, used in the nn.Embedding() layer
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.vocab_size = vocab_size
        
        # The token & position embeddings
        self.token_embedding_layer = nn.Embedding(self.vocab_size, emb_size)
        self.position_embedding_layer = nn.Embedding(seq_len, emb_size)
        # Note above, nn.Embedding() is a simple lookup table, with learnable parameter in '.weight' tensor.
        #   By default nn.Embedding() initializes the embeddings from a standard Normal distribution.
        #   If required, this layer can be made non-trainable using '.weight.requires_grad = False' and
        #   also if required, we can pre-load the embedding vectors (do a table-lookup 
        #   instead of generating on the fly) using '.load_state_dict({'weight': <lookup table as a tensor>})'

        # The Transformer Classifier will have a series of TransformerBlock instances
        t_blocks = []
        for i in num_of_blocks:
            t_blocks.append(TransformerBlock(emb_size=emb_size, heads=heads))
        # Convert this array of transformer blocks into one Sequential Module
        self.transformer_blocks = nn.Sequential(*t_blocks)

        # Now reduce the output of the final transformer block that produces a vector of size 'emb_size' 
        #   into a linear layer that outputs 'num_classes' logits
        self.output_linear_layer =  nn.Linear(emb_size, num_classes)

    def forward(self, x_batch_input):
        """Apply the transformer-based classifier on a batch of data.

        Note: This implementation uses a simple position-embedding mechanism, which is not fully effective.

        Args:
            x_batch_input (tensor): A tensor of size (batch_size, seq_len), with each row containing \
                a sequence of integers corresponding a sequence of tokens.

        Returns:
            tensor: A tensor of size (batch_size, num_classes) containing the class probabilities
        """
        # Pass the sequence of integers (token IDs) to the token embedding layer and get the embedding vectors
        x_token_embeddings = self.token_embedding_layer(x_batch_input)
        batch_size, seq_len, emb_size =  x_token_embeddings.size()

        # Generate the position embeddings for each of the tokens
        position_indices = torch.arange(seq_len)
        x_position_embeddings = self.position_embedding_layer(\
            position_indices)[None, :, :].expand(batch_size, seq_len, emb_size)
        # Note above, position embedding tensor is made to have same dimensions token \
        #  dimention tensor using below operations:
        #  - The [None, :, :] will insert a new, but empty, dimension at the top. Same as 
        #    doing 'unsqueeze'. This is required to convert the 2-D tensor into a 3-D tensor.
        #  - The expand() duplicates the position embedding vector across the 3 dimensions 
        #    namely the batch dimension, seq_len dimension and the emb_size dimension
        
        # Merge the token & position embeddings
        x_embs = x_token_embeddings + x_position_embeddings

        # Run the series of transformer blocks on the all the sequences of embedding vectors
        x_attention_weighted_embs = self.transformer_blocks(x_embs)
        # which gives the self-attention weighted embeddings (Yes, exactly same dimentions as the input tensor!)

        # Just to be sure
        assert x_embs.size() == x_attention_weighted_embs.size()

        # Now, convert the attention-weighted embeddings into a logit vector of size num_class, by \
        #   1. Average-pool over all the embedding vectors within a sequence of tokens(Eg: within a sentence); this \
        #       gives one vector of size emb_size per one sequence.
        x_avg_emb = x_attention_weighted_embs.mean(dim=1) # Note here, dim=0 is batch, dim=1 is sequence, and dim=2 is embeddings
        #   2. Convert this one vector of size emb_size into class probabilities using a Linear Layer.
        class_logits = self.output_linear_layer(x_avg_emb)

        # Finally convert the logits into class-probabilities using softmax
        class_probs = F.log_softmax(class_logits, dim=1) # Note here, class_logits is a 2-D tensor \
        # where dim=0 is batch and dim=1 is sequence

        # Note above, the difference between softmax() and log_softmax() is that \
        #   log_softmax() penalizes the larger prediction errors more when compared to \
        #   the smaller prediction errors: https://discuss.pytorch.org/t/logsoftmax-vs-softmax/21386 

        return class_probs

        
