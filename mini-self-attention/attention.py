import torch
from torch import nn
import torch.nn.functional as F

import logging
logging.basicConfig(level=logging.DEBUG)

class MultiHeadSelfAttention(nn.Module):
    """Implementation of crude(un-optimized) version of mutli-head self-attention
    Ref: http://peterbloem.nl/blog/transformers

    Essentially, each self-attention head has it's own set of 3 weight matrices: W_query, W_key and W_value
    - Each of these matrices have size (k,k) where k is the embedding size
    - If there are h heads, it is optimal to combine the weight matrices for fast matrix operations
    - So, each of the query weight matrix from h heads is combined into one matrix W_queries of size (k, h*k)
    - Similarly, key weight matrices are combined into W_keys and W_values of size (k, h*k)
    - So, we end up with only 3 weight matrices irrespective of number of heads. Of course the dimension of those
        weight matrices depend on the number of heads h.
    """
    def __init__(self, k, heads=8) -> None:
        """Initialize the multi-head self-attention module.

        Define all the weight matrices required to implement multi-head attention here.

        Args:
            k (int): Embedding dimension size
            heads (int, optional): Number of self-attention heads. Defaults to 8.
        """
        super().__init__()
        self.k, self.heads = k, heads

        # Create the 3 weighted layers for queries, keys and values of all the attention heads
        # Note, we don't use bias for attention weights
        self.layer_weight_keys    = nn.Linear(k, k * heads, bias=False)
        self.layer_weight_queries = nn.Linear(k, k * heads, bias=False)
        self.layer_weight_values  = nn.Linear(k, k * heads, bias=False)

        # Final weighted layer that combines/unifies the outputs of the h different heads into a single k-vector
        # Note, this layers takes in a vector of size (k * heads), from the multi-head attentions and produces
        #   another vector of size (k)
        self.layer_merge_attention_heads = nn.Linear(heads * k, k)

    def forward(self, x_batch_input):
        """Apply the self-attention across all the heads in one go on a batch of input vector sequence 

        Args:
            x_batch_input (3-D tensor): A tensor of shape (batch_size, max_seq_size, embedding_size)

        Returns:
            3-D tensor: A modified tensor after applying the self-attention weights to the input tensor
        """
        batch_size, seq_size, emb_size = x_batch_input.size()
        assert emb_size == self.k # They should be same

        # First Step: compute the queries, keys and values by applying the corresponding weights
        queries = self.layer_weight_queries(x_batch_input).view(batch_size, seq_size, self.heads, emb_size)
        keys = self.layer_weight_keys(x_batch_input).view(batch_size, seq_size, self.heads, emb_size)
        values = self.layer_weight_values(x_batch_input).view(batch_size, seq_size, self.heads, emb_size)
        # Note: When input (b, t, k) is passed to the queries layer (or any other two layers), 
        #   they produce an output of shape (b, t, k*heads), so we reshape this into (b, t, heads, k).
        # That means, for each embeding vector of a word, we produce h different vectors (individually
        #  for query, key and value) of same size k.

        # Second Step: we multiply the query, key and values using dot products
        # Since these dot products need to be repeated for h times for each head, it is better to do it once.
        # We achive this by merging the 'heads' dimension with the 'batch' dimension:
        keys = keys.transpose(1, 2).contiguous().view(batch_size * self.heads, seq_size, emb_size)
        queries = queries.transpose(1, 2).contiguous().view(batch_size * self.heads, seq_size, emb_size)
        values = values.transpose(1, 2).contiguous().view(batch_size * self.heads, seq_size, emb_size)
        # Note above: transpose() brings the 'heads' dimension next to the 'batch' dimension
        #             continguous() ensures that the order of memory locations & structure of tensor are aligned, mostly for efficiency
        #             view() merges the 'heads' dimension with the 'batch' dimension
        # Now, we can multiply these queries, keys and values and the operation will be applied accross all the heads

        # We scale the result matrix of dot product (queries . keys) by deviding by 'sqrt(emb_size)'
        # But, instead of scaling the result after dot product, if we scale the 'queries' and 'keys' before
        #   the dot product, it reduces the memory consumption further. But note that, we do it twice hence we do
        #   quad root twice, which is same as doing square root once
        queries = queries / (emb_size ** (1/4)) # scale using the 'quad root of emb_size'
        keys = keys / (emb_size ** (1/4)) # scale using the 'quad root of emb_size'

        # Then the dot product, within a batch
        self_attention = torch.bmm(queries, keys.transpose(1, 2))
        # Note above, the result of dot product is a tensor of size (batch_size * heads, seq_size, seq_size)

        # Third Step: apply the softmax on row-wise
        self_attention = F.softmax(self_attention, dim=2)

        # Fourth Step: apply the self_attention on the values
        output = torch.bmm(self_attention, values).view(batch_size, self.heads, seq_size, emb_size)
        # This contains the output of all the attention heads, so we separate the 'batch' and 'head' dimensions

        # Fifth Step: Finally merge the outputs from all the attention heads into one vector. 
        # 
        # In order to merge all the self.heads attention heads into one k-dimentional vector,
        #       we use the 'self.layer_merge_attention_heads' layer. This layer expects data to 
        #       have a size of (self.heads * emb_size). We reshape the 'output' to fit into this dimension as follows:
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_size, self.heads * emb_size)
        # Note above, 1 corresponds to self.heads and 2 corresponds to seq_size. By transpose() we bring the
        #   self.heads and emb_size dimensions together. Then we merge those two dimensions by reshaping the tensor.

        # Now, we are ready to merge the outputs from all the attention heads
        result = self.layer_merge_attention_heads(output)

        # The result will have "multi-head, scaled dot-product self-attention"
        return result
        







import deprecation
@deprecation.deprecated(details="This is just a basic intro to self-attention")
def basic_self_attention(x_batched_input):
    """Assume x_batched_input is a 3-dimensional batched input matrix with shape (b, t, k), where
        b -> batch size
        t -> number of input vectors (eg: one vector per word)
        k -> size of each vector (eg: word embedding dimension)
    
    The self-attention opeartion takes in a series of t vectors and produces another series of t vectors.
    This operation involves a series of dot products:
        1. First, get a W matrix of shape (t,t), where w_i_j indicates the dot product (x_i . x_j), 
            which results in a scalar value
        2. Then, to ensure the W matrix has standardized values (instead of -infinity to +infinity), we apply softmax: so that
            the values are between 0-1 and values of each row vector add upto 1. 
        3. Finally, for every input vector x_i, we get the output vector y_i using summation, over all j, of weighted product
            between the W_i_j and vector x_i

    Args:
        x_batched_input (3-dim Tensor): 3-dimensional batched input matrix with shape (b, t, k) with b batches, t vectors, each of size k

    Returns:
        (3-dim Tensor): A matrix of exactly same shape as input but with values modified by self-attention
    """

    # Find the weight matrix W_i_j (with dimension (t,t) using matrix multiplication within a batch of 2-D matrices
    w_matrix = torch.bmm(x_batched_input, x_batched_input.transpose(1, 2)) 
    # Note above, the transpose dimensions here (the 0th dimension is ignored).
    logging.debug(f'Shape of w_matrix: {w_matrix.size()}')

    # Softmax along each row
    w_softmaxed = F.softmax(w_matrix, dim=2)

    # Transform the sequence of vectors using weight matrix w_softmaxed
    y_batched = torch.bmm(w_softmaxed, x_batched_input)
    # Note above: batch multiplication of (b, t, t) sized weight matrix with (b, t, k) sized input matrix
    # The result matrix y will have (b, t, k) shape

    return y_batched