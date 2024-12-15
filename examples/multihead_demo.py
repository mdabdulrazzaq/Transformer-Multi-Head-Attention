
import numpy as np
from attention_mechanism import linear_transform, scaled_dot_product_attention

def multihead_attention(X, weights, num_heads):
    head_results = []
    attention_maps = []
    for head in range(num_heads):
        W_q, W_k, W_v = weights[head]
        Q = linear_transform(X, W_q)
        K = linear_transform(X, W_k)
        V = linear_transform(X, W_v)
        context, attention = scaled_dot_product_attention(Q, K, V)
        head_results.append(context)
        attention_maps.append(attention)
    return np.concatenate(head_results, axis=-1), attention_maps

# Input embeddings
X = np.array([
    [1.0, 0.5, 0.2],
    [0.8, 0.3, 0.7],
    [0.6, 0.9, 0.4]
])

# Multi-head weights
num_heads = 2
weights = [
    (
        np.random.rand(3, 2),  # W_q
        np.random.rand(3, 2),  # W_k
        np.random.rand(3, 2)   # W_v
    )
    for _ in range(num_heads)
]

# Compute multi-head attention
contextual_embeddings, attention_maps = multihead_attention(X, weights, num_heads)

print("Contextualized Embeddings (Multi-Head):")
print(contextual_embeddings)
