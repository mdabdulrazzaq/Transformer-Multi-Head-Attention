
import numpy as np
from attention_mechanism import linear_transform, scaled_dot_product_attention

# Input embeddings
X = np.array([
    [1.0, 0.5, 0.2],
    [0.8, 0.3, 0.7],
    [0.6, 0.9, 0.4]
])

# Trainable weight matrices
W_q = np.array([[0.5, 0.2], [0.1, 0.8], [0.3, 0.6]])
W_k = np.array([[0.6, 0.4], [0.2, 0.9], [0.7, 0.5]])
W_v = np.array([[0.8, 0.1], [0.3, 0.6], [0.4, 0.5]])

# Linear transformations
Q = linear_transform(X, W_q)
K = linear_transform(X, W_k)
V = linear_transform(X, W_v)

# Compute attention
contextual_embeddings, attention_weights = scaled_dot_product_attention(Q, K, V)

print("Contextualized Embeddings:")
print(contextual_embeddings)
print("Attention Weights:")
print(attention_weights)
