
import numpy as np

def linear_transform(X, W):
    return np.dot(X, W)

def scaled_dot_product_attention(Q, K, V, scale=True):
    scores = np.dot(Q, K.T)
    if scale:
        scores /= np.sqrt(K.shape[1])  # Scale by the square root of embedding size
    attention_weights = softmax(scores)
    return np.dot(attention_weights, V), attention_weights

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)
