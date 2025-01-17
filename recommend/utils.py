import numpy as np

def calculate_time_decay_weights(num_queries, decay_rate=0.5):
    """
    Calculate exponential decay weights for a list of queries.
    
    Args:
    - num_queries: total number of queries (latest query at index num_queries - 1)
    - decay_rate: controls how fast older queries lose importance (higher = faster decay)
    
    Returns:
    - weights: normalized time decay weights
    """
    T = num_queries - 1
    weights = np.array([np.exp(-decay_rate * (T - t)) for t in range(num_queries)])
    normalized_weights = weights / np.sum(weights)
    return normalized_weights

def calculate_weighted_user_embedding(query_embeddings, weights):
    """
    Compute the user embedding as a weighted combination of query embeddings.
    
    Args:
    - query_embeddings: list of query embeddings (shape: [num_queries, embedding_dim])
    - weights: list of weights (shape: [num_queries,])
    
    Returns:
    - user_embedding: aggregated user embedding (shape: [embedding_dim,])
    """
    weighted_embeddings = np.array([w * e for w, e in zip(weights, query_embeddings)])
    user_embedding = np.sum(weighted_embeddings, axis=0)
    return user_embedding


def remove_duplicates_keep_last(lst):
    seen = set()
    result = []
    for item in reversed(lst):
        if item not in seen:
            seen.add(item)
            result.append(item)
    return list(reversed(result)) 

