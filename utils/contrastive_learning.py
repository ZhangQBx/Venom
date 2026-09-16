import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils.clustering import adjust_learning_rate
import hnswlib

class ContrastiveEncoder(nn.Module):
    """
    SimCLR-based contrastive learning encoder.
    f(): MLP with linear layers.
    g(): MLP with ReLU activations.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, projection_dim=128, encoder_layers=2):
        super(ContrastiveEncoder, self).__init__()
        
        # f(): MLP with linear layers
        if encoder_layers not in (2, 3):
            raise ValueError("contrastive_encoder_layers must be 2 or 3")
        layers = [nn.Linear(input_dim, hidden_dim)]
        if encoder_layers == 3:
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.encoder = nn.Sequential(*layers)
        
        # g(): MLP with ReLU activations
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x):
        # Pass through encoder f()
        encoded = self.encoder(x)
        # Pass through projection head g()
        projected = self.projection_head(encoded)
        return projected
    
    def encode_only(self, x):
        """
        Use only encoder f(), bypassing projection head g().
        Following SimCLR, discard projection head g() after training.
        """
        return self.encoder(x)


# def contrastive_loss(projections, temperature):
#     """
#     Contrastive loss (NT-Xent), following the SimCLR definition
#     Paper formula: l_i,j = -log(exp(sim(z_i, z_j)/tau) / sum_{k=1}^{2N} 1_[k!=i] exp(sim(z_i, z_k)/tau))
#
#     Explanation:
#     1. There are 2N embeddings: N original and N augmented.
#     2. Positive pairs (i, i+N) and (i+N, i) link each original embedding to its augmented version.
#     3. Negatives are the other 2(N-1) embeddings in the batch.
#     4. For each positive pair (i, j), compute the loss using:
#        - Numerator: exp(sim(z_i, z_j)/tau), the positive-pair similarity.
#        - Denominator: sum exp(sim(z_i, z_k)/tau) for k!=i, over all other samples.
#
#     :param projections: Projected features [2N, projection_dim], where N is the original batch size
#     :param temperature: Temperature parameter tau
#     :return: Contrastive loss
#     """
#     batch_size = projections.shape[0]
#     device = projections.device
#     N = batch_size // 2  # Original batch size
#
#     # Apply L2 normalization
#     projections = F.normalize(projections, p=2, dim=1)
#
#     # Compute the cosine similarity matrix
#     # After L2 normalization, cosine similarity equals the dot product
#     similarity_matrix = torch.matmul(projections, projections.T) / temperature
#
#     # Create the positive-pair mask
#     # Positive pairs: (i, i+N) and (i+N, i)
#     positive_mask = torch.zeros(batch_size, batch_size, device=device)
#     for i in range(N):
#         positive_mask[i, i + N] = 1  # (i, i+N)
#         positive_mask[i + N, i] = 1  # (i+N, i)
#
#     # Compute the loss
#     total_loss = 0
#     for i in range(batch_size):
#         # Find the positive sample j for i
#         positive_indices = torch.where(positive_mask[i] == 1)[0]
#
#         for j in positive_indices:
#             # Numerator: exp(sim(z_i, z_j)/tau)
#             numerator = torch.exp(similarity_matrix[i, j])
#
#             # Denominator: sum_{k=1}^{2N} 1_[k!=i] exp(sim(z_i, z_k)/tau)
#             # Exclude i itself and compute similarities to all other samples
#             mask = torch.ones(batch_size, device=device)
#             mask[i] = 0  # Exclude i itself
#             denominator = torch.sum(mask * torch.exp(similarity_matrix[i, :]))
#
#             # Compute l_i,j = -log(numerator / denominator)
#             loss_ij = -torch.log(numerator / denominator)
#             total_loss += loss_ij
#
#     # Average the loss over all positive pairs, as in the paper
#     num_positive_pairs = torch.sum(positive_mask) / 2  # Each positive pair is counted twice
#     return total_loss / num_positive_pairs


def contrastive_loss(projections, temperature):
    batch_size = projections.shape[0]
    device = projections.device
    N = batch_size // 2

    projections = F.normalize(projections, p=2, dim=1)
    similarity_matrix = torch.matmul(projections, projections.T) / temperature
    mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)], dim=0).to(device)
    loss = F.cross_entropy(similarity_matrix, labels, reduction='mean')

    return loss



def augment_batch_embeddings(batch_embeddings, cfg):
    """
    Augment the batch embeddings to produce 2N augmented embeddings.
    Following SimCLR, generate an augmented version of each original embedding.
    
    :param batch_embeddings: Original batch embeddings [N, embedding_dim]
    :param cfg: Configuration parameters
    :return: Augmented embeddings [2N, embedding_dim]
    """
    N = batch_embeddings.shape[0]
    device = batch_embeddings.device
    
    # Method 1: Add Gaussian noise to create noise embeddings
    # Add small random Gaussian noise to each embedding dimension
    noise_mean = cfg.contrastive_noise_mean
    noise_std = cfg.contrastive_noise_std
    noise = torch.normal(mean=noise_mean, std=noise_std,
                         size=batch_embeddings.shape, device=batch_embeddings.device)
    noise_embeddings = batch_embeddings + noise
    
    # Method 2: Apply dropout to create dropout embeddings
    # Set randomly selected embedding dimensions to zero
    dropout = nn.Dropout(p=cfg.contrastive_dropout_rate)
    dropout_embeddings = dropout(batch_embeddings)
    # dropout_rate = cfg.contrastive_dropout_rate
    # dropout_mask = torch.rand_like(batch_embeddings) > dropout_rate
    # dropout_embeddings = batch_embeddings * dropout_mask.float()
    
    # Concatenate noise and dropout embeddings: [N, dim] + [N, dim] = [2N, dim]
    combined_embeddings = torch.cat([noise_embeddings, dropout_embeddings], dim=0)
    
    return combined_embeddings


class ContrastiveDataset(tud.Dataset):
    """
    Contrastive learning dataset storing only original embeddings.
    Apply augmentation dynamically during training.
    """
    def __init__(self, original_embeddings):
        self.original_embeddings = original_embeddings
    
    def __len__(self):
        return len(self.original_embeddings)
    
    def __getitem__(self, idx):
        return self.original_embeddings[idx]


def train_contrastive_encoder(cfg, client, embeddings):
    """
    Train the contrastive learning encoder.
    Follow the SimCLR workflow:
    1. Collect original embeddings into a dataset.
    2. Augment each training batch to obtain 2N augmented embeddings.
    3. Apply augmentation in the first epoch and reuse those embeddings in later epochs.
    
    :param cfg: Configuration parameters
    :param client: Client object
    :param embeddings: Original embeddings from the bottom model
    :return: Trained encoder
    """
    print("Step 2: Training contrastive encoder")
    
    # Create the encoder
    input_dim = embeddings.shape[1]
    hidden_dim = cfg.contrastive_hidden_dim
    output_dim = cfg.contrastive_output_dim
    projection_dim = cfg.contrastive_projection_dim
    
    encoder = ContrastiveEncoder(input_dim, hidden_dim, output_dim, projection_dim,
                                 cfg.get("contrastive_encoder_layers", 2)).to(client.device)
    encoder.train()
    
    # Create a dataset containing only original embeddings
    contrastive_dataset = ContrastiveDataset(embeddings)
    contrastive_loader = tud.DataLoader(contrastive_dataset, batch_size=cfg.bs_contrastive, shuffle=False)
    
    # Optimizer
    optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[cfg.get("optimizer_contrastive", "AdamW")]
    optimizer = optimizer_class(encoder.parameters(), lr=cfg.lr_contrastive)
    
    # Store augmented embeddings for each batch to avoid repeated computation
    batch_augmented_cache = {}
    
    # Training loop
    epochs = cfg.contrastive_epochs
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, batch_embeddings in enumerate(contrastive_loader):
            batch_embeddings = batch_embeddings.to(client.device)
            
            # # Augment only in the first epoch and reuse the results in later epochs
            # if epoch == 0:
            #     # Augment the batch to obtain 2N embeddings
            #     combined_embeddings = augment_batch_embeddings(batch_embeddings, cfg)
            #     # Cache the augmented embeddings
            #     batch_augmented_cache[batch_idx] = combined_embeddings.clone()
            # else:
            #     # Reuse augmented embeddings from the first epoch
            #     combined_embeddings = batch_augmented_cache[batch_idx]

            # Remove caching and apply random augmentation at every step
            combined_embeddings = augment_batch_embeddings(batch_embeddings, cfg)
            
            # Forward pass
            projections = encoder(combined_embeddings)
            
            # Compute the contrastive loss
            loss = contrastive_loss(projections, cfg.contrastive_temp)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        adjust_learning_rate(epoch, cfg.lr_contrastive, optimizer)
        if epoch % 10 == 0:
            print(f">>> Contrastive training epoch {epoch}, loss: {epoch_loss:.4f}")
    
    print(">>> Contrastive encoder training completed")
    return encoder


def generate_contrastive_embeddings(encoder, embeddings, client):
    """
    Generate contrastive embeddings with the trained encoder.
    Following SimCLR, discard projection head g() after training and use only encoder f().
    :param encoder: Trained encoder
    :param embeddings: Original embeddings
    :return: Contrastive embeddings from encoder f()
    """
    encoder.eval()
    device = client.device
    with torch.no_grad():
        # Use only encoder f() and discard projection head g()
        contrastive_embeddings = encoder.encode_only(embeddings.to(device))
        contrastive_embeddings = contrastive_embeddings.cpu()
    
    print(f">>> Generated contrastive embeddings with shape: {contrastive_embeddings.shape}")
    print(">>> Note: Using encoder f() output only, projection head g() discarded as per SimCLR paper")

    # if client.rank == 0:
    #     print(embeddings.shape, contrastive_embeddings.shape)
    #     print(embeddings[:2])
    #     print(contrastive_embeddings[:2])

    return contrastive_embeddings


# def find_knn_brute_force(contrastive_embeddings, cfg):
#     """
#     Find the KNN of each contrastive embedding by brute-force search
#
#     :param contrastive_embeddings: contrastive embeddings [N, embedding_dim]
#     :param cfg: Configuration parameters
#     :return: knn_indices, knn_distances - KNN indices and distances for each embedding
#     """
#     print("Step 3: Finding KNN using brute force search")
#
#     # Get parameters
#     k = cfg.knn_k  # Number of nearest neighbors
#     num_embeddings = contrastive_embeddings.shape[0]
#
#     # Convert to a NumPy array
#     embeddings_np = contrastive_embeddings.cpu().numpy()
#
#     print(f">>> Computing pairwise distances for {num_embeddings} embeddings...")
#
#     # Compute all pairwise Euclidean distances
#     # Use broadcasting to compute the distance matrix
#     distances = np.sqrt(((embeddings_np[:, np.newaxis, :] - embeddings_np[np.newaxis, :, :]) ** 2).sum(axis=2))
#
#     print(f">>> Distance matrix shape: {distances.shape}")
#
#     sorted_indices = np.argsort(distances, axis=1)
#     # Find the K+1 nearest neighbors of each embedding, including itself
#     # knn_indices = np.argsort(distances, axis=1)[:, :k+1]  # Select the first k+1 sorted indices
#     # knn_distances = np.sort(distances, axis=1)[:, :k+1]   # Corresponding distances
#     knn_indices = sorted_indices[:, :k + 1]
#     knn_distances = np.take_along_axis(distances, knn_indices, axis=1)
#
#     # Remove the query point itself, which is the first neighbor
#     knn_indices = knn_indices[:, 1:]  # Remove the first column (self)
#     knn_distances = knn_distances[:, 1:]  # Remove the first column (self)
#
#     farthest_indices_sorted = sorted_indices[:, -k:]
#     farthest_indices = np.fliplr(farthest_indices_sorted).copy()
#     farthest_distances = np.take_along_axis(distances, farthest_indices, axis=1).copy()
#
#     print(f">>> Found nearest and farthest KNN for {num_embeddings} embeddings.")
#     # print(f">>> KNN indices shape: {knn_indices.shape}")
#     # print(f">>> KNN distances shape: {knn_distances.shape}")
#
#
#     return knn_indices, knn_distances, farthest_indices, farthest_distances

def resolve_knn_k(cfg, num_embeddings):
    """Resolve a fixed K or the paper's fraction of actual auxiliary rows."""
    if num_embeddings < 2:
        raise ValueError("Neighbor search requires at least two auxiliary rows")
    k = cfg.get("knn_k")
    if k is None:
        ratio = float(cfg.get("knn_ratio", 0.1))
        if not np.isfinite(ratio) or not 0 < ratio < 1:
            raise ValueError("knn_ratio must be between zero and one")
        k = max(1, int(num_embeddings * ratio))
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k < num_embeddings:
        raise ValueError("knn_k must be an integer smaller than the auxiliary set")
    return k


def find_knn_hnsw(embeddings, cfg):
    """
    Perform fast approximate nearest-neighbor (ANN) search using HNSW (hnswlib).
    Replace the O(N^2) brute-force method, especially for large datasets such as SUSY.

    Approximate KFN (farthest neighbors) by efficient random sampling that excludes nearest neighbors.
    """
    print(f"Running HNSW KNN (Fast O(N log N)) for {embeddings.shape[0]} items...")

    # 1. Prepare the data
    embeddings_np = embeddings.cpu().numpy()  # HNSW requires NumPy arrays
    num_samples, dim = embeddings_np.shape
    k = resolve_knn_k(cfg, num_samples)

    # 2. Initialize the HNSW index
    # The paper (VFL_Attack.pdf) uses cosine similarity (Eqs. 3, 7, 8)
    p = hnswlib.Index(space='cosine', dim=dim)

    # M and ef_construction are key index construction parameters
    # Higher ef_construction improves accuracy but slows index construction
    p.init_index(max_elements=num_samples, ef_construction=200, M=16)

    # 3. Build the index with multiple threads
    print("Building HNSW index...")
    p.add_items(embeddings_np, num_threads=-1)

    # 4. Query KNN (K-Nearest Neighbors)
    print("Querying KNN...")
    # Higher ef improves query accuracy but slows queries
    p.set_ef(max(k * 2, 50))

    # Query k+1 neighbors because the first result is the query point itself
    # The returned indices and distances are NumPy arrays
    knn_indices, knn_distances = p.knn_query(embeddings_np, k=k + 1, num_threads=-1)

    # Remove the query point itself (the first column)
    knn_indices = knn_indices[:, 1:]
    # knn_distances = knn_distances[:, 1:] # Retain distances if needed

    print("Querying KNN finished.")

    # 5. Approximate KFN (K-Farthest Neighbors) by random sampling
    print("Querying KFN (via random sampling approximation)...")

    farthest_indices = np.zeros((num_samples, k), dtype=np.int64)
    all_indices_set = set(range(num_samples))

    for i in tqdm(range(num_samples), desc="Sampling KFN"):
        # Exclude the query point and all of its KNN
        exclude_set = set(knn_indices[i])
        exclude_set.add(i)

        # Candidate pool = all points minus excluded points
        candidate_pool = list(all_indices_set - exclude_set)

        # Allow sampling with replacement when the candidate pool is too small
        replace_needed = len(candidate_pool) < k
        if replace_needed:
            # If the candidate pool is empty, fall back to sampling from all points
            if len(candidate_pool) == 0:
                candidate_pool = list(all_indices_set - {i})
            replace_needed = len(candidate_pool) < k  # Check again

        farthest_indices[i] = np.random.choice(candidate_pool, k, replace=replace_needed)

    print("Querying KFN finished.")

    # Return a format compatible with brute_force
    return knn_indices, None, farthest_indices, None




def find_knn_brute_force(contrastive_embeddings, cfg):
    """
    Find the k nearest and k farthest neighbors of each embedding using brute-force cosine similarity.

    :param contrastive_embeddings: Contrastive embeddings [N, embedding_dim] (PyTorch tensor)
    :param cfg: Configuration parameters
    :return: (knn_indices, knn_distances, farthest_indices, farthest_distances)
    """
    print("Step 3: Finding KNN using brute force search with COSINE SIMILARITY.")
    k = resolve_knn_k(cfg, len(contrastive_embeddings))

    # Perform NumPy operations on the CPU
    embeddings_np = contrastive_embeddings.cpu().numpy()
    num_embeddings = embeddings_np.shape[0]

    # 1. Apply L2 normalization
    norm = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    normalized_embeddings = embeddings_np / norm

    # 2. Compute the cosine similarity matrix
    # A @ B.T equals cosine similarity when both A and B are normalized
    print(f">>> Computing cosine similarity matrix for {num_embeddings} embeddings...")
    st_time = time.time()
    # similarity_matrix = normalized_embeddings[0] @ normalized_embeddings.T
    similarity_matrix = normalized_embeddings @ normalized_embeddings.T
    end_time = time.time()
    print("Time taken: {}".format(end_time - st_time))


    # 3. Convert to cosine distance (1 - similarity), since KNN searches for minimum distances
    # Clip the range to avoid floating-point precision issues
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
    distance_matrix = 1.0 - similarity_matrix

    # 4. Sort to find the nearest and farthest neighbors
    sorted_indices = np.argsort(distance_matrix, axis=1)

    # Extract the k nearest neighbors, excluding self
    # The first sorted column is self (distance 0); take k entries starting from the second
    knn_indices = sorted_indices[:, 1:k + 1]
    knn_distances = np.take_along_axis(distance_matrix, knn_indices, axis=1)

    # Extract the k farthest neighbors (largest distances)
    farthest_indices = sorted_indices[:, -k:]
    # Reverse the order so the farthest neighbor comes first
    farthest_indices = np.fliplr(farthest_indices).copy()
    farthest_distances = np.take_along_axis(distance_matrix, farthest_indices, axis=1)

    print(f">>> Found nearest and farthest neighbors for {num_embeddings} embeddings.")

    return knn_indices, knn_distances, farthest_indices, farthest_distances


import numpy as np
import torch


def find_knn_random_sampling(contrastive_embeddings, cfg):
    """
    Approximate each embedding's k nearest and k farthest neighbors using random sampling and cosine similarity.

    Match the find_knn_brute_force interface with a faster, approximate method.

    :param contrastive_embeddings: Contrastive embeddings [N, embedding_dim] (PyTorch tensor)
    :param cfg: Configuration parameters; cfg.knn_k is required.
                Optionally specify cfg.random_sample_m (sample size) to override the default.
    :return: (knn_indices, knn_distances, farthest_indices, farthest_distances)
             (These neighbors are approximate.)
    """
    print("Step 3: Finding approx KNN/KFN using RANDOM SAMPLING with COSINE SIMILARITY.")
    k = resolve_knn_k(cfg, len(contrastive_embeddings))
    num_embeddings = contrastive_embeddings.shape[0]

    # --- L2 normalization, consistent with the brute-force version ---
    # Perform NumPy operations on the CPU
    embeddings_tensor = contrastive_embeddings.cpu()

    # 1. Apply L2 normalization using PyTorch normalize
    normalized_embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
    embeddings_np = normalized_embeddings_tensor.numpy()
    print(f">>> Normalized {num_embeddings} embeddings.")

    # --- Random sampling parameters ---
    # Default sample size M: min(100 * k, N-1)
    # default_m = min(100 * k, num_embeddings - 1)
    # Allow cfg.random_sample_m to override the default
    m = k * 2

    # --- Validate M ---
    if num_embeddings <= 1:
        print("Warning: Only one embedding found. Cannot find neighbors.")
        empty_shape = (num_embeddings, k)
        return (np.zeros(empty_shape, dtype=np.int64), np.zeros(empty_shape, dtype=np.float32),
                np.zeros(empty_shape, dtype=np.int64), np.zeros(empty_shape, dtype=np.float32))

    if m >= num_embeddings:
        print(
            f"Warning: random_sample_m ({m}) is >= num_embeddings ({num_embeddings}). Clamping to {num_embeddings - 1}.")
        m = num_embeddings - 1
    if m < k:
        print(f"Warning: random_sample_m ({m}) is less than k ({k}). KNN/KFN will be unreliable. Setting m = k.")
        m = k
        if m >= num_embeddings:
            m = num_embeddings - 1  # Edge case: k > N

    print(f">>> Using k={k} and random_sample_m={m}")

    # --- Prepare outputs ---
    knn_indices = np.zeros((num_embeddings, k), dtype=np.int64)
    knn_distances = np.zeros((num_embeddings, k), dtype=np.float32)
    farthest_indices = np.zeros((num_embeddings, k), dtype=np.int64)
    farthest_distances = np.zeros((num_embeddings, k), dtype=np.float32)

    # Array [0, 1, ..., N-2] used for sampling
    indices_pool = np.arange(num_embeddings - 1)

    print(f">>> Starting random sampling loop for {num_embeddings} embeddings...")

    # --- Main loop ---
    for i in range(num_embeddings):
        query_vec = embeddings_np[i]

        # 1. Sample m indices from [0, ..., N-2] without replacement
        sample_idx_raw = np.random.choice(indices_pool, m, replace=False)

        # 2. Remap indices to skip i
        #    If a sampled index j >= i, its actual index is j+1.
        #    This samples from all N-1 indices except i in O(M) time.
        sample_indices = np.where(sample_idx_raw >= i, sample_idx_raw + 1, sample_idx_raw)

        # 3. Retrieve sampled embeddings
        sample_embeddings = embeddings_np[sample_indices]  # [m, dim]

        # 4. Compute cosine similarities using the normalized vectors
        # (m, dim) @ (dim,) -> (m,)
        similarities = np.dot(sample_embeddings, query_vec)

        # 5. Convert to cosine distances
        similarities = np.clip(similarities, -1.0, 1.0)
        distances = 1.0 - similarities  # [m,]

        # 6. Sort the m distances
        # argsort returns indices in [0, ..., m-1]
        sorted_local_indices = np.argsort(distances)

        # 7. Extract KNN from the same sample
        # Select the k smallest distances
        knn_local_idx = sorted_local_indices[:k]
        # Map back to original indices
        knn_indices[i] = sample_indices[knn_local_idx]
        knn_distances[i] = distances[knn_local_idx]

        # 8. Extract KFN from the same sample
        # Select the k largest distances
        kfn_local_idx = sorted_local_indices[-k:]
        # Reverse the order so the farthest neighbor comes first
        kfn_local_idx = np.flip(kfn_local_idx)

        farthest_indices[i] = sample_indices[kfn_local_idx]
        farthest_distances[i] = distances[kfn_local_idx]

        if (i + 1) % 5000 == 0 or (i + 1) == num_embeddings:
            print(f"    ... processed {i + 1} / {num_embeddings}")

    print(f">>> Found approx nearest and farthest neighbors for {num_embeddings} embeddings.")

    return knn_indices, knn_distances, farthest_indices, farthest_distances


def find_knn_sklearn(contrastive_embeddings, cfg):
    """
    Find each contrastive embedding's KNN using sklearn.

    :param contrastive_embeddings: contrastive embeddings [N, embedding_dim]
    :param k: Number of nearest neighbors
    :return: knn_indices, knn_distances
    """
    print("Step 3: Finding KNN using sklearn.neighbors.NearestNeighbors")

    k = resolve_knn_k(cfg, len(contrastive_embeddings))
    nn_model = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')

    # Fit the data, primarily to build the index
    nn_model.fit(contrastive_embeddings)
    knn_distances, knn_indices = nn_model.kneighbors(contrastive_embeddings, return_distance=True)

    print(f">>> Found KNN for {contrastive_embeddings.shape[0]} embeddings")
    print(f">>> KNN indices shape: {knn_indices.shape}")
    print(f">>> KNN distances shape: {knn_distances.shape}")

    return knn_indices, knn_distances


# def store_knn_for_each_embedding(knn_indices, cfg):
#     """
#     Store KNN indices for each contrastive embedding
#
#     :param knn_indices: KNN indices [N, k]
#     :param cfg: Configuration parameters
#     :return: knn_records - Dictionary of KNN indices for each embedding
#     """
#     print("Step 4: Storing KNN indices for each embedding")
#
#     num_embeddings, k = knn_indices.shape
#     knn_records = [None for _ in range(num_embeddings)]
#
#     # Create a KNN record for each embedding, storing only indices
#     for i in range(num_embeddings):
#         knn_records[i] = knn_indices[i].tolist()  # Store the KNN index list directly
#
#     # Print KNN information for the first few embeddings as examples
#     print(f">>> Stored KNN indices for {num_embeddings} embeddings")
#     print(">>> Example KNN records:")
#     for i in range(min(5, num_embeddings)):  # Show KNN for the first five embeddings
#         knn_list = knn_records[i]
#         print(f">>> Embedding {i}: KNN indices = {knn_list}")
#
#     return knn_records

def sort_knn_by_distance_and_id(distances, indices):
    """
    Break equal-distance ties by sorting KNN results by index ID in ascending order.
    Assume the distances array is already sorted in ascending order.

    :param distances: Distance array of shape (N, M), sorted in ascending order
    :param indices: Index array of shape (N, M)
    :return: Sorted distances and indices arrays
    """

    # Ensure the input arrays have matching shapes
    if distances.shape != indices.shape:
        raise ValueError("distances and indices arrays must have the same shape.")

    num_points, num_neighbors = distances.shape

    # Create an empty array for the sorted results
    sorted_indices = np.zeros_like(indices)
    sorted_distances = np.zeros_like(distances)

    # Sort each row (data point)
    for i in range(num_points):
        # Use np.lexsort() to sort by multiple keys
        # Although distances[i] is already sorted, retain it as the primary key
        # Sort first by distances[i], then by indices[i]
        sorted_order = np.lexsort((indices[i], distances[i]))

        # Reorder indices and distances using the sorted order
        sorted_indices[i] = indices[i][sorted_order]
        sorted_distances[i] = distances[i][sorted_order]

    return sorted_indices, sorted_distances
