# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2024/11/15 4:03 PM
@File :defense.py
"""
import time

import torch
from torch.func import vmap, jacrev

def noisy_embedding_based_defense(embedding, std=1):
    noise = torch.normal(mean=0.0, std=std, size=embedding.shape)
    noise = noise.to(embedding.device)
    noisy_output = embedding + noise

    return noisy_output

def random_projection_based_defense(embedding, matrix):
    projected_embedding = torch.matmul(embedding, matrix)

    return projected_embedding


def pruning_embedding_based_defense_by_removing_elements(embedding, ratio):
    pruned_embedding_length = round(embedding.shape[1] * (1 - ratio))
    mask = torch.zeros_like(embedding, dtype=torch.bool)
    sorted_embedding, sorted_indices = torch.sort(embedding, dim=1)
    pruned_embedding = sorted_embedding[:, -pruned_embedding_length:]
    pruned_embedding = pruned_embedding.to(embedding.device)
    mask = mask.to(embedding.device)

    return pruned_embedding, mask

def DPSGD_based_defense(gradient, noise_std):
    clipping = max(1, gradient.norm(2) / 1)
    gradient = gradient / clipping
    noise = torch.normal(mean=0.0, std=noise_std, size=gradient.shape)
    noise = noise.to(gradient.device)
    noisy_gradient = gradient + noise

    return noisy_gradient

# def pruning_embedding_based_defense_by_setting_zeros(embedding: torch.Tensor, ratio: float) -> torch.Tensor:
#     pruned_embedding = torch.zeros_like(embedding)
#     for i in range(embedding.shape[0]):
#         vector = embedding[i]
#         num_pruning = int(len(vector) * ratio)
#         sorted_vector, _ = torch.sort(vector)
#         threshold = sorted_vector[num_pruning]
#         pruned_vector = torch.where(vector >= threshold, vector, torch.tensor(0.0, device=vector.device))
#         pruned_embedding[i] = pruned_vector
#
#     return pruned_embedding


# def invl_enp_defense(embedding, input_data, client_model, noise_std=0.1, k_ratio=0.95, j_ratio=0.60):
#     print("invRE")
#     input_data.requires_grad = True
#     if input_data.dim() == 4 and input_data.shape[0] > 1:
#         # For batched input, process only the first item to simplify this example
#         print("Warning: invl_enp_defense is processing only the first item in the batch for Jacobian calculation.")
#         input_single = input_data[0].unsqueeze(0)
#         embedding_single = embedding[0].unsqueeze(0)
#     else:
#         input_single = input_data
#         embedding_single = embedding
#
#     def model_fn(x):
#         return client_model(x)
#
#     jacobian_matrix = torch.autograd.functional.jacobian(model_fn, input_single)
#     # Reshape jacobian_matrix to (p, m) if needed
#     # p = embedding.numel(), m = input_data.numel()
#     G_x = jacobian_matrix.squeeze().view(embedding_single.numel(), -1)
#
#     # Step 1.1: Compute the singular value decomposition (SVD) of the Jacobian
#     U, S, _ = torch.linalg.svd(G_x, full_matrices=False)
#
#     # Step 2: Sample a random Gaussian noise matrix epsilon
#     # Match the noise shape to the embedding F(x)
#     noise = torch.normal(mean=0.0, std=noise_std, size=embedding_single.shape)
#     noise = noise.to(embedding.device)
#     # Reshape the noise to a column vector for matrix multiplication
#     noise_vec = noise.view(-1, 1)
#
#     # Step 3: Project noise into the singular-vector space and truncate adaptively
#     n = torch.matmul(U.T, noise_vec)  # n = U' * ε
#
#     # Compute the k and j index positions
#     total_sv_sum = torch.sum(S)
#     cumulative_sv_sum = torch.cumsum(S, dim=0)
#
#     k_index = (cumulative_sv_sum >= total_sv_sum * k_ratio).nonzero(as_tuple=True)[0][0]
#     j_index = (cumulative_sv_sum >= total_sv_sum * j_ratio).nonzero(as_tuple=True)[0][0]
#
#     # Create a mask for the dimensions receiving noise, between j and k
#     mask = torch.zeros_like(n)
#     mask[j_index:k_index] = 1.0
#
#     # Apply the mask to retain noise only in the specified dimensions
#     n_truncated = n * mask
#
#     # Step 4: Project the truncated noise back into the original embedding space
#     adaptive_noise_vec = torch.matmul(U, n_truncated)
#     adaptive_noise = adaptive_noise_vec.view(embedding_single.shape)
#
#     # Add adaptive noise to the original embedding
#     # This computes noise for one sample; extend or loop over it for batched input
#     noisy_embedding = embedding + adaptive_noise
#
#     return noisy_embedding


def invl_enp_defense(embedding, input_data, client_model, noise_std=1, k_ratio=0.95, j_ratio=0.60):
    """
    Implement an efficient batched InvL-ENP defense from "From Risk to Resilience".
    Use torch.func.vmap and torch.func.jacrev to compute Jacobians for the whole batch.

    Args:
        embedding (torch.Tensor): Original client embeddings F(x), shape [B, P] (batch size B, embedding dimension P).
        input_data (torch.Tensor): Original client input x, shape [B, C, H, W].
        client_model (torch.nn.Module): Client bottom model.
        noise_std (float): Standard deviation of the initial noise distribution.
        k_ratio (float): Determines the upper bound of the noise injection dimensions.
        j_ratio (float): Determines the lower bound of the noise injection dimensions.

    Returns:
        torch.Tensor: Embeddings with adaptive noise, shape [B, P].
    """
    client_model.eval()
    batch_size = input_data.shape[0]
    embedding_dim = embedding.shape[1]

    # The paper uses vmap and jacrev to improve efficiency
    # 1. Define the function that computes a single-sample Jacobian
    # jacrev produces shape [Embedding_Dim, *Input_Shape]
    calc_jacobian_single = jacrev(client_model)

    # 2. Use vmap to batch the single-sample function
    # in_dims=0 maps over input dimension 0 (the batch dimension)
    # Compute and stack the Jacobian for each sample in the batch
    # jacobian_batch has shape [Batch_Size, Embedding_Dim, *Input_Shape]
    calc_jacobian_batch = vmap(calc_jacobian_single, in_dims=0)

    # Run the batched computation
    jacobian_batch = calc_jacobian_batch(input_data)

    # Flatten each Jacobian to [Batch_Size, Embedding_Dim, Input_Dim_Flat]
    G_x_batch = jacobian_batch.view(batch_size, embedding_dim, -1)

    # Compute the SVD of each Jacobian in the batch
    # Shapes: U [B, P, P], S [B, P], Vh [B, M, M], where M=min(P, Input_Dim_Flat)
    # Only U and S are needed
    U, S, _ = torch.linalg.svd(G_x_batch, full_matrices=False)

    # Generate random noise for the entire batch
    laplace_dist = torch.distributions.Laplace(0, noise_std)
    noise = laplace_dist.sample(embedding.shape).to(embedding.device)
    # noise = torch.normal(mean=0.0, std=noise_std, size=embedding.shape).to(embedding.device)
    noise_vec_batch = noise.view(batch_size, embedding_dim, 1)
    # noise_vec_batch = noise.view(batch_size, embedding_dim, 1)

    # Project the batched noise into singular-value space
    # U.transpose(-1, -2) performs batched matrix transposition
    n_batch = torch.matmul(U.transpose(-1, -2), noise_vec_batch)

    # --- Compute truncation indices j and k for the batch ---
    total_sv_sum = torch.sum(S, dim=1, keepdim=True)
    cumulative_sv_sum = torch.cumsum(S, dim=1)

    # Find j_index and k_index for each sample
    k_indices = torch.argmax((cumulative_sv_sum >= total_sv_sum * k_ratio).int(), dim=1)
    j_indices = torch.argmax((cumulative_sv_sum >= total_sv_sum * j_ratio).int(), dim=1)

    # Create a batched mask
    mask = torch.zeros_like(n_batch)
    for i in range(batch_size):
        mask[i, j_indices[i]:k_indices[i], :] = 1.0

    n_truncated_batch = n_batch * mask
    # --- End truncation ---

    # Map the truncated noise back to the original space
    adaptive_noise_vec_batch = torch.matmul(U, n_truncated_batch)
    adaptive_noise = adaptive_noise_vec_batch.view(embedding.shape)

    noisy_embedding = embedding + adaptive_noise
    client_model.train()

    return noisy_embedding


# def invl_enp_defense_optimized(embedding, input_data, client_model, noise_std=1, k_ratio=0.95, j_ratio=0.60):
#     """
#     [Optimized version]
#     Compute one approximate Jacobian at the batch mean to avoid the bottleneck
#     of computing a Jacobian and SVD for each image.
#     """
#     client_model.eval()
#
#     # --- Optimization: no longer use vmap ---
#     # 1. Compute the batch mean while retaining dimensions
#     input_mean = torch.mean(input_data, dim=0, keepdim=True)
#
#     # 2. Compute the Jacobian once at the mean
#     # For output [P] and input [1, C, H, W], the Jacobian has shape [P, 1, C, H, W]
#     jacobian = jacrev(client_model)(input_mean)
#
#     # 3. Flatten the Jacobian to [P, M] (embedding dimension P, input dimension M)
#     # Fix: the first Jacobian dimension is output dimension P; using shape[1] was incorrect
#     embedding_dim = jacobian.shape[0]
#     G_x = jacobian.view(embedding_dim, -1)
#
#     # 4. Compute the SVD once
#     try:
#         U, S, _ = torch.linalg.svd(G_x, full_matrices=False)
#     except torch.linalg.LinAlgError:
#         # Fall back to simple Gaussian noise if SVD fails to converge
#         print("Warning: SVD did not converge. Falling back to standard Gaussian noise.")
#         noise = torch.randn_like(embedding) * noise_std
#         client_model.train()
#         return embedding + noise
#
#     # --- Generate noise using a single matrix without a loop ---
#
#     # Generate noise and project it into singular-value space
#     laplace_dist = torch.distributions.Laplace(0, noise_std)
#     noise = laplace_dist.sample(embedding.shape).to(embedding.device)
#     # Shapes: U.T [k, P], noise.T [P, B] -> n [k, B], where k=min(P,M)
#     n = torch.matmul(U.T, noise.T)
#
#     # Compute truncation indices j and k once
#     total_sv_sum = torch.sum(S)
#     cumulative_sv_sum = torch.cumsum(S, dim=0)
#     k_index = torch.argmax((cumulative_sv_sum >= total_sv_sum * k_ratio).int()).item()
#     j_index = torch.argmax((cumulative_sv_sum >= total_sv_sum * j_ratio).int()).item()
#
#     # Create a mask and truncate the noise
#     mask = torch.zeros_like(n)
#     if k_index > j_index:
#         mask[j_index:k_index, :] = 1.0
#
#     n_truncated = n * mask
#
#     # Map noise back to the original space and add it to the embeddings
#     # U shape [P, k], n_truncated shape [k, B] -> [P, B] -> T -> [B, P]
#     adaptive_noise = torch.matmul(U, n_truncated).T
#
#     noisy_embedding = embedding + adaptive_noise
#     client_model.train()
#
#     return noisy_embedding


def invl_enp_defense_optimized(embedding, input_data, client_model, noise_std=1, k_ratio=0.95, j_ratio=0.60):
    """
    [Corrected version]
    Use the known embedding dimension to reshape the Jacobian explicitly,
    instead of inferring it from the Jacobian shape, to fix incorrect SVD input dimensions.
    """
    client_model.eval()

    # --- 1. Get the embedding dimension directly from the embedding tensor ---
    embedding_dim = embedding.shape[1]

    # Print diagnostic information
    # print("\n--- [InvL-ENP Defense Debug] ---")
    # print(f"Input data shape: {input_data.shape}")
    # print(f"True Embedding dimension: {embedding_dim}")

    # 2. Compute the batch mean and evaluate the Jacobian at that point
    input_mean = torch.mean(input_data, dim=0, keepdim=True)
    try:
        jacobian = jacrev(client_model)(input_mean)
        # print(f"Initial Jacobian shape from jacrev: {jacobian.shape}")
    except Exception as e:
        # print(f"Error during jacobian calculation: {e}. Falling back to standard Gaussian noise.")
        noise = torch.randn_like(embedding) * noise_std
        client_model.train()
        return embedding + noise

    # --- Core fix ---
    # 3. Reshape the Jacobian to the correct two-dimensional shape [P, M]
    #    Use embedding_dim regardless of the original Jacobian shape,
    #    and flatten to (embedding_dim, N), where N is the product of all input dimensions.
    try:
        G_x = jacobian.view(embedding_dim, -1)
        # print(f"Reshaped Jacobian for SVD (G_x shape): {G_x.shape}")
    except RuntimeError as e:
        # print(f"Error: Failed to reshape Jacobian. Jacobian total elements ({jacobian.numel()}) "
        #       f"is not divisible by embedding_dim ({embedding_dim}).")
        # print(f"RuntimeError: {e}. Falling back to standard Gaussian noise.")
        noise = torch.randn_like(embedding) * noise_std
        client_model.train()
        return embedding + noise
    # ---

    # 4. Compute the SVD
    try:
        U, S, _ = torch.linalg.svd(G_x, full_matrices=False)
        # print(f"Shape of U matrix after SVD: {U.shape}")
    except torch.linalg.LinAlgError:
        # print("Warning: SVD did not converge. Falling back to standard Gaussian noise.")
        noise = torch.randn_like(embedding) * noise_std
        client_model.train()
        return embedding + noise

    # Check that the first dimension of U equals the embedding dimension
    if U.shape[0] != embedding_dim:
        # print(
        #     f"Critical Error: Shape mismatch after SVD. U shape is {U.shape}, but expected embedding_dim {embedding_dim}.")
        # print("Falling back to standard Gaussian noise.")
        noise = torch.randn_like(embedding) * noise_std
        client_model.train()
        return embedding + noise

    # 5. Generate and project noise
    laplace_dist = torch.distributions.Laplace(0, noise_std)
    noise = laplace_dist.sample(embedding.shape).to(embedding.device)
    n = torch.matmul(U.T, noise.T)

    # 6. Compute truncation indices
    total_sv_sum = torch.sum(S)
    cumulative_sv_sum = torch.cumsum(S, dim=0)
    k_indices = torch.where(cumulative_sv_sum >= total_sv_sum * k_ratio)[0]
    j_indices = torch.where(cumulative_sv_sum >= total_sv_sum * j_ratio)[0]
    k_index = k_indices[0].item() if len(k_indices) > 0 else len(S) - 1
    j_index = j_indices[0].item() if len(j_indices) > 0 else len(S) - 1

    # 7. Create a mask and truncate noise
    mask = torch.zeros_like(n)
    if k_index > j_index:
        mask[j_index:k_index, :] = 1.0
    n_truncated = n * mask

    # 8. Map noise back to the original space and add it to the embeddings
    adaptive_noise = torch.matmul(U, n_truncated).T
    noisy_embedding = embedding + adaptive_noise
    client_model.train()

    # print("--- [InvL-ENP Defense] Successfully applied adaptive noise. ---\n")
    return noisy_embedding


def invl_dnp_defense(input_data, client_model, noise_scale=0.1, k_ratio=0.95):
    """
    (Final version, Laplace noise)
    Implement an efficient batched InvL-DNP defense from "From Risk to Resilience".
    Add adaptive noise at the input-data level.

    Args:
        input_data (torch.Tensor): Original client input x, shape [B, C, H, W].
        client_model (torch.nn.Module): Client bottom model.
        noise_scale (float): Scale parameter of the Laplace distribution.
        k_ratio (float): Singular-value fraction used to retain noise in the leading dimensions.

    Returns:
        torch.Tensor: Input data with adaptive noise, shape [B, C, H, W].
    """
    batch_size = input_data.shape[0]

    # 1. Compute the Jacobian and its SVD
    calc_jacobian_batch = vmap(jacrev(client_model), in_dims=0)
    jacobian_batch = calc_jacobian_batch(input_data)

    # Run a forward pass to obtain the embedding dimension
    embedding = client_model(input_data)
    embedding_dim = embedding.shape[1]

    G_x_batch = jacobian_batch.view(batch_size, embedding_dim, -1)

    _, S, Vh = torch.linalg.svd(G_x_batch, full_matrices=False)
    V = Vh.transpose(-1, -2)

    # 2. Generate Laplace noise with the same shape as the input data
    laplace_dist = torch.distributions.Laplace(0, noise_scale)
    noise = laplace_dist.sample(input_data.shape).to(input_data.device)
    noise_flat = noise.view(batch_size, -1, 1)

    # 3. Project noise into the right singular-vector space (V)
    n_batch = torch.bmm(Vh, noise_flat)

    # 4. Truncate using k_ratio, retaining only the first k components
    total_sv_sum = torch.sum(S, dim=1, keepdim=True)
    cumulative_sv_sum = torch.cumsum(S, dim=1)
    k_indices = torch.argmax((cumulative_sv_sum >= total_sv_sum * k_ratio).int(), dim=1)

    mask = torch.zeros_like(n_batch)
    for i in range(batch_size):
        mask[i, :k_indices[i], :] = 1.0

    n_truncated_batch = n_batch * mask

    # 5. Map truncated noise back to the input space and apply it
    adaptive_noise_vec_batch = torch.bmm(V, n_truncated_batch)
    adaptive_noise = adaptive_noise_vec_batch.view(input_data.shape)

    perturbed_input = input_data + adaptive_noise

    return perturbed_input


def invl_dnp_defense_optimized(input_data, client_model, noise_scale=0.1, k_ratio=0.95):
    """
    [Optimized DNP defense]
    As in optimized ENP, compute one approximate Jacobian at the batch mean to avoid the performance bottleneck.
    DNP uses right singular vectors V, whereas ENP uses left singular vectors U.
    """
    client_model.eval()

    # 1. Compute the batch mean
    input_mean = torch.mean(input_data, dim=0, keepdim=True)

    # 2. Compute the Jacobian once at the mean
    jacobian = jacrev(client_model)(input_mean)

    # 3. Flatten the Jacobian to [P, M]
    embedding_dim = jacobian.shape[0]
    G_x = jacobian.view(embedding_dim, -1)

    # 4. Compute the SVD once; Vh is needed here
    try:
        _, S, Vh = torch.linalg.svd(G_x, full_matrices=False)
        V = Vh.T  # Obtain V
    except torch.linalg.LinAlgError:
        print("Warning: DNP SVD did not converge. Falling back to standard Gaussian noise.")
        noise = torch.randn_like(input_data) * noise_scale
        client_model.train()
        return input_data + noise

    # --- Generate noise using a single matrix ---
    # Generate noise with the same shape as the input data
    laplace_dist = torch.distributions.Laplace(0, noise_scale)
    noise = laplace_dist.sample(input_data.shape).to(input_data.device)
    noise_flat = noise.view(input_data.shape[0], -1)  # Flatten to [B, M]

    # Project noise into the right singular-vector space (V)
    n = torch.matmul(noise_flat, V)  # noise_flat:[B,M], V:[M,k] -> n:[B,k]

    # Truncate using k_ratio, retaining only the first k components
    total_sv_sum = torch.sum(S)
    cumulative_sv_sum = torch.cumsum(S, dim=0)
    k_index = torch.argmax((cumulative_sv_sum >= total_sv_sum * k_ratio).int()).item()

    # Retain only the first k_index components
    n[:, k_index:] = 0.0

    # Map truncated noise back to the input space and apply it
    adaptive_noise_flat = torch.matmul(n, Vh)  # n:[B,k], Vh:[k,M] -> adaptive_noise_flat:[B,M]
    adaptive_noise = adaptive_noise_flat.view(input_data.shape)

    perturbed_input = input_data + adaptive_noise
    client_model.train()
    return perturbed_input


# =========================
# B4B-style bucketing defense
# =========================
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class B4BContext:
    """
    A lightweight, VFL-friendly adaptation of 'bucketing' defenses:
    - Assign each embedding to a bucket via random-hyperplane hashing (SimHash).
    - Apply a bucket-specific orthogonal transform (perm + sign flip) + optional small noise.
    Notes:
      * Bucket assignment is computed on detached embeddings (non-diff), but the transform
        is applied to the original embedding so gradients still flow through the transform.
      * ctx must be created ONCE and reused (do NOT recreate per batch).
    """
    n_hash_bits: int = 12
    seed: int = 12345
    noise_std: float = 0.0  # set small value (e.g., 0.01~0.05) if you want extra ambiguity

    # internal (lazy init)
    _proj: Optional[torch.Tensor] = None   # [D, n_hash_bits]
    _dim: Optional[int] = None

    def ensure_init(self, dim: int, device: torch.device, dtype: torch.dtype):
        if self._proj is not None and self._dim == dim and self._proj.device == device and self._proj.dtype == dtype:
            return
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        # random hyperplanes
        proj = torch.randn(dim, self.n_hash_bits, generator=g, dtype=dtype)
        proj = proj.to(device=device)
        self._proj = proj
        self._dim = dim


def _bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    """
    bits: [B, n_bits] bool/int
    return: [B] int64 bucket ids
    """
    bits = bits.to(torch.int64)
    n_bits = bits.shape[1]
    shifts = torch.arange(n_bits, device=bits.device, dtype=torch.int64)
    # little-endian: bit0 is LSB
    return torch.sum(bits << shifts, dim=1)


def _bucket_perm_sign(dim: int, bucket_id: int, seed: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministically generate a bucket-specific permutation + sign vector
    using a private seed. Both are orthogonal transforms.
    """
    # mix seed with bucket_id (simple, deterministic)
    mixed = (seed * 1000003) ^ (bucket_id * 9176 + 0x9E3779B9)
    g = torch.Generator(device="cpu")
    g.manual_seed(mixed & 0xFFFFFFFF)

    perm = torch.randperm(dim, generator=g).to(device=device)
    sign = (torch.randint(0, 2, (dim,), generator=g).to(device=device) * 2 - 1).to(torch.float32)
    return perm, sign


def b4b_defense(
    embedding: torch.Tensor,
    ctx: B4BContext,
    detach_for_bucket: bool = True,
    return_bucket: bool = True
):
    """
    Args:
        embedding: [B, D]
        ctx: B4BContext (must be persistent across batches)
        detach_for_bucket: compute bucket assignment on detached embeddings
        return_bucket: whether to return bucket ids

    Returns:
        defended_embedding: [B, D]
        bucket_ids (optional): [B]
    """
    assert embedding.dim() == 2, f"Expected [B, D], got {embedding.shape}"
    B, D = embedding.shape
    ctx.ensure_init(dim=D, device=embedding.device, dtype=embedding.dtype)

    # 1) bucket assignment via SimHash on detached embeddings
    emb_for_hash = embedding.detach() if detach_for_bucket else embedding
    proj = ctx._proj  # [D, n_bits]
    hash_scores = emb_for_hash @ proj  # [B, n_bits]
    bits = (hash_scores > 0)
    bucket_ids = _bits_to_int(bits)  # [B]

    # 2) apply bucket-specific orthogonal transforms (perm + sign flip)
    out = torch.empty_like(embedding)
    # group by bucket to avoid per-sample overhead
    unique_buckets = torch.unique(bucket_ids)
    for b in unique_buckets.tolist():
        idx = (bucket_ids == b)
        perm, sign = _bucket_perm_sign(D, int(b), ctx.seed, embedding.device)
        # permutation + sign flip (orthogonal); preserves norms within bucket
        out[idx] = embedding[idx][:, perm] * sign

    # 3) optional small noise
    if ctx.noise_std and ctx.noise_std > 0:
        noise = torch.randn_like(out) * ctx.noise_std
        out = out + noise

    if return_bucket:
        return out, bucket_ids
    return out
