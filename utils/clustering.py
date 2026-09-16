import time
import numpy as np
from PIL.ImageMath import lambda_eval
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch

def euclidean_dist(vector_a, vector_b):
    return np.sqrt(sum(np.power((vector_a - vector_b), 2)))


def kmeans_cluster(k, embeddings):
     st_time = time.time()
     cluster_pred = KMeans(n_clusters=k).fit(embeddings)
     cluster_labels = cluster_pred.labels_
     cluster_centers = cluster_pred.cluster_centers_
         # print(len(cluster_labels))
         # print(len(cluster_centers))
     end_time = time.time()
     print(">>> Kmeans total time: ", end_time - st_time)

     return cluster_labels, cluster_centers


def calculate_weight(k, embeddings, T):
    st_time = time.time()
    cluster_labels, cluster_centers = kmeans_cluster(k, embeddings)
    weight_list = [None for _ in range(len(embeddings))]
    for index, data in enumerate(embeddings):
        diff = cluster_centers - data.numpy()
        dist = np.linalg.norm(diff, axis=1)
        # dist = np.sum(diff ** 2, axis=1)
        scores = -dist / T
        exp_scores = np.exp(scores)
        weights = exp_scores / np.sum(exp_scores)
        weight_list[index] = weights
    weight_list = np.array(weight_list)
    weight_list = torch.from_numpy(weight_list)
    prototypes = torch.from_numpy(cluster_centers)
    end_time = time.time()
    print(">>> Calculate weight total time: ", end_time - st_time)

    return weight_list, prototypes


def adjust_learning_rate(epoch, lr, optimizer):
    new_lr = lr * (0.1) ** (epoch // 20)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    # tqdm.write(f">>>Learning rate: {new_lr}")


def loss_fn(s, e, prototypes, weights, lambda_val):
    reconstruction_loss = torch.linalg.norm(s - e, ord=2, dim=1)

    s_expanded = s.unsqueeze(1)
    e_expanded = e.unsqueeze(1)
    c_expanded = prototypes.unsqueeze(0)

    dist_s = torch.linalg.norm(s_expanded - c_expanded, ord=2, dim=2)
    dist_e = torch.linalg.norm(e_expanded - c_expanded, ord=2, dim=2)
    total_dist = dist_s + dist_e
    # total_dist = dist_s

    weighted_term = weights * total_dist
    structure_loss = torch.sum(weighted_term, dim=1)
    final_loss = reconstruction_loss + lambda_val * structure_loss

    return torch.mean(final_loss)


