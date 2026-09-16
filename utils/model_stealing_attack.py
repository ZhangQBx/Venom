import time

import numpy
import torch
import torch.utils.data as tud
from torch import distributed as dist
from tqdm import tqdm
from utils.defense import (pruning_embedding_based_defense_by_removing_elements, noisy_embedding_based_defense,
                           random_projection_based_defense)
from utils.funcs import *
from utils.clustering import calculate_weight, adjust_learning_rate, loss_fn
from utils.contrastive_learning import (train_contrastive_encoder, generate_contrastive_embeddings,
                                        find_knn_brute_force, find_knn_random_sampling,
                                        find_knn_hnsw)


def model_stealing_attack_entrance(cfg, client, server, num_clients, original_pred_list, num_surrogates):
    # model_stealing_attack(cfg, client)
    # model_stealing_attack_one(cfg, client)
    # print("Get in contrastive")
    if num_surrogates == 1:
        st_time = time.time()
        # model_stealing_attack(cfg, client)
        model_stealing_attack_with_contrastive_and_knn(cfg, client, server, num_clients, original_pred_list)
        # model_stealing_attack_cont_steal(cfg, client)
        # model_stealing_attack_original_space_knn(cfg, client, server, num_clients, original_pred_list)
        # model_stealing_via_distill_knn(cfg, client)
        # model_stealing_attack_with_kmeans(cfg, client)
        end_time = time.time()
        print('MS Time Cost:', end_time - st_time)

        right_pred = 0
        pred_list = []
        label_list = []
        client.surrogate_model.eval()
        for data, _, y in client.test_dataloader:
            data = data.to(client.device)
            output = client.surrogate_model(data)
            gather_list = [torch.empty_like(output) for _ in range(num_clients)]
            dist.all_gather(gather_list, output)

            if client.rank == 0:
                y = y.to(server.device)
                top_input = torch.cat(gather_list, dim=1)
                pred = server.top_model(top_input)

                if cfg.type == "tabular":
                    if cfg.task == "classification":
                        _, predicted = pred.max(1)
                        pred_list.append(predicted.cpu().numpy())
                        label = torch.from_numpy(y.nonzero().cpu().numpy()[:, 1]).to(server.device)
                        right_pred += predicted.eq(label).sum().item()

                    elif cfg.task == "regression":
                        pred_list.append(pred.detach().cpu().numpy()) # regression
                        label_list.append(y.detach().cpu().numpy()) # regression
                else:
                    predicted = pred.argmax(dim=1)
                    pred_list.append(predicted.cpu().numpy())
                    right_pred += predicted.eq(y).sum().item()

        if client.rank == 0:
            pred_list = np.concatenate(pred_list)
            # print(len(original_pred_list), len(pred_list))

            if cfg.task == "classification":
                tqdm.write(f"Attack acc: {100 * (right_pred / len(pred_list))}%")
                tqdm.write(f"Agreement: {100 * (np.sum(pred_list == original_pred_list) / len(pred_list))}%")

            elif cfg.task == "regression":
                original_pred_list = np.concatenate(original_pred_list) # regression

                label_list = np.concatenate(label_list) # regression
                mse = np.mean((label_list - pred_list) ** 2) # regression
                agreement_mse = np.mean((original_pred_list - pred_list) ** 2) # regression

                tqdm.write(f"Attack MSE: {mse}") # regression
                tqdm.write(f"Agreement MSE: {agreement_mse}") # regression

    else:
        model_stealing_attack_multi_surrogates(cfg, client, num_surrogates)
        right_pred = 0
        pred_list = []
        label_list = []
        for data, hash_res, y in client.test_dataloader:
            data = data.to(client.device)
            hash_res = hash_res.to(client.device)

            output = torch.zeros((len(data), cfg.bottom_model_output_dim)).to(client.device)
            output[hash_res == 0] = client.surrogate_models[0](data[hash_res == 0])
            output[hash_res == 1] = client.surrogate_models[1](data[hash_res == 1])

            gather_list = [torch.empty_like(output) for _ in range(num_clients)]
            dist.all_gather(gather_list, output)

            if client.rank == 0:
                y = y.to(server.device)
                top_input = torch.cat(gather_list, dim=1)
                pred = server.top_model(top_input)

                if cfg.type == "tabular":
                    # _, predicted = pred.max(1)
                    # pred_list.append(predicted.cpu().numpy())
                    # label = torch.from_numpy(y.nonzero().cpu().numpy()[:, 1]).to(server.device)
                    # right_pred += predicted.eq(label).sum().item()
                    pred_list.append(pred.detach().cpu().numpy())
                    label_list.append(y.detach().cpu().numpy())
                else:
                    predicted = pred.argmax(dim=1)
                    pred_list.append(predicted.cpu().numpy())
                    right_pred += predicted.eq(y).sum().item()

        if client.rank == 0:
            if cfg.task == "classification":
                original_pred_list = np.concatenate(original_pred_list)
                # print(len(original_pred_list), len(pred_list))
                tqdm.write(f"Attack acc: {100 * (right_pred / len(pred_list))}%")
                tqdm.write(f"Agreement: {100 * (np.sum(pred_list == original_pred_list) / len(pred_list))}%")

            elif cfg.task == "regression":
                pred_list = np.concatenate(pred_list)
                label_list = np.concatenate(label_list)
                mse = np.mean((label_list - pred_list) ** 2)
                agreement_mse = np.mean((original_pred_list - pred_list) ** 2)

                tqdm.write(f"Attack MSE: {mse}")
                tqdm.write(f"Agreement MSE: {agreement_mse}")


def evaluate_surrogate_model(cfg, client, server, num_clients, original_pred_list, epoch, history_metrics):
    if client.rank != 0:
        client.surrogate_model.eval()
        for data, _, y in client.test_dataloader:
            data = data.to(client.device)
            output = client.surrogate_model(data)
            gather_list = [torch.empty_like(output) for _ in range(num_clients)]
            dist.all_gather(gather_list, output)
        client.surrogate_model.train()
        return

        # --- Run the following logic only on rank 0 ---
    right_pred = 0
    pred_list = []
    label_list = []
    client.surrogate_model.eval()

    with torch.no_grad():
        for data, _, y in client.test_dataloader:
            data = data.to(client.device)
            output = client.surrogate_model(data)
            gather_list = [torch.empty_like(output) for _ in range(num_clients)]
            dist.all_gather(gather_list, output)
            y = y.to(server.device)
            top_input = torch.cat(gather_list, dim=1)
            pred = server.top_model(top_input)

            if cfg.type == "tabular":
                if cfg.task == "classification":
                    _, predicted = pred.max(1)
                    pred_list.append(predicted.cpu().numpy())
                    label = torch.from_numpy(y.nonzero().cpu().numpy()[:, 1]).to(server.device)
                    right_pred += predicted.eq(label).sum().item()
                elif cfg.task == "regression":
                    pred_list.append(pred.detach().cpu().numpy())
                    label_list.append(y.detach().cpu().numpy())
            else:
                predicted = pred.argmax(dim=1)
                pred_list.append(predicted.cpu().numpy())
                right_pred += predicted.eq(y).sum().item()

    client.surrogate_model.train()
    pred_list = np.concatenate(pred_list)

    tqdm.write(f"--- [Epoch: {epoch}] Evaluation ---")
    # if cfg.task == "classification":
    attack_acc = 100 * (right_pred / len(pred_list))
    agreement = 100 * (np.sum(pred_list == original_pred_list) / len(pred_list))

    # Append the current epoch's results to the history list
    history_metrics['acc_history'].append(attack_acc)
    history_metrics['agr_history'].append(agreement)

    tqdm.write(f"Attack ACC: {attack_acc:.2f}%")
    tqdm.write(f"Agreement: {agreement:.2f}%")

    # elif cfg.task == "regression":
    #     original_pred_list_np = np.concatenate(original_pred_list)
    #     label_list_np = np.concatenate(label_list)
    #     mse = np.mean((label_list_np - pred_list) ** 2)
    #     agreement_mse = np.mean((original_pred_list_np - pred_list) ** 2)
    #
    #     # For MSE, find the minimum value
    #     best_metrics['mse'] = min(best_metrics['mse'], mse)
    #     best_metrics['agr_mse'] = min(best_metrics['agr_mse'], agreement_mse)
    #
    #     tqdm.write(f"Attack MSE: {mse:.4f} (Best: {best_metrics['mse']:.4f})")
    #     tqdm.write(f"Agreement MSE: {agreement_mse:.4f} (Best: {best_metrics['agr_mse']:.4f})")

    # tqdm.write("------------------------------------")
    # return best_metrics
    # tqdm.write("------------------------------------")



# def model_stealing_attack_for_one_client(cfg, client):
#     """
#     used for only one client (demo usage)
#     """
#     embedding = []
#     for data, hash_res, y in client.aux_dataloader:
#         for item in zip(data, hash_res):
#             output = client.bottom_models[item[1]](item[0])
#             embedding.append(output.detach())
#     embeddings = torch.stack(embedding)
#     aux_emb_dataset = MyDatasetTS(client.aux_dataset.aux_data_ts, embeddings)
#
#     criterion = l2_norm_loss
#     aux_emb_loader = tud.DataLoader(aux_emb_dataset, batch_size=cfg.bs_ms)
#     optimizer = torch.optim.Adam(client.surrogate_model.parameters(), lr=cfg.lr_ms)
#     epoch_loss = 0
#     for epoch in range(cfg.epochs):
#         for data, y in aux_emb_loader:
#             output = client.surrogate_model(data)
#             loss = criterion(output, y)
#             epoch_loss += loss.item()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()



def model_stealing_attack(cfg, client):
    """
    model stealing attack
    :param cfg: mlp_config / cnn_config
    :param client: each client
    :return: None
    """
    input_dim = client.train_dataset.num_features
    print("Now in model_stealing_attack")
    output_dim = cfg.bottom_model_output_dim

    if cfg.defense == "pruning":
        output_dim = round(output_dim * (1 - cfg.pruning_ratio))

    if cfg.type == "tabular":
        client.surrogate_model = MLPBottomModel(client.bottom_models[0].in_dim, output_dim,
                                                cfg.bottom_model_layers).to(client.device)
    elif cfg.type == "image":
        if cfg.dataset == "mnist":
            client.surrogate_model = ResBottomModel(1, cfg.surrogate_model_layers, output_dim).to(client.device)
        elif cfg.dataset == "cifar10":
            client.surrogate_model = ResBottomModel(3, cfg.surrogate_model_layers, output_dim).to(client.device)
    else:
        print("Invalid type")
        return

    client.surrogate_model.train()
    embedding = []
    for data, hash_res, y in client.aux_dataloader:
        data = data.to(client.device)
        hash_res = hash_res.to(client.device)
        for item in zip(data, hash_res):
            output = client.bottom_models[item[1]](item[0])
            embedding.append(output.detach())
    embeddings = torch.stack(embedding)
    # if client.rank == 0:
    #     numpy.save("../plots/vfl-attack//tnse/ground-truth-embeddings.npy", embeddings.cpu())

    if cfg.defense == "noise":
        embeddings = noisy_embedding_based_defense(embeddings, cfg.noise_std)
    elif cfg.defense == "pruning":
        embeddings, _ = pruning_embedding_based_defense_by_removing_elements(embeddings, cfg.pruning_ratio)
    elif cfg.defense == 'projection':
        embeddings = random_projection_based_defense(embeddings, client.projection_matrix)
    elif cfg.defense == "ResSFL":
        pass
    elif cfg.defense == "DPSGD":
        pass

    # for i in range(len(client.aux_dataloaders)):
    #     for data, y in client.aux_dataloaders[i]:
    #         # for item in zip(data, hash_res):
    #         #     output = client.bottom_models[item[1]](item[0])
    #         #     embedding.append(output.detach())
    #         output = client.bottom_models[i](data)
    #         embedding.append(output.detach().numpy())
    # embeddings = np.concatenate(embedding)
    # embeddings = torch.from_numpy(embeddings)

    aux_emb_dataset = MyDatasetTS(client.aux_dataset.aux_data_ts, embeddings)

    criterion = l2_norm_loss_1
    aux_emb_loader = tud.DataLoader(aux_emb_dataset, batch_size=cfg.bs_ms)
    optimizer = torch.optim.Adam(client.surrogate_model.parameters(), lr=cfg.lr_ms)

    # for epoch in tqdm(range(60), desc="Attack Train", dynamic_ncols=True):
    for epoch in range(cfg.epochs):
        epoch_loss = 0
        for data, y in aux_emb_loader:
            data = data.to(client.device)
            y = y.to(client.device)
            output = client.surrogate_model(data)
            loss = criterion(output, y)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # adjust_learning_rate(epoch, cfg.lr_ms, optimizer)
        tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss}")
    # extract_embedding_from_surrogate_model(client)
    extract_both_embeddings(client)



def model_stealing_attack_multi_surrogates(cfg, client, num_surrogates):
    """
    model stealing attack for multiple surrogate models
    :param cfg:
    :param client:
    :param num_surrogates:
    :return:
    """
    print("Now in model_stealing_attack_multi_surrogates")
    # print(client.surrogate_model)
    client.surrogate_models.append(
        MLPBottomModel(client.bottom_models[0].in_dim, cfg.bottom_model_output_dim, cfg.bottom_model_layers).to(client.device))
    client.surrogate_models.append(
        MLPBottomModel(client.bottom_models[0].in_dim, cfg.bottom_model_output_dim, cfg.bottom_model_layers).to(client.device))
    for model in client.surrogate_models:
        model.train()

    embedding = []
    for data, hash_res, y in client.aux_dataloader:
        data = data.to(client.device)
        hash_res = hash_res.to(client.device)
        y = y.to(client.device)
        for item in zip(data, hash_res):
            output = client.bottom_models[item[1]](item[0])
            embedding.append(output.detach())
    embeddings = torch.stack(embedding)

    # for i in range(len(client.aux_dataloaders)):
    #     for data, y in client.aux_dataloaders[i]:
    #         # for item in zip(data, hash_res):
    #         #     output = client.bottom_models[item[1]](item[0])
    #         #     embedding.append(output.detach())
    #         output = client.bottom_models[i](data)
    #         embedding.append(output.detach().numpy())
    # embeddings = np.concatenate(embedding)
    # embeddings = torch.from_numpy(embeddings)

    embeddings = embeddings.cpu().detach()

    aux_emb_dataset = AttackDatasetTS(cfg, client.rank, client.aux_dataset.aux_data_ts, embeddings, num_surrogates, True)

    criterion = l2_norm_loss
    aux_emb_loader = tud.DataLoader(aux_emb_dataset, batch_size=cfg.bs_ms)
    optimizer0 = torch.optim.Adam(client.surrogate_models[0].parameters(), lr=cfg.lr_ms)
    optimizer1 = torch.optim.Adam(client.surrogate_models[1].parameters(), lr=cfg.lr_ms)

    # for epoch in tqdm(range(cfg.epochs), desc="Attack Train", dynamic_ncols=True):
    for epoch in range(cfg.epochs):
        epoch_loss = 0
        for data, hash_res, y in aux_emb_loader:
            # surrogate_embedding = []
            # for item in zip(data, hash_res):
            #     output = client.surrogate_models[item[1]](item[0])
            #     surrogate_embedding.append(output)
            # surrogate_embeddings = torch.stack(surrogate_embedding)
            data = data.to(client.device)
            hash_res = hash_res.to(client.device)
            y = y.to(client.device)

            surrogate_embeddings = torch.zeros((data.shape[0], cfg.bottom_model_output_dim)).to(client.device)
            surrogate_embeddings[hash_res == 0] = client.surrogate_models[0](data[hash_res == 0])
            surrogate_embeddings[hash_res == 1] = client.surrogate_models[1](data[hash_res == 1])

            loss = criterion(surrogate_embeddings, y)
            epoch_loss += loss.item()
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            loss.backward()
            optimizer0.step()
            optimizer1.step()
        # tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss}")


def extract_embedding_from_surrogate_model_knn(client):
    embedding = []
    for data, hash_res, y in client.aux_dataloader:
        data = data.to(client.device)
        for item in data:
            output = client.surrogate_model(item)
            embedding.append(output.detach())

    embeddings = torch.stack(embedding)
    if client.rank == 0:
        numpy.save("../plots/vfl-attack/tnse/surrogate-embeddings-knn.npy", embeddings.cpu())

    return embeddings


def extract_both_embeddings(client):
    embedding_original = []
    embedding_surrogate = []
    labels = []
    with torch.no_grad():
        for data, hash_res, y in client.aux_dataloader:
            data = data.to(client.device)
            hash_res = hash_res.to(client.device)
            labels.append(y.cpu())

            for item in zip(data, hash_res):
                output_original = client.bottom_models[item[1]](item[0])
                embedding_original.append(output_original.detach())
                output_surrogate = client.surrogate_model(item[0])
                embedding_surrogate.append(output_surrogate.detach())

    embedding_original = torch.stack(embedding_original)
    embedding_surrogate = torch.stack(embedding_surrogate)
    all_labels_tensor = torch.cat(labels, dim=0)
    all_labels_numpy = all_labels_tensor.cpu().numpy()

    if client.rank == 0:
        numpy.save("../plots/vfl-attack/tnse/ground-truth-embeddings.npy", embedding_original.cpu())
        numpy.save("../plots/vfl-attack/tnse/surrogate-embeddings.npy", embedding_surrogate.cpu())
        numpy.save("../plots/vfl-attack/tnse/labels.npy", all_labels_numpy)
        print("Done!")


def extract_embedding_from_surrogate_model(client):
    embedding = []
    for data, hash_res, y in client.aux_dataloader:
        data = data.to(client.device)
        for item in data:
            output = client.surrogate_model(item)
            embedding.append(output.detach())

    embeddings = torch.stack(embedding)
    if client.rank == 0:
        numpy.save("../plots/vfl-attack/tnse/surrogate-embeddings.npy", embeddings.cpu())

def extract_embeddings_from_bottom_model(cfg, client):
    print("Step 1: Extracting embeddings from bottom model using auxiliary data")
    embedding = []
    labels = []
    with torch.no_grad():
        for data, hash_res, y in client.aux_dataloader:
            data = data.to(client.device)
            hash_res = hash_res.to(client.device)
            labels.append(y.cpu())

            for item in zip(data, hash_res):
                output = client.bottom_models[item[1]](item[0])
                embedding.append(output.detach())

    embeddings = torch.stack(embedding)
    all_labels_tensor = torch.cat(labels, dim=0)
    all_labels_numpy = all_labels_tensor.cpu().numpy()
    # if client.rank == 0:
    #     numpy.save("../plots/vfl-attack/tnse/labels.npy", all_labels_numpy)

    # Apply the defense using the existing logic
    if cfg.defense == "noise":
        embeddings = noisy_embedding_based_defense(embeddings, cfg.noise_std)
        print(">>> Applied noise defense to embeddings")
    elif cfg.defense == "pruning":
        embeddings, _ = pruning_embedding_based_defense_by_removing_elements(embeddings, cfg.pruning_ratio)
        print(">>> Applied pruning defense to embeddings")
    elif cfg.defense == 'projection':
        embeddings = random_projection_based_defense(embeddings, client.projection_matrix)
    elif cfg.defense == "ResSFL":
        print(">>> ResSFL defense - no modification to embeddings")
    elif cfg.defense == "DPSGD":
        print(">>> DPSGD defense - no modification to embeddings")

    print(f">>> Successfully obtained {len(embeddings)} embeddings with shape {embeddings.shape}")
    return embeddings


def model_stealing_attack_with_contrastive(cfg, client):
    """
    Model stealing attack with contrastive learning.
    Step 1: Query the bottom model with auxiliary data to obtain embeddings.
    Step 2: Train the encoder with contrastive learning.
    :param cfg: Configuration parameters
    :param client: Client object
    :return: Trained encoder and contrastive embeddings
    """
    print("Now in new_model_stealing_attack_with_contrastive")

    # Step 1: Query the bottom model with auxiliary data using the existing logic
    embeddings = extract_embeddings_from_bottom_model(cfg, client)

    # Step 2: Train the encoder with contrastive learning
    encoder = train_contrastive_encoder(cfg, client, embeddings)

    # Obtain contrastive embeddings
    contrastive_embeddings = generate_contrastive_embeddings(encoder, embeddings, client)

    return encoder, contrastive_embeddings


# def model_stealing_attack_with_contrastive_and_knn(cfg, client):
#     """
#     Model stealing attack with contrastive learning and KNN analysis
#     Step 1: Query the bottom model with auxiliary data to obtain embeddings
#     Step 2: Train the encoder with contrastive learning
#     Step 3: Find KNN using brute-force search
#     Step 4: Store KNN indices for each embedding
#
#     :param cfg: Configuration parameters
#     :param client: Client object
#     :return: Trained encoder, contrastive embeddings, KNN results, and KNN records
#     """
#     print("Now in new_model_stealing_attack_with_contrastive_and_knn")
#
#     embeddings = extract_embeddings_from_bottom_model(cfg, client)
#     encoder = train_contrastive_encoder(cfg, client, embeddings)
#     contrastive_embeddings = generate_contrastive_embeddings(encoder, embeddings, client)
#     # print(">>> Rank", client.rank, contrastive_embeddings[:3])
#     # time.sleep(100000)
#     # Step 3: Find KNN using brute-force search
#     # knn_indices, knn_distances = find_knn_brute_force(contrastive_embeddings, cfg)
#     knn_indices, knn_distances, farthest_indices, farthest_distances = find_knn_brute_force(contrastive_embeddings, cfg)
#     # knn_indices, knn_distances = find_knn_sklearn(contrastive_embeddings, cfg)
#
#     knn_indices, knn_distances = sort_knn_by_distance_and_id(knn_distances, knn_indices)
#     knn_indices = torch.tensor(knn_indices)
#     farthest_indices = torch.tensor(farthest_indices)
#
#     output_dim = cfg.bottom_model_output_dim
#     if cfg.defense == "pruning":
#         output_dim = round(output_dim * (1 - cfg.pruning_ratio))
#
#     if cfg.type == "tabular":
#         client.surrogate_model = MLPBottomModel(client.bottom_models[0].in_dim, output_dim,
#                                                 cfg.bottom_model_layers).to(client.device)
#     elif cfg.type == "image":
#         client.surrogate_model = ResBottomModel(1, cfg.bottom_model_output_dim).to(client.device)
#     else:
#         print("Invalid type")
#         return
#
#     # aux_emb_dataset = MyDatasetTS(client.aux_dataset.aux_data_ts, embeddings)
#     # aux_emb_dataset = MyDatasetTS(client.aux_dataset.aux_data_ts, knn_indices)
#     aux_emb_dataset = MyDatasetKnn_Near_Far(client.aux_dataset.aux_data_ts, knn_indices, farthest_indices)
#
#
#     criterion = l2_norm_loss
#     aux_emb_loader = tud.DataLoader(aux_emb_dataset, batch_size=cfg.bs_ms)
#     optimizer = torch.optim.AdamW(client.surrogate_model.parameters(), lr=cfg.lr_ms)
#
#     client.surrogate_model.train()
#     embeddings = embeddings.to(client.device)
#
#     for epoch in range(cfg.epochs):
#         epoch_loss = 0
#         for data, near_knn, far_knn in aux_emb_loader:
#             data = data.to(client.device)
#             near_knn = near_knn.to(client.device)
#             far_knn = far_knn.to(client.device)
#             output = client.surrogate_model(data)
#
#             # loss = criterion(output, embeddings[y[:,0]])
#             # loss = criterion(output, y)
#             loss = criterion(output, embeddings[near_knn[:,0]])
#             near_loss = 0
#             far_loss = 0
#             for i in range(cfg.knn_k):
#                 near_loss += criterion(output, embeddings[near_knn[:,i+1]])
#                 far_loss += criterion(output, embeddings[far_knn[:,i]])
#             near_loss /= cfg.knn_k
#             far_loss = far_loss / cfg.knn_k
#
#             # dist_true = get_similarity_distribution(embeddings[near_knn[:,0]], temperature=2)
#             # dist_surrogate = get_similarity_distribution(output, temperature=2)
#             # loss_kl = F.kl_div(dist_surrogate.log(), dist_true, reduction='batchmean')
#
#             dist_true = F.softmax(embeddings[near_knn[:,0]] / 2, dim=1)
#
#             # b. Convert each surrogate output feature vector to a probability distribution
#             dist_surrogate = F.softmax(output / 2, dim=1)
#
#             # c. Compute KL divergence; F.kl_div expects log-probabilities as its first input
#             loss_kl = F.kl_div(dist_surrogate.log(), dist_true, reduction='batchmean')
#
#             loss = (1.2 * loss + 0.2 * near_loss - 0.2 * far_loss + 1.2 * loss_kl).mean()
#
#             epoch_loss += loss.item()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         adjust_learning_rate(epoch, cfg.lr_ms, optimizer)
#         if client.rank == 0:
#             tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss}")


class MydatasetForHybridLoss(tud.Dataset):
    """
    [Dataset for the combined loss]
    To support the combined loss function, this dataset returns:
    1. Anchor data (anchor_data)
    2. The anchor's true embedding (anchor_true_embedding)
    3. Indices of K nearest neighbors (near_indices)
    4. Indices of K farthest neighbors (far_indices)
    """

    def __init__(self, aux_data, all_embeddings, near_indices, far_indices):
        self.aux_data = aux_data
        self.all_embeddings = all_embeddings
        self.near_indices = near_indices
        self.far_indices = far_indices

    def __len__(self):
        return len(self.aux_data)

    def __getitem__(self, index):
        return (self.aux_data[index],
                self.all_embeddings[index],
                self.near_indices[index],
                self.far_indices[index])


def get_similarity_distribution(embeddings, temperature=1.0):
    """
    [Helper function]
    Compute pairwise cosine similarities within a batch and convert them to probability distributions with softmax.
    """
    sim_matrix = F.normalize(embeddings, p=2, dim=1) @ F.normalize(embeddings, p=2, dim=1).T
    return F.softmax(sim_matrix / temperature, dim=1)


def model_stealing_attack_with_contrastive_and_knn(cfg, client, server, num_clients, original_pred_list):
    """
    [Revised version]
    Add KL divergence loss to the existing contrastive learning and KNN framework.
    """
    print("Now in new_model_stealing_attack_with_contrastive_and_knn (Hybrid Loss: KNN + KL)")

    tau_soft = float(cfg.get("tau_soft", 0.5))
    if not numpy.isfinite(tau_soft) or tau_soft <= 0:
        raise ValueError("tau_soft must be a finite positive temperature")

    # --- Steps 1 and 2: unchanged ---
    embeddings = extract_embeddings_from_bottom_model(cfg, client)
    # if client.rank == 0:
    #     numpy.save("../plots/vfl-attack/tnse/ground-truth-embeddings.npy", embeddings.cpu())
    encoder = train_contrastive_encoder(cfg, client, embeddings)
    contrastive_embeddings = generate_contrastive_embeddings(encoder, embeddings, client)
    # if client.rank == 0:
    #     numpy.save("../plots/vfl-attack/contrastive-embeddings.npy", contrastive_embeddings.cpu())

    st_time = time.time()
    knn_indices, _, farthest_indices, _ = find_knn_brute_force(contrastive_embeddings, cfg)
    # knn_indices, _, farthest_indices, _ = find_knn_random_sampling(contrastive_embeddings, cfg)
    # knn_indices, _, farthest_indices, _ = find_knn_hnsw(contrastive_embeddings, cfg)
    end_time = time.time()
    print("find knn time: {}".format(end_time - st_time))


    # --- Step 4: Prepare training ---
    knn_indices_np = knn_indices
    farthest_indices_np = farthest_indices

    # Use the revised dataset to provide the required data
    aux_emb_dataset = MydatasetForHybridLoss(
        client.aux_dataset.aux_data_ts,
        embeddings,
        knn_indices_np,
        farthest_indices_np
    )

    aux_emb_loader = tud.DataLoader(
        aux_emb_dataset,
        batch_size=cfg.bs_ms,
        shuffle=True,
    )

    # Create the surrogate model using the existing logic
    output_dim = embeddings.shape[1]
    if cfg.type == "tabular":
        in_dim = client.aux_dataset.aux_data_ts.shape[1]
        client.surrogate_model = MLPBottomModel(in_dim, output_dim, cfg.bottom_model_layers).to(client.device)
    elif cfg.type == "image":
        if cfg.dataset == "mnist":
            client.surrogate_model = ResBottomModel(1, cfg.surrogate_model_layers, output_dim).to(client.device)
        elif cfg.dataset == "cifar10":
            client.surrogate_model = ResBottomModel(3, cfg.surrogate_model_layers, output_dim).to(client.device)


    optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[cfg.get("optimizer_ms", "AdamW")]
    optimizer = optimizer_class(client.surrogate_model.parameters(), lr=cfg.lr_ms)
    embeddings_device = embeddings.to(client.device)
    client.surrogate_model.train()

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    contrastive_embeddings_device = contrastive_embeddings.to(client.device)

    history_metrics = {}
    if client.rank == 0:
        history_metrics = {
            'acc_history': [], 'agr_history': [],
            'mse_history': [], 'agr_mse_history': []
        }

    for epoch in range(cfg.epochs):
        epoch_loss = 0
        for data, true_emb, near_indices, far_indices in aux_emb_loader:
            data = data.to(client.device)
            true_emb = true_emb.to(client.device)
            near_indices = near_indices.to(client.device)
            far_indices = far_indices.to(client.device)

            optimizer.zero_grad()
            surrogate_output = client.surrogate_model(data)
            # 1. Core stealing loss (L2 distance)
            # loss_steal = l2_norm_loss(surrogate_output, true_emb)
            loss_steal = F.mse_loss(surrogate_output, true_emb, reduction='none').mean(dim=1)

            # 2. Neighborhood loss based on L2 distance
            #    a. Nearest-neighbor attraction loss
            # output_expanded = surrogate_output.unsqueeze(1).expand(-1, cfg.knn_k, -1)
            # near_embeddings = embeddings_device[near_indices]
            # # loss_attraction = F.mse_loss(output_expanded, near_embeddings)
            # loss_attraction = l2_norm_loss(output_expanded, near_embeddings).mean()

            surrogate_con_output = encoder.encode_only(surrogate_output)
            output_normalized = F.normalize(surrogate_con_output, p=2, dim=1)
            near_embeddings = contrastive_embeddings_device[near_indices]
            near_embeddings_normalized = F.normalize(near_embeddings, p=2, dim=2)
            cos_sim_near = F.cosine_similarity(output_normalized.unsqueeze(1), near_embeddings_normalized, dim=2)
            loss_attraction = cos_sim_near.mean(dim=1)

            # output_normalized = F.normalize(surrogate_output, p=2, dim=1)
            # near_embeddings = embeddings_device[near_indices]
            # near_embeddings_normalized = F.normalize(near_embeddings, p=2, dim=2)
            # cos_sim_near = F.cosine_similarity(output_normalized.unsqueeze(1), near_embeddings_normalized, dim=2)
            # loss_attraction = cos_sim_near.mean(dim=1)

            #    b. Farthest-neighbor repulsion loss
            # far_embeddings = embeddings_device[far_indices]
            # # loss_repulsion = F.mse_loss(output_expanded, far_embeddings)
            # loss_repulsion = l2_norm_loss(output_expanded, far_embeddings).mean()

            far_embeddings = contrastive_embeddings_device[far_indices]
            far_embeddings_normalized = F.normalize(far_embeddings, p=2, dim=2)
            cos_sim_far = F.cosine_similarity(output_normalized.unsqueeze(1), far_embeddings_normalized, dim=2)
            loss_repulsion = cos_sim_far.mean(dim=1)

            # 3. KL divergence loss for distribution matching
            # dist_true = get_similarity_distribution(true_emb, temperature=2)
            # dist_surrogate = get_similarity_distribution(surrogate_output, temperature=2)
            # loss_kl = F.kl_div(dist_surrogate.log(), dist_true, reduction='batchmean')

            dist_true = F.softmax(true_emb / tau_soft, dim=1)
            # dist_true = F.softmax(true_emb / 2, dim=1)

            # b. Convert each surrogate output feature vector to a probability distribution
            dist_surrogate = F.softmax(surrogate_output / tau_soft, dim=1)
            # dist_surrogate = F.softmax(surrogate_output / 2, dim=1)

            # c. Compute KL divergence; F.kl_div expects log-probabilities as its first input
            # loss_kl = F.kl_div(dist_surrogate.log(), dist_true)
            loss_kl = F.kl_div(dist_surrogate.log(), dist_true, reduction='none').sum(dim=1)

            # 4. Combine the total loss
            #    - Subtract loss_repulsion to maximize this distance
            #    - lambda_kl balances the KL divergence loss
            total_loss = (1 * loss_steal - cfg.alpha_ms * loss_attraction + cfg.beta_ms * loss_repulsion + 1 * loss_kl).mean()

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
        # evaluate_surrogate_model(cfg, client, server, num_clients, original_pred_list, epoch, history_metrics)
        adjust_learning_rate(epoch, cfg.lr_ms, optimizer)
        if client.rank == 0:
            tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss / len(aux_emb_loader):.4f}")
    # if client.rank == 0:
    #     best_acc = max(history_metrics['acc_history']) if history_metrics['acc_history'] else 0
    #     best_agr = max(history_metrics['agr_history']) if history_metrics['agr_history'] else 0
    #     tqdm.write(f"Highest Attack ACC: {best_acc:.2f}%")
    #     tqdm.write(f"Highest Agreement: {best_agr:.2f}%")

    extract_both_embeddings(client)
    # extract_embedding_from_surrogate_model_knn(client)


def model_stealing_attack_original_space_knn(cfg, client, server, num_clients, original_pred_list):
    """
    [Ablation experiment] (the paper's "w/o contrast" method)
    Find neighbors and compute losses directly in the original space without contrastive learning.
    """
    print("--- Running Ablation Study: w/o Contrastive Learning (KNN in Original Space) ---")

    # --- Step 1: unchanged ---
    embeddings = extract_embeddings_from_bottom_model(cfg, client)

    # --- Ablation: remove step 2 ---
    # Remove train_contrastive_encoder and generate_contrastive_embeddings

    # --- Ablation: find neighbors in the original space in step 3 ---
    print("Step 2: Finding neighbors in ORIGINAL embedding space...")
    knn_indices, _, farthest_indices, _ = find_knn_brute_force(embeddings, cfg)  # Use original embeddings directly

    # --- Step 4: Prepare training as in the original function ---
    knn_indices_np = knn_indices
    farthest_indices_np = farthest_indices
    aux_emb_dataset = MydatasetForHybridLoss(
        client.aux_dataset.aux_data_ts,
        embeddings,
        knn_indices_np,
        farthest_indices_np
    )
    aux_emb_loader = tud.DataLoader(
        aux_emb_dataset,
        batch_size=cfg.bs_ms,
        shuffle=True,
    )

    # Create the surrogate model using the existing logic
    output_dim = embeddings.shape[1]
    if cfg.type == "tabular":
        in_dim = client.aux_dataset.aux_data_ts.shape[1]
        client.surrogate_model = MLPBottomModel(in_dim, output_dim, cfg.bottom_model_layers).to(client.device)
    elif cfg.type == "image":
        if cfg.dataset == "mnist":
            client.surrogate_model = ResBottomModel(1, cfg.surrogate_model_layers, output_dim).to(client.device)
        elif cfg.dataset == "cifar10":
            client.surrogate_model = ResBottomModel(3, cfg.surrogate_model_layers, output_dim).to(client.device)

    optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[cfg.get("optimizer_ms", "AdamW")]
    optimizer = optimizer_class(client.surrogate_model.parameters(), lr=cfg.lr_ms)
    client.surrogate_model.train()

    # --- Ablation: remove code related to encoder and contrastive_embeddings ---
    embeddings_device = embeddings.to(client.device)  # Only original embeddings are needed

    # Initialize the history dictionary
    history_metrics = {}
    if client.rank == 0:
        history_metrics = {'acc_history': [], 'agr_history': [], 'mse_history': [], 'agr_mse_history': []}

    # --- Training loop for the ablation experiment ---
    for epoch in range(cfg.epochs):
        epoch_loss = 0
        for data, true_emb, near_indices, far_indices in aux_emb_loader:
            data = data.to(client.device)
            true_emb = true_emb.to(client.device)
            near_indices = near_indices.to(client.device)
            far_indices = far_indices.to(client.device)

            optimizer.zero_grad()
            surrogate_output = client.surrogate_model(data)

            # 1. Core stealing loss in the original space, unchanged
            loss_steal = F.mse_loss(surrogate_output, true_emb, reduction='none').mean(dim=1)

            # --- Ablation: compute neighborhood loss in the original space ---
            output_normalized = F.normalize(surrogate_output, p=2, dim=1)  # Use surrogate_output directly

            near_embeddings = embeddings_device[near_indices]  # Use original embeddings
            near_embeddings_normalized = F.normalize(near_embeddings, p=2, dim=2)
            cos_sim_near = F.cosine_similarity(output_normalized.unsqueeze(1), near_embeddings_normalized, dim=2)
            loss_attraction = cos_sim_near.mean(dim=1)

            far_embeddings = embeddings_device[far_indices]  # Use original embeddings
            far_embeddings_normalized = F.normalize(far_embeddings, p=2, dim=2)
            cos_sim_far = F.cosine_similarity(output_normalized.unsqueeze(1), far_embeddings_normalized, dim=2)
            loss_repulsion = cos_sim_far.mean(dim=1)

            # 3. KL divergence loss in the original space, unchanged
            dist_true = F.softmax(true_emb / 2, dim=1)
            dist_surrogate = F.softmax(surrogate_output / 2, dim=1)
            loss_kl = F.kl_div(dist_surrogate.log(), dist_true, reduction='none').sum(dim=1)

            # 4. Combine the total loss using the same formula
            total_loss = (
                        1 * loss_steal - cfg.alpha_ms * loss_attraction + cfg.beta_ms * loss_repulsion + 1 * loss_kl).mean()

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        # Evaluation and logging
        adjust_learning_rate(epoch, cfg.lr_ms, optimizer)
        if client.rank == 0:
            tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss / len(aux_emb_loader):.4f}")
        evaluate_surrogate_model(cfg, client, server, num_clients, original_pred_list, epoch, history_metrics)

    # Print the final summary
    if client.rank == 0:
        tqdm.write("\n--- [Ablation: w/o Contrast] Final Summary ---")
        if cfg.task == "classification":
            best_acc = max(history_metrics['acc_history']) if history_metrics['acc_history'] else 0
            best_agr = max(history_metrics['agr_history']) if history_metrics['agr_history'] else 0
            tqdm.write(f"Highest Attack ACC: {best_acc:.2f}%")
            tqdm.write(f"Highest Agreement: {best_agr:.2f}%")
        # ... Regression-task output omitted ...
        tqdm.write("==============================================\n")



class MydatasetForHybridLoss_idx(tud.Dataset):
    """
    [Dataset for the combined loss]
    To support the combined loss function, this dataset returns:
    1. Anchor data (anchor_data)
    2. The anchor's true embedding (anchor_true_embedding)
    3. Indices of K nearest neighbors (near_indices)
    4. Indices of K farthest neighbors (far_indices)
    """

    def __init__(self, aux_data, all_embeddings, near_indices, far_indices):
        self.aux_data = aux_data
        self.all_embeddings = all_embeddings
        self.idx = [i for i in range(len(all_embeddings))]
        self.near_indices = near_indices
        self.far_indices = far_indices

    def __len__(self):
        return len(self.aux_data)

    def __getitem__(self, index):
        return (self.aux_data[index],
                self.all_embeddings[index],
                self.idx[index],
                self.near_indices[index],
                self.far_indices[index])


def model_stealing_attack_with_contrastive_and_knn_1(cfg, client):

    # --- Steps 1 and 2: unchanged ---
    embeddings = extract_embeddings_from_bottom_model(cfg, client)
    encoder = train_contrastive_encoder(cfg, client, embeddings)
    contrastive_embeddings = generate_contrastive_embeddings(encoder, embeddings, client)

    # --- Step 3: Find neighbors using the existing logic ---
    knn_indices, _, farthest_indices, _ = find_knn_brute_force(contrastive_embeddings, cfg)

    # --- Step 4: Prepare training ---
    knn_indices_np = knn_indices
    farthest_indices_np = farthest_indices

    # Use the revised dataset to provide the required data
    aux_emb_dataset = MydatasetForHybridLoss_idx(
        client.aux_dataset.aux_data_ts,
        embeddings,
        knn_indices_np,
        farthest_indices_np
    )

    aux_emb_loader = tud.DataLoader(
        aux_emb_dataset,
        batch_size=cfg.bs_ms,
        shuffle=True,
    )

    # Create the surrogate model using the existing logic
    output_dim = embeddings.shape[1]
    if cfg.type == "tabular":
        in_dim = client.aux_dataset.aux_data_ts.shape[1]
        client.surrogate_model = MLPBottomModel(in_dim, output_dim, cfg.bottom_model_layers).to(client.device)
    elif cfg.type == "image":
        if cfg.dataset == "mnist":
            client.surrogate_model = ResBottomModel(1, cfg.surrogate_model_layers, output_dim).to(client.device)
        elif cfg.dataset == "cifar10":
            client.surrogate_model = ResBottomModel(3, cfg.surrogate_model_layers, output_dim).to(client.device)


    optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[cfg.get("optimizer_ms", "AdamW")]
    optimizer = optimizer_class(client.surrogate_model.parameters(), lr=cfg.lr_ms)
    embeddings_device = embeddings.to(client.device)
    client.surrogate_model.train()

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    contrastive_embeddings_device = contrastive_embeddings.to(client.device)

    for epoch in range(cfg.epochs):
        epoch_loss = 0
        for data, true_emb, idx, near_indices, far_indices in aux_emb_loader:
            data = data.to(client.device)
            true_emb = true_emb.to(client.device)
            idx = idx.to(client.device)
            near_indices = near_indices.to(client.device)
            far_indices = far_indices.to(client.device)

            optimizer.zero_grad()
            surrogate_output = client.surrogate_model(data)

            loss_steal = F.mse_loss(surrogate_output, true_emb, reduction='none').mean(dim=1)

            surrogate_con_output = encoder.encode_only(surrogate_output)
            anchor_con_embs = contrastive_embeddings_device[idx]
            near_con_embs = contrastive_embeddings_device[near_indices]  # Shape: [B, k, D]
            far_con_embs = contrastive_embeddings_device[far_indices]
            # loss_direct_con = F.mse_loss(surrogate_con_output, anchor_con_embs, reduction='none').mean(dim=1)

            sim_near_true = F.cosine_similarity(anchor_con_embs.unsqueeze(1), near_con_embs, dim=2)  # Shape: [B, k]
            sim_far_true = F.cosine_similarity(anchor_con_embs.unsqueeze(1), far_con_embs, dim=2)  # Shape: [B, k]

            sim_near_surrogate = F.cosine_similarity(surrogate_con_output.unsqueeze(1), near_con_embs, dim=2)
            sim_far_surrogate = F.cosine_similarity(surrogate_con_output.unsqueeze(1), far_con_embs, dim=2)

            dist_near_true = F.softmax(sim_near_true / 3, dim=1)
            dist_far_true = F.softmax(sim_far_true / 3, dim=1)

            dist_near_surrogate = F.softmax(sim_near_surrogate / 3, dim=1)
            dist_far_surrogate = F.softmax(sim_far_surrogate / 3, dim=1)

            loss_kl_knn = F.kl_div(dist_near_surrogate.log(), dist_near_true, reduction='none').sum(dim=1)
            loss_kl_kfn = F.kl_div(dist_far_surrogate.log(), dist_far_true, reduction='none').sum(dim=1)

            dist_true = F.softmax(true_emb / 5, dim=1)

            # b. Convert each surrogate output feature vector to a probability distribution
            dist_surrogate = F.softmax(surrogate_output / 5, dim=1)

            # c. Compute KL divergence; F.kl_div expects log-probabilities as its first input
            # loss_kl = F.kl_div(dist_surrogate.log(), dist_true)
            loss_kl = F.kl_div(dist_surrogate.log(), dist_true, reduction='none').sum(dim=1)

            # 4. Combine the total loss
            #    - Subtract loss_repulsion to maximize this distance
            #    - lambda_kl balances the KL divergence loss
            total_loss = (1 * loss_steal + cfg.alpha_ms * loss_kl_knn + cfg.beta_ms * loss_kl_kfn + 1 * loss_kl).mean()

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        adjust_learning_rate(epoch, cfg.lr_ms, optimizer)
        if client.rank == 0:
            tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss / len(aux_emb_loader):.4f}")


# def model_stealing_attack_with_contrastive_and_knn(cfg, client):
#     """
#     Corrected model stealing attack
#     """
#     print("Now in the FIXED model_stealing_attack_with_contrastive_and_knn")
#
#     # Step 1: Obtain original embeddings
#     embeddings = extract_embeddings_from_bottom_model(cfg, client)
#
#     # Step 2: Train the contrastive encoder
#     encoder = train_contrastive_encoder(cfg, client, embeddings)
#     contrastive_embeddings = generate_contrastive_embeddings(encoder, embeddings, client)
#
#     # Step 3: Find neighbors using cosine similarity
#     # Ensure the called function is the cosine-similarity version, even if its name differs
#     knn_indices, _, farthest_indices, _ = find_knn_brute_force(contrastive_embeddings, cfg)
#
#     # Step 4: Prepare training
#     knn_indices = torch.from_numpy(knn_indices)
#     farthest_indices = torch.from_numpy(farthest_indices)
#
#     # Create the surrogate model
#     output_dim = cfg.bottom_model_output_dim
#     if cfg.defense == "pruning":
#         output_dim = round(output_dim * (1 - cfg.pruning_ratio))
#
#     if cfg.type == "tabular":
#         client.surrogate_model = MLPBottomModel(client.bottom_models[0].in_dim, output_dim,
#                                                 cfg.bottom_model_layers).to(client.device)
#     elif cfg.type == "image":
#         client.surrogate_model = ResBottomModel(1, cfg.bottom_model_output_dim).to(client.device)
#     else:
#         raise ValueError(f"Invalid data type: {cfg.type}")
#
#     # Use the corrected dataset
#     aux_emb_dataset = MyDatasetKnn_Near_Far(client.aux_dataset.aux_data_ts, knn_indices, farthest_indices)
#     aux_emb_loader = tud.DataLoader(aux_emb_dataset, batch_size=cfg.bs_ms, shuffle=True)
#
#     optimizer = torch.optim.AdamW(client.surrogate_model.parameters(), lr=cfg.lr_ms)
#
#     # Move all true embeddings to the device for fast indexing
#     embeddings_device = embeddings.to(client.device)
#     client.surrogate_model.train()
#
#     # --- Step 2: Update the training loop and loss function ---
#     for epoch in range(cfg.epochs):
#         epoch_loss = 0
#         for data, indices, near_knn, far_knn in aux_emb_loader:
#             data = data.to(client.device)
#             indices = indices.to(client.device)
#             near_knn = near_knn.to(client.device)
#             far_knn = far_knn.to(client.device)
#
#             # Run a forward pass to obtain surrogate model outputs
#             output = client.surrogate_model(data)
#
#             # --- Core changes ---
#
#             # 1. Corrected stealing loss L_steal (L2 distance)
#             # Target the input sample's own true embedding
#             true_embeddings = embeddings_device[indices]
#             # loss_steal = l2_norm_loss(output, true_embeddings)
#             loss_steal_l2 = l2_norm_loss(output, true_embeddings).mean()
#
#             # Prepare to compute cosine similarity
#             output_normalized = F.normalize(output, p=2, dim=1)
#             true_embeddings_normalized = F.normalize(true_embeddings, p=2, dim=1)
#
#             # 1b. Cosine similarity loss to match direction
#             # Maximize cosine similarity by minimizing (1 - cos_sim)
#             cos_sim_steal = F.cosine_similarity(output_normalized, true_embeddings_normalized, dim=1)
#             loss_steal_cosine = (1 - cos_sim_steal).mean()
#
#             # 2. Corrected nearest-neighbor attraction loss L_knn (maximize cosine similarity)
#             # PyTorch optimizers minimize loss, so minimize `1 - cos_sim`
#             near_embeddings = embeddings_device[near_knn]  # Shape: [batch, k, dim]
#             near_embeddings_normalized = F.normalize(near_embeddings, p=2, dim=2)
#             # output_normalized shape: [batch, dim] -> [batch, 1, dim] for broadcasting
#             cos_sim_near = F.cosine_similarity(output_normalized.unsqueeze(1), near_embeddings_normalized, dim=2)
#             loss_attraction = cos_sim_near.mean()
#
#             # 3. Corrected farthest-neighbor repulsion loss L_kfn (minimize cosine similarity)
#             # Minimize `cos_sim`, equivalently the negation of `-cos_sim`
#             far_embeddings = embeddings_device[far_knn]  # Shape: [batch, k, dim]
#             far_embeddings_normalized = F.normalize(far_embeddings, p=2, dim=2)
#             cos_sim_far = F.cosine_similarity(output_normalized.unsqueeze(1), far_embeddings_normalized, dim=2)
#             loss_repulsion = cos_sim_far.mean()
#
#             # 4. Combine the corrected total loss
#             # Add repulsion loss because the objective is to minimize cos_sim_far
#             # total_loss = loss_steal_l2 + loss_steal_cosine + cfg.alpha_ms * loss_attraction + cfg.beta_ms * loss_repulsion
#             total_loss = 0.8 * loss_steal_l2 + (loss_steal_cosine - 0.5 * loss_attraction  + 0.5 * loss_repulsion)
#
#             epoch_loss += total_loss.item()
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()
#
#         adjust_learning_rate(epoch, cfg.lr_ms, optimizer)
#         if client.rank == 0:
#             tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss / len(aux_emb_loader):.4f}")
#

# -------------------------------------------------------------------------------------------------------------------

def euclidean_dist(vector_a, vector_b):
    """

    :param vector_a: a
    :param vector_b: b
    :return: Euclidean distance between a and b
    """
    return np.sqrt(sum(np.power((vector_a - vector_b), 2)))


# def model_stealing_attack_with_kmeans(cfg, client):
#     """
#         model stealing attack
#         :param cfg: mlp_config / cnn_config
#         :param client: each client
#         :return: None
#         """
#     # input_dim = client.train_dataset.num_features
#     print("Now in model_stealing_attack_with_kmeans.")
#     output_dim = cfg.bottom_model_output_dim
#
#
#     if cfg.defense == "pruning":
#         output_dim = round(output_dim * (1 - cfg.pruning_ratio))
#
#     if cfg.type == "tabular":
#         client.surrogate_model = MLPBottomModel(client.bottom_models[0].in_dim, output_dim,
#                                                 cfg.bottom_model_layers).to(client.device)
#     elif cfg.type == "image":
#         client.surrogate_model = ResBottomModel(1, cfg.bottom_model_output_dim).to(client.device)
#     else:
#         print("Invalid type")
#         return
#
#     client.surrogate_model.train()
#     embeddings = extract_embeddings_from_bottom_model(cfg, client)
#     # embedding = []
#     # for data, hash_res, y in client.aux_dataloader:
#     #     data = data.to(client.device)
#     #     hash_res = hash_res.to(client.device)
#     #     for item in zip(data, hash_res):
#     #         output = client.bottom_models[item[1]](item[0])
#     #         embedding.append(output.detach())
#     # embeddings = torch.stack(embedding)
#     #
#     # if cfg.defense == "noise":
#     #     embeddings = noisy_embedding_based_defense(embeddings, cfg.noise_std)
#     # elif cfg.defense == "pruning":
#     #     embeddings, _ = pruning_embedding_based_defense_by_removing_elements(embeddings, cfg.pruning_ratio)
#     # elif cfg.defense == "ResSFL":
#     #     pass
#     # elif cfg.defense == "DPSGD":
#     #     pass
#
#     weights, prototypes = calculate_weight(cfg.k, embeddings.cpu(), cfg.temperature)
#     prototypes = prototypes.to(client.device)
#
#     # aux_emb_dataset = MyDatasetTS(client.aux_dataset.aux_data_ts, embeddings)
#     aux_emb_dataset = MyDatasetWeightsTS(client.aux_dataset.aux_data_ts, embeddings, weights)
#
#     # criterion = l2_norm_loss
#     criterion = loss_fn
#     aux_emb_loader = tud.DataLoader(aux_emb_dataset, batch_size=cfg.bs_ms)
#     optimizer = torch.optim.Adadelta(client.surrogate_model.parameters(), lr=cfg.lr_ms)
#
#     # for epoch in tqdm(range(cfg.epochs), desc="Attack Train", dynamic_ncols=True):
#     with tqdm(range(cfg.epochs), desc="Train", dynamic_ncols=True) as pbar:
#         for epoch in pbar:
#             epoch_loss = 0
#             for data, y, weight in aux_emb_loader:
#                 data = data.to(client.device)
#                 y = y.to(client.device)
#                 weight = weight.to(client.device)
#                 output = client.surrogate_model(data)
#                 loss = criterion(output, y, prototypes, weight, cfg.lambda_val)
#                 epoch_loss += loss.item()
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#             # adjust_learning_rate(epoch, cfg.lr_ms, optimizer)
#             pbar.set_postfix(Epoch=f"{epoch}", loss=f"{epoch_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.4f}")


class DistillKnnDataset(tud.Dataset):
    """
    Dataset for the revised method.
    Return: 1. Anchor data, 2. Its true original embedding, 3. Nearest-neighbor indices, 4. Farthest-neighbor indices.
    (Neighbor input data and original embeddings are unnecessary because neighbor operations use contrastive space.)
    """

    def __init__(self, aux_data, all_embeddings, near_indices, far_indices):
        self.aux_data = aux_data
        self.all_embeddings = all_embeddings
        self.near_indices = near_indices
        self.far_indices = far_indices

    def __len__(self):
        return len(self.aux_data)

    def __getitem__(self, index):
        return (self.aux_data[index],
                self.all_embeddings[index],
                self.near_indices[index],
                self.far_indices[index])


# def model_stealing_via_distill_knn(cfg, client):
#     """
#     Revised method: k-NN knowledge distillation in contrastive space
#     """
#     print("--- Running Model Stealing via k-NN Distillation in Contrastive Space ---")
#
#     # --- Steps 1 and 2: unchanged ---
#     embeddings = extract_embeddings_from_bottom_model(cfg, client)
#     encoder = train_contrastive_encoder(cfg, client, embeddings)
#     contrastive_embeddings = generate_contrastive_embeddings(encoder, embeddings, client)
#
#     # --- Step 3: Find k-NN in contrastive space ---
#     knn_indices, _, farthest_indices, _ = find_knn_brute_force(contrastive_embeddings, cfg)
#
#     # --- Step 4: Prepare training ---
#     dataset = DistillKnnDataset(client.aux_dataset.aux_data_ts, embeddings, knn_indices, farthest_indices)
#     loader = tud.DataLoader(dataset, batch_size=cfg.bs_ms, shuffle=True)
#
#     # Create the surrogate model using the existing logic
#     output_dim = embeddings.shape[1]
#     if cfg.type == "tabular":
#         in_dim = client.aux_dataset.aux_data_ts.shape[1]
#         client.surrogate_model = MLPBottomModel(in_dim, output_dim, cfg.bottom_model_layers).to(client.device)
#     elif cfg.type == "image":
#         client.surrogate_model = ResBottomModel(1, output_dim).to(client.device)
#
#     optimizer = torch.optim.AdamW(client.surrogate_model.parameters(), lr=cfg.lr_ms)
#     client.surrogate_model.train()
#
#     # Set the reference encoder to evaluation mode and freeze its gradients
#     encoder.eval()
#     for param in encoder.parameters():
#         param.requires_grad = False
#
#     # Move all contrastive embeddings to the device for fast lookup
#     contrastive_embeddings_device = contrastive_embeddings.to(client.device)
#
#     # --- Training loop for the revised method ---
#     for epoch in range(cfg.epochs):
#         epoch_loss = 0
#         for data, true_emb, near_indices, far_indices in loader:
#             data = data.to(client.device)
#             true_emb = true_emb.to(client.device)
#             near_indices = near_indices.to(client.device)
#             far_indices = far_indices.to(client.device)
#
#             optimizer.zero_grad()
#
#             # 1. Generate surrogate model outputs
#             surrogate_output = client.surrogate_model(data)
#
#             # --- Revised loss function ---
#
#             # a. Core stealing loss for the anchor in the original space
#             # loss_steal = F.mse_loss(surrogate_output, true_emb)
#             loss_steal = F.mse_loss(surrogate_output, true_emb, reduction='none').mean(dim=1)
#
#
#             # b. Project surrogate model outputs into contrastive space
#             surrogate_con_output = encoder.encode_only(surrogate_output)
#
#             # c. Prepare to compute neighborhood loss
#             #    Expand surrogate outputs in contrastive space to match neighbor shapes
#             k = cfg.knn_k
#             surrogate_con_output_expanded = surrogate_con_output.unsqueeze(1).expand(-1, k, -1)
#
#             #    Look up true neighbors in contrastive_embeddings_device
#             near_con_embeddings = contrastive_embeddings_device[near_indices]
#             far_con_embeddings = contrastive_embeddings_device[far_indices]
#
#             # d. Nearest-neighbor attraction loss in contrastive space
#             loss_attraction = F.mse_loss(surrogate_con_output_expanded, near_con_embeddings)
#
#             # e. Farthest-neighbor repulsion loss in contrastive space
#             loss_repulsion = F.mse_loss(surrogate_con_output_expanded, far_con_embeddings)
#
#             # f. Total loss
#             total_loss = loss_steal + cfg.alpha_ms * loss_attraction - cfg.beta_ms * loss_repulsion
#
#             total_loss.backward()
#             optimizer.step()
#             epoch_loss += total_loss.item()
#
#         if client.rank == 0:
#             tqdm.write(f">>> Epoch: {epoch}, Train Loss: {epoch_loss / len(loader):.4f}")




def _contsteal_info_nce_cross(z_s, z_t, tau: float = 0.1):
    """Cross-model InfoNCE: make surrogate embeddings match target embeddings sample-wise.
    Args:
        z_s: surrogate embeddings, shape [B, D]
        z_t: target embeddings, shape [B, D]
        tau: temperature
    Returns:
        scalar loss
    """
    import torch
    import torch.nn.functional as F

    z_s = F.normalize(z_s, p=2, dim=1)
    z_t = F.normalize(z_t, p=2, dim=1)
    logits = (z_s @ z_t.t()) / tau
    labels = torch.arange(logits.size(0), device=logits.device)
    # Symmetric cross-entropy (rows + cols)
    loss_row = F.cross_entropy(logits, labels)
    loss_col = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_row + loss_col)


def _contsteal_augment_image(x, pad: int = 4, p_flip: float = 0.5):
    """Lightweight image augmentation without torchvision.
    Works for x of shape [C,H,W] or [B,C,H,W] in [0,1] or standard normalized space.
    """
    if isinstance(p_flip, (list, tuple)):
        p_flip = p_flip[0] if len(p_flip) > 0 else 0.0
    if p_flip is None:
        p_flip = 0.0
    p_flip = float(p_flip)
    if x.dim() == 3:
        x = x.unsqueeze(0)  # [1,C,H,W]
        squeeze_back = True
    else:
        squeeze_back = False

    b, c, h, w = x.shape
    if pad and pad > 0:
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        _, _, hp, wp = x.shape
        top = torch.randint(0, hp - h + 1, (b,), device=x.device)
        left = torch.randint(0, wp - w + 1, (b,), device=x.device)
        # crop per sample
        crops = []
        for i in range(b):
            crops.append(x[i:i+1, :, top[i]:top[i]+h, left[i]:left[i]+w])
        x = torch.cat(crops, dim=0)

    if p_flip is not None and p_flip > 0:
        flip_mask = (torch.rand((b,), device=x.device) < p_flip)
        if flip_mask.any():
            x[flip_mask] = torch.flip(x[flip_mask], dims=[3])  # horizontal

    return x.squeeze(0) if squeeze_back else x


def _contsteal_augment_tabular(x, noise_std: float = 0.01, drop_prob: float = 0.0):
    """Simple tabular augmentation: Gaussian noise + optional feature dropout."""
    import torch
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    if noise_std and noise_std > 0:
        x = x + noise_std * torch.randn_like(x)

    if drop_prob and drop_prob > 0:
        mask = (torch.rand_like(x) < drop_prob)
        x = x.masked_fill(mask, 0.0)

    return x.squeeze(0) if squeeze_back else x


def model_stealing_attack_cont_steal(cfg, client, tau: float = 0.1, pad: int = 4, p_flip: float = 0.5,
                                    tab_noise_std: float = 0.01, tab_drop_prob: float = 0.0,
                                    use_mse_aux: bool = False, mse_weight: float = 1.0):
    """Cont-Steal style baseline: contrastive stealing against the client bottom model.
    This implements a *teacher-student* contrastive objective between the surrogate (student) and
    the victim bottom model outputs (teacher), similar to Cont-Steal (CVPR'23), adapted to this codebase.

    Usage:
        model_stealing_attack_cont_steal(cfg, client)

    Notes:
        - For image data, we create two augmentations per sample and query the victim bottom model
          to obtain teacher embeddings for both views, then train the surrogate using cross-model InfoNCE.
        - For tabular data, we apply lightweight noise/dropout augmentations.
        - If use_mse_aux=True, we additionally add an MSE term to stabilize training (optional).

    Side effect:
        Sets client.surrogate_model.
    """
    import torch
    import torch.nn.functional as F
    import torch.utils.data as tud
    from tqdm import tqdm

    # 1) Build surrogate model (match your existing baseline construction)
    output_dim = cfg.bottom_model_output_dim
    if getattr(cfg, "defense", None) == "pruning":
        output_dim = round(output_dim * (1 - cfg.pruning_ratio))

    if cfg.type == "tabular":
        client.surrogate_model = MLPBottomModel(client.bottom_models[0].in_dim, output_dim,
                                                cfg.bottom_model_layers).to(client.device)
    elif cfg.type == "image":
        if cfg.dataset == "mnist":
            client.surrogate_model = ResBottomModel(1, cfg.surrogate_model_layers, output_dim).to(client.device)
        elif cfg.dataset == "cifar10":
            client.surrogate_model = ResBottomModel(3, cfg.surrogate_model_layers, output_dim).to(client.device)
        else:
            # default to 3-channel
            client.surrogate_model = ResBottomModel(3, cfg.surrogate_model_layers, output_dim).to(client.device)
    else:
        raise ValueError(f"Unsupported cfg.type={cfg.type}")

    client.surrogate_model.train()

    # 2) Optimizer
    optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[cfg.get("optimizer_ms", "AdamW")]
    optimizer = optimizer_class(client.surrogate_model.parameters(), lr=cfg.lr_ms)

    # 3) Training loop on auxiliary queries
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in client.aux_dataloader:
            # aux_dataloader yields (data, hash_res, y) in your code
            if len(batch) == 3:
                data, hash_res, _ = batch
            else:
                data, hash_res = batch[0], batch[1]

            data = data.to(client.device)
            hash_res = hash_res.to(client.device)

            # Build two augmented views and query victim bottom model for teacher embeddings
            x1_list, x2_list, t1_list, t2_list = [], [], [], []
            for x, h in zip(data, hash_res):
                if cfg.type == "image":
                    x1 = _contsteal_augment_image(x, pad=pad, p_flip=p_flip)
                    x2 = _contsteal_augment_image(x, pad=pad, p_flip=p_flip)
                else:
                    x1 = _contsteal_augment_tabular(x, noise_std=tab_noise_std, drop_prob=tab_drop_prob)
                    x2 = _contsteal_augment_tabular(x, noise_std=tab_noise_std, drop_prob=tab_drop_prob)

                # teacher embeddings from the (possibly defense-wrapped) bottom model output
                # NOTE: if you apply defenses at the embedding channel in evaluation, keep it here consistent.
                with torch.no_grad():
                    t1 = client.bottom_models[int(h)](x1)
                    t2 = client.bottom_models[int(h)](x2)

                x1_list.append(x1)
                x2_list.append(x2)
                t1_list.append(t1.detach())
                t2_list.append(t2.detach())

            x1b = torch.stack(x1_list, dim=0)
            x2b = torch.stack(x2_list, dim=0)
            t1b = torch.stack(t1_list, dim=0)
            t2b = torch.stack(t2_list, dim=0)

            # If defenses are applied to embeddings (noise/pruning/projection), apply them to teacher embeddings here
            if getattr(cfg, "defense", None) == "noise":
                t1b = noisy_embedding_based_defense(t1b, cfg.noise_std)
                t2b = noisy_embedding_based_defense(t2b, cfg.noise_std)
            elif getattr(cfg, "defense", None) == "pruning":
                t1b, _ = pruning_embedding_based_defense_by_removing_elements(t1b, cfg.pruning_ratio)
                t2b, _ = pruning_embedding_based_defense_by_removing_elements(t2b, cfg.pruning_ratio)
            elif getattr(cfg, "defense", None) == "projection":
                t1b = random_projection_based_defense(t1b, client.projection_matrix)
                t2b = random_projection_based_defense(t2b, client.projection_matrix)

            # surrogate embeddings
            s1 = client.surrogate_model(x1b)
            s2 = client.surrogate_model(x2b)

            # Cont-Steal style cross-model contrastive loss
            loss = _contsteal_info_nce_cross(s1, t2b, tau=tau) + _contsteal_info_nce_cross(s2, t1b, tau=tau)
            loss = 0.5 * loss

            if use_mse_aux:
                loss = loss + mse_weight * (F.mse_loss(s1, t1b) + F.mse_loss(s2, t2b)) * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1
        # evaluate_surrogate_model(cfg, client, server, num_clients, original_pred_list, epoch, history_metrics)
        adjust_learning_rate(epoch, cfg.lr_ms, optimizer)
        if client.rank == 0:
            tqdm.write(f">>> [Cont-Steal] Epoch: {epoch}, train loss: {epoch_loss / max(n_batches, 1):.4f}")

    extract_both_embeddings(client)

    # return client.surrogate_model
