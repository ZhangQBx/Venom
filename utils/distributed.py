# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2024/12/17 4:29 PM
@File :distributed.py
"""
import time
from itertools import count
import os
# from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from utils.defense import *
from utils.model_stealing_attack import *

def train_one_model(cfg, client, server, num_clients):
    """
    training stage for one bottom model
    :param cfg:
    :param client:
    :param server:
    :param num_clients:
    :return:
    """
    print("trainning mode: 1 model")

    if cfg.task == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    elif cfg.task == 'regression':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError('invalid task type')

    bottom_output_dim = cfg.bottom_model_output_dim

    if cfg.defense == 'pruning':
        bottom_output_dim = round(cfg.bottom_model_output_dim * (1 - cfg.pruning_ratio))

    b4b_ctx = None
    if cfg.defense == "b4b":
        # ===== hard-coded B4B params =====
        B4B_HASH_BITS = 12
        B4B_BASE_SEED = 12345
        B4B_NOISE_STD = 0.0  # e.g., 0.01~0.05 if you want extra randomization
        # Each client uses a different secret to prevent cross-client alignment
        b4b_ctx = B4BContext(
            n_hash_bits=B4B_HASH_BITS,
            seed=B4B_BASE_SEED + int(client.rank),
            noise_std=B4B_NOISE_STD,
        )

    print(f"client output dim:{bottom_output_dim}")

    for model in client.bottom_models:
        model.to(client.device)

    # for epoch in tqdm(range(cfg.epochs), desc="Train", dynamic_ncols=True):
    with tqdm(range(cfg.epochs), desc="Train", dynamic_ncols=True) as pbar:
        for epoch in pbar:
            epoch_loss = 0
            right_pred = 0
            for data, _, y in client.train_dataloader:
                data = data.to(client.device)
                if cfg.defense == "dnp":
                    # data =  invl_dnp_defense(data, client.bottom_models[0])
                    data =  invl_dnp_defense_optimized(data, client.bottom_models[0])
                batch_outputs = client.bottom_models[0](data)

                if cfg.defense == "noise":
                    batch_outputs = noisy_embedding_based_defense(batch_outputs, cfg.noise_std)
                elif cfg.defense == "pruning":
                    batch_outputs, mask = pruning_embedding_based_defense_by_removing_elements(batch_outputs, cfg.pruning_ratio)
                    mask.to(client.device)
                elif cfg.defense == 'projection':
                    batch_outputs = random_projection_based_defense(batch_outputs, client.projection_matrix)
                elif cfg.defense == "enp":
                    # batch_outputs = invl_enp_defense(batch_outputs, data, client.bottom_models[0])
                    batch_outputs = invl_enp_defense_optimized(batch_outputs, data, client.bottom_models[0])
                elif cfg.defense == "b4b":
                    batch_outputs, _ = b4b_defense(batch_outputs, ctx=b4b_ctx, return_bucket=True)
                elif cfg.defense == "ResSFL":
                    pass
                elif cfg.defense == "DPSGD":
                    pass
                elif cfg.defense == "no":
                    pass
                elif cfg.defense == "dnp":
                    pass
                else:
                    raise ValueError('invalid defense type')

                gather_list = [torch.empty_like(batch_outputs) for _ in range(num_clients)]
                dist.all_gather(gather_list, batch_outputs)
                grad = torch.zeros((len(data), num_clients * bottom_output_dim))

                if client.rank == 0:
                    server.top_model.to(server.device)
                    y = y.to(server.device)
                    top_input = torch.cat(gather_list, dim=1).requires_grad_(True)
                    top_input.retain_grad()
                    top_output = server.top_model(top_input)

                    # pred = top_output.detach()
                    # _, predicted = pred.max(1)
                    # label = torch.from_numpy(y.nonzero().cpu().numpy()[:, 1]).to(server.device)
                    # right_pred += predicted.eq(label).sum().item()

                    loss = criterion(top_output, y)
                    server.top_optimizer.zero_grad()
                    loss.backward()
                    server.top_optimizer.step()
                    epoch_loss += loss.item()
                    grad = top_input.grad

                dist.broadcast(grad, src=0)
                client_grad = grad[:, client.rank * bottom_output_dim:(client.rank + 1) * bottom_output_dim]
                client_grad = client_grad.to(client.device)

                if cfg.defense == "DPSGD":
                    client_grad = DPSGD_based_defense(client_grad, cfg.noise_std)

                for optim in client.bottom_optimizers:
                    optim.zero_grad()
                batch_outputs.backward(client_grad)
                for optim in client.bottom_optimizers:
                    optim.step()
            # if client.rank == 0:
            pbar.set_postfix(Epoch=f"{epoch}", loss=f"{epoch_loss:.4f}")

    save_checkpoints(cfg, client, server)


def train_model(cfg, client, server, num_clients):
    """
    training stage for Model Rake
    :param cfg:
    :param client:
    :param server:
    :param num_clients:
    :return:
    """
    print(f">>> Training mode: Model Rake, rank: {client.rank}")

    if cfg.task == 'classification':
        criterion = torch.nn.CrossEntropyLoss() # classification
    elif cfg.task == 'regression':
        criterion = torch.nn.MSELoss() # regression
    else:
        raise ValueError('invalid task type')

    for model in client.bottom_models:
        model.to(client.device)

    for epoch in tqdm(range(cfg.epochs), desc="Train", dynamic_ncols=True):
        epoch_loss = 0
        for data, hash_res, y in client.train_dataloader:
            data = data.to(client.device)
            hash_res = hash_res.to(client.device)

            batch_outputs = torch.zeros((len(data), cfg.bottom_model_output_dim)).to(client.device)
            bottom_0_embeddings = client.bottom_models[0](data[hash_res == 0])
            bottom_1_embeddings = client.bottom_models[1](data[hash_res == 1])

            batch_outputs[hash_res == 0] = bottom_0_embeddings
            batch_outputs[hash_res == 1] = bottom_1_embeddings

            # Oversampling
            len_0, len_1 = bottom_0_embeddings.shape[0], bottom_1_embeddings.shape[0]

            if len_0 < len_1:
                diff = len_1 - len_0
                indices = torch.randint(0, len_0, (diff,))
                bottom_0_embeddings = torch.cat([bottom_0_embeddings, bottom_0_embeddings[indices]], dim=0)
            elif len_1 < len_0:
                diff = len_0 - len_1
                indices = torch.randint(0, len_1, (diff,))
                bottom_1_embeddings = torch.cat([bottom_1_embeddings, bottom_1_embeddings[indices]], dim=0)

            # Gather clients' embeddings
            gather_list = [torch.empty_like(batch_outputs) for _ in range(num_clients)]
            dist.all_gather(gather_list, batch_outputs)
            grad = torch.zeros((len(data), num_clients * cfg.bottom_model_output_dim))

            if client.rank == 0:
                server.top_model.to(server.device)
                y = y.to(server.device)
                top_input = torch.cat(gather_list, dim=1).requires_grad_(True)
                top_input.retain_grad()
                top_output = server.top_model(top_input)
                loss = criterion(top_output, y)
                server.top_optimizer.zero_grad()
                loss.backward()
                server.top_optimizer.step()
                epoch_loss += loss.item()
                grad = top_input.grad

            dist.broadcast(grad, src=0)
            client_grad = grad[:, client.rank * cfg.bottom_model_output_dim:(client.rank + 1) * cfg.bottom_model_output_dim]
            client_grad = client_grad.to(client.device)

            # inter_loss
            cos_sim = F.cosine_similarity(
                bottom_0_embeddings.unsqueeze(1),
                bottom_1_embeddings.unsqueeze(0),
                dim=2)

            cos_loss = torch.mean(cos_sim)


            # intra_loss
            intra_cos_sim_0 = F.cosine_similarity(bottom_0_embeddings.unsqueeze(1),
                                                  bottom_0_embeddings.unsqueeze(0)).mean()
            intra_cos_sim_1 = F.cosine_similarity(bottom_1_embeddings.unsqueeze(1),
                                                  bottom_1_embeddings.unsqueeze(0)).mean()
            intra_cos_sim_loss_0 = torch.mean(torch.abs(intra_cos_sim_0))
            intra_cos_sim_loss_1 = torch.mean(torch.abs(intra_cos_sim_1))
            intra_loss = -(intra_cos_sim_loss_0 + intra_cos_sim_loss_1)

            # update gradients
            for optim in client.bottom_optimizers:
                optim.zero_grad()

            cos_loss.backward(retain_graph=True)
            grads_cos_0 = [param.grad.clone() for param in client.bottom_models[0].parameters()]
            grads_cos_1 = [param.grad.clone() for param in client.bottom_models[1].parameters()]

            client.bottom_models[0].zero_grad()
            client.bottom_models[1].zero_grad()

            intra_loss.backward(retain_graph=True)
            grads_intra_0 = [param.grad.clone() for param in client.bottom_models[0].parameters()]
            grads_intra_1 = [param.grad.clone() for param in client.bottom_models[1].parameters()]

            client.bottom_models[0].zero_grad()
            client.bottom_models[1].zero_grad()

            batch_outputs.backward(client_grad)
            grads_task_0 = [param.grad.clone() for param in client.bottom_models[0].parameters()]
            grads_task_1 = [param.grad.clone() for param in client.bottom_models[1].parameters()]

            client.bottom_models[0].zero_grad()
            client.bottom_models[1].zero_grad()

            for param, grad_cos_0, grad_intra_0, grad_task_0 in \
                    (zip(client.bottom_models[0].parameters(), grads_cos_0, grads_intra_0, grads_task_0)):
                param.grad = (cfg.cosine_alpha * grad_cos_0 +
                              cfg.intra_beta * grad_intra_0 +
                              cfg.task_theta * grad_task_0)

            for param, grad_cos_1, grad_intra_1, grad_task_1 in (
                    zip(client.bottom_models[1].parameters(), grads_cos_1, grads_intra_1, grads_task_1)):
                param.grad = (cfg.cosine_alpha * grad_cos_1 +
                              cfg.intra_beta * grad_intra_1 +
                              cfg.task_theta * grad_task_1)

            for optim in client.bottom_optimizers:
                optim.step()

        if client.rank == 0:
            # tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss}")
            print(f">>> Epoch: {epoch}, train loss: {epoch_loss}")

    save_checkpoints(cfg, client, server)

def test_model(cfg, client, server, num_clients):
    """
    test stage
    :param cfg:
    :param client:
    :param server:
    :param num_clients:
    :return:
    """
    load_checkpoints(cfg, client, server)
    # server.top_model.eval()
    right_pred = 0
    pred_list = []
    label_list = []
    embeddings = []
    embeddings_labels = []

    # ---- B4B context: init once for the whole test ----
    b4b_ctx = None
    if cfg.defense == "b4b":
        B4B_HASH_BITS = 12
        B4B_BASE_SEED = 12345
        B4B_NOISE_STD = 0.0
        b4b_ctx = B4BContext(
            n_hash_bits=B4B_HASH_BITS,
            seed=B4B_BASE_SEED + int(client.rank),
            noise_std=B4B_NOISE_STD,
        )

    for model in client.bottom_models:
        model.to(client.device)
        model.eval()
    for data, hash_res, y in client.test_dataloader:
        data = data.to(client.device)
        hash_res = hash_res.to(client.device)
        origin_bottom_outputs = []
        for item in zip(data, hash_res):
            output = client.bottom_models[item[1]](item[0])
            origin_bottom_outputs.append(output)

        batch_outputs = torch.stack(origin_bottom_outputs)

        embeddings = batch_outputs.detach().cpu().numpy()
        embeddings_labels = hash_res.detach().cpu().numpy()

        if cfg.defense == "noise":
            batch_outputs = noisy_embedding_based_defense(batch_outputs, cfg.noise_std)
        elif cfg.defense == "pruning":
            batch_outputs, mask = pruning_embedding_based_defense_by_removing_elements(batch_outputs, cfg.pruning_ratio)
            mask.to(client.device)
        elif cfg.defense == 'projection':
            batch_outputs = random_projection_based_defense(batch_outputs, client.projection_matrix)
        elif cfg.defense == "b4b":
            batch_outputs, _ = b4b_defense(batch_outputs, ctx=b4b_ctx, return_bucket=True)
        elif cfg.defense == "ResSFL":
            pass
        elif cfg.defense == "DPSGD":
            pass

        gather_list = [torch.empty_like(batch_outputs) for _ in range(num_clients)]
        dist.all_gather(gather_list, batch_outputs)

        if client.rank == 0:
            server.top_model.eval()
            y = y.to(server.device)
            server.top_model.to(server.device)
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
                    raise ValueError("Invalid task")
            else:
                # print(pred.shape)
                predicted = pred.argmax(dim=1)
                pred_list.append(predicted.cpu().numpy())
                right_pred += predicted.eq(y).sum().item()

    if client.rank == 0:
        pred_list = np.concatenate(pred_list)

        # embeddings = torch.cat(embeddings, dim=0)
        # embeddings_labels = np.concatenate(embeddings_labels)
        # tsne_plot(embeddings, labels=embeddings_labels, save_path="./tsne.pdf")

        if cfg.task == "classification":
            tqdm.write(f">>> Test acc: {100 * (right_pred / len(client.test_dataset))}%") # classification
        elif cfg.task == "regression":
            label_list = np.concatenate(label_list)
            mse = np.mean((label_list - pred_list) ** 2)
            tqdm.write(f">>> Test MSE: {mse}")

    return pred_list



def tsne_plot(embeddings, labels=None, title="t-SNE Visualization", save_path=None, random_state=42):
    pass
    # if hasattr(embeddings, 'detach'):
    #     embeddings = embeddings.detach().cpu().numpy()
    #
    # np.save("./embeddings.npy", embeddings)
    # np.save("./labels.npy", labels)
    #
    # tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, random_state=random_state)
    # embeddings_2d = tsne.fit_transform(embeddings)
    #
    # plt.figure(figsize=(10, 8))
    # if labels is not None:
    #     scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', s=2, alpha=0.7)
    #     plt.colorbar(scatter, ticks=np.unique(labels))
    # else:
    #     plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=2, alpha=0.7)
    #
    # plt.title(title)
    #
    # plt.show()
    #
    # if save_path:
    #     plt.savefig(save_path)
    #     print(f"t-SNE plot saved to: {save_path}")


# GAN distract (abandoned)
#
# def discriminator_loss(d_real, d_fake):
#     return -torch.log(d_real + 1e-8).mean() - torch.log(1 - d_fake + 1e-8).mean()
#
# def adversarial_loss(d_real, d_fake):
#     return -torch.log(1 - d_real + 1e-8).mean() - torch.log(d_fake + 1e-8).mean()
#
# def optimize_model_with_discriminator(cfg, client):
#     discriminator = Discriminator(cfg.bottom_model_output_dim)
#     discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr)
#     for epoch in tqdm(range(cfg.epochs), desc=f"Client {client.rank} Optimize", dynamic_ncols=True):
#
#         # train discriminator
#         client.bottom_models[0].eval()
#         client.bottom_models[1].eval()
#         discriminator.train()
#         for batch0, batch1 in zip(client.train_dataloaders[0], client.train_dataloaders[1]):
#             if batch0[0].shape[0] != batch1[0].shape[0]:
#                 continue
#
#             data0, _ = batch0
#             data1, _ = batch1
#
#             output0 = client.bottom_models[0](data0)
#             output1 = client.bottom_models[1](data1)
#
#             discriminator_optimizer.zero_grad()
#
#             pred0 = discriminator(output0.detach())
#             pred1 = discriminator(output1.detach())
#
#             loss = discriminator_loss(pred0, pred1)
#             loss.backward()
#
#             discriminator_optimizer.step()
#
#         # train bottom models
#         client.bottom_models[0].train()
#         client.bottom_models[1].train()
#         discriminator.eval()
#         for batch0, batch1 in zip(client.train_dataloaders[0], client.train_dataloaders[1]):
#             if batch0[0].shape[0] != batch1[0].shape[0]:
#                 continue
#
#             data0, _ = batch0
#             data1, _ = batch1
#
#             output0 = client.bottom_models[0](data0)
#             output1 = client.bottom_models[1](data1)
#
#             client.bottom_optimizers[0].zero_grad()
#             client.bottom_optimizers[1].zero_grad()
#
#             pred0 = discriminator(output0)
#             pred1 = discriminator(output1)
#
#             loss = adversarial_loss(pred0, pred1)
#             loss.backward()
#
#             client.bottom_optimizers[0].step()
#             client.bottom_optimizers[1].step()
#
#
#             # tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss}")
#
#     print(f"Client {client.rank}'s Discriminator Training Finished.")


def save_checkpoints(cfg, client, server):
    model_states = {}
    for i, model in enumerate(client.bottom_models):
        model_states[f'model_{i}_state_dict'] = model.state_dict()
    if client.rank == 0:
        model_states['top_model_state_dict'] = server.top_model.state_dict()

    os.makedirs(cfg.checkpoint_path, exist_ok=True)
    checkpoint_path = os.path.join(cfg.checkpoint_path, f"client_{client.rank}.pth")
    torch.save(model_states, checkpoint_path)
    print(f">>> Save client {client.rank} checkpoints.")


def load_checkpoints(cfg, client, server):
    checkpoint_path = os.path.join(cfg.checkpoint_path, f"client_{client.rank}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"--- ERROR: Checkpoint file not found at {checkpoint_path}. Cannot run test.")
    # checkpoint = torch.load(checkpoint_path)
    saved_weights = torch.load(checkpoint_path, map_location=client.device)
    for i, model in enumerate(client.bottom_models):
        key = f'model_{i}_state_dict'
        model.load_state_dict(saved_weights[key])
        model.to(client.device)
        model.eval()
    if client.rank == 0:
        server.top_model.load_state_dict(saved_weights['top_model_state_dict'])
        server.top_model.to(client.device)
        server.top_model.eval()
    print(f">>> Load client {client.rank} checkpoints.")