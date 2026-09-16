# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2024/10/25 2:55 PM
@File :train_demo.py
"""
import time

import torch
import torch.utils.data as tud
import torch.nn as nn
from tqdm import tqdm
from utils.defense import *
import numpy as np

def train_one_bottom_model(cfg, bottom_model, top_model, train_dataset, test_dataset,
                           lr, epochs, batch_size, defense="no"):
    train_loader = tud.DataLoader(train_dataset, batch_size)
    test_loader = tud.DataLoader(test_dataset, batch_size)
    bottom_optimizer = torch.optim.Adam(bottom_model.parameters(), lr=lr)
    top_optimizer = torch.optim.Adam(top_model.parameters(), lr=lr)

    # device = torch.device('cpu')
    device = torch.device('cuda', 0)
    bottom_model.to(device)
    top_model.to(device)

    bottom_model.train()
    top_model.train()
    criterion = torch.nn.CrossEntropyLoss()

    # train
    for epoch in tqdm(range(epochs), desc="Train one bottom model", dynamic_ncols=True):
        epoch_loss = 0
        right_train = 0
        for x, _, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            b_output = bottom_model(x)
            pruned_output = b_output
            mask = None
            if defense == 'no':
                pass
            else:
                if defense == 'noisy':
                    b_output = noisy_embedding_based_defense(b_output, cfg.noise_std)
                if defense == 'pruning':
                    # pruned_output = pruning_embedding_based_defense_by_setting_zeros(b_output, cfg.pruning_ratio)
                    pruned_output, mask = pruning_embedding_based_defense_by_removing_elements(b_output, cfg.pruning_ratio)
                    # pruned_output = pruned_output.to(device)
                    # print(pruned_output.shape)
            b_output.retain_grad()
            if defense == 'pruning':
                t_output = top_model(pruned_output)
            else:
                t_output = top_model(b_output)
            loss = criterion(t_output, y)
            epoch_loss += loss.item()

            right_train += t_output.argmax(dim=1).eq(y.argmax(dim=1)).sum().item()

            bottom_optimizer.zero_grad()
            top_optimizer.zero_grad()

            loss.backward()

            if defense == 'pruning' and mask is not None:
                b_output.grad *= mask

            epoch_loss += loss.item()
            top_optimizer.step()
            bottom_optimizer.step()

        tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss}, train acc: {100 * (right_train / len(train_dataset))}%")

    # test
    right_pred = 0
    pred_list = []
    for x, _, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        b_output = bottom_model(x)
        if defense == "no":
            pass
        else:
            if defense == 'noisy':
                b_output = noisy_embedding_based_defense(b_output, cfg.noise_std)
            if defense == 'pruning':
                # b_output = pruning_embedding_based_defense_by_setting_zeros(b_output, cfg.pruning_ratio)
                b_output, _ = pruning_embedding_based_defense_by_removing_elements(b_output, cfg.pruning_ratio)
        pred = top_model(b_output)
        _, predicted = pred.max(1)
        predicted = predicted.cpu()
        pred_list.append(predicted.numpy())
        label = torch.from_numpy(y.nonzero().cpu().numpy()[:, 1])
        right_pred += predicted.eq(label).sum().item()

    print(f">>> Test acc: {100 * (right_pred / len(test_dataset))}%")

    return pred_list

def train_two_bottom_model(bottom_models, top_model, train_datasets, test_datasets,
                           lr, epochs, batch_size):
    train_loader_0 = tud.DataLoader(train_datasets[0], batch_size)
    train_loader_1 = tud.DataLoader(train_datasets[1], batch_size)
    test_loader_0 = tud.DataLoader(test_datasets[0], batch_size)
    test_loader_1 = tud.DataLoader(test_datasets[1], batch_size)

    bottom_model_0 = bottom_models[0]
    bottom_model_1 = bottom_models[1]
    bottom_optimizer_0 = torch.optim.Adam(bottom_model_0.parameters(), lr=lr)
    bottom_optimizer_1 = torch.optim.Adam(bottom_model_1.parameters(), lr=lr)
    top_optimizer = torch.optim.Adam(top_model.parameters(), lr=lr)

    # device = torch.device('cpu')
    device = torch.device('cuda', 0)
    bottom_model_0.to(device)
    bottom_model_1.to(device)
    top_model.to(device)

    bottom_model_0.train()
    bottom_model_1.train()
    top_model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="Train two bottom models", dynamic_ncols=True):
        epoch_loss = 0
        for x, y in train_loader_0:
            x = x.to(device)
            y = y.to(device)
            b_output_0 = bottom_model_0(x)
            b_output_0.retain_grad()

            t_output = top_model(b_output_0)
            loss = criterion(t_output, y)
            epoch_loss += loss.item()

            bottom_optimizer_0.zero_grad()
            top_optimizer.zero_grad()

            loss.backward()
            epoch_loss += loss.item()
            top_optimizer.step()
            bottom_optimizer_0.step()

        for x, y in train_loader_1:
            x = x.to(device)
            y = y.to(device)
            b_output_1 = bottom_model_1(x)
            b_output_1.retain_grad()

            t_output = top_model(b_output_1)
            loss = criterion(t_output, y)
            epoch_loss += loss.item()

            bottom_optimizer_1.zero_grad()
            top_optimizer.zero_grad()

            loss.backward()
            epoch_loss += loss.item()
            top_optimizer.step()
            bottom_optimizer_1.step()

        tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss}")

    right_pred = 0
    pred_list = []
    for x, y in test_loader_0:
        x = x.to(device)
        y = y.to(device)
        b_output = bottom_model_0(x)
        pred = top_model(b_output)
        _, predicted = pred.max(1)
        predicted = predicted.cpu()
        pred_list.append(predicted.numpy())
        label = torch.from_numpy(y.nonzero().cpu().numpy()[:, 1])
        right_pred += predicted.eq(label).sum().item()

    for x, y in test_loader_1:
        x = x.to(device)
        y = y.to(device)
        b_output = bottom_model_1(x)
        pred = top_model(b_output)
        _, predicted = pred.max(1)
        predicted = predicted.cpu()
        pred_list.append(predicted.numpy())
        label = torch.from_numpy(y.nonzero().cpu().numpy()[:, 1])
        right_pred += predicted.eq(label).sum().item()

    print(f">>> Test acc: {100 * (right_pred / (len(test_datasets[0]) +len(test_datasets[1])))}%")

    return pred_list

def train_client_and_server(cfg, clients, server):
    batch_num = clients[0].train_batch
    for epoch in tqdm(range(cfg.epochs), desc="Train", dynamic_ncols=True):
        epoch_loss = 0
        train_right = 0
        for batch in range(batch_num):
            embeddings = []

            for client in clients:
                embedding = client.get_embeddings(1, batch)
                # tqdm.write(f"Embedding: {embedding}")
                embeddings.append(embedding)

            # print(embeddings_out)
            embeddings = np.array(embeddings)
            embeddings = np.concatenate(embeddings, axis=1)
            top_input = torch.from_numpy(embeddings).requires_grad_(True)

            # print(top_input.shape)

            loss, right_num, grads = server.train(top_input, batch)

            epoch_loss += loss
            train_right += right_num

            # loss.backward()

            server.update_top_model()

            grads = grads.cpu().detach().numpy()

            for i in range(len(clients)):
                client_grad = grads[:, cfg.bottom_model_output_dim * i: cfg.bottom_model_output_dim * (i + 1)]
                # print(i, "\t", client_grad)
                client_grad = torch.from_numpy(client_grad)
                clients[i].update_bottom_models(client_grad, batch)

                # time.sleep(1000000000)

        tqdm.write(
            f">>>Epoch: {epoch}, Loss: {epoch_loss}, Train Acc: {100 * (train_right / len(server.train_dataset))}%")

    print("Model Training Success.")

def test_client_and_server(cfg, clients, server):
    batch_num = clients[0].test_batch
    test_right = 0
    pred_list = []
    for batch in range(batch_num):
        embeddings = []
        tag = False

        for client in clients:
            embedding = client.get_embeddings(2, batch)
            if embedding is None:
                tag = True
                break
            embeddings.append(embedding)

        if tag:
            break

        embeddings = np.array(embeddings)
        embeddings = np.concatenate(embeddings, axis=1)
        embeddings = torch.from_numpy(embeddings)

        outputs, right_pred = server.get_predictions(embeddings, batch)
        pred_list.append(outputs)
        test_right += right_pred

    tqdm.write(f"Test Acc: {100 * (test_right / len(server.test_dataset))}%")

    return pred_list, test_right