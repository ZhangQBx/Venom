# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2024/10/23 3:28 PM
@File :attacks_demo.py
"""
import numpy as np
import torch
import torch.utils.data as tud
from tqdm import tqdm
import time

from utils.defense import pruning_embedding_based_defense_by_removing_elements
from utils.funcs import *


def get_aux_dataset_two(bottom_models, aux_datasets):
    bottom_model_0 = bottom_models[0]
    bottom_model_1 = bottom_models[1]
    aux_loader_0 = tud.DataLoader(aux_datasets[0], 40)
    aux_loader_1 = tud.DataLoader(aux_datasets[1], 40)

    device = torch.device('cpu')
    bottom_model_0.to(device)
    bottom_model_1.to(device)
    bottom_model_0.train()
    bottom_model_1.train()

    x_aux = []
    y = []
    for x, _ in aux_loader_0:
        x = x.to(device)
        # y = y.to(device)
        output_0 = bottom_model_0(x)
        output_0 = output_0.detach().numpy()
        x_0 = x.detach().numpy()

        y.append(output_0)
        x_aux.append(x_0)

    for x, _ in aux_loader_1:
        x = x.to(device)
        # y = y.to(device)
        output_1 = bottom_model_1(x)
        output_1 = output_1.detach().numpy()
        x_1 = x.detach().numpy()

        y.append(output_1)
        x_aux.append(x_1)

    y = np.concatenate(y, axis=0)
    y = torch.from_numpy(y).float()
    x_aux = np.concatenate(x_aux, axis=0)
    x_aux = torch.from_numpy(x_aux).float()
    dataset = MyDatasetTS(x_aux, y)

    # print(y[:2])
    # print(y.shape)
    # time.sleep(100000)
    return dataset
def get_aux_dataset_one(cfg, bottom_model, aux_dataset):
    aux_loader = tud.DataLoader(aux_dataset, 40)

    device = torch.device('cpu')
    bottom_model.to(device)
    bottom_model.train()
    x_aux = []
    y = []

    for x, _, _ in aux_loader:
        x = x.to(device)
        # y = y.to(device)
        output = bottom_model(x)
        if cfg.defense == "pruning":
            output, _ = pruning_embedding_based_defense_by_removing_elements(output, cfg.pruning_ratio)
        output = output.detach().numpy()
        x_ = x.detach().numpy()

        y.append(output)
        x_aux.append(x_)

    y = np.concatenate(y, axis=0)
    y = torch.from_numpy(y).float()
    x_aux = np.concatenate(x_aux, axis=0)
    x_aux = torch.from_numpy(x_aux).float()
    dataset = MyDatasetTS(x_aux, y)

    print(y.shape)

    return dataset

def get_two_bottom_model_train_embeddings(bottom_models, train_datasets):
    bottom_model_0 = bottom_models[0]
    bottom_model_1 = bottom_models[1]
    train_loader_0 = tud.DataLoader(train_datasets[0], 50)
    train_loader_1 = tud.DataLoader(train_datasets[1], 50)

    device = torch.device('cpu')
    bottom_model_0.to(device)
    bottom_model_1.to(device)

    embs = []
    for x, _ in train_loader_0:
        output_0 = bottom_model_0(x)
        output_0 = output_0.detach().numpy()
        embs.append(output_0)

    for x, _ in train_loader_1:
        output_1 = bottom_model_1(x)
        output_1 = output_1.detach().numpy()
        embs.append(output_1)

    embs = np.concatenate(embs, axis=0)
    embs = torch.from_numpy(embs).float()

    return embs


def get_knn(bottom_models, train_datasets):
    pass

def model_stealing_attack_two(surrogate_model, bottom_models, top_model, aux_datasets, test_dataset,
                              two_bottom_model_pred, lr, epochs, batch_size):
    aux_embedding_dataset = get_aux_dataset_two(bottom_models, aux_datasets)
    # print(aux_embedding_dataset[:4])
    aux_emb_loader = tud.DataLoader(aux_embedding_dataset, batch_size)
    test_loader = tud.DataLoader(test_dataset, batch_size)
    aux_optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=lr)

    # device = torch.device('cpu')
    device = torch.device('cuda', 0)
    surrogate_model.to(device)
    surrogate_model.train()
    criterion = l2_norm_loss

    # train
    for epoch in tqdm(range(epochs), desc="Model stealing attack", dynamic_ncols=True):
        epoch_loss = 0
        for x, y in aux_emb_loader:
            x = x.to(device)
            y = y.to(device)
            b_output = surrogate_model(x)
            loss = criterion(b_output, y)
            epoch_loss += loss.item()

            aux_optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            aux_optimizer.step()

        tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss}")

    # print(len(test_dataset))
    # test
    right_pred = 0
    pred_list = []
    for x, _, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        b_output = surrogate_model(x)
        pred = top_model(b_output)
        _, predicted = pred.max(1)
        predicted = predicted.cpu()
        pred_list.append(predicted.numpy())
        label = torch.from_numpy(y.nonzero().cpu().numpy()[:, 1])
        right_pred += predicted.eq(label).sum().item()

    two_bottom_model_pred = np.concatenate(two_bottom_model_pred, axis=0)
    # print(bibmodel_pred.shape)
    # bibmodel_pred = np.array(bibmodel_pred)
    pred_list = np.concatenate(pred_list, axis=0)
    print(pred_list)
    print(two_bottom_model_pred)
    print(f">>> Test acc: {100 * (right_pred / len(test_dataset))}%")
    print(f">>> Agreement: {100 * np.sum(two_bottom_model_pred == pred_list) / len(test_dataset)}%")


def model_stealing_attack_one_old(cfg, surrogate_model, bottom_model, top_model, aux_dataset, test_dataset,
                              bottom_model_pred, lr, epochs, batch_size):
    aux_embedding_dataset = get_aux_dataset_one(cfg, bottom_model, aux_dataset)
    aux_emb_loader = tud.DataLoader(aux_embedding_dataset, batch_size)
    test_loader = tud.DataLoader(test_dataset, batch_size)
    aux_optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=lr)

    # device = torch.device('cpu')
    device = torch.device('cuda', 0)
    surrogate_model.to(device)
    top_model.to(device)
    surrogate_model.train()
    criterion = l2_norm_loss

    # train
    for epoch in tqdm(range(epochs), desc="Model stealing attack", dynamic_ncols=True):
        epoch_loss = 0
        for x, y in aux_emb_loader:
            x = x.to(device)
            y = y.to(device)
            b_output = surrogate_model(x)
            loss = criterion(b_output, y)
            epoch_loss += loss.item()

            aux_optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            aux_optimizer.step()

        tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss}")

    # test
    right_pred = 0
    pred_list = []
    for x, _, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        b_output = surrogate_model(x)
        pred = top_model(b_output)
        _, predicted = pred.max(1)
        predicted = predicted.cpu()
        pred_list.append(predicted.numpy())
        label = torch.from_numpy(y.nonzero().cpu().numpy()[:, 1])
        right_pred += predicted.eq(label).sum().item()

    bottom_model_pred = np.concatenate(bottom_model_pred, axis=0)
    pred_list = np.concatenate(pred_list, axis=0)
    print(pred_list)
    print(bottom_model_pred)
    # time.sleep(100000)
    print(f">>> Test acc: {100 * (right_pred / len(test_dataset))}%")
    print(f">>> Agreement: {100 * np.sum(bottom_model_pred == pred_list) / len(test_dataset)}%")





    # batch_num = clients[0].aux_batch
    # for client in clients:
    #     aux_dataset = client.aux_dataset
    #     aux_embeddings = []
    #
    #     for batch in range(batch_num):
    #         aux_embedding = client.get_embeddings(3, batch)
    #         if aux_embedding is None:
    #             break
    #         aux_embeddings.append(aux_embedding)
    #
    #     # embeddings = np.array(embeddings)
    #     aux_embeddings = np.concatenate(aux_embeddings, axis=0)
    #     aux_embeddings = torch.from_numpy(aux_embeddings)
    #
    #     aux_emb_dataset = MyDatasetTS(aux_dataset.aux_data_ts, aux_embeddings)
    #
    #     client.train_surrogate_model(aux_emb_dataset)

def model_stealing_test(cfg, clients, server):
    batch_num = clients[0].test_batch
    test_right = 0
    pred_list = []
    for batch in range(batch_num):
        embeddings = []
        tag = False

        for client in clients:
            embedding = client.get_surrogate_model_output(batch)
            if embedding is None:
                tag = True
                break
            embeddings.append(embedding)

        if tag:
            break

        embeddings = np.concatenate(embeddings, axis=1)
        embeddings = torch.from_numpy(embeddings)

        outputs, right_pred = server.get_predictions(embeddings, batch)
        pred_list.append(outputs)
        test_right += right_pred

    return pred_list, test_right


def data_reconstruction_attack_one(cfg, generator, surrogate_model, bottom_model, train_dataset, lr, epochs, batch_size):
    embs = []
    train_loader = tud.DataLoader(train_dataset, batch_size)
    for x, _, _ in train_loader:
        output = bottom_model(x)
        embs.append(output.detach().numpy())
    embs = np.concatenate(embs, axis=0)
    embs = torch.from_numpy(embs).float()

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    surrogate_optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=lr)
    device = torch.device('cuda', 0)
    generator.to(device)
    surrogate_model.to(device)

    input_noise = torch.randn(len(train_dataset), train_dataset.train_data_ts.shape[1])
    dr_attack_dataset = MyDatasetTS(input_noise, embs)

    dr_attack_loader = tud.DataLoader(dr_attack_dataset, batch_size)

    criterion = nn.MSELoss()
    for epoch in tqdm(range(epochs), desc="Data reconstruction attack", dynamic_ncols=True):
        epoch_loss = 0
        for x, y in dr_attack_loader:
            x = x.to(device)
            y = y.to(device)
            output_data = generator(x)
            output_emb = surrogate_model(output_data)
            loss = criterion(output_emb, y)

            generator_optimizer.zero_grad()
            surrogate_optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            generator_optimizer.step()
        tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss}")

    generator.to('cpu')
    reconstructed_data = generator(input_noise).detach()
    origin_data = train_dataset.train_data_ts

    mse = torch.mean((reconstructed_data - origin_data) ** 2)
    relative_error = torch.mean(torch.norm(reconstructed_data - origin_data, dim=1) / torch.norm(origin_data, dim=1))

    print(f">>> Mean square error: {mse}")
    print(f">>> Relative error: {relative_error}")


def data_reconstruction_attack_two(bottom_models, train_datasets, generator, surrogate_model,
                                   lr, epochs, batch_size):
    embs = get_two_bottom_model_train_embeddings(bottom_models, train_datasets)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    surrogate_optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=lr)
    device = torch.device('cuda', 0)
    generator.to(device)
    surrogate_model.to(device)

    input_noise = torch.randn(len(train_datasets[0]) + len(train_datasets[1]), train_datasets[0].data.shape[1])
    # print(input_noise.shape)
    dr_attack_dataset = MyDatasetTS(input_noise, embs)
    dr_attack_loader = tud.DataLoader(dr_attack_dataset, batch_size)
    # print(dr_attack_dataset.data.shape)
    # print(dr_attack_dataset.label.shape)
    criterion = nn.MSELoss()
    for epoch in tqdm(range(epochs), desc="Data reconstruction attack", dynamic_ncols=True):
        epoch_loss = 0
        for x, y in dr_attack_loader:
            x = x.to(device)
            y = y.to(device)
            output_data = generator(x)
            output_emb = surrogate_model(output_data)
            loss = criterion(output_emb, y)

            generator_optimizer.zero_grad()
            surrogate_optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            generator_optimizer.step()
        tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss}")

    # print(torch.sum(input_noise == dr_attack_dataset.data))
    generator.to('cpu')
    reconstructed_data = generator(input_noise).detach()
    origin_data = np.concatenate([train_datasets[0].data.numpy(), train_datasets[1].data.numpy()], axis=0)
    origin_data = torch.from_numpy(origin_data).float()
    # print(reconstructed_data)
    # print(origin_data)
    mse = torch.mean((reconstructed_data - origin_data) ** 2)
    relative_error = torch.mean(torch.norm(reconstructed_data - origin_data, dim=1) /
                                torch.norm(origin_data, dim=1))
    # print(torch.norm(reconstructed_data - origin_data, dim=1))
    # mse = nn.MSELoss(torch.from_numpy(reconstruct_data).float(), torch.from_numpy(origin_data).float())
    print(f">>> Mean square error: {mse}")
    print(f">>> Relative error: {relative_error}")

    # time.sleep(100000)
