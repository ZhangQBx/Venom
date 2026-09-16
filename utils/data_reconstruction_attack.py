import numpy as np
import torch
# from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data as tud
from tqdm import tqdm

from models import Generator
from models.Generator import ImageGenerator
from utils import AttackDatasetTS, MyDatasetTS, total_variation_loss, ssim


def data_reconstruction_attack_entrance(cfg, client, num_surrogates):
    if cfg.type == "tabular":
        data_reconstruction_attack(client, num_surrogates)
    elif cfg.type == "image":
        data_reconstruction_attack_based_image(cfg, client)
    else:
        print("Invalid type")
        return
    # input_noise = input_noise.to(client.device)
    # client.generator.eval()
    #
    # # origin_data = client.train_dataset.train_data_ts
    # origin_data = client.train_dataset.train_data_ts.to(client.device)
    # # origin_data.to(client.device)
    #
    # # input_noise = torch.randn(origin_data.shape).to(client.device)
    # reconstructed_data = client.generator(input_noise)
    #
    # mse = torch.mean((origin_data - reconstructed_data) ** 2)
    #
    # # mse = F.mse_loss(origin_data, reconstructed_data)
    #
    # print(f"Client {client.rank}'s Reconstruction MSE: {mse}")
    #



def data_reconstruction_attack(client, num_surrogates):
    """
    data reconstruction attack for tabular datasets
    :param client:
    :param num_surrogates:
    :return:
    """
    client.generator = Generator(client.bottom_models[0].in_dim, client.bottom_models[0].in_dim).to(client.device)

    # print("surrogate_model's parameters(before dr):")
    # for param in client.surrogate_model.parameters():
    #     print(param)
    #
    # print("bottom_model's parameters(before dr):")
    # for param in client.bottom_models[0].parameters():
    #     print(param)

    embeddings = []
    for data, hash, _ in client.train_dataloader:
        data = data.to(client.device)
        hash = hash.to(client.device)
        for item in zip(data, hash):
            output = client.bottom_models[item[1]](item[0])
            embeddings.append(output.detach().cpu().numpy())

    # embeddings = np.concatenate(embeddings, axis=0)
    embeddings = np.array(embeddings)
    embeddings = torch.from_numpy(embeddings)
    # print(embeddings.shape)

    # time.sleep(100000000)
    input_noise = torch.randn(len(client.train_dataset), client.train_dataset.train_data_ts.shape[1])
    # print(input_noise.shape)

    dr_attack_dataset = AttackDatasetTS(client.cfg, client.rank, input_noise, embeddings, num_surrogates, True)
    # dr_attack_dataset = MyDatasetTS(input_noise, client.train_dataset.train_data_ts)
    dr_attack_loader = tud.DataLoader(dr_attack_dataset, batch_size=client.cfg.bs_dr, shuffle=False)
    generator_optimizer = torch.optim.Adam(client.generator.parameters(), lr=client.cfg.lr_dr)
    # surrogate_optimizer = torch.optim.Adam(client.surrogate_model.parameters(), lr=client.cfg.lr_dr)

    criterion = nn.MSELoss()

    client.generator.train()
    client.surrogate_model.eval().to(client.device)

    for param in client.surrogate_model.parameters():
        param.requires_grad = False

    if num_surrogates == 2:
        print("in data_reconstruction_attack_multi_surrogates")
        for model in client.surrogate_models:
            model.eval().to(client.device)
            for param in model.parameters():
                param.requires_grad = False

    for epoch in range(client.cfg.epochs):
        epoch_loss = 0
        for x, hash_res, y in dr_attack_loader:
            if num_surrogates == 1:
                x = x.to(client.device)
                y = y.to(client.device)
                output_data = client.generator(x)
                output_emb = client.surrogate_model(output_data)
                loss = criterion(output_emb, y)
                generator_optimizer.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                generator_optimizer.step()
            else:

                x = x.to(client.device)
                y = y.to(client.device)
                output_data = client.generator(x)
                # output_data.retain_grad()
                output_emb = torch.zeros((x.shape[0], client.surrogate_models[0].out_dim)).to(client.device)

                output_emb[hash_res == 0] = client.surrogate_models[0](output_data[hash_res == 0])
                output_emb[hash_res == 1] = client.surrogate_models[1](output_data[hash_res == 1])

                # print(output_emb.shape, y.shape)
                loss = criterion(output_emb, y)
                # loss = criterion(output_data, y)
                # loss = torch.mean(torch.abs(output_emb - y))

                # surrogate_optimizer.zero_grad()
                generator_optimizer.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                generator_optimizer.step()

        # tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss}")

    client.generator.eval()

    input_noise = input_noise.to(client.device)

    reconstructed_data = client.generator(input_noise).detach()

    origin_data = client.train_dataset.train_data_ts.to(client.device)

    # print("surrogate_model's parameters(after dr):")
    # for param in client.surrogate_model.parameters():
    #     print(param)
    #
    # print("bottom_model's parameters(after dr):")
    # for param in client.bottom_models[0].parameters():
    #     print(param)
    reconstructed_data_np = reconstructed_data.cpu().numpy()
    # origin_data_np = client.train_dataset.train_data_np
    # euclidean_distance = 0
    # for index in range(len(origin_data_np)):
    #     a = math.sqrt(np.linalg.norm(reconstructed_data_np[index] - origin_data_np[index]))
    #     b = math.sqrt(np.linalg.norm(origin_data_np[index]))
    #     value = a / b
    #     euclidean_distance += value

    # print("error: ", euclidean_distance / len(origin_data_np))
    # time.sleep(1000000)


    mse = torch.mean((reconstructed_data - origin_data) ** 2)
    relative_error_ts = torch.abs(reconstructed_data - origin_data) / torch.norm(origin_data, dim=1, keepdim=True)
    # relative_error_ts = torch.abs(reconstructed_data - origin_data) / (torch.abs(origin_data) + 0.0001)
    # print(relative_error_ts.shape)
    # time.sleep(10000000)

    relative_error_ts = relative_error_ts.flatten()
    relative_error_ts = torch.sort(relative_error_ts, descending=True).values
    median = torch.median(relative_error_ts)

    # for i in range(2):
    #     print(f"reconstructed_data{i+1000}: {reconstructed_data[i+1000]}")
    #     print(f"origin_data{i+1000}: {origin_data[i+1000]}")

    # relative_error = torch.mean(torch.abs(reconstructed_data - origin_data) / (torch.abs(origin_data) + 0.0001))
    relative_error = torch.mean(relative_error_ts)

    print(f">>> Client {client.rank}'s Reconstruction MSE: {mse}")
    # print(f">>> Client {client.rank}'s Relative error: {relative_error}")
    # print(f">>> Client {client.rank}'s Median relative error: {median.item()}")



def data_reconstruction_attack_based_image(cfg, client):
    """
    data reconstruction attack for image datasets
    :param cfg:
    :param client:
    :return:
    """
    noise_dim = 100
    channels = client.train_dataset.train_data_ts[0].shape[0]
    img_w = client.train_dataset.train_data_ts[0].shape[2]
    img_h = client.train_dataset.train_data_ts[0].shape[1]
    generator = ImageGenerator(noise_dim, channels, img_w, img_h).to(client.device)

    random_index = np.random.randint(0, len(client.train_dataset), 100)
    print(random_index)
    origin_data = []
    # hash_res = client.train_dataset.train_hash_res[random_index]
    embeddings = []
    for i in random_index:
        output = client.bottom_models[client.train_dataset.train_hash_res[i]](client.train_dataset.train_data_ts[i].to(client.device))
        embeddings.append(output.cpu().detach().numpy())
        origin_data.append(client.train_dataset.train_data_ts[i].cpu().detach().numpy())

    embeddings = np.array(embeddings)
    embeddings = torch.from_numpy(embeddings)

    origin_data = np.array(origin_data)
    origin_data = torch.from_numpy(origin_data).to(client.device)

    input_noise = torch.randn(len(origin_data), noise_dim).to(client.device)

    # input_noise = torch.randn_like(client.train_dataset.train_data_ts).to(client.device)
    # input_noise = torch.randn(len(client.train_dataset), 100).to(client.device)

    dr_attack_dataset = MyDatasetTS(input_noise, embeddings)
    # dr_attack_dataset = MyDatasetTS(input_noise, client.train_dataset.train_data_ts)
    dr_attack_loader = tud.DataLoader(dr_attack_dataset, batch_size=client.cfg.bs_dr, shuffle=True)

    generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=client.cfg.lr_dr)
    # surrogate_optimizer = torch.optim.Adam(client.surrogate_model.parameters(), lr=client.cfg.lr_dr)

    generator.train()
    client.surrogate_model.eval().to(client.device)

    for param in client.surrogate_model.parameters():
        param.requires_grad = False

    for epoch in tqdm(range(client.cfg.epochs), desc=f"Client {client.rank} D-R Attack", dynamic_ncols=True):
        epoch_loss = 0
        for x, y in dr_attack_loader:
            x = x.to(client.device)
            y = y.to(client.device)
            output_data = generator(x)
            output_emb = client.surrogate_model(output_data)
            # print(output_data.shape, output_emb.shape, y.shape)
            # time.sleep(100000)
            # loss = torch.mean((output_data - y) ** 2) + cfg.R_tv_alpha * total_variation_loss(output_data)
            loss = torch.mean((output_emb - y) ** 2) + cfg.R_tv_alpha * total_variation_loss(output_data)

            # surrogate_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            generator_optimizer.step()

        tqdm.write(f">>> Epoch: {epoch}, train loss: {epoch_loss}")

    generator.eval()

    reconstruct_data_list = []
    for noise, _ in dr_attack_loader:
        re_data = generator(noise)
        reconstruct_data_list.append(re_data.cpu().detach().numpy())

    reconstructed_data = np.concatenate(reconstruct_data_list, axis=0)
    reconstructed_data = torch.from_numpy(reconstructed_data).to(client.device)
    # origin_data = client.train_dataset.train_data_ts.to(client.device)

    # print(reconstructed_data.shape, origin_data.shape)

    mse = torch.mean((reconstructed_data - origin_data) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    ssim_value = ssim(reconstructed_data, origin_data)

    print(f">>> Client {client.rank}'s Reconstruction MSE: {mse}")
    print(f">>> Client {client.rank}'s PSNR: {psnr}")
    print(f">>> Client {client.rank}'s SSIM: {ssim_value}")

    # fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    # axes = axes.flatten()

    # for i in range(5):
    #     image = origin_data[i].cpu().detach().numpy()
    #
    #     axes[i].imshow(image.squeeze(), cmap='gray')  # Remove the single-channel dimension
    #     axes[i].set_title(f'Label: {i}')  # Add the label
    #     axes[i].axis('off')  # Hide axes
    #
    # for i in range(5):
    #     image = reconstructed_data[i].cpu().detach().numpy()
    #
    #     axes[i+5].imshow(image.squeeze(), cmap='gray')  # Remove the single-channel dimension
    #     axes[i+5].set_title(f'Label: {i+5}')  # Add the label
    #     axes[i+5].axis('off')  # Hide axes

    # plt.tight_layout()
    # plt.show()
