# -*- coding: utf-8 -*-
"""
@Auth : ZQB
@Time : 2024/9/28 4:19 PM
@File :run.py
"""

import hydra
import torch.distributed
from omegaconf import DictConfig
import os

# os.environ["GLOO_USE_LIBUV"] = "0"
import torch.distributed as dist
import torch.multiprocessing as mp

from roles import Client, Server
from utils import *
from utils.data_reconstruction_attack import data_reconstruction_attack_entrance
from utils.demos.train_demo import train_client_and_server, test_client_and_server
from utils.distributed import train_model, test_model, train_one_model, load_checkpoints
from utils.model_stealing_attack import model_stealing_attack_entrance



def init_process(cfg, port, rank, num_clients, num_models, num_surrogates):
    if rank == 0:
        # print("---------------------------------------")
        print(">>> num_clients: {}, num_models: {}, num_surrogates: {}".format(num_clients, num_models, num_surrogates))
        print(">>> Defense method: ",cfg.defense)
        # print("---------------------------------------")

    os.environ['MASTER_ADDR'] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["GLOO_USE_LIBUV"] = "0"
    dist.init_process_group("gloo", rank=rank, world_size=num_clients)

    torch.manual_seed(cfg.seed)
    dataset_name = cfg.dataset
    client_info_dict = init_client(cfg, dataset_name, rank, num_clients, num_models)
    # client = Client(cfg, rank, client_info_dict, torch.device(cfg.client_device), cfg.defense)
    client = Client(cfg, rank, client_info_dict, torch.device("cuda", 0), cfg.defense)
    print("client {} init success.".format(rank))

    top_input_dim = cfg.bottom_model_output_dim * num_clients

    if cfg.defense == 'pruning':
        bottom_output_dim = round(cfg.bottom_model_output_dim * (1 - cfg.pruning_ratio))
        top_input_dim = bottom_output_dim * num_clients

    torch.distributed.barrier()

    if client.rank == 0:
        server_info_dict = init_server(cfg, dataset_name, top_input_dim, num_clients, num_models)
        # server = Server(cfg, server_info_dict, torch.device(cfg.server_device))
        server = Server(cfg, server_info_dict, torch.device("cuda", rank))
        print("server {} init success.".format(rank))
    else:
        server = None

    # optimize_model_with_discriminator(cfg, client)
    # dist.barrier()
    # if num_models == 1:
    #     train_one_model(cfg, client, server, num_clients)
    # else:
    #     train_model(cfg, client, server, num_clients)
    pred_list = test_model(cfg, client, server, num_clients)
    model_stealing_attack_entrance(cfg, client, server, num_clients, pred_list, num_surrogates)
    # data_reconstruction_attack_entrance(cfg, client, num_surrogates)


@hydra.main(version_base=None, config_path="./conf", config_name="conf")
def launch(cfg: DictConfig):
    port = cfg.localhost
    num_models = cfg.num_models
    num_clients = cfg.num_clients
    num_surrogates = cfg.num_surrogates
    if cfg.model_type == "mlp":
        cfg = cfg.mlp_conf
    elif cfg.model_type == "cnn":
        cfg = cfg.cnn_conf
    else:
        pass

    process = []
    mp.set_start_method("spawn")
    for rank in range(num_clients):
        p = mp.Process(target=init_process, args=(cfg, port, rank, num_clients, num_models, num_surrogates))
        p.start()
        process.append(p)

    for p in process:
        p.join()

if __name__ == '__main__':
    start = time.time()
    launch() # multi-process
    end = time.time()
    print('>>> Time cost:', end - start)