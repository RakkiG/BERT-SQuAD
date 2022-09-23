import argparse
import os

import torch

# from trainer import train, inference
from knn_trainer import knn_inference
from gnn_trainer import train, inference
from gnn_aug.build_datastore import build

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='/home/rakki/GNN-QA/data')
    parser.add_argument('--model_path', default='/home/rakki/GNN-QA/models/new_64/model_backup.pt')
    parser.add_argument('--bert_dir', default='/home/rakki/GNN-QA/bert')
    parser.add_argument('--visible_gpus', default='0', type=str)
    parser.add_argument('--dataset', default='v1.1.json', type=str, choices=['v1.1.json', 'v2.0.json'])
    parser.add_argument('--version_2_with_negative', default=False, type=bool)
    parser.add_argument('--seed', default=12345, type=int)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--n_best_size', default=10, type=int)
    parser.add_argument('--max_answer_len', default=30, type=int)
    parser.add_argument('--max_sen_len', default=384, type=int)
    parser.add_argument('--max_query_len', default=64, type=int)
    parser.add_argument('--doc_stride', default=128, type=int)
    parser.add_argument('--max_position_embeddings', default=512, type=int)
    parser.add_argument('--is_sample_shuffle', default=True, type=bool)

    parser.add_argument('--lr', default=3.5e-5, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--model_val_per_epoch', default=1, type=int)

    parser.add_argument('--k', default=64, type=int)
    parser.add_argument('--probe', default=32, type=int)
    parser.add_argument('--lamb', default=0.8, type=float)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpus
    device = torch.device('cpu' if args.visible_gpus == '-1' else 'cuda')
    args.device = device

    train(args, device)
    # inference(args, device)

