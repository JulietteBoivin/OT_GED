import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from arguments import get_args
from utils import StatsTracker
from utils import load_model

from train_base import train_model

from graph import MolGraph
from models import GCN
import pdb

def main(args=None, quiet=False, splits=None, abs_output_dir=False):
    if args is None:
        args = get_args()

    train_stats = train_model(
        model_class=GCN, run_func=run_func, args=args, quiet=quiet,
        splits=splits, abs_output_dir=abs_output_dir)
    return train_stats

def run_func(model, optim, data_loader, data_type, args, write_path=None,
             quiet=False):
    is_train = data_type == 'train'
    if is_train:
        model.train()
    else:
        model.eval()

    # Keeps track of epoch stats by aggregating batch stats
    stats_tracker = StatsTracker()
    write_output = []
    all_preds, all_labels = [], []
    if args.n_labels > 1:
        for _ in range(args.n_labels):
            all_preds.append([])
            all_labels.append([])

    for _, batch_data in enumerate(tqdm(data_loader, disable=quiet)):
        if is_train:
            optim.zero_grad()
        if args.n_labels == 1:
            smiles_list, labels_list = batch_data
        else:
            smiles_list, labels_list, mask = batch_data
        n_data = len(smiles_list)

        labels = torch.tensor(labels_list, device=args.device).float()
        if args.n_labels == 1:
            labels = labels.unsqueeze(1)
        if args.n_labels > 1:
            mask = torch.tensor(mask, device=args.device).float()

        mol_graph = MolGraph(smiles_list)
        _, preds = model(mol_graph)

        if args.val_stat == 'rmse':
            loss = torch.mean((labels - preds) ** 2)
            stats_tracker.add_stat('mse', loss.item() * n_data, n_data)
        else:
            assert False

        if write_path is not None and args.n_labels == 1:
            for smiles_idx, smiles in enumerate(smiles_list):
                label = labels[smiles_idx].item()
                pred = preds[smiles_idx].item()

                write_output.append({
                    'smiles': smiles,
                    'label': label,
                    'pred': pred})

        if is_train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()

    if args.n_labels == 1:
        if args.val_stat == 'rmse':
            mse = stats_tracker.get_stats()['mse']
            stats_tracker.add_stat('rmse', mse ** 0.5, 1)
    else:
        all_stats = []
        for label_idx in range(args.n_labels):
            cur_preds = np.concatenate(all_preds[label_idx])
            cur_labels = np.concatenate(all_labels[label_idx])

            data_name = args.data.split('_')[label_idx]
            if args.label_task_list[i] == 'rmse':
                mse = np.mean((cur_preds - cur_labels) ** 2)
                stats_tracker.add_stat('%s_mse' % data_name, mse, 1)
                stats_tracker.add_stat('%s_rmse' % data_name, mse ** 0.5, 1)
            else:
                assert False
        stats_tracker.add_stat('multi_obj', np.mean(np.array(all_stats)), 1)

    if write_path is not None and args.n_labels == 1:
        with open(write_path, 'w+') as write_file:
            for write_ele in write_output:
                json.dump(write_ele, write_file)
                write_file.write('\n')

    return stats_tracker

if __name__ == '__main__':
    main()
