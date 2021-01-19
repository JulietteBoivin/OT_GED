import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from arguments import get_args
from utils import StatsTracker, PlotTracker, get_data_examples

from train_base import train_model, test_model

from graph import MolGraph
from datasets import get_loader
from models import ProtoNet


def init_plot_tracker(args):
    # Text data to store are the epoch and the val_stat, either rmse or auc
    text_names = ['epoch', 'train %s' % args.val_stat, 'val %s' % args.val_stat]
    args.plot_tracker = PlotTracker(
        plot_freq=args.plot_freq, plot_max=args.plot_max,
        save_dir=args.output_dir, title=args.data, text_names=text_names)

def main(args=None, quiet=False, splits=None, abs_output_dir=False):
    if args is None:
        args = get_args()

    if args.pretrain_model is not None:
        test_model(model_class=ProtoNet, run_func=run_func, args=args, quiet=quiet)
        exit()

    train_results = train_model(
        model_class=ProtoNet, run_func=run_func, args=args, quiet=quiet,
        splits=splits, abs_output_dir=abs_output_dir)

    return train_results

def run_func(model, optim, data_loader, data_type, args, write_path=None,
             debug=False, quiet=False):
    is_train = data_type == 'train'
    if is_train:
        model.train()
    else:
        model.eval()

    # Keeps track of epoch stats by aggregating batch stats
    stats_tracker = StatsTracker()
    all_preds, all_labels = [], []
    if args.n_labels > 1:
        for _ in range(args.n_labels):
            all_preds.append([])
            all_labels.append([])
    write_output = []

    # Add the pc ot distance of the first two point clouds
    for _, batch_data in enumerate(tqdm(data_loader, disable=quiet)):
        if is_train:
            optim.zero_grad()

        if args.n_labels == 1:
            smile1, smile2, labels_list = batch_data
        else:
            smile1, smile2, labels_list, mask = batch_data
        n_data = len(smile1)

        labels = torch.tensor(labels_list, device=args.device).float()
        if args.n_labels == 1:
            labels = labels.unsqueeze(1)
        if args.n_labels > 1:
            mask = torch.tensor(mask, device=args.device).float()

        # mol_graph is the abstraction of the model input
        mol_graph1 = MolGraph(smile1)
        mol_graph2 = MolGraph(smile2)
        output_dict = model(mol_graph1, mol_graph2, debug=debug)

        preds = output_dict['preds']
        if args.val_stat == 'rmse':
            loss = torch.mean((labels - preds) ** 2)
            stats_tracker.add_stat('mse', loss.item() * n_data, n_data)
            all_preds.append(preds.detach().cpu().numpy())
        else:
            assert False
            
        all_labels.append(np.array(labels_list))
        
        if write_path is not None and args.n_labels == 1:
            for smiles_idx, smiles in enumerate(smile1):
                label = labels#[smiles_idx].item()
                pred = preds#[smiles_idx].item()
                write_output.append({'smiles': (smile1, smile2), 'label': label, 'pred': pred})

        # Add stats to keep track of
        stats_tracker.add_stat('loss', loss.item() * n_data, n_data)

        if is_train:
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()

    if args.n_labels == 1:
        all_preds = np.squeeze(np.reshape(np.concatenate(all_preds), (np.concatenate(all_preds).shape[0], 1)), axis=1)
        all_labels = np.concatenate(all_labels)

        preds_mean = np.mean(all_preds)
        preds_std = np.std(all_preds)
        stats_tracker.add_stat('prediction_mean', preds_mean, 1)
        stats_tracker.add_stat('prediction_std', preds_std, 1)

        if args.val_stat == 'rmse':
            mse = stats_tracker.get_stats()['mse']
            stats_tracker.add_stat('rmse', mse ** 0.5, 1)
            
        if data_type == 'train':
                with open('train_loss.txt', 'a') as f:
                    f.write(str(stats_tracker.get_stats()['mse']) + '\n')
                    f.close()
                    
                with open('train_preds.txt', 'a') as f:
                    f.write(str(all_preds) + '\n')
                    f.close()
                    
                with open('train_labels.txt', 'a') as f:
                    f.write(str(all_labels) + '\n')
                    f.close()
                    
        if data_type == 'val':
            with open('valid_loss.txt', 'a') as f:
                f.write(str(stats_tracker.get_stats()['mse']) + '\n')
                f.close()
            
            with open('valid_preds.txt', 'a') as f:
                f.write(str(all_preds) + '\n')
                f.close()
                
            with open('valid_labels.txt', 'a') as f:
                f.write(str(all_labels) + '\n')
                f.close()
                    
        if data_type == 'test':
            with open('test_preds.txt', 'a') as f:
                f.write(str(all_preds) + '\n')
                f.close()
                
            with open('test_labels.txt', 'a') as f:
                f.write(str(all_labels) + '\n')
                f.close()

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

    return stats_tracker

def reshape_tensor_to_bond_list(list_of_tensors):
    """
    Input: list of tensors i each of shape (ni, ni, d)
    Output: list of bond lists each of shape [bond1,...,bond_m] with bond_j.shape=(d)
    """
    bond_list = []
    for i in range(len(list_of_tensors)):
        tensor = list_of_tensors[i]
        current_list = []
        n = tensor.shape[0]
        assert tensor.shape[1] == n
        for j in range(n):
            for k in range(n):
                current_list.append(tensor[j][k])
        bond_list.append(torch.stack(current_list))
    return bond_list

if __name__ == '__main__':
    main()
