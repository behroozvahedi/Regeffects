"""
test.py

This script loads the top X checkpoints from the hyperparameter optimization, evaluates each model on the test set,
collects their predictions, computes an ensemble average, and saves a single CSV containing:
  - gene name
  - group ID
  - observed TPM
  - predictions from each of the top-X models (columns: model_1, model_2, ...)
  - the ensemble (average) prediction

All shared classes and functions are imported from utils.py
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import (
    set_random_seeds,
    get_device,
    get_indices,
    DNADualDatasetMeta,
    TwoBranchCNN,
    DummyTrial,
    evaluate_model
)

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate top-X 2BCNN models and save combined predictions.'
    )
    parser.add_argument('--val_group', type=str, required=True,
                        help='Validation group ID (e.g. "4")')
    parser.add_argument('--test_group', type=str, required=True,
                        help='Test group ID (e.g. "5")')
    parser.add_argument('--top_x', type=int, default=1,
                        help='Number of top models to include in ensemble')
    args = parser.parse_args()

    # Reproducibility and device
    set_random_seeds(42)
    device = get_device()
    print('Using device:', device)

    # Base directories
    base_dir = f"/faststorage/project/sieve/Behrooz/WP2/Models/2BCNN/Allspecies_PCembed/val{args.val_group}_test{args.test_group}/round8_Bdi_Osa/"
    ckpt_dir = base_dir
    output_dir = os.path.join(base_dir, 'test_results')
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    data_dir = "/faststorage/project/sieve/Behrooz/PC_Embeddings/PC_embeddings_corrected/npy_files/"
    tss    = np.load(os.path.join(data_dir, 'PCembed_allspecies_tss_new.npy'), mmap_mode='r')
    tts    = np.load(os.path.join(data_dir, 'PCembed_allspecies_tts_new.npy'), mmap_mode='r')
    TPM    = np.load(os.path.join(data_dir, 'PCembed_allspecies_TPM_new.npy'), mmap_mode='r')
    groups = np.load(os.path.join(data_dir, 'PCembed_allspecies_group_for_cross_validation_new.npy'), mmap_mode='r')
    gene   = np.load(os.path.join(data_dir, 'PCembed_allspecies_gene_new.npy'), allow_pickle=True)

    TPM = np.log10(1 + TPM)  # ensure log-transform

    # Compute splits
    train_idx, val_idx, test_idx = get_indices(args.val_group, args.test_group, groups)

    # Load global stats
    train_groups = np.unique(groups[train_idx])
    train_groups_str = '_'.join(map(str, np.sort(train_groups)))
    stats = np.load(os.path.join(data_dir, f'global_stats_train_{train_groups_str}.npz'))
    tss_mean, tss_std = stats['tss_mean'], stats['tss_std']
    tts_mean, tts_std = stats['tts_mean'], stats['tts_std']
    stats.close()

    # Build test dataset and loader
    test_dataset = DNADualDatasetMeta(
        test_idx, tss, tts, TPM,
        tss_mean, tss_std, tts_mean, tts_std,
        gene, groups
    )
    # Use default batch size for ensemble evaluation
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Read trial results
    trials_csv = os.path.join(ckpt_dir, 'trial_results.csv')
    df_trials = pd.read_csv(trials_csv)
    top_trials = df_trials.head(args.top_x).reset_index(drop=True)
    print('Top trials:')
    print(top_trials[['checkpoint_file','val_loss','batch_size']])

    # Storage for predictions and metadata
    all_predictions = []  # list of np arrays, one per model
    model_labels    = []  # model_1, model_2, ...
    criterion = nn.MSELoss()

    # Iterate over top-X trials
    for i, row in top_trials.iterrows():
        model_label = f'model_{i+1}'
        model_labels.append(model_label)

        # Load checkpoint
        ckpt_path = os.path.join(ckpt_dir, row['checkpoint_file'])
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Prepare dummy trial and model
        hp = {
            'n_conv_layers': int(row['n_conv_layers']),
            'n_filters': int(row['n_filters']),
            'kernel_size': int(row['kernel_size']),
            'n_dense_layers': int(row['n_dense_layers']),
            'dense_units': int(row['dense_units']),
            'n_post_dense_layers': int(row['n_post_dense_layers']),
            'dropout_rate': float(row['dropout_rate']),
            'batch_norm': True
        }
        dummy = DummyTrial(hp)
        model = TwoBranchCNN(dummy).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        loss, preds = evaluate_model(model, test_loader, device, criterion)
        all_predictions.append(preds.squeeze())
        print(f"{model_label}: test MSE = {loss:.6f}")

    # Stack predictions: shape (top_x, N)
    preds_stack = np.stack(all_predictions, axis=0)
    ensemble_preds = preds_stack.mean(axis=0)

    # Gather metadata per sample
    genes     = []
    groups_out= []
    observed  = []
    for tss_t, tts_t, target, gene_name, grp in test_dataset:
        genes.append(gene_name)
        groups_out.append(grp)
        observed.append(target.item())
    observed = np.array(observed)

    # Build DataFrame
    df_out = pd.DataFrame({
        'gene': genes,
        'group': groups_out,
        'observed_TPM': observed
    })
    # Add per-model predictions
    for label, preds in zip(model_labels, all_predictions):
        df_out[label] = preds.squeeze()
    # Add ensemble column
    df_out['ensemble_pred'] = ensemble_preds.squeeze()

    # Save single CSV
    out_csv = os.path.join(output_dir, 'test_predictions.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"Saved combined predictions to {out_csv}")

