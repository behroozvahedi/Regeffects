#!/usr/bin/env python
import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import tables as tb

from utils_PC_a2z import (
    set_random_seeds,
    get_device,
    get_indices,
    DNADualDataset,
    TwoBranchCNN,
    DummyTrial,
    evaluate_model,
)

def GetTranslations(file):
    translation = dict()
    file = open(file,'r')

    counter = 0
    for line in file:
        items = line.strip('\n').split('\t')
        translation[items[1]] = items[0].split('.')[1]
    return(translation)

def GetGroups(file, translation):
    groups = dict()
    file = open(file,'r')

    counter = 0
    header = True
    for line in file:
        if header:
            header = False
            continue
        items = line.strip('\n').split(',')
        gene_id = items[1].strip('"')

        if gene_id in translation:
            gene_id = translation[gene_id]
            groups[gene_id] = items[8]

    return(groups)

class H5Dataset(Dataset):
    def __init__(self, file_caduceus, file_a2z, models, stats, test_groups):

        self.caduceus = tb.open_file(file_caduceus, mode="r")
        self.a2z = tb.open_file(file_a2z, mode="r")
        self.table_caduceus = self.caduceus.get_node('/data')
        self.table_a2z = self.a2z.get_node('/data')
        self.models = models
        self.stats = stats
        self.test_groups = test_groups

        if len(self.table_caduceus) != len(self.table_a2z):
            print('Error: caduceus data length is %i while a2z is %i'%(len(self.table_caduceus),len(self.table_a2z)))
            #exit(0)

    def __len__(self):
        return len(self.table_caduceus)
    
    def __getitem__(self, idx):
        row_caduceus = self.table_caduceus[idx]
        gene = row_caduceus['gene'].decode()

        if gene in self.test_groups:
            test_group = self.test_groups[gene]
            EXTRA = self.stats[test_group]['EXTRA']

            if EXTRA == 'a2z':
                row_a2z = self.table_a2z[idx]
                tss_mean = self.stats[test_group]['tss_emb_mean']
                tss_std = self.stats[test_group]['tss_emb_std']
                tss_mean = np.squeeze(tss_mean, axis = 0)
                tss_std = np.squeeze(tss_std, axis = 0)
                tss_sample = np.transpose(row_a2z['tss_embed'])
                tss_sample = (tss_sample - tss_mean) / tss_std

                tts_mean = self.stats[test_group]['tts_emb_mean']
                tts_std = self.stats[test_group]['tts_emb_std']
                tts_mean = np.squeeze(tts_mean, axis = 0)
                tts_std = np.squeeze(tts_std, axis = 0)
                tts_sample = np.transpose(row_a2z['tts_embed'])
                tts_sample = (tts_sample - tts_mean) / tts_std
            else:
                tss_mean = self.stats[test_group]['tss_mean']
                tss_std = self.stats[test_group]['tss_std']
                tts_mean = self.stats[test_group]['tts_mean']
                tts_std = self.stats[test_group]['tts_std']

                tss_mean = np.squeeze(tss_mean, axis = 0)
                tss_std = np.squeeze(tss_std, axis = 0)
                tts_mean = np.squeeze(tts_mean, axis = 0)
                tts_std = np.squeeze(tts_std, axis = 0)

                #print(tss_mean.shape)
                #print(tss_std.shape)

                # Load and standardize tss
                tss_sample = np.transpose(row_caduceus['tss'])
                #print(tss_sample.shape)
                tss_sample = (tss_sample - tss_mean) / tss_std
                #print(tss_sample.shape)

                #print(tss_sample)

                # Load and standardize tts
                tts_sample = np.transpose(row_caduceus['tts'])
                tts_sample = (tts_sample - tts_mean) / tts_std
                #print(tss_sample)

                #if row_caduceus['hash'] != row_a2z['hash']:
                #    print('Warning: gene %s has differing caduceus and a2z hashes'%gene)

                if EXTRA != 'none':
                    row_a2z = self.table_a2z[idx]
                    if gene != row_a2z['gene'].decode():
                        print('Error: caduceus and a2z row mismatch at index: %'%idx)
                        exit(0)
                    extra_tss_mean = self.stats[test_group]['tss_%s_mean'%EXTRA]
                    extra_tss_std = self.stats[test_group]['tss_%s_std'%EXTRA]
                    extra_tss_mean = np.squeeze(extra_tss_mean, axis = 0)
                    extra_tss_std = np.squeeze(extra_tss_std, axis = 0)

                    #extra_tss_sample = np.transpose(row_a2z['tss_%s'%EXTRA])
                    #extra_tss_sample = row_a2z['tss_%s'%EXTRA].reshape(1,C_EXTRA,20)
                    if EXTRA != 'emb':
                        extra_tss_sample = np.transpose(row_a2z['tss_%s'%EXTRA])
                    else:
                        extra_tss_sample = np.transpose(row_a2z['tss_embed'])

                    #extra_tss_sample = np.expand_dims(extra_tss_sample, axis=0)
                    #if EXTRA == 'pred':
                    #    extra_tss_sample = np.expand_dims(extra_tss_sample, axis=0)
                    #print('extra_tss_sample:')
                    #print(extra_tss_sample.shape)
                    #print('extra_tss_mean:')
                    #print(extra_tss_mean.shape)
                    #print('extra_tss_std:')
                    #print(extra_tss_std.shape)
                    #print('- standardize -')
                    extra_tss_sample = (extra_tss_sample - extra_tss_mean) / extra_tss_std
                    #print('extra_tss_sample:')
                    #print(extra_tss_sample.shape)

                    #extra_tss_sample = np.squeeze(extra_tss_sample, axis=0)  # shape: (C_extra, P)
                    #print(extra_tss_sample.shape)
                    #print('tss_sample:')
                    #print(tss_sample.shape)
                    #print('- concat -')
                    tss_sample = np.concatenate([tss_sample, extra_tss_sample], axis=0)   # shape: (1, C + C_extra, P)
                    #print('tss_sample:')
                    #print(tss_sample.shape)

                    extra_tts_mean = self.stats[test_group]['tts_%s_mean'%EXTRA]
                    extra_tts_std = self.stats[test_group]['tts_%s_std'%EXTRA]
                    extra_tts_mean = np.squeeze(extra_tts_mean, axis = 0)
                    extra_tts_std = np.squeeze(extra_tts_std, axis = 0)
                    #extra_tts_sample = np.transpose(row_a2z['tts_%s'%EXTRA])
                    if EXTRA != 'emb':
                        extra_tts_sample = np.transpose(row_a2z['tts_%s'%EXTRA])
                    else:
                        extra_tts_sample = np.transpose(row_a2z['tts_embed'])

                    #extra_tts_sample = np.expand_dims(extra_tts_sample, axis=0)
                    #if EXTRA == 'pred':
                    #    extra_tts_sample = np.expand_dims(extra_tts_sample, axis=0)

                    extra_tts_sample = (extra_tts_sample - extra_tts_mean) / extra_tts_std
                    #extra_tts_sample = np.squeeze(extra_tts_sample, axis=0)  # shape: (C_extra, P)
                    tts_sample = np.concatenate([tts_sample, extra_tts_sample], axis=0)   # shape: (1, C + C_extra, P)

            #tss_sample = np.expand_dims(tss_sample, axis=0)
            #tts_sample = np.expand_dims(tts_sample, axis=0)
            return self.models[test_group], str(row_caduceus['group'].decode()), row_caduceus['gene'].decode(), row_caduceus['transcript'].decode(), int(row_caduceus['hash']), test_group, torch.tensor(tss_sample, dtype=torch.float32), torch.tensor(tts_sample, dtype=torch.float32)
        else:
            print('skipping gene: %s, transcript: %s'%(row['gene'].decode(), row['transcript'].decode()))
            return None, None, None, None, None, None, None, None

    def done(self):
        self.caduceus.close()
        self.a2z.close()

def Predic(models, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []

    with torch.no_grad():
        for x_tss, x_tts, target in dataloader:
            x_tss = x_tss.to(device)
            x_tts = x_tts.to(device)
            target = target.to(device).unsqueeze(1)

            output = model(x_tss, x_tts)
            loss = criterion(output, target)

            batch_size = x_tss.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_predictions.append(output.cpu().numpy())

    avg_loss = total_loss / total_samples
    predictions = np.concatenate(all_predictions, axis=0)
    return avg_loss, predictions

def main():
    workdir = '/Volumes/N1/Embeddings/DATA/'
    # Set to either: none, pred, emb or a2z
    #EXTRA = "emb"
    for EXTRA in ['none', 'pred', 'emb', 'a2z']:
        print('Generating predictions for %s'%EXTRA)
        caduceus_file = workdir + 'data.csv'
        translation_file = workdir + 'gene.id.translation.tsv'

        translations = GetTranslations(translation_file)
        test_groups = GetGroups(caduceus_file, translations)

        groups = np.load(workdir+'group_for_cross_validation.npy', mmap_mode="r", allow_pickle=True)

        caduceus_embeddings_file = workdir+'embeddings.bd.caduceus.h5'
        a2z_embeddings_file = workdir+'embeddings.bd.a2z.h5'

        TOP_X = 5
        device = 'cpu'

        group_stats = dict()
        group_models = dict()

        for val_group in range(1, 6):
            test_group = (val_group + 1) % 6
            if test_group == 0:
                test_group = 1

            test_group = str(test_group)
            val_group = str(val_group)

            print('Loading: val%s_test%s'%(val_group, test_group))

            train_idx, _, _ = get_indices(val_group, test_group, groups)
            train_groups = np.unique(groups[train_idx])
            train_groups_str = "_".join(map(str, np.sort(train_groups)))

            stats = np.load(workdir+'global_stats_train_%s.npz'%train_groups_str)
            group_stats[test_group] = {'EXTRA':EXTRA,'tss_mean':stats['tss_mean'],'tss_std':stats['tss_std'],'tts_mean':stats['tts_mean'],'tts_std':stats['tts_std'],'tss_pred_mean':stats['tss_pred_mean'],'tss_pred_std':stats['tss_pred_std'],'tts_pred_mean':stats['tts_pred_mean'],'tts_pred_std':stats['tts_pred_std'],'tss_emb_mean':stats['tss_emb_mean'],'tss_emb_std':stats['tss_emb_std'],'tts_emb_mean':stats['tts_emb_mean'],'tts_emb_std':stats['tts_emb_std']}
            stats.close()

            if EXTRA == 'a2z':
                in_channels = group_stats[test_group]['tss_emb_mean'].shape[1]

                # ------------------------------------------------------------------------
                # 7. Locate top-X trials CSV
                # ------------------------------------------------------------------------
                run_dir = os.path.join(
                    workdir,
                    f"val{val_group}_test{test_group}",
                    "a2z_emb_only"
                )
            else:
                base_channels = group_stats[test_group]['tss_mean'].shape[1]

                if EXTRA == "none":
                    extra_channels = 0
                elif EXTRA == "pred":
                    extra_channels = group_stats[test_group]['tss_pred_std'].shape[1]
                elif EXTRA == "emb":
                    extra_channels = group_stats[test_group]['tss_emb_std'].shape[1]
                else:
                    print('Unknown EXTRA: %s'%EXTRA)
                    exit(0)

                #extra_channels = group_stats[test_group]['extra_tss_std'].shape[1] if EXTRA != "none" else 0
                in_channels = base_channels + extra_channels

                # ------------------------------------------------------------------------
                # 7. Locate top-X trials CSV
                # ------------------------------------------------------------------------
                run_dir = os.path.join(
                    workdir,
                    f"val{val_group}_test{test_group}",
                    "base_models" if EXTRA=="none" else f"full_models_{EXTRA}"
                )
            CHECKPOINTS_DIR = run_dir
            TRIALS_CSV = os.path.join(CHECKPOINTS_DIR, "trial_results.csv")

            df_trials = pd.read_csv(TRIALS_CSV)
            top_trials = df_trials.head(TOP_X).reset_index(drop=True)

            # ------------------------------------------------------------------------
            # 8. Evaluate each of the top-X models
            # ------------------------------------------------------------------------

            group_models[test_group] = []

            for _, row in top_trials.iterrows():
                hp = {
                    "n_conv_layers":       int(row["n_conv_layers"]),
                    "n_filters":           int(row["n_filters"]),
                    "kernel_size":         int(row["kernel_size"]),
                    "n_dense_layers":      int(row["n_dense_layers"]),
                    "dense_units":         int(row["dense_units"]),
                    "n_post_dense_layers": int(row["n_post_dense_layers"]),
                    "dropout_rate":        float(row["dropout_rate"]),
                    "batch_norm":          True,
                }
                trial_obj = DummyTrial(hp)

                model = TwoBranchCNN(trial_obj, in_channels=in_channels).to(device)
                checkpoint_file = row["checkpoint_file"].strip()
                checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_file)
                #print(checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                group_models[str(test_group)].append(model)

        dataset = H5Dataset(caduceus_embeddings_file, a2z_embeddings_file, group_models, group_stats, test_groups)

        counter = 0
        out = open(workdir+'predictions_%s.tsv'%EXTRA,'w')
        out.write('id\tgene\ttranscript\thash(seq)\ttest_group\tmodel_1_pred\tmodel_2_pred2\tmodel_3_pred\tmodel_4_pred\tmodel_5_pred\n')
        with torch.no_grad():
            for index in range(dataset.__len__()):
                counter += 1
                if not counter % 10000:
                    print(counter)
                models, ids, gene, transcript, h, test_group, tss, tts = dataset.__getitem__(index)
                if model is None:
                    continue
                #print(tss.shape)
                #print(tts.shape)
                #tss = np.transpose(tss)
                #tts = np.transpose(tts)
                tss = torch.tensor(np.expand_dims(tss, axis=0), dtype=torch.float32)
                tts = torch.tensor(np.expand_dims(tts, axis=0), dtype=torch.float32)
                #tss = np.transpose(tss)
                #tts = np.transpose(tts)
                tss = tss.to(device)
                tts = tts.to(device)
                preds = ''
                for model in models:
                    preds += '\t%f'%model(tss, tts).numpy()

                for id in ids.split(' '):
                    out.write('%s\t%s\t%s\t%d\t%s%s\n'%(id,gene,transcript,h,test_group,preds))
        dataset.done()
        out.close()

if __name__ == "__main__":
    main()
