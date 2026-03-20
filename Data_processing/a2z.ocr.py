from FASTA import FASTA
import numpy as np
import torch
import logging
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
from a2z.data import seq_one_hot_encode
from tensorflow.keras import Model 
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, ref_sequences,alt_sequences,variants):        
        self.ref_sequences = ref_sequences
        self.alt_sequences = alt_sequences
        self.variants = variants

    def __len__(self):
        return len(self.variants)
    
    def __getitem__(self, idx):
        ref_seq = seq_one_hot_encode(self.ref_sequences[idx])
        alt_seq = seq_one_hot_encode(self.alt_sequences[idx])
        variant = self.variants[idx]
        return (ref_seq, alt_seq, variant)


def GetSequences(fasta_file, bim_file):
    chromosomes = FASTA(fasta_file)

    counter = 0
    skipped = []
    seq_size = 300

    bim_file = open(bim_file,'r')

    ref_seqs = []
    alt_seqs = []
    variants = []

    for line in bim_file:
        counter += 1
        items = line.strip('\n').split('\t')
        chromosome = items[0]
        position = int(items[3])
        allele1 = items[5]
        allele2 = items[4]
        variant = '%s:%i:%s/%s'%(chromosome,position,allele1,allele2)
        if allele2 == '*':
            allele2 = 'N'
        ref_seq = chromosomes[chromosome][(position-1)-seq_size:(position-1)+seq_size]
        alt_seq = chromosomes[chromosome][(position-1)-seq_size:(position-1)] + allele2 + chromosomes[chromosome][(position-1)+1:(position-1)+seq_size]
        if len(ref_seq) != 600 or len(alt_seq) != 600:
            skipped.append(variant)
            continue            
        ref_seqs.append(ref_seq)
        alt_seqs.append(alt_seq)
        variants.append(variant)

    print('Processed %i variants, %i skipped: '%(counter, len(skipped)))
    print(skipped)
    return ref_seqs, alt_seqs, variants

def a2z_ocr(model, dataset, batch_size, file):
    logging.info("Making a2z ocr scores")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    file = open(file,'w')
    file.write('variant\tref.ocr\talt.ocr\n')

    counter = 0
    for ref_seqs, alt_seqs, variants in data_loader:
        ref_preds = model.predict(tf.convert_to_tensor(ref_seqs))
        alt_preds = model.predict(tf.convert_to_tensor(alt_seqs))

        for index in range(len(variants)):
            file.write('%s\t%f\t%f\n'%(variants[index], ref_preds[index], alt_preds[index]))
            counter += 1

    print('Made predictions for %i variants.'%(counter))
    file.close()

def a2z_ocr_old(model, ref_sequences, alt_sequences, variants, file):
    logging.info("Extracting a2z embeddings")

    file = open(file,'w')
    file.write('variant\tref.orc\talt.ocr\n')

    counter = 0

    for index in range(len(ref_sequences)):
        ref_seq = ref_sequences[index]
        alt_seq = alt_sequences[index]
        variant = variants[index]

        ref_pred = model.predict(tf.convert_to_tensor([seq_one_hot_encode(ref_seq)]))
        alt_pred = model.predict(tf.convert_to_tensor([seq_one_hot_encode(alt_seq)]))
        file.write('%s\t%f\t%f\n'%(variant, ref_pred, alt_pred))
        counter += 1

    print('Made predictions for %i variants.'%(counter))
    file.close()

def main():
    indir = '/Volumes/N1/a2z/'
    outdir = '/Volumes/N1/a2z/'

    # Fasta file from BD21 Reference genome
    fastafile = 'BdistachyonBd21_3_537_v1.0.fa'

    # bim file generated from BD21.3 VCF file
    bimfile = 'snps.combined.bim'

    # a2z trained model file 
    modelfile = 'model-accessibility-full.h5'

    model = tf.keras.models.load_model(modelfile)
    ref_sequences, alt_sequences, variants = GetSequences(indir+fastafile, indir+bimfile)
    dataset = MyDataset(ref_sequences, alt_sequences, variants)
    a2z_ocr(model, dataset, 10000, outdir+'a2z.ocr.tsv')

if __name__ == "__main__":
    main()

