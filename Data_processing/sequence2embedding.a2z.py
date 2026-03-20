import json
import gzip
import pickle
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
import numpy as np
from a2z.data import seq_one_hot_encode
from tensorflow.keras import Model 

#Loads sequences prepared by "prepare.sequences.py"
def LoadSequences(filepath, max_sequences = None):
    sequences = []
    genes = []
    families = []
    tpms = []
    features = []
    chunks = []
    chunk_size = 0
    data = open(filepath)

    counter = 0
    header = True
    begining = True
    for line in data:
        if header:
            header = False
            continue
        items = line.strip('\n').split('\t')

        if not begining and int(items[5]) == 0:
            counter+=1
        if max_sequences is not None and counter >= max_sequences:
            break

        genes.append(items[0])
        families.append(items[1])
        tpms.append(items[2])
        sequences.append(items[3])
        features.append('tss')
        chunks.append(items[5])

        genes.append(items[0])
        families.append(items[1])
        tpms.append(items[2])
        sequences.append(items[4])
        features.append('tts')
        chunks.append(items[5])

        if chunk_size < int(items[5])+1:
            chunk_size = int(items[5])+1
        
        begining = False
    data.close()

    print('Loaded %i sequences.'%counter)

    return sequences, genes, families, tpms, features, chunks, chunk_size

#Extracts embeddings for the loaded sequences 
def extract_a2z_embeddings(model, sequences, genes, families, tpms, features, chunks, file, species, compressed = False):
    logging.info("Extracting a2z embeddings")

    if compressed:
        file = gzip.open(file+'.bin', mode='wt', compresslevel=9, encoding=None, errors=None, newline=None)
    else:
        file = open(file+'.tsv','w')
    file.write('gene\tfamily\tTPM\tdimensions\ttss\ttts\n')

    tss_sequences = []
    tts_sequences = []
    gene = None
    family = None
    tpm = None
    current_gene = None
    current_family = None
    current_tpm = None
    counter = 0

    for index in range(len(sequences)):
        feature = features[index]
        gene = genes[index]
        family = families[index]
        tpm = tpms[index]

        if current_gene is None:
            current_gene = gene
            current_family = family
            current_tpm = tpm

        if gene != current_gene or index == len(sequences)-1:
            if index == len(sequences)-1:
                if feature == 'tss':
                    tss_sequences.append(seq_one_hot_encode(sequences[index]))
                if feature == 'tts':
                    tts_sequences.append(seq_one_hot_encode(sequences[index]))

            prediction = model.predict(tf.convert_to_tensor(tss_sequences))
            tss_embeddings = prediction[0]
            tss_predictions = prediction[1] 

            prediction = model.predict(tf.convert_to_tensor(tts_sequences))
            tts_embeddings = prediction[0]
            tts_predictions = prediction[1]

            file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(current_gene, current_family, current_tpm, json.dumps(tss_embeddings.shape),json.dumps(tss_embeddings.reshape(-1).tolist()), json.dumps(tts_embeddings.reshape(-1).tolist()),json.dumps(tss_predictions.reshape(-1).tolist()),json.dumps(tts_predictions.reshape(-1).tolist()) ))
            counter += 1

            current_gene = gene
            current_family = family
            current_tpm = tpm
            tss_sequences = []
            tts_sequences = []
            tf.keras.backend.clear_session()

        if feature == 'tss':
            tss_sequences.append(seq_one_hot_encode(sequences[index]))
        if feature == 'tts':
            tts_sequences.append(seq_one_hot_encode(sequences[index]))

    print('Exported %i sequence embeddings for species %s.'%(counter,species))
    file.close()

#Example code for reading extracted embeddings
def Import(file):
    print('Starting import.')

    if file[-3:] == 'bin':
        print('Assuming compressed file.')
        file = gzip.open(file, mode='rt', compresslevel=9, encoding=None, errors=None, newline=None)
    else:
        print('Assuming uncompressed file.')
        file = open(file,'r')

    counter = 0
    header = True
    for line in file:
        if header:
            header = False
            continue
        counter += 1

        items = line.strip('\n').split('\t')

        gene = items[0]
        family = items[1]
        TPM = float(items[2])
        dimensions = json.loads(items[3])
        tss_embeddings = np.array(json.loads(items[4])).reshape(dimensions)
        tts_embeddings = np.array(json.loads(items[5])).reshape(dimensions)
        tss_predictions = np.array(json.loads(items[6]))
        tts_predictions = np.array(json.loads(items[7]))

        print('gene: %s'%gene)
        print('family: %s'%family)
        print('TPM: %f'%TPM)
        print(tss_embeddings.shape)
        print(tts_embeddings.shape)
        print(tss_predictions.shape)
        print(tts_predictions.shape)
        #print(tss_embeddings)
        #print(tts_embeddings)
        #print(tss_predictions)
        #print(tts_predictions)

    file.close()
    print('Imported %i embeddings.'%counter)

def main():
    workdir = '/Users/au468646/Downloads/Embeddings/'
    outdir = '/Volumes/N1/a2z/'

    model = tf.keras.models.load_model('/Users/au468646/Downloads/Embeddings/model-accessibility-full.h5')
    new_model = Model(inputs=model.input,
                      outputs=[model.get_layer('dense').output,model.get_layer('dense_1').output])

    #Set to None to load all sequences
    max_sequences_loaded = None

    species_set = ['Csa', 'Sly', 'Sbi', 'Ath', 'Vvi', 'Osa', 'Mtr', 'Ppa', 'Gma', 'Bdi', 'Sit', 'Ptr', 'Svi', 'Bvu', 'Cre', 'Zma', 'Stu']

    for species in species_set:
        print(species)
        sequences, genes, families, tpms, features, chunks, chunk_size = LoadSequences(workdir+'a2z.sequences.%s.tsv'%species, max_sequences_loaded)
        extract_a2z_embeddings(new_model, sequences, genes, families, tpms, features, chunks,  outdir+'a2z.training.data.%s'%species, species, True)

    #Import(workdir+'training.data.Bdi.bin')
if __name__ == "__main__":
    main()

