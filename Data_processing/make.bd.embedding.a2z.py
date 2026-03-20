import numpy as np
import logging
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
from a2z.data import seq_one_hot_encode
from tensorflow.keras import Model 
import tables as tb

def ProcessSequences(datafile, h5file, model, max_sequences = None):
    logging.info("Extracting a2z embeddings")

    sequences = []
    genes = []
    transcripts = []
    groups = []
    features = []
    chunks = []
    hashes = []
    chunk_size = 0

    dtype = np.dtype([
        ("gene", "S40"),       # String (fixed-length)
        ("transcript", "S55"),       # String (fixed-length)
        ("group", "S8000"),       # String (fixed-length)
        ("hash", int),       # integer
        ("tss_embed", float, (20, 925)),   # NumPy array (e.g., shape=(20,384))
        ("tts_embed", float, (20, 925)),   # NumPy array (e.g., shape=(20,384))
        ("tss_pred", float, (20,1)),   # NumPy array (e.g., shape=(20,384))
        ("tts_pred", float, (20,1))   # NumPy array (e.g., shape=(20,384))
    ])

    h5file = tb.open_file(h5file, 'w', buffering=False)
    h5 = h5file.create_table("/", 'data', dtype, expectedrows=1943255, filters=tb.Filters(complib="zlib", complevel=5))


    datafile = open(datafile, 'r')

    exported = 0
    counter = 0
    header = True
    begining = True
    for line in datafile:
        if header:
            header = False
            continue
        items = line.strip('\n').split('\t')

        if not begining and int(items[5]) == 0:
            counter+=1
        if max_sequences is not None and counter >= max_sequences:
            extract_a2z_embeddings(model, sequences, genes, transcripts, groups, hashes, features, chunks, chunk_size, h5)
            exported += counter

            sequences = []
            genes = []
            transcripts = []
            groups = []
            features = []
            chunks = []
            hashes = []
            counter = 0

        genes.append(items[0])
        transcripts.append(items[1])
        groups.append(items[2])
        sequences.append(items[3])
        features.append('tss')
        chunks.append(items[5])
        hashes.append(int(items[6]))

        genes.append(items[0])
        transcripts.append(items[1])
        groups.append(items[2])
        sequences.append(items[4])
        features.append('tts')
        chunks.append(items[5])
        hashes.append(int(items[6]))

        if chunk_size < int(items[5])+1:
            chunk_size = int(items[5])+1
        
        begining = False

    if counter > 0:
        extract_a2z_embeddings(model, sequences, genes, transcripts, groups, hashes, features, chunks, chunk_size, h5)
        exported += counter

    h5file.close()
    datafile.close()
    print('Exported %i a2z sequence embeddings.'%(exported))

def extract_a2z_embeddings(model, sequences, genes, transcripts, groups, hashes, features, chunks,chunk_size, h5):
    tss_sequences = []
    tts_sequences = []
    counter = 0
    chunk_counter = 0
    
    row = h5.row
    for index in range(len(sequences)):
        chunk_counter += 1

        chunk = int(chunks[index])
        feature = features[index]
        gene = genes[index]
        transcript = transcripts[index]
        group = groups[index]
        h = hashes[index]

        if feature == 'tss':
            tss_sequences.append(seq_one_hot_encode(sequences[index]))
        if feature == 'tts':
            tts_sequences.append(seq_one_hot_encode(sequences[index]))

        if chunk_counter == chunk_size*2:
            chunk_counter = 0
            prediction = model.predict(tf.convert_to_tensor(tss_sequences))

            tss_embeddings = prediction[0]
            tss_predictions = prediction[1]

            prediction = model.predict(tf.convert_to_tensor(tts_sequences))
            tts_embeddings = prediction[0]
            tts_predictions = prediction[1]

            row["gene"] = gene.encode("utf-8")
            row["transcript"] = transcript.encode("utf-8")
            row["group"] = group.encode("utf-8")
            row["hash"] = h
            row["tss_embed"] = tss_embeddings
            row["tts_embed"] = tts_embeddings
            row["tss_pred"] = tss_predictions
            row["tts_pred"] = tts_predictions
            row.append()
            h5.flush()

            counter += 1

            tss_sequences = []
            tts_sequences = []
            tf.keras.backend.clear_session()


#class H5Dataset(Dataset):
class H5Dataset():
    def __init__(self, file):

        self.file = tb.open_file(file, mode="r")
        self.table = self.file.get_node('/data')

    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        row = self.table[idx]

        return row['gene'].decode(), row['transcript'].decode(), row['group'].decode(), int(row['hash']), row['tss_embed'], row['tts_embed'], row['tss_pred'], row['tts_pred']
        #return row['gene'].decode(), row['transcript'].decode(), row['group'].decode(), int(row['hash']), torch.tensor(row['tss_embed'], dtype=torch.float32), torch.tensor(row['tts_embed'], dtype=torch.float32), torch.tensor(row['tss_pred'], dtype=torch.float32), torch.tensor(row['tts_pred'], dtype=torch.float32)

    def done(self):
        self.file.close()

def main():
    #workdir = '/Volumes/N1/Embeddings/'
    #outdir = '/Volumes/N1/Embeddings/'
    workdir = '/usr/home/qgg/camo/Embeddings/'
    outdir = '/usr/home/qgg/camo/Embeddings/'
    in_file = 'bd.sequences.a2z.tsv'
    out_file = 'embeddings.bd.a2z.h5'

    perform_extraction = True
    perform_testing = False

    if perform_extraction:
        model = tf.keras.models.load_model(workdir+'model-accessibility-full.h5')
        new_model = Model(inputs=model.input, outputs=[model.get_layer('dense').output,model.get_layer('dense_1').output])

        #Set to None to load all sequences at once
        max_sequences_loaded = 10000
        ProcessSequences(workdir+in_file, outdir+out_file, new_model, max_sequences_loaded)

    if perform_testing:
        print('Reading h5 data.')
        h5data = H5Dataset(outdir+out_file)
        #dataloader = DataLoader(h5data, batch_size=1, shuffle=False)
        for index in range(h5data.__len__()):

            gene, transcript, group, h, tss_embed, tts_embed, tss_pred, tts_pred  = h5data.__getitem__(index)
            print('gene: %s'%gene)
            print('transcript: %s'%transcript)
            print('group: %i'%len(group.split(' ')))
            print('hash: %i'%h)
            print(tss_embed.shape)
            print(tts_embed.shape)
            print(tss_pred.shape)
            print(tts_pred.shape)

            #print('gene: %s'%gene[0])
            #print('transcript: %s'%transcript[0])
            #print('group: %i'%len(group[0].split(' ')))
            #print('hash: %i'%h[0])
            #print(tss_embed[0].shape)
            #print(tts_embed[0].shape)
            #print(tss_pred[0].shape)
            #print(tts_pred[0].shape)
            break
        h5data.done()

if __name__ == "__main__":
    main()
