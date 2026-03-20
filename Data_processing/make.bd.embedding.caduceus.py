import json
import gzip
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import logging
import time
import tables as tb

def ProcessSequences(datafile, h5file, model, tokenizer, device, core_sequence_size, batch_size, max_sequences = None):
    logging.info("Extracting caduceus embeddings")

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
        ("tss", float, (20, 384)),   # NumPy array (e.g., shape=(20,384))
        ("tts", float, (20, 384))   # NumPy array (e.g., shape=(20,384))
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
            adjusted_batch_size = (batch_size//(chunk_size*2))*(chunk_size*2)
            if adjusted_batch_size < chunk_size*2:
                adjusted_batch_size = chunk_size*2
            data_loader = create_dataloader(sequences, genes, transcripts, groups, hashes, features, chunks, tokenizer, adjusted_batch_size)
            extract_embeddings(model, data_loader, device, core_sequence_size, chunk_size, h5)
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
        adjusted_batch_size = (batch_size//(chunk_size*2))*(chunk_size*2)
        if adjusted_batch_size < chunk_size*2:
            adjusted_batch_size = chunk_size*2
        data_loader = create_dataloader(sequences, genes, transcripts, groups, hashes, features, chunks, tokenizer, adjusted_batch_size)
        extract_embeddings(model, data_loader, device, core_sequence_size, chunk_size, h5)
        exported += counter

    h5file.close()
    datafile.close()
    print('Exported %i caduceus sequence embeddings.'%(exported))

def create_dataloader(sequences, genes, transcripts, groups, hashes, features, chunks, tokenizer, batch_size):
    logging.info(f"Creating DataLoader with batch size {batch_size}")
    dataset = SequenceDataset(sequences, genes, transcripts, groups, hashes, features, chunks, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

class SequenceDataset(Dataset):
    def __init__(self, sequences, genes, transcripts, groups, hashes, features, chunks, tokenizer):
        self.sequences = sequences
        self.genes = genes
        self.transcripts = transcripts
        self.groups = groups
        self.hashes = hashes
        self.sequences = sequences
        self.features = features
        self.chunks = chunks
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        gene = self.genes[idx]
        transcript = self.transcripts[idx]
        group = self.groups[idx]
        h = self.hashes[idx]
        feature = self.features[idx]
        chunk = self.chunks[idx]
        tokenizer = self.tokenizer
        encoding = self.tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )
        input_ids = encoding['input_ids']
        #print('gene: %s, transcript: %s, feature: %s, chunk: %s'%(gene,transcript, feature, chunk))
        return {
            'sequence': sequence,
            'gene': gene,
            'transcript': transcript,
            'group': group,
            'hash': h,
            'feature': feature,
            'chunk': chunk,
            'input_ids': input_ids.squeeze()
        }

def load_model_and_tokenizer(model_dir, device):
    logging.info(f"Loading model and tokenizer from {model_dir}")
    
    # Determine the appropriate dtype based on the GPU capabilities
    def get_optimal_dtype():
        if not torch.cuda.is_available():
            logging.info("Using float32 as no GPU is available.")
            return torch.float32  

        device_index = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device_index)

        if capability[0] >= 8:  # sm_80 or higher
            logging.info("Using bfloat16 as the GPU supports sm_80 or higher.")
            return torch.bfloat16
        elif capability[0] >= 6:  # sm_60 or higher
            logging.info("Using float16 as the GPU supports sm_60 or higher.")
            return torch.float16
        else:
            logging.info("Using float32 as the GPU does not support float16 or bfloat16.")
            return torch.float32

    optimal_dtype = get_optimal_dtype()

    # Load the model with the selected dtype
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=optimal_dtype)
    except Exception as e:
        logging.error(f"Failed to load model with {optimal_dtype}, falling back to float32. Error: {e}")
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    return model, tokenizer

def extract_embeddings(model, dataloader, device, core_sequence_size, chunk_size, h5):
    counter = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].to(torch.float32).cpu().numpy()
            hidden_size = embeddings.shape[-1] // 2
            forward = embeddings[..., 0:hidden_size]
            reverse = embeddings[..., hidden_size:]
            reverse = reverse[..., ::-1]
            averaged_embeddings = (forward + reverse) / 2
            input_sequence_size = embeddings.shape[-2]
            core_sequence_start = (input_sequence_size//2)-(core_sequence_size//2)
            core_sequence_end = core_sequence_start + core_sequence_size
            averaged_embeddings = averaged_embeddings[:,  core_sequence_start:core_sequence_end , :]
            
            averaged_embeddings = np.expand_dims(averaged_embeddings.mean(axis = 1), axis = 1)
            genes = batch['gene']
            transcripts = batch['transcript']
            groups = batch['group']
            features = batch['feature']
            chunks = batch['chunk']
            hashes = batch['hash']
            counter += export(averaged_embeddings, genes, transcripts, groups, hashes, features, chunks, chunk_size, h5)
            if not counter % 5000:
                print('processed: %i'%counter)

    print('Exported %i sequence embeddings.'%(counter))

def export(embeddings, genes, transcripts, groups, hashes, features, chunks, chunk_size, h5):
    tss_embeddings = []
    tts_embeddings = []
    row = h5.row

    counter = 0
    chunk_counter = 0
    for index in range(len(embeddings)):
        chunk_counter += 1

        chunk = int(chunks[index])
        feature = features[index]
        gene = genes[index]
        transcript = transcripts[index]
        group = groups[index]
        h = hashes[index]
        #print(group)
        #print(h)
        #exit(0)

        if feature == 'tss':
            tss_embeddings.append(embeddings[index])
        if feature == 'tts':
            tts_embeddings.append(embeddings[index])

        if chunk_counter == chunk_size*2:
            chunk_counter = 0
            
            tss_embeddings = np.concatenate(tss_embeddings, axis=0)
            tts_embeddings = np.concatenate(tts_embeddings, axis=0)
            #print(group.encode("utf-8"))
            #exit(0)
            row["gene"] = gene.encode("utf-8")
            row["transcript"] = transcript.encode("utf-8")
            row["group"] = group.encode("utf-8")
            row["hash"] = h
            row["tss"] = tss_embeddings
            row["tts"] = tts_embeddings
            row.append()
            h5.flush()
            #print('flush!')

            counter += 1
            
            tss_embeddings = []
            tts_embeddings = []


    return counter

class H5Dataset(Dataset):
    def __init__(self, file):

        self.file = tb.open_file(file, mode="r")
        self.table = self.file.get_node('/data')

    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        row = self.table[idx]

        return row['gene'].decode(), row['transcript'].decode(), row['group'].decode(), int(row['hash']), torch.tensor(row['tss'], dtype=torch.float32), torch.tensor(row['tts'], dtype=torch.float32)

    def done(self):
        self.file.close()

def main():
    workdir = '/usr/home/qgg/camo/Embeddings/'
    outdir = '/usr/home/qgg/camo/Embeddings/'
    #workdir = '/Volumes/N1/Embeddings/'
    #outdir = '/Volumes/N1/Embeddings/'
    in_file = 'bd.sequences.caduceus.tsv'
    out_file = 'embeddings.bd.caduceus.h5'

    perform_extraction = True
    perform_testing = False

    if perform_extraction:
        #Set to None to load all sequences at once
        max_sequences_loaded = 10000

        batch_size = 200
        core_sequence_size = 250

        device = 'cuda'
        model, tokenizer = load_model_and_tokenizer(workdir+'PlantCaduceus_l20', device)
        ProcessSequences(workdir+in_file, outdir+out_file, model, tokenizer, device, core_sequence_size, batch_size, max_sequences_loaded)

    if perform_testing:
        print('Reading h5 data.')
        h5data = H5Dataset(outdir+out_file)
        dataloader = DataLoader(h5data, batch_size=1, shuffle=False)
        for gene, transcript, group, h, tss, tts in dataloader:
            print('gene: %s'%gene[0])
            print('transcript: %s'%transcript[0])
            print('group: %i'%len(group[0].split(' ')))
            print('hash: %i'%h[0])
            print(tss[0].shape)
            print(tts[0].shape)
            break
        h5data.done()

if __name__ == "__main__":
    main()
