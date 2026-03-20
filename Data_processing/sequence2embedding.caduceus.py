import json
import gzip
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import logging

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

def create_dataloader(sequences, genes, families, tpms, features, chunks, tokenizer, batch_size):
    logging.info(f"Creating DataLoader with batch size {batch_size}")
    dataset = SequenceDataset(sequences, genes, families, tpms, features, chunks, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

class SequenceDataset(Dataset):
    def __init__(self, sequences, genes, families, tpms, features, chunks, tokenizer):
        self.sequences = sequences
        self.genes = genes
        self.families = families
        self.tpms = tpms
        self.sequences = sequences
        self.features = features
        self.chunks = chunks
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        gene = self.genes[idx]
        family = self.families[idx]
        tpm = self.tpms[idx]
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
        return {
            'sequence': sequence,
            'gene': gene,
            'family': family,
            'tpm': tpm,
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

#Extracts embeddings for the loaded sequences 
def extract_caduceus_embeddings(model, dataloader, device, core_sequence_size, file, species, compressed = False, average_chunks = False):
    logging.info("Extracting embeddings")

    if compressed:
        file = gzip.open(file+'.bin', mode='wt', compresslevel=9, encoding=None, errors=None, newline=None)
    else:
        file = open(file+'.tsv','w')
    file.write('gene\tfamily\tTPM\tdimensions\ttss\ttts\n')

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
            
            # Average by chunk (250 basepaire core sequence) 250x384 -> 1x384
            if average_chunks:
                averaged_embeddings = np.expand_dims(averaged_embeddings.mean(axis = 1), axis = 1)
            genes = batch['gene']
            families = batch['family']
            tpms = batch['tpm']
            features = batch['feature']
            chunks = batch['chunk']
            counter += Export(averaged_embeddings, genes, families, tpms, features, chunks, file)

    file.close()

    print('Exported %i sequence embeddings for species %s.'%(counter,species))

#Exports extrated embeddings for the loaded sequences 
def Export(embeddings, genes, families, tpms, features, chunks, file):
    tss_embeddings = []
    tts_embeddings = []
    gene = None
    family = None
    tpm = None
    current_gene = None
    current_family = None
    current_tpm = None

    counter = 0
    for index in range(len(embeddings)):
        chunk = int(chunks[index])
        feature = features[index]
        gene = genes[index]
        family = families[index]
        tpm = tpms[index]

        if current_gene is None:
            current_gene = gene
            current_family = family
            current_tpm = tpm

        if gene != current_gene or index == len(embeddings)-1:
            if index == len(embeddings)-1:
                if feature == 'tss':
                    tss_embeddings.append(embeddings[index])
                if feature == 'tts':
                    tts_embeddings.append(embeddings[index])
            tss_embeddings = np.concatenate(tss_embeddings, axis=0)
            tts_embeddings = np.concatenate(tts_embeddings, axis=0)

            file.write('%s\t%s\t%s\t%s\t%s\t%s\n'%(current_gene, current_family, current_tpm, json.dumps(tss_embeddings.shape),json.dumps(tss_embeddings.reshape(-1).tolist()), json.dumps(tts_embeddings.reshape(-1).tolist())))
            counter += 1
            
            current_gene = gene
            current_family = family
            current_tpm = tpm
            tss_embeddings = []
            tts_embeddings = []

        if feature == 'tss':
            tss_embeddings.append(embeddings[index])
        if feature == 'tts':
            tts_embeddings.append(embeddings[index])

    return counter

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


        print('gene: %s'%gene)
        print('family: %s'%family)
        print('TPM: %f'%TPM)
        print(tss_embeddings.shape)
        print(tts_embeddings.shape)

        #print(tss_embeddings)

    file.close()
    print('Imported %i embeddings.'%counter)

def main():
    workdir = '/home/shd-sieve/camous/PlantCaduceus/'
    
    #Set to None to load all sequences
    max_sequences_loaded = None

    batch_size = 100
    core_sequence_size = 250

    species_set = ['Csa', 'Sly', 'Sbi', 'Ath', 'Vvi', 'Osa', 'Mtr', 'Ppa', 'Gma', 'Bdi', 'Sit', 'Ptr', 'Svi', 'Bvu', 'Cre', 'Zma', 'Stu']
    device = 'cuda'
    model, tokenizer = load_model_and_tokenizer(workdir+'PlantCaduceus_l20', device)

    for species in species_set:
        sequences, genes, families, tpms, features, chunks, chunk_size = LoadSequences(workdir+'caduceus.sequences.%s.tsv'%species, max_sequences_loaded)

        print('Chunk size: %i'%chunk_size)

        #Batch size needs to be divisible with chunk size, or else...
        adjusted_batch_size = (batch_size//(chunk_size*2))*(chunk_size*2)
        if adjusted_batch_size < chunk_size*2:
            adjusted_batch_size = chunk_size*2

        print('batch size: %i'%adjusted_batch_size)

        data_loader = create_dataloader(sequences, genes, families, tpms, features, chunks, tokenizer, adjusted_batch_size)
        extract_caduceus_embeddings(model, data_loader, device, core_sequence_size, workdir+'caduceus.training.data.%s'%species, species, True, True)
    
    #Import(workdir+'training.data.Bdi.bin')

if __name__ == "__main__":
    main()
