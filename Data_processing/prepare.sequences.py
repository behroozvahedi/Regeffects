
#Partition DNA sequence into several chunks, each of which will fit the input sequence length of the chosen model
def SequenceToChunk(sequence, subsequence_start, subsequence_length, core_sequence_length, model_input_size):
    chunks = []
    for chunk in range(int(subsequence_length / core_sequence_length)):
        start = subsequence_start + (chunk * core_sequence_length) - ((model_input_size - core_sequence_length) / 2)
        end = start + model_input_size
        input_sequence = sequence[int(start):int(end)]
        chunks.append(input_sequence)

    return(chunks)

def IterateSequence(seq, downstream_len, upstream_len, core_len, input_size):
    return SequenceToChunk(seq, (len(seq)/2)-downstream_len, upstream_len + downstream_len, core_len, input_size)

#Loads in the TSS and TTS sequences used to train PlantCaduceus, and splits them up into a number of chunks
#Each chunk is be the size of the sequence input length of the given species.
#Some additional data such as gene, gene family and TPM is also read and recorded.
def ProcessSpecies(speciesin, filein, fileout, core_sequence_length = 250, model_input_size = 512):
    #We extract 5000bp: 4000bp up/downstream, plus 1000bp inside of TSS and TTS as defined by mRNA locus
    tss_upstream = 4000
    tss_downstream = 1000

    tts_upstream = 1000
    tts_downstream = 4000

    filein = open(filein,'r')
    fileout = open(fileout,'w')
    fileout.write('gene\tfamily\tTPM\ttss\ttts\tchunk\n')

    counter = 0

    header = True
    for line in filein:
        if header:
            header = False
            continue

        items = line.strip('\n').split(',')
        gene = items[1].strip('"')
        species = items[2].strip('"')
        transcript = items[3].strip('"')
        promoter =  items[4].strip('"')
        terminator =  items[5].strip('"')
        TPM =  items[6].strip('"')
        family =  items[7].strip('"')

        if promoter == 'NA' or terminator == 'NA':
            continue


        if species == speciesin and len(promoter) == 10000:
            counter += 1
            #Make the TSS chunks
            tss_chunks = IterateSequence(promoter, tss_downstream, tss_upstream, core_sequence_length, model_input_size)
            #Make the TTS chunks
            tts_chunks = IterateSequence(terminator, tts_downstream, tts_upstream, core_sequence_length, model_input_size)
            #Save the TSS and TTS chunks to file
            for chunk in range(len(tss_chunks)):
                fileout.write('%s\t%s\t%s\t%s\t%s\t%i\n'%(gene, family, TPM, tss_chunks[chunk], tts_chunks[chunk], chunk))

    filein.close()
    fileout.close()
    print('Processed %i sequences for species %s'%(counter, speciesin))

def main():
    workdir = '/Users/au468646/Downloads/Embeddings/'

    species_set = ['Csa', 'Sly', 'Sbi', 'Ath', 'Vvi', 'Osa', 'Mtr', 'Ppa', 'Gma', 'Bdi', 'Sit', 'Ptr', 'Svi', 'Bvu', 'Cre', 'Zma', 'Stu']
    for species in species_set:
        #Process sequences for plant caduceus feature extraction
        ProcessSpecies(species, workdir+'data.csv', workdir+'caduceus.sequences.%s.tsv'%species, 250, 512)
        #Process sequences for a2z feature extraction
        ProcessSpecies(species, workdir+'data.csv', workdir+'a2z.sequences.%s.tsv'%species, 250, 600)

if __name__ == "__main__":
    main()
