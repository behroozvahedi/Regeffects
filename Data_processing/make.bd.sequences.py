def SequenceToChunk(sequence, subsequence_start, subsequence_length, core_sequence_length, model_input_size):
    seq_start = None
    seq_end = None
    chunks = []
    for chunk in range(int(subsequence_length / core_sequence_length)):
        start = subsequence_start + (chunk * core_sequence_length) - ((model_input_size - core_sequence_length) / 2)
        end = start + model_input_size

        if seq_start is None or start < seq_start:
            seq_start = int(start)
        if seq_end is None or end > seq_end:
            seq_end = int(end)
        input_sequence = sequence[int(start):int(end)]
        chunks.append(input_sequence)

    return(chunks, sequence[seq_start:seq_end])

def IterateSequence(seq, downstream_len, upstream_len, core_len, input_size):
    return SequenceToChunk(seq, (len(seq)/2)-downstream_len, upstream_len + downstream_len, core_len, input_size)

def ExportSequences(filein, fileout, core_sequence_length = 250, model_input_size = 512):
    tss_upstream = 4000
    tss_downstream = 1000

    tts_upstream = 1000
    tts_downstream = 4000

    filein = open(filein,'r')
    fileout = open(fileout,'w')
    fileout.write('gene\ttranscript\tgroup\ttss\ttts\tchunk\thash\n')

    counter = 0
    skipped = 0
    header = True
    for line in filein:
        if header:
            header = False
            continue

        items = line.strip('\n').split(',')
        gene = items[1].strip('"')
        transcript = items[3].strip('"')
        promoter =  items[4].strip('"')
        terminator =  items[5].strip('"')
        #promoter = ('N'*500)+promoter[2000:len(promoter)-2000]+('N'*3500)
        #terminator = ('N'*3500) + terminator[2000:len(terminator)-2000] + ('N'*500)
        group =  items[8].strip('"')

        if promoter == 'NA' or terminator == 'NA':
            continue

        if len(promoter) == 10000 and len(terminator) == 10000:
            counter += 1
            if not counter % 10000:
                print(counter) 
            tss_chunks, tss_seq = IterateSequence(promoter, tss_downstream, tss_upstream, core_sequence_length, model_input_size)
            tts_chunks, tts_seq = IterateSequence(terminator, tts_downstream, tts_upstream, core_sequence_length, model_input_size)
            for chunk in range(len(tss_chunks)):
                fileout.write('%s\t%s\t%s\t%s\t%s\t%i\t%i\n'%(gene, transcript, group, tss_chunks[chunk], tts_chunks[chunk], chunk, hash(tss_seq+tts_seq)))
        else:
            skipped += 1

    filein.close()
    fileout.close()
    print('Exported %i sequences, skipped %i'%(counter, skipped))

def main():
    workdir = '/Volumes/N1/Embeddings/DATA/'

    print('Exporting Caduceus sequences.')
    ExportSequences(workdir+'data.bd.csv', workdir+'bd.sequences.caduceus.tsv', 250, 512)
    print('Exporting a2z sequences.')
    ExportSequences(workdir+'data.bd.csv', workdir+'bd.sequences.a2z.tsv', 250, 600)

if __name__ == "__main__":
    main()
