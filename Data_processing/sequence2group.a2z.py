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
    fileout.write('gene\tfamily\tTPM\tgroup\n')

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
        group =  items[8].strip('"')

        if promoter == 'NA' or terminator == 'NA':
            continue

        if species == speciesin and len(promoter) == 10000:
            counter += 1
            fileout.write('%s\t%s\t%s\t%s\n'%(gene, family, TPM, group))

    filein.close()
    fileout.close()
    print('Processed %i sequences for species %s'%(counter, speciesin))

def main():
    workdir = '/Users/au468646/Downloads/Embeddings/'

    species_set = ['Csa', 'Sly', 'Sbi', 'Ath', 'Vvi', 'Osa', 'Mtr', 'Ppa', 'Gma', 'Bdi', 'Sit', 'Ptr', 'Svi', 'Bvu', 'Cre', 'Zma', 'Stu']
    for species in species_set:
        ProcessSpecies(species, workdir+'data.csv', workdir+'caduceus.training.groups.%s.tsv'%species, 250, 512)
        ProcessSpecies(species, workdir+'data.csv', workdir+'a2z.training.groups.%s.tsv'%species, 250, 600)

if __name__ == "__main__":
    main()
