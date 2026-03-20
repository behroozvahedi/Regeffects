from bedbug import BEDBUG
from FASTA import FASTA
from mRNA import mRNA
import numpy as np

def TranslateID(file):
    translation = dict()
    file = open(file,'r')

    counter = 0
    for line in file:
        items = line.strip('\n').split('\t')
        translation[items[0]] = items[1]
    return(translation)

def Invert(sequence):
    sequence = sequence.upper()
    inverted = ''
    for index in range(len(sequence)):
        if sequence[index] == 'A':
            nucleotide = 'T'
        elif sequence[index] == 'T':
            nucleotide = 'A'
        elif sequence[index] == 'C':
            nucleotide = 'G'
        elif sequence[index] == 'G':
            nucleotide = 'C'
        #Adenine: A – – – V
        elif sequence[index] == 'A':
            nucleotide = 'V'
        #Cytosine: – C – – H
        elif sequence[index] == 'C':
            nucleotide = 'H'
        #Guanine: – – G – D
        elif sequence[index] == 'G':
            nucleotide = 'D'
        #Thymine: – – – T B
        elif sequence[index] == 'T':
            nucleotide = 'B'
        #Weak: A – – T S
        elif sequence[index] == 'W':
            nucleotide = 'S'
        #Strong: – C G – W
        elif sequence[index] == 'S':
            nucleotide = 'W'
        #aMino: A C – – K
        elif sequence[index] == 'M':
            nucleotide = 'K'
        #Keto: – – G T M
        elif sequence[index] == 'K':
            nucleotide = 'M'
        #puRine: A – G – Y
        elif sequence[index] == 'R':
            nucleotide = 'Y'
        #pYrimidine: – C – T R
        elif sequence[index] == 'Y':
            nucleotide = 'R'
        #not A: – C G T A
        elif sequence[index] == 'B':
            nucleotide = 'A'
        #not C: A – G T C
        elif sequence[index] == 'D':
            nucleotide = 'C'
        #not G: A C – T G
        elif sequence[index] == 'H':
            nucleotide = 'G'
        #not T: A C G – T
        elif sequence[index] == 'V':
            nucleotide = 'T'
        #any Nucleotide: A C G T N
        #elif sequence[index] == 'Z'
            #nucleotide = 'N'
        #Zero: – – – – Z
        #elif sequence[index] == 'N'
            #nucleotide = 'Z'
        elif sequence[index] == 'N':
            nucleotide = 'N'
        else:
            print('Error: Unrecognized nucleotide %s at %i in %s'%(sequence[index], index, sequence))
            exit(0)

        inverted = inverted+nucleotide

    return(inverted[::-1])

class Sequence:
    def __init__(self, id, tss, tts, strand):
        self.cohort = [id]

        if strand == '+':
            self.tss = tss
            self.tts = tts
        elif strand == '-':
            self.tss = Invert(tts)
            self.tts = Invert(tss)
        else:
            print('Error: Unrecognized strand %s'%strand)
            exit(0)

    def add(self, id):
        self.cohort.append(id)

def SNP(input):
    A = False
    T = False
    C = False
    G = False

    if '*' in input:
        print('- * -')
        exit(0)
        #return 'N'

    if 'A' in input:
        A = True
    if 'T' in input:
        T = True
    if 'C' in input:
        C = True
    if 'G' in input:
        G = True

    #Adenine: A – – – V
    if A and not C and not G and not T:
        return 'A'

    #Cytosine: – C – – H
    if not A and C and not G and not T:
        return 'C'

    #Guanine: – – G – D
    if not A and not C and G and not T:
        return 'G'

    #Thymine: – – – T B
    if not A and not C and not G and T:
        return 'T'

    #Weak: A – – T S
    if A and not C and not G and T:
        return 'W'

    #Strong: – C G – W
    if not A and C and G and not T:
        return 'S'

    #aMino: A C – – K
    if A and C and not G and not T:
        return 'M'
    
    #Keto: – – G T M
    if not A and not C and G and T:
        return 'K'

    #puRine: A – G – Y
    if A and not C and G and not T:
        return 'R'

    #pYrimidine: – C – T R
    if not A and C and not G and T:
        return 'Y'
    
    #not A: – C G T A
    if not A and C and G and T:
        return 'B'
    
    #not C: A – G T C
    if A and not C and G and T:
        return 'D'

    #not G: A C – T G
    if A and C and not G and T:
        return 'H'

    #not T: A C G – T
    if A and C and G and not T:
        return 'V'

    #any Nucleotide: A C G T N
    #if A and C and G and T:
    #    return 'Z'

    #Zero: – – – – Z
    if not A and not C and not G and not T:
        return 'N'

    print('Error: Unrecognized SNP input %s'%input)
    exit(0)

def Mutate(bed, chromosomes, chromosome, cohort, seq_start, seq_end):
    sequence = chromosomes[chromosome][seq_start:seq_end]
    variants, cases, _, genotypes = bed.regionGenotypes(chromosome, seq_start+1, seq_end, cohort)
    genotypes = np.array(genotypes,dtype='int')
    genotypes = genotypes.reshape(len(variants), len(cases)).transpose()

    for variant in variants:
        if sequence[(variant.pos-1)-seq_start] != variant.allele2:
            print('@sequence: %i-%i'%(seq_start,seq_end))
            print('@variant: %s:%i:%s/%s'%(variant.chr, variant.pos, variant.allele2, variant.allele1))
            print('mismatch: %s<->%s'%(sequence[(variant.pos-1)-seq_start], variant.allele2))
            print('%i]-[%i'%((variant.pos-1)-seq_start,(variant.pos-1)-seq_start+1))
            print('seq: %s'%sequence)
            print('seq.head: %s'%sequence[:(variant.pos-1)-seq_start])
            print('seq.tail: %s'%sequence[(variant.pos-1)-seq_start+1:])
            sequence = sequence[:(variant.pos-1)-seq_start] + variant.allele2 + sequence[(variant.pos-1)-seq_start+1:]

    case_sequences = dict()
    for i_case in range(len(cases)):
        case = cases[i_case]
        case_sequences[case] = sequence
        i_hets = np.where(genotypes[i_case] == 1)[0]
        i_homs = np.where(genotypes[i_case] == 0)[0]

        for index in i_hets:
            variant = variants[index]
            if variant.allele1 == '*':
                continue

            snp = SNP(variant.allele2+variant.allele1)
            case_sequences[case] = case_sequences[case][:(variant.pos-1)-seq_start] + snp + case_sequences[case][(variant.pos-1)-seq_start+1:]

        for index in i_homs:
            variant = variants[index]
            if variant.allele1 == '*':
                continue
            case_sequences[case] = case_sequences[case][:(variant.pos-1)-seq_start] + variant.allele1 + case_sequences[case][(variant.pos-1)-seq_start+1:]

    return case_sequences

def Cohort(fam_file):
    fam_file = open(fam_file,'r')
    cohort = []
    for line in fam_file:
        items = line.strip('\n').split(' ')

        cohort.append(items[0])
    fam_file.close()
    return cohort

def main():
    workdir = '/Volumes/N1/Embeddings/DATA/'

    fasta_file = workdir+'BdistachyonBd21_3_537_v1.0.fa'
    gff_file = workdir+'BdistachyonBd21_3_537_v1.2.gene.gff3'
    bed_file = workdir+'snps.combined.M5.filtered.renamed'
    sequences_file = workdir+'data.bd.csv'
    stats_file = workdir+'stats.tsv'
    translation_file = workdir+'gene.id.translation.tsv'

    stats_file = open(stats_file,'w')
    stats_file.write('transcript\tsequences\n')
    
    cohort = Cohort(bed_file+'.fam')
    chromosomes = FASTA(fasta_file)
    bed = BEDBUG(bed_file)
    translation = TranslateID(translation_file)

    sequence_size = 10000

    counter = 0
    done = False
    skipped = 0
    gff_file = open(gff_file,'r')
    sequences_file = open(sequences_file, 'w')
    sequences_file.write('"","gene","species","transcript","promoter","terminator","median_TPM","gene_family","group_for_cross_validation"\n')

    total_sequences = 0
    total_transcripts = 0

    for line in gff_file:
        if line[:2] == '##':
            continue
        items = line.strip('\n').split('\t')
        if items[2] == 'mRNA':
            mrna = mRNA(items[0], int(items[3])-1,int(items[4])-1,items[6], items[8])
            bd21 = ''
            if mrna.id in translation:
                bd21 = translation[mrna.id]
            else:
                skipped += 1
                continue

            tss_start = mrna.start - int(sequence_size/2)
            tss_end = mrna.start + int(sequence_size/2)
            tts_start = mrna.end - int(sequence_size/2)
            tts_end = mrna.end + int(sequence_size/2)

            if tss_start < 0 or tts_end >= len(chromosomes[mrna.chr]):
                skipped += 1
                continue

            tss_seqs = Mutate(bed, chromosomes, mrna.chr, cohort, tss_start, tss_end)
            tts_seqs = Mutate(bed, chromosomes, mrna.chr, cohort, tts_start, tts_end)

            sequences = dict()
            for case in tss_seqs:
                h = hash(tss_seqs[case]+tts_seqs[case])
                if h not in sequences:
                    sequences[h] = Sequence(case,tss_seqs[case],tts_seqs[case], mrna.strand)
                else:
                    sequences[h].add(case)
            total_sequences += len(sequences)
            total_transcripts += 1
            stats_file.write('%s\t%i\n'%(mrna.id,len(sequences)))

            for h in sequences:
                counter+=1
                sequences_file.write('"%s","%s","%s","%s","%s","%s",%s,%s,%s\n'%(counter,mrna.gene,bd21,mrna.id,sequences[h].tss,sequences[h].tts,'NA','NA',' '.join(sequences[h].cohort)))
                if not counter % 5000:
                    print('seqs: %i'%counter)
        if done:
            break

    print('processed %i sequences (skipped %i), average sequence per transcript %f'%(total_sequences, skipped, float(total_sequences)/float(total_transcripts)))
    sequences_file.close()
    stats_file.close()

if __name__ == "__main__":
    main()
