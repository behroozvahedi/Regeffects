#!/usr/bin/env python
import pandas as pd

def GetTranslation(file):
    translation = dict()
    file = open(file,'r')

    counter = 0
    header = True
    for line in file:
        if header:
            header = False
            continue
        items = line.strip('\n').split('\t')
        translation['%i'%int(items[0])] = items[1]
    return(translation)

def MakeObservations(file_in, file_out, translation):
    file_in = open(file_in,'r')
    file_out = open(file_out,'w')
    file_out.write('id\tgene\ttpm\n')

    counter = 0
    header = True
    ids = []
    for line in file_in:
        if header:
            header = False
            ids = line.strip('\n').split('\t')[1:]
            #print(ids)
            continue
        items = line.strip('\n').split('\t')
        gene = items[0].split('.')[1]
        tmps = items[1:]

        for index in range(len(ids)):
            id = translation[ids[index].split('_')[0]]
            file_out.write('%s\t%s\t%s\n'%(id, gene, tmps[index]))
    file_in.close()
    file_out.close()

def GetObservations(file_in, translation):
    file_in = open(file_in,'r')
    observations = dict()

    header = True
    ids = []
    for line in file_in:
        if header:
            header = False
            ids = line.strip('\n').split('\t')[1:]
            continue
        items = line.strip('\n').split('\t')
        gene = items[0].split('.')[1]
        tmps = items[1:]

        for index in range(len(ids)):
            id = translation[ids[index].split('_')[0]]
            observations['%s-%s'%(id,gene)] = tmps[index]
    file_in.close()
    return(observations)

def MakePEERDataset(file_in, dir_out, translation):
    file_in = open(file_in,'r')
    file_expression = open(dir_out+'peer.expression.csv','w')
    file_genes = open(dir_out+'peer.genes.csv','w')
    file_samples = open(dir_out+'peer.samples.csv','w')

    counter = 0
    header = True
    ids = []
    for line in file_in:
        if header:
            header = False
            samples = line.strip('\n').split('\t')[1:]
            for sample in samples:
                file_samples.write('%s\n'%translation[sample.split('_')[0]])
            continue
        items = line.strip('\n').split('\t')
        gene = items[0].split('.')[1]
        tmps = items[1:]

        file_genes.write('%s\n'%gene)

        for index in range(len(tmps)):
            if index > 0:
                file_expression.write(',')
            file_expression.write('%s'%(tmps[index]))
        file_expression.write('\n')
    file_in.close()
    file_genes.close()
    file_samples.close()
    file_expression.close()

    #Transpose matrix
    pd.read_csv(dir_out+'peer.expression.csv').transpose().to_csv(dir_out+'peer.expression.csv', header=False)

def MakeRDataset(file_in, file_out, observations):
    file_in = open(file_in,'r')
    file_out = open(file_out,'w')

    counter = 0
    header = True
    for line in file_in:
        if header:
            header = False
            file_out.write('%s\ttpm\n'%line.strip('\n') )
            continue

        items = line.strip('\n').split('\t')
        id = items[0]
        gene = items[1]

        key = '%s-%s'%(id,gene)
        if key in observations:
            file_out.write('%s\t%s\n'%(line.strip('\n'), observations[key]))

    file_in.close()
    file_out.close()

def main():
    workdir = '/Volumes/N1/Embeddings/DATA/'
    translation = GetTranslation(workdir + 'RNAseq/RNAseq.ids.tsv')
    
    print('Making PEER dataset')
    MakePEERDataset(workdir + 'RNAseq/RNAseq.reads.tsv', workdir, translation)
    observations = GetObservations(workdir + 'RNAseq/RNAseq.reads.tsv', translation)

    for type in ['none', 'pred', 'emb', 'a2z']:
        print('Making WP2 dataset for %s'%type)
        MakeRDataset(workdir+'predictions_%s.tsv'%type, workdir+'wp2.dataset_%s.tsv'%type, observations)

if __name__ == "__main__":
    main()
