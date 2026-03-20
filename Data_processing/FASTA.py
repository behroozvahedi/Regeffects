def FASTA(file):
    sequences = dict()
    file = open(file,'r')
    counter = 0
    transcript = None
    sequence = ''
    for line in file:
        if transcript != None:
            if line[0] != '>':
                sequence += line.strip('\n')
            else:
                sequences[transcript] = sequence
                transcript = None
                sequence = ''
        if transcript == None and line[0] == '>':
            transcript = line[1:].strip('\n')
    file.close()
    return sequences
