class mRNA:
    id = None
    gene = None
    chr = None
    start = None
    end = None
    strand = None

    def __init__(self, chr, start, end, strand, id = None):
        if id is not None:
            items = id.strip('\n').split(';')
            for item in items:
                if item[:5] == 'Name=':
                    id = item[5:].split('.v')[0]
                if item[:7] == 'Parent=':
                    gene = item[7:].split('.')[1]

        self.id = id
        self.gene = gene
        self.chr = chr
        self.start = int(start)
        self.end = int(end)
        self.strand = strand
