#!/bin/bash

#List of lines to process
LINES='100 130 156 181 26 53 802 827 852 102 131 157 182 27 54 803 828 853 103 132 158 183 28 55 804 829 854 104 133 159 185 2 56 805 82 855 105 134 15 186 30 57 806 830 856 106 135 160 187 31 58 807 831 857 107 136 162 188 32 59 808 832 85 108 137 163 189 33 5 809 833 86 109 138 164 18 34 60 80 834 87 10 139 165 190 35 61 810 835 88 110 13 166 191 37 62 811 836 89 111 140 167 192 38 63 812 837 8 112 141 168 193 39 64 813 839 90 113 142 169 194 3 65 814 83 91 114 143 16 196 40 67 815 840 92 117 144 170 197 41 69'
#The folder containing the tools binaries used for processing
TFOLDER=/home/camous/sieve/Camous/TOOLS
#The folders that contain 00_fastq and 01_output where where RNAseqs are read and written respectively
IFOLDER=/home/camous/sieve/90-1080311147/00_fastq
OFOLDER=/home/camous/sieve/Camous/90-1080311147/01_output

$TFOLDER/cufflinks/gffread /Volumes/N1/INPUT/GENOME/BD/REF/annotation/BdistachyonBd21_3_537_v1.2.gene.gff3 -T -o $OFOLDER/BD.gtf
python3 $TFOLDER/hisat2/hisat2_extract_splice_sites.py $OFOLDER/BD.gtf > $OFOLDER/BD.ss
python3 $TFOLDER/hisat2/hisat2_extract_exons.py $OFOLDER/BD.gtf > $OFOLDER/BD.exon
$TFOLDER/hisat2/hisat2-build -p 8 --exon $OFOLDER/BD.exon --ss $OFOLDER/BD.ss /Volumes/N1/INPUT/GENOME/BD/REF/assembly/BdistachyonBd21_3_537_v1.0.fa $OFOLDER/BD
for i in $LINES
do
    $TFOLDER/fastp -w 8 -i $IFOLDER'/'$i'A1_R1_001.fastq.gz' -I $IFOLDER'/'$i'A1_R2_001.fastq.gz' -o $OFOLDER'/'$i'_R1.fq.gz' -O $OFOLDER'/'$i'_R2.fq.gz'  -j $OFOLDER'/'$i'_fastp.json' -h $OFOLDER'/'$i'_fastp.html'
    $TFOLDER/hisat2/hisat2 -p 8 -x $OFOLDER/BD -1 $OFOLDER'/'$i'_R1.fq.gz' -2 $OFOLDER'/'$i'_R2.fq.gz' --new-summary --summary-file $OFOLDER'/'$i'_summary.txt' | $TFOLDER/samtools/samtools view -bSh > $OFOLDER'/'$i'_aligned.bam'
    $TFOLDER/subread/featureCounts -T 8 -p -a $OFOLDER'/BD.gtf' -o $OFOLDER'/'$i'_aligned.FC' $OFOLDER'/'$i'_aligned.bam'
done
