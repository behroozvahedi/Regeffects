workdir = '/Volumes/N1/Embeddings/DATA/'
filein = open(workdir+'data.bd.csv','r')
fileout = open(workdir+'data.bdi.csv','w')

counter = 0
header = None
for line in filein:
	if header == None:
		header = line
		fileout.write(line)
		continue

	if not counter % 5000:
		print(counter)
	items = line.strip('\n').split(',')
	fileout.write('%s,%s,%s,%s,%s,%s,%s,%s,%s\n'%(items[0],items[2],'Bdi',items[3],items[4],items[5],items[6],items[7],items[8]))
	counter += 1

filein.close()
fileout.close()