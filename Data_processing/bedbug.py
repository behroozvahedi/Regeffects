from time import time
import math
import numpy as np

quiet = True
verbose = False

class Variant:
	chr = ""
	pos = 0
	allele1 = ""
	allele2 = ""
	maf = float('nan')
	na = float('nan')
	homA1 = 0
	het = 0
	homA2 = 0
	missing = 0

	def __init__(self, variant,chr=None,pos=None,allele1=None,allele2=None):
		if chr == None:
			items = variant.strip('\n').split('\t')
			self.chr = items[0]
			self.pos = int(items[3])
			self.allele1 = items[4]
			self.allele2 = items[5]
		else:
			self.chr = chr
			self.pos = pos
			self.allele1 = allele1
			self.allele2 = allele2

	def id(self):
		return('%s:%i:%s:%s'%(self.chr,self.pos,self.allele2,self.allele1))

class BEDBUG:
	def __init__(self, filename):
		if verbose:
			print('call: init')
		timestamp = time()
		self.indices_first = {}
		self.indices_last = {}
		self.filename = filename
		self.cases = []
		#self.firstVariant = {}
		#self.lastVariant = {}
		self.chromosomes = {}

		bim = open(filename + '.bim', 'r')
		index = 0
		for line in bim:
			if line.strip('\n') == '':
				continue
			variant = Variant(line)

			key = variant.chr + ':' + str(variant.pos)

			if key not in self.indices_first:
				self.indices_first[key] = index
			elif self.indices_first[key] > index:
				self.indices_first[key] = index
			if key not in self.indices_last:
				self.indices_last[key] = index
			elif self.indices_last[key] < index:
				self.indices_last[key] = index

			#if chr not in self.firstVariant or self.firstVariant[chr].pos > variant.pos:
			#	self.firstVariant[chr] = variant

			#if chr not in self.lastVariant or self.lastVariant[chr].pos < variant.pos:
			#	self.lastVariant[chr] = variant

			if variant.chr not in self.chromosomes:
				self.chromosomes[variant.chr] = []
			self.chromosomes[variant.chr].append(variant)
		bim.close()
		fam = open(filename + '.fam', 'r')
		for line in fam:
			if line.strip('\n') == '':
				continue

			items = line.strip('\n').split('\t')
			if len(items) == 1:
				items = line.strip('\n').split(' ')
			self.cases.append(items[0])
		fam.close()
		if verbose:
			print('call took: %.1fs'%(time()-timestamp))

	def findIndex(self, chr, start, end):
		if chr not in self.chromosomes:
			return None, None

		key_start = chr + ':' + start
		key_end = chr + ':' + end

		if key_start in self.indices_first and key_end in self.indices_last:
			return self.indices_first[key_start], self.indices_last[key_end]

		variants = self.chromosomes[chr]
		first = variants[0].pos
		last = variants[-1].pos

		avg = (last - first) / len(variants)

		indexStart = int(start / avg)
		indexEnd = int(end / avg)

		if indexStart >= len(variants):
			indexStart = len(variants) - 1

		if indexEnd >= len(variants):
			indexEnd = len(variants) - 1

		# Find the variant at or closest above the desired start point
		counter = 0
		while True:
			counter += 1
			if variants[indexStart].pos == start:
				break

			if variants[indexStart].pos < start:
				if indexStart < len(variants) - 1:
					indexStart += 1
					if variants[indexStart].pos >= start:
						break
					else:
						continue
				else:
					break

			if variants[indexStart].pos > start:
				if indexStart > 0:
					if variants[indexStart - 1].pos <= start:
						break
					else:
						indexStart -= 1
						continue
				else:
					break

		# Find the variant at or closest below the desired end point
		while True:
			counter += 1
			if variants[indexEnd].pos == end:
				break

			if variants[indexEnd].pos < end:
				if indexEnd < len(variants) - 1:
					if variants[indexEnd + 1].pos >= end:
						break
					else:
						indexEnd += 1
						continue
				else:
					break

			if variants[indexEnd].pos > end:
				if indexEnd > 0:
					indexEnd -= 1
					if variants[indexEnd].pos <= end:
						break
					else:
						continue
				else:
					break
		if indexStart > indexEnd:
			return None, None
		else:
			return indexStart, indexEnd

	def regionCount(self, chr, start, end):
		iStart, iEnd = self.findIndex(chr, start, end)

		if iStart is None or iEnd is None:
			return 0
		else:
			return iEnd-iStart

	def regionVariants(self, chr, start, end):
		iStart, iEnd = self.findIndex(chr, start, end)

		if iStart is None or iEnd is None:
			return []
		else:
			return self.chromosomes[chr][iStart:iEnd]

	def chromosomeGenotypes(self, chr, csubset):
		if verbose:
			print('call: chromosome variants')
		timestamp = time()
		vindices = []
		cindices = []
		cases = []
		variants = []

		if type(chr) != str:
			chr = str(chr)
		vindex = 0
		vseekindex = 0
		for c in self.chromosomes:
			if c == chr:
				for variant in self.chromosomes[chr]:
					vindices.append(vindex)
					vseekindices.append(vseekindex)
					vindex +=  1
					vseekindex += 1
				break
			else:
				vseekindex += len(self.chromosomes[c])

		cindex = 0
		for case in self.cases:
			if case in csubset or len(csubset) == 0:
				cindices.append(cindex)
				cases.append(case)
			cindex = cindex + 1
		if not quiet:
			print('Found ' + str(len(vindices)) + ' variants for ' + str(len(cindices)) + ' cases in region ' + str(chr) + ':' + str(start) + '-' + str(end) )

		if verbose:
			print('call took: %.1fs'%(time()-timestamp))

		if len(vindices) > 0:
			(perfect, genotypes) = self.extract(chr,vindices,vseekindices,cindices)
		else:
			perfect = []
			genotypes = []

		return(variants, cases, perfect, genotypes)

	def regionGenotypes(self, chr, start, end, csubset):
		if verbose:
			print('call: region genotypes')
		timestamp = time()
		vindices = []
		vseekindices = []
		cindices = []
		cases = []
		variants = []

		if type(chr) != str:
			chr = str(chr)
		vindex = 0
		vseekindex = 0
		for c in self.chromosomes:
			if c == chr:
				for variant in self.chromosomes[chr]:
					if variant.pos >= start and variant.pos <= end:
						vindices.append(vindex)
						vseekindices.append(vseekindex)
						variants.append(variant)
					vindex +=  1
					vseekindex +=  1 
				break
			else:
				vseekindex += len(self.chromosomes[c])

		cindex = 0
		for case in self.cases:
			if case in csubset or len(csubset) == 0:
				cindices.append(cindex)
				cases.append(case)
			cindex = cindex + 1
		if not quiet:
			print('Found ' + str(len(vindices)) + ' variants for ' + str(len(cindices)) + ' cases in region ' + str(chr) + ':' + str(start) + '-' + str(end) )

		if verbose:
			print('call took: %.1fs'%(time()-timestamp))

		if len(vindices) > 0:
			(perfect, genotypes) = self.extract(chr,vindices,vseekindices,cindices)
		else:
			perfect = []
			genotypes = []

		return(variants, cases, perfect, genotypes)

	def variant(self, chr, pos, a2, a1, csubset):
		if verbose:
			print('call: variant')
		timestamp = time()
		cindices = []
		cases = []

		if type(chr) != str:
			chr = str(chr)

		cindex = 0
		for case in self.cases:
			if case in csubset or len(csubset) == 0:
				cindices.append(cindex)
				cases.append(case)
			cindex += 1

		vindex = 0
		vseekindex = 0
		for c in self.chromosomes:
			if c == chr:
				for variant in self.chromosomes[chr]:
					if variant.chr == chr and variant.pos == pos:
						if variant.allele1 == a1 and variant.allele2 == a2:
							if verbose:
								print('call took: %.1fs'%(time()-timestamp))
							return(variant, cases, self.extract(chr, [vindex], [vseekindex],cindices)[1])
						elif variant.allele1 == a2 and variant.allele2 == a1:
							print('Note: Opposite variant match ' + str(chr) + ':' + str(pos) + ':' + a2 + '/' + a1)
					vseekindex += 1
					vindex += 1 
				break
			else:
				vseekindex += len(self.chromosomes[c])

		if verbose:
			print('call took: %.1fs'%(time()-timestamp))
		return(None, None, None)

	def stats(self, genotypes, variants, cases):
		if verbose:
			print('call: stats')
		timestamp = time()

		data = np.array(genotypes,dtype='int')
		data = data.reshape(variants, cases)
		#data = np.transpose(data[:,complete])
		data = np.transpose(data)

		completeset = set()
		incompleteset = set()
		completes = 0
		incompletes = 0

		for casevars in data:
			h = hash(casevars.tostring())

			if casevars.min() < 0:
				incompletes = incompletes + 1
				incompleteset.add(h)
			else:
				completes = completes + 1
				completeset.add(h)
		if verbose:
			print('call took: %.1fs'%(time()-timestamp))
		return(completes, len(completeset), incompletes, len(incompleteset))

	def verify(self, genotypes, variants, cases, file):
		if verbose:
			print('call: verify')
		file = open(file, 'r')

		snps = 0
		donors = 0
		data = []
		for line in file:
			if line[0] == '#':
				continue

			snps = snps + 1
			items = line.rstrip('\n').split('\t')[9:]
			donors = len(items)
			for item in items:
				if item == '0/0':
					data.append(0)
				elif item == '0/1':
					data.append(1)
				elif item == '1/1':
					data.append(2)
				elif item == './.':
					data.append(-1)
				else:
					print('Error: Unknown genotype ' + item)
					exit(0)
		file.close()

		print(variants)
		print(snps)
		print(cases)
		print(donors)
		print(len(genotypes))
		print(len(data))

		for index in range(0,len(data)):
			if data[index] != genotypes[index]:
				print(index)

				file = open('/home/cammos/out.txt', 'w')
				file.write(str(genotypes[index:index+200])+'\n')
				file.write(str(data[index:index+200])+'\n')
				file.close()
				exit(1)

		if genotypes != data:
			print('Verification: Failed!')
		else:
			print('Verification: Passed!')
		exit(0)

	def setVariantStats(self):
		if verbose:
			print('call: set variant stats')
		timestamp = time()
		avgmiss = 0.0
		genotypes = []
		#perfect = [True] * len(cindices)
		bed = open(self.filename + '.bed', 'rb')

		magic_number=int.from_bytes(bed.read(2), byteorder='little')
		SNP_type=int.from_bytes(bed.read(1), byteorder='little')

		if magic_number != 7020:
			print("Error: File is not a BED file.")
		if SNP_type != 1:
			print("Error: Only SNP-major BED files supported.")

		residual = len(self.cases) % 4
		clen = math.floor(len(self.cases) / 4) + (residual != 0)

		vseekindex = 0
		for chromosome in self.chromosomes:
			if verbose:
				print('Processing chromosome: %s'%chromosome)
			variants = self.chromosomes[chromosome]
			for vindex in range(len(variants)):
				bed.seek((vseekindex*clen)+3, 0)
				vseekindex += 1
				line = bed.read(clen)

				homA1 = 0
				het = 0
				homA2 = 0
				missing = 0

				pindex = 0
				for cindex in range(len(self.cases)):
					pos = int(math.floor(cindex / 4))
					res = int((cindex % 4) * 2)
					genotype = (line[pos] >> res) & 0x03

					if genotype == 0:
						genotypes.append(0)
						homA1 = homA1 + 1
					elif genotype == 2:
						genotypes.append(1)
						het = het + 1
					elif genotype == 3:
						genotypes.append(2)
						homA2 = homA2 + 1
					elif genotype == 1:
						genotypes.append(-1)
						missing = missing + 1
					else:
						print("Error: Unexpected genotype.")
						exit(1)

					pindex = pindex + 1

				variants[vindex].homA1 = homA1
				variants[vindex].homA2 = homA2
				variants[vindex].het = het
				variants[vindex].missing = missing

				if homA1+homA2+het > 0:
					if homA1*2+het > homA2*2+het:
						variants[vindex].maf = float(homA2*2+het)/((homA1+homA2+het)*2)
					else:
						variants[vindex].maf = float(homA1*2+het)/((homA1+homA2+het)*2)
					variants[vindex].na = float(missing)/(homA1+homA2+het+missing)
					avgmiss = avgmiss + variants[vindex].na
			if verbose:
				print('Processing complete.')

		bed.close()
		if verbose:
			print('call took: %.1fs'%(time()-timestamp))

	def regionGenotypes(self, chr, start, end, csubset):
		if verbose:
			print('call: region genotypes')
		timestamp = time()
		vindices = []
		vseekindices = []
		cindices = []
		cases = []
		variants = []

		if type(chr) != str:
			chr = str(chr)
		vindex = 0
		vseekindex = 0
		for c in self.chromosomes:
			if c == chr:
				for variant in self.chromosomes[chr]:
					if variant.pos >= start and variant.pos <= end:
						vindices.append(vindex)
						vseekindices.append(vseekindex)
						variants.append(variant)
					vindex +=  1
					vseekindex +=  1 
				break
			else:
				vseekindex += len(self.chromosomes[c])

		cindex = 0
		for case in self.cases:
			if case in csubset or len(csubset) == 0:
				cindices.append(cindex)
				cases.append(case)
			cindex = cindex + 1
		if not quiet:
			print('Found ' + str(len(vindices)) + ' variants for ' + str(len(cindices)) + ' cases in region ' + str(chr) + ':' + str(start) + '-' + str(end) )

		if verbose:
			print('call took: %.1fs'%(time()-timestamp))

		if len(vindices) > 0:
			(perfect, genotypes) = self.extract(chr,vindices,vseekindices,cindices)
		else:
			perfect = []
			genotypes = []

		return(variants, cases, perfect, genotypes)

	def regionSingletons(self, chr, start, end, csubset):
		if verbose:
			print('call: chromosome singletons')
		timestamp = time()
		vindices = []
		vseekindices = []
		cindices = []
		cases = []
		variants = []

		if type(chr) != str:
			chr = str(chr)

		vindex = 0
		vseekindex = 0
		for c in self.chromosomes:
			if c == chr:
				for variant in self.chromosomes[chr]:
					if variant.pos >= start and variant.pos <= end:
						vindices.append(vindex)
						vseekindices.append(vseekindex)
						variants.append(variant)
					vindex += 1
					vseekindex += 1
				break
			else:
				vseekindex += len(self.chromosomes[c])

		cindex = 0
		for cindex in range(len(self.cases)):
			case = self.cases[cindex]
		#for case in self.cases:
			if case in csubset or len(csubset) == 0:
				cindices.append(cindex)
				cases.append(case)
			cindex = cindex + 1
		if not quiet:
			print('Found ' + str(len(vindices)) + ' variants for ' + str(len(cindices)) + ' cases in region ' + str(chr) + ':' + str(start) + '-' + str(end) )

		if verbose:
			print('call took: %.1fs'%(time()-timestamp))

		if len(vindices) > 0:
			(is_singleton, singleton_hom, singleton_het, is_multix, is_alt) = self.singletons(chr,vindices,vseekindices,cindices)
		else:
			is_singleton = []
			singleton_hom = []
			singleton_het = []
			is_multix = []
			is_alt = []

		return(variants, cases, is_singleton, singleton_hom, singleton_het, is_multix)

	def chromosomeSingletons(self, chr, csubset):
		if verbose:
			print('call: chromosome singletons')
		timestamp = time()
		vindices = []
		vseekindices = []
		cindices = []
		cases = []
		variants = []

		if type(chr) != str:
			chr = str(chr)

		vindex = 0
		vseekindex = 0
		for c in self.chromosomes:
			if c == chr:
				for variant in self.chromosomes[chr]:
					variants.append(variant)
					vindices.append(vindex)
					vseekindices.append(vseekindex)
					vindex += 1
					vseekindex += 1
				break
			else:
				vseekindex += len(self.chromosomes[c])

		cindex = 0
		for case in self.cases:
			if case in csubset or len(csubset) == 0:
				cindices.append(cindex)
				cases.append(case)
			cindex = cindex + 1
		if not quiet:
			print('Found ' + str(len(vindices)) + ' variants for ' + str(len(cindices)) + ' cases in region ' + str(chr) + ':' + str(start) + '-' + str(end) )

		if verbose:
			print('call took: %.1fs'%(time()-timestamp))

		if len(vindices) > 0:
			(is_singleton, singleton_hom, singleton_het, is_multix, is_alt) = self.singletons(chr,vindices,vseekindices,cindices)
		else:
			perfect = []
			genotypes = []

		return(variants, cases, is_singleton, singleton_hom, singleton_het, is_multix, is_alt)

	def singletons(self, chr, vindices, vseekindices, cindices):
		if verbose:
			print('call: singletons')

		timestamp = time()
		avgmiss = 0.0
		genotypes = []
		is_singleton = [False] * len(vindices)
		singleton_het = [-1] * len(vindices) 
		singleton_hom = [-1] * len(vindices)
		is_multix =  [False] * len(vindices)
		is_alt =  [False] * len(vindices)
		bed = open(self.filename + '.bed', 'rb')

		magic_number=int.from_bytes(bed.read(2), byteorder='little')
		SNP_type=int.from_bytes(bed.read(1), byteorder='little')

		if magic_number != 7020:
			print("Error: File is not a BED file.")
		if SNP_type != 1:
			print("Error: Only SNP-major BED files supported.")

		residual = len(self.cases) % 4
		clen = math.floor(len(self.cases) / 4) + (residual != 0)
		if chr not in self.chromosomes:
			print("Error: chromosome (%s) is not in loaded data"%chr)
			return(is_singleton, singleton_hom, singleton_het)

		variants = self.chromosomes[chr]
		#multi = 0
		for index in range(len(vindices)):
			#vindex = vindices[index]
			vseekindex = vseekindices[index]
			bed.seek((vseekindex*clen)+3, 0)
			line = bed.read(clen)

			homA1 = 0
			het = 0
			homA2 = 0
			missing = 0

			sindex_hom = -1
			sindex_het = -1
			sindex_a2 = -1
			multix = False

			for cindex in range(len(cindices)):
				pos = int(math.floor(cindices[cindex] / 4))
				res = int((cindices[cindex] % 4) * 2)
				genotype = (line[pos] >> res) & 0x03

				#print(genotype)

				if genotype == 0:
					homA1 += 1
					if het == 0:
						sindex_hom = cindex
				elif genotype == 2:
					het += 1
					if homA1 == 0 or homA2 == 0:
						sindex_het = cindex
				elif genotype == 3:
					homA2 = homA2 + 1
					if homA2 == 1:
						sindex_a2 = cindex
				elif genotype == 1:
					missing += 1
				else:
					print("Error: Unexpected genotype.")
					exit(1)

				if missing > 0 or (homA1 > 0 and het > 0 and homA2 > 0) or (homA1 > 1 and homA2 > 1) or (het > 1 and homA2 > 1) or (het > 1 and homA1 > 1):
					sindex_hom = -1
					sindex_het = -1
					sindex_a2 = -1
					#multi += 1
					multix = True
					break

			#homA1
			if missing == 0 and (homA2 == 1 and het == 0 and homA1 > 1):
				is_singleton[index] = True
				sindex_hom = sindex_a2
				sindex_het = -1
				multix = False
			elif missing == 0 and (homA2 == 0 and het == 1 and homA1 > 1):
				is_singleton[index] = True
				sindex_hom = -1
				multix = False
			elif (sindex_hom != -1 and sindex_het == -1 and homA1 == 1) or (sindex_het != -1 and sindex_hom == -1 and het == 1):
				is_singleton[index] = True
				is_alt[index] = True
				multix = False
			else:
				sindex_hom = -1
				sindex_het = -1

			#print('homA1: %i, het: %i, homA2: %i,  sindex_hom: %i, sindex_het: %i'%(homA1,het,homA2,sindex_hom, sindex_het))

			singleton_hom[index] = sindex_hom
			singleton_het[index] = sindex_het
			is_multix[index] = multix

		#print('Multix: %i'%multi)
		bed.close()
		if verbose:
			print('call took: %.1fs'%(time()-timestamp))
			print('completes: %i unique: %i (%.2f%%), incompletes: %i unique: %i (%.2f%%) '%(comp,compset,compercent,incomp,incompset,incompercent))

		return(is_singleton, singleton_hom, singleton_het, is_multix, is_alt)

	def extract(self, chr, vindices, vseekindices, cindices):
		if verbose:
			print('call: extract')
		timestamp = time()
		avgmiss = 0.0
		genotypes = []
		perfect = [True] * len(cindices) 
		bed = open(self.filename + '.bed', 'rb')

		magic_number=int.from_bytes(bed.read(2), byteorder='little')
		SNP_type=int.from_bytes(bed.read(1), byteorder='little')

		if magic_number != 7020:
			print("Error: File is not a BED file.")
		if SNP_type != 1:
			print("Error: Only SNP-major BED files supported.")

		residual = len(self.cases) % 4
		clen = math.floor(len(self.cases) / 4) + (residual != 0)
		if chr not in self.chromosomes:
			print("Error: chromosome (%s) is not in loaded data"%chr)

		variants = self.chromosomes[chr]
		for index in range(len(vindices)):
			vindex = vindices[index]
			vseekindex = vseekindices[index]
			bed.seek((vseekindex*clen)+3, 0)
			line = bed.read(clen)

			homA1 = 0
			het = 0
			homA2 = 0
			missing = 0

			pindex = 0
			for cindex in cindices:
				pos = int(math.floor(cindex / 4))
				res = int((cindex % 4) * 2)
				try:
					genotype = (line[pos] >> res) & 0x03
				except Exception as e:
					print('Exception: pos ' + str(pos) + ' cindex ' + str(cindex) + ' len(clen) ' + str(clen) + ' len(line) ' + str(len(line)) + ' len(cindices) ' + str(len(cindices)) + ' len(self.cases) ' + str(len(self.cases)) )
					raise e

				#00b = 0  Homozygote "1"/"1"
				if genotype == 0:
					genotypes.append(0)
					homA1 = homA1 + 1
				#01b = 1  Heterozygote
				elif genotype == 2:
					genotypes.append(1)
					het = het + 1
				#11b = 3  Homozygote "2"/"2"
				elif genotype == 3:
					genotypes.append(2)
					homA2 = homA2 + 1
				#10b = 2  Missing genotype
				elif genotype == 1:
					genotypes.append(-1)
					missing = missing + 1
					perfect[pindex] = False
				else:
					print("Error: Unexpected genotype.")
					exit(1)

				pindex = pindex + 1

			variants[vindex].homA1 = homA1
			variants[vindex].homA2 = homA2
			variants[vindex].het = het
			variants[vindex].missing = missing
			if homA1+homA2+het > 0:
				if homA1*2+het > homA2*2+het:
					variants[vindex].maf = float(homA2*2+het)/((homA1+homA2+het)*2)
				else:
					variants[vindex].maf = float(homA1*2+het)/((homA1+homA2+het)*2)
				variants[vindex].na = float(missing)/(homA1+homA2+het+missing)
				avgmiss = avgmiss + variants[vindex].na

		bed.close()
		if verbose:
			print('call took: %.1fs'%(time()-timestamp))

		(comp, compset, incomp, incompset) = self.stats(genotypes, len(vindices), len(cindices))

		if not quiet:
			misspercent = 'nan'
			if len(perfect) != 0:
				misspercent = str(round(sum(perfect)*100/len(perfect),2))
			print('avg variant missingness: ' + str(round(avgmiss*100.0 / len(vindices),2)) +'%, perfectly typed cases: ' + str(sum(perfect)) + ' (' + misspercent +'%)' )
			compercent = float('nan')
			incompercent = float('nan')
			if comp != 0:
				compercent = compset*100/(comp)
			if incomp != 0:
				incompercent = incompset*100/(incomp)
			print('completes: %i unique: %i (%.2f%%), incompletes: %i unique: %i (%.2f%%) '%(comp,compset,compercent,incomp,incompset,incompercent))

		return(perfect, genotypes)
