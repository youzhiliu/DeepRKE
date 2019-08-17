from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import gensim
import gzip
import os
import glob
import csv
import multiprocessing
import numpy as np


word2vec_model = './RNAShapes2Vec_GraphProt_4mer'
data_dir = './struct_data_GraphProt/'
Embsize = 100
stride = 1
Embepochs = 100
kmer_len = 4
def Gen_Words(sequences,kmer_len,s):
		out=[]

		for i in sequences:

				kmer_list=[]
				for j in range(0,(len(i)-kmer_len)+1,s):

							kmer_list.append(i[j:j+kmer_len])

				out.append(kmer_list)

		return out

def load_sequence(data_dir):
	sequences = []
	train_file_list = sorted(glob.glob(data_dir + '*train*'))
	for file in train_file_list:
		with gzip.open(file, 'rt') as data:
			next(data)
			reader = csv.reader(data,delimiter='\t')
			for row in reader:
				sequences.append(row[0])
	# print(sequences)
	# for file in train_file_list:
	# 	print(file)
	return sequences

def train(sequences):
	print('training word2vec model')
	document= Gen_Words(sequences,kmer_len,stride)
	# print(document)
	model = gensim.models.Word2Vec (document, window=int(12 / stride), min_count=0, size=Embsize,workers=multiprocessing.cpu_count())
	model.train(document,total_examples=len(document),epochs=Embepochs)
	model.save(word2vec_model)

if __name__ == '__main__':
	sequences = []
	sequences = load_sequence(data_dir)
	# test = list(set(sequences))
	# print(test)
	# print(np.array(sequences).shape)
	train(sequences)
