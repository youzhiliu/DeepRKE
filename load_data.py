import time
import csv
import math 
import random
import gzip
import torch
from sklearn import metrics
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import gensim
import multiprocessing
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import argparse
import warnings

torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)

class Chip():
	def __init__(self,filename):
			self.file = filename

	def openFile(self):
		train_dataset=[]
		sequences=[]
		with gzip.open(self.file, 'rt') as data:
			next(data)
			reader = csv.reader(data,delimiter='\t')
			for row in reader:
				## When using Embedding
				# if len(row[0]) < 41:
				# 	row[0] = row[0].center(41,'N')
				sequences.append(row[0])			
				train_dataset.append([row[0],[float(row[1])]])
		# Triple cross validation
		random.shuffle(train_dataset)
		size=int(len(train_dataset)/3)
		firstvalid=train_dataset[:size]
		secondvalid=train_dataset[size:size+size]
		thirdvalid=train_dataset[size+size:]
		firsttrain=secondvalid+thirdvalid
		secondtrain=firstvalid+thirdvalid
		thirdtrain=firstvalid+secondvalid
		return firsttrain,firstvalid,secondtrain,secondvalid,thirdtrain,thirdvalid,train_dataset,sequences

def Gen_Words(sequences,kmer_len,s):
		out=[]

		for i in sequences:

				kmer_list=[]
				for j in range(0,(len(i)-kmer_len)+1,s):

							kmer_list.append(i[j:j+kmer_len])

				out.append(kmer_list)

		return out

class chipseq_dataset_embd(Dataset):
		""" Diabetes dataset."""

		def __init__(self,xy=None,model=None,kmer_len=5,stride=2):
			
				self.kmer_len= kmer_len
				self.stride= stride
				data=[el[0] for el in xy]
				words_doc= self.Gen_Words(data,self.kmer_len,self.stride)
				# print(words_doc[0])
				x_data=[self.convert_data_to_index(el,model.wv) for el in words_doc]
				# print(x_data[0])
			 
				
				self.x_data=np.asarray(x_data,dtype=np.float32)
				self.y_data =np.asarray([el[1] for el in xy ],dtype=np.float32)
				self.x_data = torch.LongTensor(self.x_data)
				self.y_data = torch.from_numpy(self.y_data)
				self.len=len(self.x_data)
			

		def __getitem__(self, index):
				return self.x_data[index], self.y_data[index]

		def __len__(self):
				return self.len
			
		def Gen_Words(self,pos_data,kmer_len,s):
				out=[]
				
				for i in pos_data:

						kmer_list=[]
						for j in range(0,(len(i)-kmer_len)+1,s):

									kmer_list.append(i[j:j+kmer_len])
								
						out.append(kmer_list)
						
				return out

		def convert_data_to_index(self, string_data, wv):
			index_data = []
			for word in string_data:
					if word in wv:
							index_data.append(wv.vocab[word].index)
			return index_data

# load test data		
class Chip_test():
	def __init__(self,filename):
		self.file = filename
	def openFile(self):
		test_dataset=[]
		seq=[]

		with gzip.open(self.file, 'rt') as data:
			next(data)
			reader = csv.reader(data,delimiter='\t')
			for row in reader:
				# if len(row[0]) < 41:
				# 	row[0] = row[0].center(41,'N')
				## When using Embedding
				test_dataset.append([row[0],[float(row[1])]])
				seq.append(row[0])
				
		return test_dataset,seq
		
batch_size = 128
Embsize = 100
stride = 1
Embepochs = 100
kmer_len = 3
word2vec_model_31='./RNA2Vec'
word2vec_model_24='./RNA2Vec_GraphProt'

# word2vec_model='models_seq/word2vec_struct_model_3mer_kernel8'
def Load_Data(train_file,test_file, flag):

	chipseq=Chip(train_file)
	train1,valid1,train2,valid2,train3,valid3,alldataset,sequences=chipseq.openFile()

	#### word2vect model training
	
	print('training word2vec model')
	document= Gen_Words(sequences,kmer_len,stride)
	# print(document)
	# model = gensim.models.Word2Vec (document, window=int(12 / stride), min_count=0, size=Embsize,workers=multiprocessing.cpu_count())
	# model.train(document,total_examples=len(document),epochs=Embepochs)
	# model.save(word2vec_model)

	if flag == 'True':
		model1 = gensim.models.Word2Vec.load(word2vec_model_31)
	else:
		model1 = gensim.models.Word2Vec.load(word2vec_model_24)

	train1_dataset=chipseq_dataset_embd(train1,model1,kmer_len,stride)
	train2_dataset=chipseq_dataset_embd(train2,model1,kmer_len,stride)
	train3_dataset=chipseq_dataset_embd(train3,model1,kmer_len,stride)
	valid1_dataset=chipseq_dataset_embd(valid1,model1,kmer_len,stride)
	valid2_dataset=chipseq_dataset_embd(valid2,model1,kmer_len,stride)
	valid3_dataset=chipseq_dataset_embd(valid3,model1,kmer_len,stride)
	alldataset_dataset=chipseq_dataset_embd(alldataset,model1,kmer_len,stride)
	

	train_loader1 = DataLoader(dataset=train1_dataset,batch_size=batch_size,shuffle=True)
	train_loader2 = DataLoader(dataset=train2_dataset,batch_size=batch_size,shuffle=True)
	train_loader3 = DataLoader(dataset=train3_dataset,batch_size=batch_size,shuffle=True)
	valid1_loader = DataLoader(dataset=valid1_dataset,batch_size=batch_size,shuffle=True)
	valid2_loader = DataLoader(dataset=valid2_dataset,batch_size=batch_size,shuffle=True)
	valid3_loader = DataLoader(dataset=valid3_dataset,batch_size=batch_size,shuffle=True)
	alldataset_loader=DataLoader(dataset=alldataset_dataset,batch_size=batch_size,shuffle=True)

	train_dataloader=[train_loader1,train_loader2,train_loader3]
	valid_dataloader=[valid1_loader,valid2_loader,valid3_loader]

	#### test dataset

	
	if flag == 'True':
		model1 = gensim.models.Word2Vec.load(word2vec_model_31)
	else:
		model1 = gensim.models.Word2Vec.load(word2vec_model_24)
	chipseq_test=Chip_test(test_file)	 
	motif_test=Chip_test(test_file)
	
	test_data, seq=chipseq_test.openFile()
	motif_data, seq_motif=motif_test.openFile()
	test_dataset=chipseq_dataset_embd(test_data,model1,kmer_len,stride)
	motif_dataset=chipseq_dataset_embd(motif_data,model1,kmer_len,stride)
	
	test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=True)
	motif_loader = DataLoader(dataset=motif_dataset,batch_size=10000000,shuffle=False)

	return train_dataloader, valid_dataloader, test_loader, motif_loader
if __name__ == "__main__":
	# chipseq=Chip("../example-input-data.gz")
	# chipseq=Chip("../data/train/RNCMPT00001-train.gz")
	# train1,valid1,train2,valid2,train3,valid3,alldataset,sequences=chipseq.openFile()
	# out = Gen_Words(sequences,3,1)
	# model = gensim.models.Word2Vec(out, window=12, min_count=0, size=100,workers=multiprocessing.cpu_count())
	# model.train(out,total_examples=len(out),epochs=100)
	# model.save("../models/word2vec_model_1")
	# model1 = gensim.models.Word2Vec.load("./models/word2vec_model")
	# for i in model1.wv.vocab:
	# 	print(model1[i])
	# 	print(i)
	# x_data=[convert_data_to_index(el,model1.wv) for el in out]
	# x_data=np.asarray(x_data,dtype=np.float32)
	# print(x_data.shape)
	train_file = "./data/1_PARCLIP_AGO1234_CLIP_train.gz"
	test_file = "./data/1_PARCLIP_AGO1234_CLIP_test.gz"
	train_dataloader, valid_dataloader, test_loader, motif_loader = Load_Data(train_file, test_file)
	for i, (data, target) in enumerate(train_dataloader[0]):
		print(data)
		print(target.shape)
	# model1 = gensim.models.Word2Vec.load(word2vec_model)
	# weights = torch.FloatTensor(model1.wv.vectors)
	# embedding = nn.Embedding.from_pretrained(weights, freeze=False)
	# print(model1.wv.vectors)
	# print(weights)