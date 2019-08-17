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

def padding_sequence(seq, max_len = 375, repkey = 'N'):
	seq_len = len(seq)
	if seq_len < max_len:
		gap_len = max_len -seq_len
		new_seq = seq + repkey * gap_len
	else:
		new_seq = seq[:max_len]
	return new_seq

def seqtopad(sequence,motlen):
		rows=len(sequence)+2*motlen-2
		S=np.empty([rows,4])
		base= 'ACGT'
		for i in range(rows):
				for j in range(4):
						if i-motlen+1<len(sequence) and sequence[i-motlen+1]=='N' or i<motlen-1 or i>len(sequence)+motlen-2:
								S[i,j]=np.float32(0.25)
						elif sequence[i-motlen+1]==base[j]:
								S[i,j]=np.float32(1)
						else:
								S[i,j]=np.float32(0)
		return np.transpose(S)

class Chip():
	def __init__(self,filename, maxlen, len_motif):
			self.file = filename
			self.maxlen = maxlen
			self.len = len_motif
	def openFile(self):
		train_dataset=[]
		sequences=[]
		with gzip.open(self.file, 'rt') as data:
			next(data)
			reader = csv.reader(data,delimiter='\t')
			for row in reader:
				train_dataset.append([seqtopad(padding_sequence(row[0], self.maxlen),self.len),[int(row[1])]])
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

class chipseq_dataset(Dataset):
		""" Diabetes dataset."""

		def __init__(self,xy=None):
			self.x_data=np.asarray([el[0] for el in xy],dtype=np.float32)
			self.y_data =np.asarray([el[1] for el in xy ],dtype=np.float32)
			self.x_data = torch.from_numpy(self.x_data)
			self.y_data = torch.from_numpy(self.y_data)
			self.len=len(self.x_data)
			

		def __getitem__(self, index):
			return self.x_data[index], self.y_data[index]

		def __len__(self):
			return self.len

# load test data		
class Chip_test():
	def __init__(self,filename, maxlen, len_motif):
			self.file = filename
			self.maxlen = maxlen
			self.len = len_motif
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
				test_dataset.append([seqtopad(padding_sequence(row[0], self.maxlen),self.len),[int(row[1])]])
				seq.append(row[0])
				
		return test_dataset,seq
		
batch_size = 128
Embsize = 100



def Load_Data(train_file,test_file,flag):

	if flag == 'True':
		chipseq=Chip(train_file, 101, 1)
	else:
		chipseq=Chip(train_file, 375, 1)
	train1,valid1,train2,valid2,train3,valid3,alldataset,sequences=chipseq.openFile()


	train1_dataset=chipseq_dataset(train1)
	train2_dataset=chipseq_dataset(train2)
	train3_dataset=chipseq_dataset(train3)
	valid1_dataset=chipseq_dataset(valid1)
	valid2_dataset=chipseq_dataset(valid2)
	valid3_dataset=chipseq_dataset(valid3)
	alldataset_dataset=chipseq_dataset(alldataset)
	

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
		chipseq_test=Chip_test(test_file, 101, 1)	 
		motif_test=Chip_test(test_file, 101, 1)
	else:
		chipseq_test=Chip_test(test_file, 375, 1)	 
		motif_test=Chip_test(test_file, 375, 1)
	
	
	test_data, seq=chipseq_test.openFile()
	motif_data, seq_motif=motif_test.openFile()
	test_dataset=chipseq_dataset(test_data)
	motif_dataset=chipseq_dataset(motif_data)
	
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
	train_dataloader, valid_dataloader, test_loader, motif_loader = Load_Data(train_file, test_file, flag='TRUE')
	for i, (data, target) in enumerate(train_dataloader[0]):
		print(data)
		print(target.shape)
	# model1 = gensim.models.Word2Vec.load(word2vec_model)
	# weights = torch.FloatTensor(model1.wv.vectors)
	# embedding = nn.Embedding.from_pretrained(weights, freeze=False)
	# print(model1.wv.vectors)
	# print(weights)