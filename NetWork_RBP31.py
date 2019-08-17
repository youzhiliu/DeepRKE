import torch.nn as nn
import torch
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import gensim

torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)

# Neural Net definition

# word2vec_model='models_seq/word2vec_struct_model_3mer_kernel8'
word2vec_model='./RNA2Vec'

class BCL_Network(nn.Module):
		def __init__(self):
				super(BCL_Network, self).__init__()

				self.cnn = nn.Sequential(
						nn.Conv1d(in_channels=4,
											out_channels=128,
											kernel_size=8,
											stride=1,
											padding=4),		
						# nn.BatchNorm1d(128),
						nn.ReLU(True),
            			nn.MaxPool1d(2),

						nn.Conv1d(in_channels=128,
											out_channels=64,
											kernel_size=16,
											stride=1,
											padding=8),
						# nn.BatchNorm1d(64),
						nn.ReLU(True),
            			nn.MaxPool1d(2),
						#
						nn.Conv1d(in_channels=64,
											out_channels=32,
											kernel_size=32,
											stride=1,
											padding=16),
						# nn.BatchNorm1d(32),
						nn.ReLU(True),
            			nn.MaxPool1d(2),

						# nn.Conv1d(in_channels=32,
						# 					out_channels=16,
						# 					kernel_size=4,
						# 					stride=1,
						# 					padding=2),
						# # nn.BatchNorm1d(16),
						# nn.ReLU(True),
						# nn.MaxPool1d(2),
				)
				self.cnn_1 = nn.Sequential(
						nn.Conv1d(in_channels=32,
											out_channels=128,
											kernel_size=8,
											stride=1,
											padding=4),		
						# nn.BatchNorm1d(128),
						nn.ReLU(True),
            			nn.MaxPool1d(2),

						nn.Conv1d(in_channels=128,
											out_channels=64,
											kernel_size=16,
											stride=1,
											padding=8),
						# nn.BatchNorm1d(64),
						nn.ReLU(True),
            			nn.MaxPool1d(2),
						#
						nn.Conv1d(in_channels=64,
											out_channels=32,
											kernel_size=32,
											stride=1,
											padding=16),
						# nn.BatchNorm1d(32),
						nn.ReLU(True),
            			nn.MaxPool1d(2),

						# nn.Conv1d(in_channels=32,
						# 					out_channels=16,
						# 					kernel_size=4,
						# 					stride=1,
						# 					padding=2),
						# # nn.BatchNorm1d(16),
						# nn.ReLU(True),
						# nn.MaxPool1d(2),
				)

				self.BiLSTM = nn.Sequential(
					nn.LSTM(input_size=2,
									hidden_size=32,
									num_layers=1,
									batch_first=True,
									bidirectional=True,
									bias=True),
				)

				self.Prediction = nn.Sequential(
					nn.Linear(64, 32),
					nn.Dropout(0.2),
					# nn.Linear(32,1),
					nn.Linear(32, 1),
					nn.Sigmoid(),
				)
				model1 = gensim.models.Word2Vec.load(word2vec_model)
				weights = torch.FloatTensor(model1.wv.vectors)
				self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
		def forward(self, input):
			# input = self.embedding(input)
			# input = input.permute(0,2,1)
			# print("input shape is {}".format(input.shape))
			# print(input)
			cnn_output = self.cnn(input)
			
			cnn_output = self.cnn_1(cnn_output)
			# print(cnn_output.shape)
			# cnn_output = cnn_output.view(cnn_output.size(0), -1)
			bilstm_out, _ = self.BiLSTM(cnn_output)
			bilstm_out = bilstm_out[:, -1, :]
			result = self.Prediction(bilstm_out)
			return result
