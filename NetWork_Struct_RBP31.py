import torch.nn as nn
import torch
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import gensim
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)

# Neural Net definition

# struct_word2vec_model='models_seq+struct/word2vec_struct_model_3mer_kernel8'
# seq_word2vec_model='models_seq+struct/word2vec_seq_model_3mer_kernel8'
struct_word2vec_model='./RNAShapes2Vec'
seq_word2vec_model='./RNA2Vec'

class BCL_Network(nn.Module):
		def __init__(self):
				super(BCL_Network, self).__init__()

				self.cnn = nn.Sequential(
						nn.Conv1d(in_channels=100,
											out_channels=128,
											kernel_size=8,
											stride=1,
											padding=4),		
						# nn.BatchNorm1d(64),
						nn.ReLU(True),
            			nn.MaxPool1d(2),

						nn.Conv1d(in_channels=128,
											out_channels=64,
											kernel_size=16,
											stride=1,
											padding=8),
						# nn.BatchNorm1d(32),
						nn.ReLU(True),
            			nn.MaxPool1d(2),
						
						nn.Conv1d(in_channels=64,
											out_channels=32,
											kernel_size=32,
											stride=1,
											padding=16),
						# nn.BatchNorm1d(16),
						nn.ReLU(True),
            			nn.MaxPool1d(2),

						# nn.Conv1d(in_channels=32,
						# 					out_channels=16,
						# 					kernel_size=8,
						# 					stride=1,
						# 					padding=4),
						# nn.BatchNorm1d(16),
						# nn.ReLU(True),
						# nn.MaxPool1d(2),
				)

				self.cnn_2 = nn.Sequential(
						nn.Conv1d(in_channels=64,
											out_channels=128,
											kernel_size=8,
											stride=1,
											padding=4),		
						# nn.BatchNorm1d(64),
						nn.ReLU(True),
            			nn.MaxPool1d(2),

						nn.Conv1d(in_channels=128,
											out_channels=64,
											kernel_size=16,
											stride=1,
											padding=8),
						# nn.BatchNorm1d(32),
						nn.ReLU(True),
            			nn.MaxPool1d(2),
						
						nn.Conv1d(in_channels=64,
											out_channels=32,
											kernel_size=32,
											stride=1,
											padding=16),
						# nn.BatchNorm1d(16),
						nn.ReLU(True),
            			nn.MaxPool1d(2),

						# nn.Conv1d(in_channels=32,
						# 					out_channels=16,
						# 					kernel_size=8,
						# 					stride=1,
						# 					padding=4),
						# nn.BatchNorm1d(16),
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
						# nn.Linear(128,1),
						nn.Linear(32, 1),
						nn.Sigmoid(),
				)
				model1 = gensim.models.Word2Vec.load(seq_word2vec_model)
				weights_1 = torch.FloatTensor(model1.wv.vectors)
				self.embedding_seq = nn.Embedding.from_pretrained(weights_1, freeze=False)

				model2 = gensim.models.Word2Vec.load(struct_word2vec_model)
				weights_2 = torch.FloatTensor(model2.wv.vectors)
				self.embedding_struct = nn.Embedding.from_pretrained(weights_2, freeze=False)
		def forward(self, seq, struct):
				seq = self.embedding_seq(seq)
				seq = seq.permute(0,2,1)
				# print("input shape is {}".format(input.shape))
				# print(input)
				struct = self.embedding_struct(struct)
				struct = struct.permute(0,2,1)

				seq_out = self.cnn(seq)
				struct_out = self.cnn(struct)
				# 序列和结构横向拼接
				cnn_output_1 = torch.cat((seq_out, struct_out), 1)
				cnn_output = self.cnn_2(cnn_output_1)

				bilstm_out, _ = self.BiLSTM(cnn_output)
				bilstm_out = bilstm_out[:, -1, :]
				# print("input shape is {}".format(input.shape))
				# cnn_output = self.cnn(input)
				
				# print(cnn_output.shape)
				# cnn_output = cnn_output.view(cnn_output.size(0), -1)
				# bilstm_out_seq, _ = self.BiLSTM(seq_out)
				# bilstm_out_seq = bilstm_out_seq[:, -1, :]

				# bilstm_out_struct, _ = self.BiLSTM(struct_out)
				# bilstm_out_struct = bilstm_out_struct[:, -1, :]
				# bilstm_out = cnn_output[:, -1, :]
				# bilstm_out = torch.cat((bilstm_out_seq, bilstm_out_struct), 1)
				# result = self.Prediction(bilstm_out)
				cnn_output = cnn_output.view(cnn_output.size(0), -1)
				result = self.Prediction(cnn_output)
				return result
