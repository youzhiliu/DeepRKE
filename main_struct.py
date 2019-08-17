import load_data_struct as ld
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, recall_score, f1_score, \
	precision_recall_curve
from NetWork_Struct_RBP31 import BCL_Network
from NetWork_Struct_RBP24 import BCL_Network_GraphProt
import torch.nn as nn
import time
import csv
import numpy as np
import torch
import math
import glob
import os
import argparse
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 返回每一折多个epoch中的最优模型
def train(model, loss_func, optimizer, scheduler, struct_train_loader, struct_valid_loader, seq_train_loader, seq_valid_loader, path, fold):
	best = 0
	for epoch in range(Epoch):
		for step, ((struct_x, struct_y), (seq_x, seq_y)) in enumerate(zip(struct_train_loader,seq_train_loader)):
			struct_x = struct_x.cuda()
			struct_y = struct_y.cuda()

			seq_x = seq_x.cuda()
			seq_y = seq_y.cuda()

			model.train()
			output = model(seq_x, struct_x)
			loss = loss_func(output, seq_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		# ROC, PR, F1, test_loss, accuracy = validate(valid_loader, epoch)
		ROC, test_loss, accuracy = validate(model, loss_func, struct_valid_loader, seq_valid_loader, fold, epoch)
		if ROC > best:
			best = ROC
			torch.save(model.state_dict(),
					   path + '_validate_params_' + str(fold) + '_' + str(epoch) + '.pkl')
			model_name = path + '_validate_params_' + str(fold) + '_' + str(epoch) + '.pkl'
	scheduler.step(test_loss)
	print("The best mode_name is {}".format(model_name))
	return model_name


def validate(model, loss_func, struct_valid_loader, seq_valid_loader, fold, epoch):
	output_list = []
	output_result_list = []
	correct_list = []
	test_loss = 0
	for step, ((struct_x, struct_y), (seq_x, seq_y)) in enumerate(zip(struct_valid_loader, seq_valid_loader)):
		struct_x = struct_x.cuda()
		struct_y = struct_y.cuda()

		seq_x = seq_x.cuda()
		seq_y = seq_y.cuda()
		model.eval()
		output = model(seq_x, struct_x)
		loss = loss_func(output, seq_y)
		test_loss += float(loss)
		output_list += output.cpu().detach().numpy().tolist()
		output = (output > 0.5).int()
		output_result_list += output.cpu().detach().numpy().tolist()
		correct_list += seq_y.cpu().detach().numpy().tolist()
	y_pred = np.array(output_result_list)
	y_true = np.array(correct_list)
	y_score = np.array(output_list)
	accuracy = accuracy_score(y_true, y_pred)
	test_loss /= seq_valid_loader.__len__()
	print('Validate set: Average loss:{:.4f}\tAccuracy:{:.3f}'.format(test_loss, accuracy))
	print('第{}折_第{}轮: '.format(fold, epoch))
	fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1, sample_weight=None, drop_intermediate=True)

	ROC = auc(fpr, tpr)
	# ROC, PR, F1 = util.draw_ROC_Curve(output_list, output_result_list, correct_list, path + '/' + 'test')
	# print('第{}折_第{}轮_ROC:{}\tPR:{}\tF1:{} '.format(fold, epoch, ROC, PR, F1))
	# return ROC, PR, F1, test_loss, accuracy
	return ROC, test_loss, accuracy

def test(model, loss_func, struct_test_loader, seq_test_loader, path, fold, best_model_name):
	name = 'validate_params_' + str(fold)
	model.load_state_dict(torch.load(best_model_name))
	output_list = []
	output_result_list = []
	correct_list = []
	for step, ((struct_x, struct_y), (seq_x, seq_y)) in enumerate(zip(struct_test_loader, seq_test_loader)):
		struct_x = struct_x.cuda()
		struct_y = struct_y.cuda()

		seq_x = seq_x.cuda()
		seq_y = seq_y.cuda()
		model.eval()
		output = model(seq_x, struct_x)
		output_list += output.cpu().detach().numpy().tolist()
		output = (output > 0.5).int()
		output_result_list += output.cpu().detach().numpy().tolist()
		correct_list += seq_y.cpu().detach().numpy().tolist()
	y_pred = np.array(output_result_list)
	y_true = np.array(correct_list)
	y_score = np.array(output_list)
	accuracy = accuracy_score(y_true, y_pred)
	fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1, sample_weight=None, drop_intermediate=True)
	ROC = auc(fpr, tpr)
	# ROC, PR, F1 = util.draw_ROC_Curve(output_list, output_result_list, correct_list, path + '/' + name)
	# return ROC, PR, F1
	return ROC, accuracy

Epoch = 30
K_Fold = 3

def run_rbp31():
	LR = 0.001
	seq_data_dir = "./data/"
	struct_data_dir = "./struct_data/"
	struct_train_file_list = sorted(glob.glob(struct_data_dir + '*train*'))
	struct_test_file_list = sorted(glob.glob(struct_data_dir + '*test*'))
	seq_train_file_list = sorted(glob.glob(seq_data_dir + '*train*'))
	seq_test_file_list = sorted(glob.glob(seq_data_dir + '*test*'))
	for struct_train_file, struct_test_file, seq_train_file, seq_test_file in zip(struct_train_file_list, struct_test_file_list, seq_train_file_list, seq_test_file_list):
		print(struct_train_file + '\t' + struct_test_file + '\t' + seq_train_file + '\t' + seq_test_file)

		struct_train_dataloader, struct_valid_dataloader, struct_test_loader, struct_motif_loader = ld.Load_Data(struct_train_file, struct_test_file, flag='RBP31_struct')

		seq_train_dataloader, seq_valid_dataloader, seq_test_loader, seq_motif_loader = ld.Load_Data(seq_train_file, seq_test_file, flag='RBP31_seq')

		# model_auc=[[],[],[]]
		path = "./params_rbp31_seq+struct/" + struct_train_file.split('.')[1].split('/')[2]
		roc_total = 0
		# pr_total = 0
		# F1_total = 0
		best_roc = 0
		best_model_name_final = ""
		for kk in range(K_Fold):
			# 结构
			struct_train_loader = struct_train_dataloader[kk]
			struct_valid_loader=struct_valid_dataloader[kk]
			# 序列
			seq_train_loader = seq_train_dataloader[kk]
			seq_valid_loader = seq_valid_dataloader[kk]
			model = BCL_Network().cuda()
			# print(model)
			# model = nn.parallel.DataParallel(model, device_ids=[0, 1])
			#  优化器和损失函数writer
			optimizer = torch.optim.Adam(model.parameters(), lr=LR)
			# 动态学习率
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3)
			loss_func = nn.BCELoss()
			
			best_model_name = train(model, loss_func, optimizer, scheduler, struct_train_loader, struct_valid_loader, seq_train_loader, seq_valid_loader, path, kk + 1)
			# ROC, PR, F1 = test(test_DataLoader, path, kk + 1, best_model_name)
			ROC, accuracy = test(model, loss_func, struct_test_loader, seq_test_loader, path, kk + 1, best_model_name)
			if ROC > best_roc:
				best_roc = ROC
				best_model_name_final = best_model_name
			# roc_total += ROC
			# pr_total += PR
			# F1_total += F1
			# fold += 1
		# 获得三折的平均AUC值
		# roc_average = roc_total / K_Fold
		# pr_average = pr_total / K_Fold
		# f1_average = F1_total / K_Fold
		print("The best model is :{}".format(best_model_name_final))
		print("The finall ROC is : {}".format(best_roc))
		# print("Average ROC:{}".format(roc_average))
		# print("Average ROC:{}\tPR:{}\tF1:{}".format(roc_average, pr_average, f1_average))
		print("#################################")

def run_rbp24():
	LR = 0.001
	seq_data_dir = "./data_GraphProt/"
	struct_data_dir = "./struct_data_GraphProt/"
	struct_train_file_list = sorted(glob.glob(struct_data_dir + '*train*'))
	struct_test_file_list = sorted(glob.glob(struct_data_dir + '*test*'))
	seq_train_file_list = sorted(glob.glob(seq_data_dir + '*train*'))
	seq_test_file_list = sorted(glob.glob(seq_data_dir + '*test*'))
	for struct_train_file, struct_test_file, seq_train_file, seq_test_file in zip(struct_train_file_list, struct_test_file_list, seq_train_file_list, seq_test_file_list):
		print(struct_train_file + '\t' + struct_test_file + '\t' + seq_train_file + '\t' + seq_test_file)

		struct_train_dataloader, struct_valid_dataloader, struct_test_loader, struct_motif_loader = ld.Load_Data(struct_train_file, struct_test_file, flag='RBP24_struct')

		seq_train_dataloader, seq_valid_dataloader, seq_test_loader, seq_motif_loader = ld.Load_Data(seq_train_file, seq_test_file, flag='RBP24_seq')

		# model_auc=[[],[],[]]
		path = "./params_rbp24_seq+struct/" + struct_train_file.split('.')[1].split('/')[2]
		roc_total = 0
		# pr_total = 0
		# F1_total = 0
		best_roc = 0
		best_model_name_final = ""
		for kk in range(K_Fold):
			# 结构
			struct_train_loader = struct_train_dataloader[kk]
			struct_valid_loader=struct_valid_dataloader[kk]
			# 序列
			seq_train_loader = seq_train_dataloader[kk]
			seq_valid_loader = seq_valid_dataloader[kk]
			model = BCL_Network_GraphProt().cuda()
			# print(model)
			# model = nn.parallel.DataParallel(model, device_ids=[0, 1])
			#  优化器和损失函数writer
			optimizer = torch.optim.Adam(model.parameters(), lr=LR)
			# 动态学习率
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=3)
			loss_func = nn.BCELoss()
			
			best_model_name = train(model, loss_func, optimizer, scheduler, struct_train_loader, struct_valid_loader, seq_train_loader, seq_valid_loader, path, kk + 1)
			# ROC, PR, F1 = test(test_DataLoader, path, kk + 1, best_model_name)
			ROC, accuracy = test(model, loss_func, struct_test_loader, seq_test_loader, path, kk + 1, best_model_name)
			if ROC > best_roc:
				best_roc = ROC
				best_model_name_final = best_model_name
			# roc_total += ROC
			# pr_total += PR
			# F1_total += F1
			# fold += 1
		# 获得三折的平均AUC值
		# roc_average = roc_total / K_Fold
		# pr_average = pr_total / K_Fold
		# f1_average = F1_total / K_Fold
		print("The best model is :{}".format(best_model_name_final))
		print("The finall ROC is : {}".format(best_roc))
		# print("Average ROC:{}".format(roc_average))
		# print("Average ROC:{}\tPR:{}\tF1:{}".format(roc_average, pr_average, f1_average))
		print("#################################")

def run_deepRKE(parser):
	dataset = parser.dataset
	if dataset == "RBP-24":
		run_rbp24()
	if dataset == "RBP-31":
		run_rbp31()

def parse_arguments(parser):
	parser.add_argument('--dataset', type=str, default="RBP-31", required=True)
   
	args = parser.parse_args()
	return args

		 
if __name__ == "__main__":
	start_time = time.time()
	parser = argparse.ArgumentParser()
	args = parse_arguments(parser)
	run_deepRKE(args)
	end_time = time.time()
	duration = end_time - start_time
	day = math.floor(duration / (3600*24))
	h = math.floor((duration % (3600*24)) / 3600)
	minute = math.floor(((duration % (3600*24)) % 3600) / 60)
	print ('{} 天 {} 小时 {} 分钟'.format(day,h,minute))