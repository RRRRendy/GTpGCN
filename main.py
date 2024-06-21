import torch
import argparse
import os
import numpy as np
import argparse
import torch.optim as optim
from model import BrainNetworkTransformer,GCN
from train import test_transformer,train_transformer,train_gcn, test_gcn
from dataloader import get_dataloader,get_gcn_data_v2,get_gcn_data_val

imaging = False # 用于控制阶段一和阶段二
non_imaging = True
get_weight = False
grad_cam = False

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='848MDDvs794NC', help='Name of the task') # 232FEDNvs394NC 119FEDNvs72Recurrent 189Recurrentvs427NC 848MDDvs794NC
parser.add_argument('--altas_name', type=str, default='AAL', help='Name of the altas') # CC200 Dosenbach HO AAL
base_path = '/home/rendy/fMRI/DATASETS/MDD'
args = parser.parse_args()
data_path = f"{base_path}/{args.task_name}/{args.task_name}_{args.altas_name}.npy"
csv_path = f"{base_path}/{args.task_name}/{args.task_name}_{args.altas_name}.csv"
#---不同模板---
parser.add_argument('--data_path', type=str, default=data_path, help='Path to the data file')
parser.add_argument('--csv_path', type=str, default=csv_path, help='Path to the CSV file')

# ------experiment config------ #
parser.add_argument('--n_split', type=int, default=5, help='n-fold cross validation')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

# ------Transformer config------ #
parser.add_argument('--num_epochs', type=int, default=400, help='num_epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--saved_model_dir', type=str, default='./saved_model/', help='root of saved_model')


# ------ext feature------ #
parser.add_argument('--ext_featuresize', type=int, default=16, help='exc_featuresize')
# ------GNN config------ #

parser.add_argument('--gcn_num_epochs', type=int, default=500, help='num_epochs')
parser.add_argument('--numeric_features', type=str, default=['Age', 'Edu'], help='numeric features')# 'Age', 'Edu'
parser.add_argument('--text_features', type=str, default=['Site' , 'Sex'], help='text features ')#'Site' , 'Sex'
parser.add_argument('--num_features', type=int, default=16, help='num_features')# 和ext_featuresize一样，一会改一下
parser.add_argument('--nhid', type=int, default=256, help='hidden size ')
parser.add_argument('--dropout_ratio', type=float, default=0.3, help='dropout ratio')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

args = parser.parse_args()

print(args.task_name)
print(f"data_path: {args.data_path}")
print(f"csv_path: {args.csv_path}")
torch.manual_seed(args.seed)
saved_model_dir = os.path.join(args.saved_model_dir, f"{args.task_name}_{args.altas_name}") # saved_model创建任务_模版对应的文件夹
os.makedirs(saved_model_dir, exist_ok=True)

feature_dir = os.path.join('./data/', f"{args.task_name}_{args.altas_name}")
os.makedirs(feature_dir, exist_ok=True)
dataloaders_list = get_dataloader(args)
if imaging:
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    all_best_metrics = []
    for fold, (train_dataloader, test_dataloader) in enumerate(dataloaders_list):

        model = BrainNetworkTransformer(args).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
        # optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=5e-5)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000, eta_min=1.0e-5)

        best_epoch = 0
        best_acc = 0
        best_auc = 0
        
        for epoch in range(args.num_epochs):
            # 重置准确率等
            # train_loss, train_accuracy = train_one_epoch(model, train_dataloader, loss_fn, optimizer, args)
            train_loss, train_ACC, train_SEN, train_SPE, train_AUC = train_transformer(model, train_dataloader, loss_fn, optimizer, args)
            test_loss, test_ACC, test_SEN, test_SPE, test_AUC = test_transformer(model, test_dataloader, loss_fn, args)
            lr_scheduler.step()

            print(f'Epoch {epoch+1}/{args.num_epochs}, '
              f'Train Loss: {train_loss:.3f}, Train ACC: {train_ACC:.3f}, '
              f'Test Loss: {test_loss:.3f}, Test ACC: {test_ACC:.3f}, '
              f'Test AUC: {test_AUC:.3f}, SEN: {test_SEN:.3f}, SPE: {test_SPE:.3f}')

            if test_ACC > best_acc:
                best_acc = test_ACC
                best_acc_epoch = epoch
                best_model_path = os.path.join(saved_model_dir, f'best_acc_model_{fold}.pth')
                
                torch.save(model.state_dict(), best_model_path)
                best_metrics = {
                    'ACC': best_acc,
                    'AUC': test_AUC,
                    'SEN': test_SEN,
                    'SPE': test_SPE
                }

        print(f'Best Epoch for Accuracy in Fold {fold}: {best_acc_epoch+1}, '
        f'Best ACC: {best_metrics["ACC"]:.3f}, Best AUC: {best_metrics["AUC"]:.3f}, '
        f'Best SEN: {best_metrics["SEN"]:.3f}, Best SPE: {best_metrics["SPE"]:.3f}')

        all_best_metrics.append(best_metrics)
        # -----结束-----#

        # -----------------提取特征-----------------#

        # best_model_path 是最佳模型的路径
        
        # 加载最佳模型
        model = BrainNetworkTransformer(args).to(args.device)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()  # 确保模型处于评估模式

        # 初始化列表以收集特征和标签
        all_features_list = []
        all_labels_list = []
        all_indices_list = [] # 对应于原始数据的索引
        train_test_list = [] # 用于指示是属于训练集（0）还是测试集（1）

        with torch.no_grad():
        # 处理训练数据
            for time_series, node_feature, labels, indices in train_dataloader:
                labels = labels.float()
                time_series, node_feature, labels = time_series.to(args.device), node_feature.to(args.device), labels.to(args.device)
                _, extract_feature = model(node_feature)# time_series, 
                all_features_list.append(extract_feature.cpu().numpy())
                all_labels_list.append(labels.cpu().numpy())
                all_indices_list.append(indices.cpu().numpy())
                train_test_list.append(np.zeros(indices.size(0)))  # 对于训练数据，添加全为0的数组
                
            # 处理测试数据
            for time_series, node_feature, labels, indices in test_dataloader:
                labels = labels.float()
                time_series, node_feature, labels = time_series.to(args.device), node_feature.to(args.device), labels.to(args.device)
                _, extract_feature = model(node_feature) # time_series, 
                all_features_list.append(extract_feature.cpu().numpy())
                all_labels_list.append(labels.cpu().numpy())
                all_indices_list.append(indices.cpu().numpy())
                train_test_list.append(np.ones(indices.size(0)))  # 对于测试数据，添加全为1的数组
                
            # 将特征、标签、索引和集合类型保存到文件中
            all_features_array = np.concatenate(all_features_list, axis=0)
            all_labels_array = np.concatenate(all_labels_list, axis=0)
            all_indices_array = np.concatenate(all_indices_list, axis=0)
            train_test_array = np.concatenate(train_test_list, axis=0)  # 新增，合并训练集和测试集的标志
            
            np.save(os.path.join(feature_dir, f'all_features_fold{fold}_X.npy'), all_features_array)
            np.save(os.path.join(feature_dir, f'all_labels_fold{fold}_Y.npy'), all_labels_array)
            np.save(os.path.join(feature_dir, f'all_indices_fold{fold}.npy'), all_indices_array)
            np.save(os.path.join(feature_dir, f'train_test_fold{fold}.npy'), train_test_array)  # 新增，保存集合类型数组
        print("------Feature extraction completed.------")
        # ----------------提取特征------------------       
    #训练中所有折，最好的，以及五折平均       
    average_metrics = {
        'ACC': np.mean([m['ACC'] for m in all_best_metrics]),
        'SEN': np.mean([m['SEN'] for m in all_best_metrics]),
        'SPE': np.mean([m['SPE'] for m in all_best_metrics]),
        'AUC': np.mean([m['AUC'] for m in all_best_metrics]),
    }
    print(f"All best ACC: {[m['ACC'] for m in all_best_metrics]}, Average ACC: {average_metrics['ACC']:.3f}")
    print(f"All best SEN: {[m['SEN'] for m in all_best_metrics]}, Average SEN: {average_metrics['SEN']:.3f}")
    print(f"All best SPE: {[m['SPE'] for m in all_best_metrics]}, Average SPE: {average_metrics['SPE']:.3f}")
    print(f"All best AUC: {[m['AUC'] for m in all_best_metrics]}, Average AUC: {average_metrics['AUC']:.3f}")

    print("------Starting GCN fusion with non-imaging features.-------")

# ---------------可解释分析---------------
# if grad_cam:

if get_weight:
    attention_weights_dir = os.path.join(saved_model_dir, 'attention_weight')
    if not os.path.exists(attention_weights_dir):
        os.makedirs(attention_weights_dir)
    for fold, (train_dataloader, test_dataloader) in enumerate(dataloaders_list):
        model = BrainNetworkTransformer(args).to(args.device)
        fold_model_path = os.path.join(saved_model_dir, f'best_acc_model_{fold}.pth')
        model.load_state_dict(torch.load(fold_model_path))
        model.eval()

        # 为训练数据集保存注意力权重
        train_attention_weights = []
        train_labels = []
        for time_series, node_feature, labels, indices in train_dataloader:
            labels = labels.float()
            time_series, node_feature, labels = time_series.to(args.device), node_feature.to(args.device), labels.to(args.device)
            model(node_feature)# time_series, 
            attention_weights = model.transformer1.get_attention_weights().detach()
            train_attention_weights.append(attention_weights.cpu())
            train_labels.extend(labels.cpu().tolist())
        train_attention_weights = torch.cat(train_attention_weights, dim=0)
        train_attention_weights_path = os.path.join(attention_weights_dir, f'train_attention_weights_fold{fold}.npy')
        np.save(train_attention_weights_path, train_attention_weights.numpy())
        train_labels_path = os.path.join(attention_weights_dir, f'train_labels_fold{fold}.npy')
        np.save(train_labels_path, np.array(train_labels))

        # 为测试数据集保存注意力权重
        test_attention_weights = []
        test_labels = []
        for time_series, node_feature, labels, indices in test_dataloader:
            labels = labels.float()
            time_series, node_feature, labels = time_series.to(args.device), node_feature.to(args.device), labels.to(args.device)
            model( node_feature)# time_series,
            attention_weights = model.transformer1.get_attention_weights().detach()
            test_attention_weights.append(attention_weights.cpu())
            test_labels.extend(labels.cpu().tolist()) 
        test_attention_weights = torch.cat(test_attention_weights, dim=0)
        test_attention_weights_path = os.path.join(attention_weights_dir, f'test_attention_weights_fold{fold}.npy')
        np.save(test_attention_weights_path, test_attention_weights.numpy())
        test_labels_path = os.path.join(attention_weights_dir, f'test_labels_fold{fold}.npy')
        np.save(test_labels_path, np.array(test_labels))
    print("------Attention weights extraction completed.-------")

    
from torch_geometric.data import DataLoader
# if non_imaging:
#     gcn_all_best_metrics = []
#     gcn_test_metrics = []
#     for fold in range(args.n_split):pip install grad-cam

#         data = get_gcn_data_val(args,fold,feature_dir)
#         data_loader = DataLoader([data], batch_size=1, shuffle=True)
#         model = GCN(args.num_features, num_classes=2, dropout=0.2, hgc=16, lg=3).to(args.device) #  切比雪夫阶数K = 3
#         optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

#         gcn_best_acc = 0
#         for epoch in range(args.gcn_num_epochs):
#             train_loss, train_ACC, train_SEN, train_SPE, train_AUC = train_gcn(data_loader, model, optimizer, args)
#             val_loss, val_ACC, val_SEN, val_SPE, val_AUC = test_gcn(data_loader, model, args, is_test_phase=False)

#             print(f'Epoch {epoch+1}/{args.gcn_num_epochs}, '
#               f'Train Loss: {train_loss:.4f}, Train ACC: {train_ACC:.4f}, '
#               f'Test Loss: {val_loss:.4f}, ACC: {val_ACC:.4f}, '
#               f'AUC: {val_AUC:.4f}, SEN: {val_SEN:.4f}, SPE: {val_SPE:.4f}')
#             best_model_path = os.path.join(saved_model_dir, f'best_gcn_model_{fold}.pth')
#             if val_ACC > gcn_best_acc:
#                 gcn_best_acc = val_ACC
#                 best_acc_epoch = epoch
#                 gcn_best_metrics = {
#                         'ACC': val_ACC,
#                         'AUC': val_AUC,
#                         'SEN': val_SEN,
#                         'SPE': val_SPE
#                     }
#                 torch.save(model.state_dict(), best_model_path)

#         print(f'Best Epoch for Accuracy in Fold {fold}: {best_acc_epoch+1}, '
#         f'Best ACC: {gcn_best_metrics["ACC"]:.3f}, Best AUC: {gcn_best_metrics["AUC"]:.3f}, '
#         f'Best SEN: {gcn_best_metrics["SEN"]:.3f}, Best SPE: {gcn_best_metrics["SPE"]:.3f}')
#         #---加载最佳模型
#         model.load_state_dict(torch.load(best_model_path))
#         test_loss, test_ACC, test_SEN, test_SPE, test_AUC = test_gcn(data_loader, model, args, is_test_phase=True)
#         gcn_test_metrics.append({
#                     'ACC': test_ACC,
#                     'AUC': test_AUC,
#                     'SEN': test_SEN,
#                     'SPE': test_SPE
#                 })
#     test_average_metrics = {
#         'ACC': np.mean([m['ACC'] for m in gcn_test_metrics]),
#         'SEN': np.mean([m['SEN'] for m in gcn_test_metrics]),
#         'SPE': np.mean([m['SPE'] for m in gcn_test_metrics]),
#         'AUC': np.mean([m['AUC'] for m in gcn_test_metrics]),
#     }
#     print(f"Test ACC: {[m['ACC'] for m in gcn_test_metrics]}, Average ACC: {test_average_metrics['ACC']:.3f}")
#     print(f"Test SEN: {[m['SEN'] for m in gcn_test_metrics]}, Average SEN: {test_average_metrics['SEN']:.3f}")
#     print(f"Test SPE: {[m['SPE'] for m in gcn_test_metrics]}, Average SPE: {test_average_metrics['SPE']:.3f}")
#     print(f"Test AUC: {[m['AUC'] for m in gcn_test_metrics]}, Average AUC: {test_average_metrics['AUC']:.3f}")
    
if non_imaging:
    gcn_all_best_metrics = []
    for fold in range(args.n_split):
        
        data = get_gcn_data_v2(args,fold,feature_dir)
        data_loader = DataLoader([data], batch_size=1, shuffle=True)
        model = GCN(args.num_features, num_classes=2, dropout=0.2, hgc=16, lg=3).to(args.device) #  切比雪夫阶数K = 3
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        gcn_best_epoch = 0
        gcn_best_acc = 0
        gcn_best_auc = 0
        for epoch in range(args.gcn_num_epochs):
            train_loss, train_ACC, train_SEN, train_SPE, train_AUC = train_gcn(data_loader, model, optimizer, args)
            test_loss, test_ACC, test_SEN, test_SPE, test_AUC = test_gcn(data_loader, model, args, is_test_phase=True)

            print(f'Epoch {epoch+1}/{args.gcn_num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train ACC: {train_ACC:.4f}, '
              f'Test Loss: {test_loss:.4f}, ACC: {test_ACC:.4f}, '
              f'AUC: {test_AUC:.4f}, SEN: {test_SEN:.4f}, SPE: {test_SPE:.4f}')
            
            if test_ACC > gcn_best_acc:
                gcn_best_acc = test_ACC
                best_acc_epoch = epoch
                gcn_best_metrics = {
                        'ACC': gcn_best_acc,
                        'AUC': test_AUC,
                        'SEN': test_SEN,
                        'SPE': test_SPE
                    }
        print(f'Best Epoch for Accuracy in Fold {fold}: {best_acc_epoch+1}, '
        f'Best ACC: {gcn_best_metrics["ACC"]:.4f}, Best AUC: {gcn_best_metrics["AUC"]:.4f}, '
        f'Best SEN: {gcn_best_metrics["SEN"]:.4f}, Best SPE: {gcn_best_metrics["SPE"]:.4f}')
        gcn_all_best_metrics.append(gcn_best_metrics)
    gcn_average_metrics = {
        'ACC': np.mean([m['ACC'] for m in gcn_all_best_metrics]),
        'SEN': np.mean([m['SEN'] for m in gcn_all_best_metrics]),
        'SPE': np.mean([m['SPE'] for m in gcn_all_best_metrics]),
        'AUC': np.mean([m['AUC'] for m in gcn_all_best_metrics]),
    }
    print(f"All best ACC: {[m['ACC'] for m in gcn_all_best_metrics]}, Average ACC: {gcn_average_metrics['ACC']:.3f}")
    print(f"All best SEN: {[m['SEN'] for m in gcn_all_best_metrics]}, Average SEN: {gcn_average_metrics['SEN']:.3f}")
    print(f"All best SPE: {[m['SPE'] for m in gcn_all_best_metrics]}, Average SPE: {gcn_average_metrics['SPE']:.3f}")
    print(f"All best AUC: {[m['AUC'] for m in gcn_all_best_metrics]}, Average AUC: {gcn_average_metrics['AUC']:.3f}")

    