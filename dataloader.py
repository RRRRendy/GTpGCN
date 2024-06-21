from operator import index
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
import torch.utils.data as utils
from sklearn.model_selection import StratifiedKFold

def get_dataloader(args):
    
    final_timeseires, final_pearson, labels, site, index = load_mdd_data(args)
    dataloaders_list = Kflod_dataloader(args, final_timeseires, final_pearson, labels, site, index, args.seed)

    return dataloaders_list

def load_mdd_data(args):

    data_path = args.data_path
    data = np.load(data_path, allow_pickle=True).item()
    final_timeseires = data["timeseries"]
    final_pearson = data["corr"]
    labels = data["labels"]
    site = data['sites']
    index = data['index']

    # final_timeseires = standardize(final_timeseires)

    final_timeseires, final_pearson, labels ,index= [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels, index)]


    args.node_sz, args.node_feature_sz = final_pearson.shape[1:]
    args.num_samples = final_pearson.shape[0]
    args.timeseries_sz = final_pearson.shape[2] # 200

    return final_timeseires, final_pearson, labels, site, index
    # final_pearson是相关系数矩阵，final_timeseires是时间序列，labels是标签，site是数据集来源

# def stratified_dataloader(args, final_timeseires, final_pearson, labels, site):

#     labels = F.one_hot(labels.to(torch.int64))
#     length = final_timeseires.shape[0] # 数据总长度
#     train_length = int(length*args.len_train_set) # 训练集长度
#     val_length = int(length*args.len_val_set) # 验证集长度
#     test_length = length-train_length-val_length # 测试集长度

#     split = StratifiedShuffleSplit(
#         n_splits=1, test_size=val_length + test_length, train_size=train_length, random_state=42)

#     stratified = labels.argmax(axis=1).numpy()  # 从 one-hot 编码中获取原始标签用于分层

def Kflod_dataloader(args, final_timeseires, final_pearson, labels, site, index, seed):
    labels = F.one_hot(labels.to(torch.int64))
    
    # 定义五折交叉验证
    skf = StratifiedKFold(n_splits=args.n_split, shuffle=True, random_state=seed)

    dataloaders_list = []

    # 迭代每个折叠
    for fold, (train_index, test_index) in enumerate(skf.split(final_timeseires, site)):
        # 将train和test对应index保存
        # train_index_path = os.path.join(args.data_dir, f'train_index_fold{fold}.csv')
        # test_index_path = os.path.join(args.data_dir, f'test_index_fold{fold}.csv')
        # pd.DataFrame(train_index).to_csv(train_index_path, index=False, header=False)
        # pd.DataFrame(test_index).to_csv(test_index_path, index=False, header=False)
        # 划分数据
        final_timeseires_train, final_pearson_train, labels_train, index_train= ( final_timeseires[train_index], final_pearson[train_index], labels[train_index],index[train_index])
        final_timeseires_test, final_pearson_test, labels_test, index_test= (final_timeseires[test_index], final_pearson[test_index], labels[test_index],index[test_index])

        # 创建数据集
        train_dataset = utils.TensorDataset(final_timeseires_train, final_pearson_train, labels_train, index_train)
        test_dataset = utils.TensorDataset(final_timeseires_test, final_pearson_test, labels_test,index_test)

        # 创建 DataLoader
        train_dataloader = utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        test_dataloader = utils.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        dataloaders_list.append((train_dataloader, test_dataloader))

    return dataloaders_list

def standardize(data):

    mean = np.mean(data)
    std = np.std(data)
    standardized_data = (data - mean) / std
    return standardized_data


import os
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

def get_gcn_data(args,fold,fold_path):

    #---features--- 提取上一步提取特征时相关的信息
    features = np.load(os.path.join(fold_path, f'all_features_fold{fold}_X.npy'))
    features_index = np.load(os.path.join(fold_path, f'all_indices_fold{fold}.npy'))
    labels = np.load(os.path.join(fold_path, f'all_labels_fold{fold}_Y.npy'))
    if labels.ndim > 1: 
        labels = np.argmax(labels, axis=1) 
    train_test_sets = np.load(os.path.join(fold_path, f'train_test_fold{fold}.npy')) # 用于指示是属于训练集（0）还是测试集（1）

    train_test_sets = train_test_sets.astype(bool)
    train_mask = torch.tensor(~train_test_sets, dtype=torch.bool)
    test_mask = torch.tensor(train_test_sets, dtype=torch.bool)

    features_tensor = torch.tensor(features, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    # -------------------------------------------提取表型信息-------------------------------------------
    phenotypic_data = pd.read_csv(args.csv_path)
    if features_index is not None:
        phenotypic_data.set_index('Index', inplace=True)
        phenotypic_data = phenotypic_data.reindex(features_index) # 如果提供了features_index，按这个索引顺序来排序phenotypic_data
        phenotypic_data.to_csv('./sorted_phenotypic.csv')
    if args.text_features:
        encoder = OneHotEncoder()
        text_feature = encoder.fit_transform(phenotypic_data[args.text_features]).toarray()
    else:
        text_feature = None

    if args.numeric_features:
        numeric_info = phenotypic_data[args.numeric_features].values
        numeric_feature = (numeric_info - numeric_info.min(axis=0)) / (numeric_info.max(axis=0) - numeric_info.min(axis=0))
    else:
        numeric_feature = None
    
    if text_feature is not None and numeric_feature is not None:
        cluster_features = np.concatenate([text_feature, numeric_feature], axis=1)
    elif numeric_feature is not None:
        cluster_features = numeric_feature
    elif text_feature is not None:
        cluster_features = text_feature
    else:
        raise ValueError("No valid features provided.")

    sim_matrix = cosine_similarity(cluster_features)
    adj, attr = [], []
    for i in range(len(phenotypic_data)):
        for j in range(i):
            if sim_matrix[i, j] > args.edge_threshold:
                adj.append([i, j])
                attr.append(sim_matrix[i, j])

    # 转置以符合邻接矩阵形式
    adj = np.array(adj).T
    attr = np.array(attr).T

    feature_str = '_'.join(args.numeric_features + args.text_features).lower() # 设置文件名
    filename_suffix = f'{feature_str}_{args.edge_threshold}'
    print(filename_suffix)
    os.makedirs(args.GCN_data, exist_ok=True)
    pd.DataFrame(adj).to_csv(os.path.join(args.GCN_data, f'MDD_adj_{filename_suffix}.csv'), index=False, header=False)
    pd.DataFrame(attr).to_csv(os.path.join(args.GCN_data, f'MDD_attr_{filename_suffix}.csv'), index=False, header=False)

    edge_index = torch.tensor(adj, dtype=torch.long)
    if attr.ndim > 1:
        edge_attr = torch.tensor(attr.squeeze(1), dtype=torch.float)
    else:
        edge_attr = torch.tensor(attr, dtype=torch.float)

    # -------------------------------------------提取表型信息-------------------------------------------
        
    # --------------------------保存数据--------------------------
    data = Data(x= features_tensor, edge_index=edge_index, edge_attr=edge_attr, y=labels_tensor)
    data.train_mask = train_mask
    data.test_mask = test_mask

    data_file = os.path.join(args.GCN_data, f'gcn_data_fold{fold}.pt')
    
    torch.save(data, data_file)
    
    return data


from scipy.spatial import distance
def get_gcn_data_v2(args,fold,feature_dir):
    
    #---features--- 提取上一步提取特征时相关的信息
    features = np.load(os.path.join(feature_dir, f'all_features_fold{fold}_X.npy'))
    features_index = np.load(os.path.join(feature_dir, f'all_indices_fold{fold}.npy'))
    labels = np.load(os.path.join(feature_dir, f'all_labels_fold{fold}_Y.npy'))
    if labels.ndim > 1: 
        labels = np.argmax(labels, axis=1) 
    train_test_sets = np.load(os.path.join(feature_dir, f'train_test_fold{fold}.npy')) # 用于指示是属于训练集（0）还是测试集（1）

    train_test_sets = train_test_sets.astype(bool)
    train_mask = torch.tensor(~train_test_sets, dtype=torch.bool)
    test_mask = torch.tensor(train_test_sets, dtype=torch.bool)

    features_tensor = torch.tensor(features, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    # -------------------------------------------提取表型信息-------------------------------------------
    phenotypic_data = pd.read_csv(args.csv_path)
    if phenotypic_data.shape[0] != features_tensor.shape[0]:
        raise ValueError("The number of phenotypic samples does not match the number of features")
    if features_index is not None:
        phenotypic_data.set_index('Index', inplace=True)
        phenotypic_data = phenotypic_data.reindex(features_index) # 如果提供了features_index，按这个索引顺序来排序phenotypic_data
    

    num_nodes = features.shape[0]
    graph = np.zeros((num_nodes, num_nodes))# 初始化图结构  
    all_features = args.numeric_features + args.text_features
    print("使用的非成像特征:", all_features)
    for feature_name in all_features:
        nonimage_features = phenotypic_data[feature_name].values
        if feature_name in args.numeric_features:  # 检查是否为数值特征
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    val = abs(nonimage_features[k] - nonimage_features[j])
                    if val < 5:  # 数值特征的阈值判断
                        graph[k, j] += 1
                        graph[j, k] += 1

        elif feature_name in args.text_features:  # 检查是否为文本特征
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if nonimage_features[k] == nonimage_features[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1
            
    pd_affinity = graph
    distv = distance.pdist(features, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
    # adj = pd_affinity * feature_sim
    adj = pd_affinity
    # adj = feature_sim
    adj = adj / np.max(adj)# 边权的归一化

    num_edge = num_nodes*(1+num_nodes)//2 - num_nodes
    edge_index = np.zeros([2, num_edge], dtype=np.int64) # 存储图中所有可能的边的索引
    edge_attr = np.zeros([num_edge, 1], dtype=np.float32)
    flatten_ind = 0
    for i in range(num_nodes): # 以建立边的索引和相应的权重
        for j in range(i+1, num_nodes):
            edge_index[:, flatten_ind] = [i, j]
            edge_attr[flatten_ind] = adj[i][j]
            flatten_ind += 1

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    if edge_attr.ndim > 1:
        edge_attr = torch.tensor(edge_attr.squeeze(1), dtype=torch.float)
    else:
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # -------------------------------------------提取表型信息-------------------------------------------
        
    # --------------------------保存数据--------------------------
        
    zero_tensor = torch.zeros_like(features_tensor)

    # 创建和 features_tensor 维度相同的全一矩阵
    one_tensor = torch.ones_like(features_tensor)

    # 创建和 features_tensor 维度相同的随机矩阵
    random_tensor = torch.rand_like(features_tensor)
    # ---
        # features_tensor
    data = Data(x= random_tensor, edge_index=edge_index, edge_attr=edge_attr, y=labels_tensor)
    data.train_mask = train_mask
    data.test_mask = test_mask

    data_file = os.path.join(feature_dir, f'gcn_data_fold{fold}.pt')
    
    torch.save(data, data_file)
    
    return data

def get_gcn_data_val(args,fold,feature_dir):
    
    #---features--- 提取上一步提取特征时相关的信息
    features = np.load(os.path.join(feature_dir, f'all_features_fold{fold}_X.npy'))
    features_index = np.load(os.path.join(feature_dir, f'all_indices_fold{fold}.npy'))
    labels = np.load(os.path.join(feature_dir, f'all_labels_fold{fold}_Y.npy'))
    if labels.ndim > 1: 
        labels = np.argmax(labels, axis=1) 
    train_test_sets = np.load(os.path.join(feature_dir, f'train_test_fold{fold}.npy')) # 用于指示是属于训练集（0）还是测试集（1）

    train_test_sets = train_test_sets.astype(bool)

    # train_indices = np.where(~train_test_sets)[0]
    # val_indices = np.random.choice(train_indices, size=int(len(train_indices) * 0.2), replace=False)

    # train_mask = torch.zeros_like(torch.tensor(train_test_sets), dtype=torch.bool)
    # val_mask = torch.zeros_like(torch.tensor(train_test_sets), dtype=torch.bool)
    # test_mask = torch.tensor(train_test_sets, dtype=torch.bool)
    # train_mask[train_indices] = True
    # val_mask[val_indices] = True
    # train_mask[val_indices] = False
    #---测试集里划出来一半给验证---
    train_indices = np.where(~train_test_sets)[0]
    test_indices = np.where(train_test_sets)[0]
    val_indices = np.random.choice(test_indices, size=len(test_indices) // 2, replace=False)
    test_indices = np.setdiff1d(test_indices, val_indices)
    train_mask = torch.zeros_like(torch.tensor(train_test_sets), dtype=torch.bool)
    val_mask = torch.zeros_like(torch.tensor(train_test_sets), dtype=torch.bool)
    test_mask = torch.zeros_like(torch.tensor(train_test_sets), dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    features_tensor = torch.tensor(features, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    # -------------------------------------------提取表型信息-------------------------------------------
    phenotypic_data = pd.read_csv(args.csv_path)
    if phenotypic_data.shape[0] != features_tensor.shape[0]:
        raise ValueError("The number of phenotypic samples does not match the number of features")
    if features_index is not None:
        phenotypic_data.set_index('Index', inplace=True)
        phenotypic_data = phenotypic_data.reindex(features_index) # 如果提供了features_index，按这个索引顺序来排序phenotypic_data
    

    num_nodes = features.shape[0]
    graph = np.zeros((num_nodes, num_nodes))# 初始化图结构  
    all_features = args.numeric_features + args.text_features
    for feature_name in all_features:
        nonimage_features = phenotypic_data[feature_name].values
        if feature_name in args.numeric_features:  # 检查是否为数值特征
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    val = abs(nonimage_features[k] - nonimage_features[j])
                    if val < 5:  # 数值特征的阈值判断
                        graph[k, j] += 1
                        graph[j, k] += 1

        elif feature_name in args.text_features:  # 检查是否为文本特征
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if nonimage_features[k] == nonimage_features[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1
            
    pd_affinity = graph
    distv = distance.pdist(features, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
    # adj = pd_affinity * feature_sim
    adj = pd_affinity
    # adj = feature_sim
    adj = adj / np.max(adj)# 边权的归一化

    num_edge = num_nodes*(1+num_nodes)//2 - num_nodes
    edge_index = np.zeros([2, num_edge], dtype=np.int64) # 存储图中所有可能的边的索引
    edge_attr = np.zeros([num_edge, 1], dtype=np.float32)
    flatten_ind = 0
    for i in range(num_nodes): # 以建立边的索引和相应的权重
        for j in range(i+1, num_nodes):
            edge_index[:, flatten_ind] = [i, j]
            edge_attr[flatten_ind] = adj[i][j]
            flatten_ind += 1

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    if edge_attr.ndim > 1:
        edge_attr = torch.tensor(edge_attr.squeeze(1), dtype=torch.float)
    else:
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # -------------------------------------------提取表型信息-------------------------------------------
        
    # --------------------------保存数据--------------------------
    data = Data(x= features_tensor, edge_index=edge_index, edge_attr=edge_attr, y=labels_tensor)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    data_file = os.path.join(feature_dir, f'gcn_data_fold{fold}.pt')
    
    torch.save(data, data_file)
    
    return data