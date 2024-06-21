import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
# 数据增强操作
def continus_mixup_data(*xs, y=None, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha) #  Mixup 参数 使用 Beta 分布生成一个随机数 lam，控制插值的强度
    else:
        lam = 1 # 如果 alpha 不大于0，则直接将 lam 设置为1，这意味着不进行混合，保持原样本不变。
    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)  # 从批次中随机选择数据。这个索引数组被移动到指定的设备（如GPU）
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]  # 新样本[16*200*100] 时间序列，在输入数据和标签级别进行线性插值来生成新的训练样本
    y = lam * y + (1-lam) * y[index]
    return *new_xs, y
def compute_metrics(all_labels, all_probs):
    tn, fp, fn, tp = confusion_matrix(all_labels, (all_probs > 0.5).astype(int)).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(all_labels, all_probs)
    return acc, sen, spe, auc

def train_transformer(model, train_dataloader, loss_fn, optimizer, args):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    for time_series, node_feature, labels, indices in train_dataloader:
        labels = labels.float()
        time_series, node_feature, labels = time_series.to(args.device), node_feature.to(args.device), labels.to(args.device)
        # node_feature, labels = continus_mixup_data(node_feature, y=labels, alpha=1.0, device='cuda')
        predict, extract_feature = model(node_feature)# [16, 140, 200] & node_feature
        loss = loss_fn(predict, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        labels_cpu = torch.argmax(labels, dim=1).detach().cpu()  # 获取最大概率的标签
        preds_cpu = torch.softmax(predict, dim=1)[:, 1].detach().cpu()
        all_labels.append(labels_cpu)
        all_preds.append(preds_cpu)

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    acc, sen, spe, auc = compute_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(train_dataloader.dataset)

    return avg_loss, acc, sen, spe, auc

def test_transformer(model, dataloader, loss_fn, args):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for time_series, node_feature, labels, indices in dataloader:
            time_series, node_feature, labels = time_series.to(args.device), node_feature.to(args.device), labels.to(args.device)
            predict, extract_feature = model(node_feature)
            loss = loss_fn(predict, labels.float())
            total_loss += loss.item()

            labels_cpu = torch.argmax(labels, dim=1).detach().cpu()
            preds_cpu = torch.softmax(predict, dim=1)[:, 1].detach().cpu()
            all_labels.append(labels_cpu)
            all_preds.append(preds_cpu)

    avg_loss = total_loss / len(dataloader.dataset)
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    acc, sen, spe, auc = compute_metrics(all_labels, all_preds)

    return avg_loss, acc, sen, spe, auc




def train_gcn(train_loader, model, optimizer, args):
    model.train()  # Set model to training mode
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    all_labels = []  # To store all labels
    all_probs = []  # To store all predictions

    for data in train_loader:
        data = data.to(args.device)
        optimizer.zero_grad()  # Clear gradients

        logits, edge_weights = model(data.x, data.edge_index, data.edge_attr)
        train_mask = data.train_mask

        loss = criterion(logits[train_mask], data.y[train_mask].long())
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()  # Sum up loss

        # Get predictions and labels
        probs = logits[train_mask].softmax(dim=1).detach().cpu().numpy()[:, 1]
        labels = data.y[train_mask].cpu().numpy()

        all_labels.extend(labels)
        all_probs.extend(probs)

    # Compute metrics using all collected labels and predictions
    acc, sen, spe, auc = compute_metrics(np.array(all_labels), np.array(all_probs))
    average_loss = total_loss / len(train_loader)

    return average_loss, acc, sen, spe, auc

def test_gcn(test_loader, model, args, is_test_phase):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    all_labels = []  # To store all labels
    all_probs = []  # To store all probabilities (for AUC calculation)

    with torch.no_grad():  # Inference mode, no backpropagation
        for data in test_loader:
            data = data.to(args.device)
            logits, edge_weights = model(data.x, data.edge_index, data.edge_attr)

            mask = data.test_mask if is_test_phase else data.val_mask
            loss = criterion(logits[mask], data.y[mask].long())
            total_loss += loss.item()

            probs = logits[mask].softmax(dim=1).detach().cpu().numpy()[:, 1] 
            labels = data.y[mask].cpu().numpy()

            all_labels.extend(labels)
            all_probs.extend(probs)

    average_loss = total_loss / len(test_loader)
    
    # Compute ACC, SEN, SPE, AUC using the compute_metrics function
    acc, sen, spe, auc = compute_metrics(np.array(all_labels), np.array(all_probs))

    return average_loss, acc, sen, spe, auc