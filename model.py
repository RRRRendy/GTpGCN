import torch
import torch.nn as nn
from torch.nn import functional as F
# from .transformer_encoder import InterpretableTransformerEncoder
from abc import abstractmethod
from torch.nn import TransformerEncoderLayer
from torch import Tensor
from typing import Optional
import numpy as np
import torch_geometric as tg
#---GCN网络---
class GCN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, hgc, lg, K = 3):
        super(GCN, self).__init__()
        hidden = [hgc for i in range(lg)] 
        self.dropout = dropout
        bias = False
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg
        self.gconv = nn.ModuleList()
        for i in range(lg):
            in_channels = input_dim if i == 0 else hidden[i-1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K=K, normalization='sym', bias=bias))

        self.cls = nn.Sequential(
                # torch.nn.Linear(cls_input_dim, 256),
                torch.nn.Linear(hidden[lg-1], 256),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(256), 
                torch.nn.Linear(256, num_classes))

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):  # 修改为正确的类别检查
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, features, edge_index, edge_weight):
        x = self.relu(self.gconv[0](features, edge_index, edge_weight))
        for i in range(1, self.lg):
            x = F.dropout(x, self.dropout, self.training)
            x = self.relu(self.gconv[i](x, edge_index, edge_weight))
            # jk = torch.cat((x0, x), axis=1)
            # x0 = jk
        logit = self.cls(x)

        return logit, edge_weight

# ------一些模块的定义------#

class BaseModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, time_seires: torch.tensor, node_feature: torch.tensor) -> torch.tensor:
        pass

# ------一些模块的定义------#

# class GNN(torch.nn.Module):

    
# 获取模型在处理数据时生成的注意力权重。这对于理解和解释模型的决策过程非常有用。
class InterpretableTransformerEncoder(TransformerEncoderLayer):
    # d_model：这是模型的输入和输出尺寸
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: Optional[Tensor] = None
    # 重写了父类torch.nn自带的TransformerEncoderLayer中的自注意力函数，调用这个函数时就直接执行了
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True)
        self.attention_weights = weights # 经过了一个自注意力过程，x:[16, 200, 200] weights:[16, 200, 200] 这个权重矩阵显示了序列中每个元素对每个其他元素的影响程度。
        return self.dropout1(x)

    def get_attention_weights(self) -> Optional[Tensor]:
        return self.attention_weights
    
#------搭建网络------#
    


class BrainNetworkTransformer(BaseModel):

    def __init__(self, args):
        super().__init__()
        ext_featuresize = args.ext_featuresize
        input_feature_size = args.node_sz  # 节点数量，例如200
        self.transformer1 = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=4, dim_feedforward=1024, batch_first=True)
        self.transformer2 = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=4, dim_feedforward=1024, batch_first=True)
        # self.SAGPool = SAGPool(input_feature_size, ratio=0.5)
        self.activations = None
        self.gradients = None

        self.reduce_dim1 = nn.Linear(input_feature_size, 100) # (16, 200, 100)
        self.reduce_dim2 = nn.Linear(100, 50) # (16, 200, 50)
        self.reduce_dim3 = nn.Linear(50, 20) # (16, 200, 20)
        self.leaky_relu = nn.LeakyReLU()

        self.fc1 = nn.Linear(input_feature_size*20, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, ext_featuresize)
        self.fc4 = nn.Linear(ext_featuresize, 2)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(ext_featuresize)
        self.dropout = nn.Dropout(0.3)


    def forward(self,  node_feature: torch.Tensor):# time_series: torch.Tensor,
        bz, _, _ = node_feature.shape  # 获取batch size 输入的数据是[16, 200, 200]
        assignments = []  # 存储每一层的池化结果（如果有）

        node_feature = self.transformer1(node_feature) # node_feature:[16, 200, 200]
        # node_feature = self.transformer2(node_feature) # 如果需要第二层，把这句取消注释

        node_feature = self.reduce_dim1(node_feature)
        node_feature = self.leaky_relu(node_feature)
        node_feature = self.reduce_dim2(node_feature)
        node_feature = self.leaky_relu(node_feature)
        node_feature = self.reduce_dim3(node_feature)
        node_feature = self.leaky_relu(node_feature)

        node_feature = node_feature.view(bz, -1)

        node_feature = self.fc1(node_feature) # 降到256
        node_feature = self.batch_norm1(node_feature)
        node_feature = self.leaky_relu(node_feature)
        node_feature = self.dropout(node_feature)
        node_feature = self.fc2(node_feature) # 降到64
        node_feature = self.batch_norm2(node_feature)
        node_feature = self.leaky_relu(node_feature)

        #----经修改----

        node_feature = self.fc3(node_feature)
        node_feature = self.batch_norm3(node_feature)
        node_feature = self.leaky_relu(node_feature)
        
        # --------------------------
        extract_feature = node_feature
        # -------------------------

        output = self.fc4(node_feature)

        return output , extract_feature # 输出层

    def loss(self, assignments):
        # 此示例中没有实现特定的损失计算，这个方法应根据你的需要来定义
        pass


