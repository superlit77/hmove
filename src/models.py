# -*- encoding = utf-8 -*-
"""
@description: 构造混合关联结构模型
@date: 2023/10/17
@File : models.py
@Software : PyCharm
"""
import dhg
import torch
import torch.nn as nn
from dhg.structure.graphs import Graph
from dhg.nn import HyperGCNConv
import torch.nn.functional as F
# gcn + hgnn  基于谱域的混合模型
class GCN_HGNNConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,  # 帮助模型更灵活地拟合数据
            drop_rate: float = 0.1,
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, g: dhg.Graph, hg: dhg.Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        X_g = g.smoothing_with_GCN(X)  # gcn
        X_hg = hg.smoothing_with_HGNN(X)  # hgnn
        X1 = (X_g + X_hg) / 2
        X_ = self.drop(self.act(X1))
        return X_


class GCN_HGNN(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels):
        super(GCN_HGNN, self).__init__()
        self.conv1 = GCN_HGNNConv(in_channels, h_channels)
        self.conv2 = GCN_HGNNConv(h_channels, out_channels)
        # self.fc = nn.Sequential(
        #     nn.Linear(out_channels, 1),
        #     nn.ReLU()
        # )
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )  # 请确保你的损失函数与此相匹配，例如使用二元交叉熵损失

    def encode(self, X, g, hg):
        #x = self.conv1(X, g, hg).relu()
        x = self.conv1(X, g, hg).relu()
        return self.conv2(x, g, hg)

    def forward(self, X, g, hg):
        # 编码器, 混合编码
        x = self.encode(X, g, hg)

        out = []

        for edge in hg.e[0]:
            # chatgpt推荐最大池化
            edge_x = torch.stack([x[i] for i in edge], dim=0).max(dim=0).values
            # 对每条超边的顶点特征进行融合(平均池化),记得节点编号从0开始，如果从1开始，请修改x[i]->x[i-1]
            #edge_x = torch.cat([x[i].unsqueeze(0) for i in edge], dim=0).mean(dim=0)
            out.append(edge_x)

        out = torch.stack(out)  # 转换为张量
        out = self.fc(out)  # 预测

        return out


# GraphSAGE + hgnn+  基于空域的混合模型
class GraphSAGE_HGNNPLUSConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            aggr: str = "mean",
            bias: bool = True,
            drop_rate: float = 0.1,
    ):
        super().__init__()
        self.aggr = aggr
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels * 2, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, g: dhg.Graph, hg: dhg.Hypergraph) -> torch.Tensor:
        X_nbr = g.v2v(X, aggr="mean")
        X_g = torch.cat([X, X_nbr], dim=1)
        Y = hg.v2e(X_g, aggr="mean")
        X_hg = hg.e2v(Y, aggr="mean")
        X1 = self.theta(X_hg)
        X_ = self.drop(self.act(X1))
        return X_


class GraphSAGE_HGNNPLUS(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels):
        super(GraphSAGE_HGNNPLUS, self).__init__()
        self.conv1 = GraphSAGE_HGNNPLUSConv(in_channels, h_channels)
        self.conv2 = GraphSAGE_HGNNPLUSConv(h_channels, out_channels)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )  # 请确保你的损失函数与此相匹配，例如使用二元交叉熵损失

    def encode(self, X, g, hg):
        x = self.conv1(X, g, hg).relu()
        return self.conv2(x, g, hg)

    def forward(self, X, g, hg):
        # 编码器, 混合编码
        x = self.encode(X, g, hg)
        out = []

        for edge in hg.e[0]:
            # chatgpt推荐最大池化
            edge_x = torch.stack([x[i] for i in edge], dim=0).max(dim=0).values
            # 对每条超边的顶点特征进行融合(平均池化),记得节点编号从0开始，如果从1开始，请修改x[i]->x[i-1]
            #edge_x = torch.cat([x[i].unsqueeze(0) for i in edge], dim=0).mean(dim=0)
            out.append(edge_x)

        out = torch.stack(out)  # 转换为张量
        out = self.fc(out)  # 预测

        return out

# gcn + hgcn  基于谱域的混合模型
class GCN_HGCNConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,  # 帮助模型更灵活地拟合数据
            drop_rate: float = 0.1,
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, g: dhg.Graph, hg: dhg.Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        X_g = g.smoothing_with_GCN(X)  # gcn
        g1 = Graph.from_hypergraph_hypergcn(
            hg, X, False #, device=X.device
        )
        X1 = g1.smoothing_with_GCN(X_g)
        #X_end = (X_g + X1) / 2
        X_ = self.drop(self.act(X1))
        return X_


class GCN_HGCN(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels):
        super(GCN_HGCN, self).__init__()
        self.conv1 = GCN_HGCNConv(in_channels, h_channels)
        self.conv2 = GCN_HGCNConv(h_channels, out_channels)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )  # 请确保你的损失函数与此相匹配，例如使用二元交叉熵损失

    def encode(self, X, g, hg):
        x = self.conv1(X, g, hg).relu()
        return self.conv2(x, g, hg)

    def forward(self, X, g, hg):
        # 编码器, 混合编码
        x = self.encode(X, g, hg)
        out = []

        for edge in hg.e[0]:
            # chatgpt推荐最大池化
            edge_x = torch.stack([x[i] for i in edge], dim=0).max(dim=0).values
            # 对每条超边的顶点特征进行融合(平均池化),记得节点编号从0开始，如果从1开始，请修改x[i]->x[i-1]
            #edge_x = torch.cat([x[i].unsqueeze(0) for i in edge], dim=0).mean(dim=0)
            out.append(edge_x)

        out = torch.stack(out)  # 转换为张量
        out = self.fc(out)  # 预测

        return out


# 基于空域的操作
class HSMConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            drop_rate: float = 0.1,
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        X_hg = hg.smoothing_with_HGNN(X)  # HGNN
        # -----------------------
        g1 = Graph.from_hypergraph_hypergcn(
            hg, X, False  # , device=X.device
        )
        X_hg = g1.smoothing_with_GCN(X)   # Hypergcn
        # -----------------------
        Y = hg.v2e(X, aggr="mean")    # HGNN+
        X_hg = hg.e2v(Y, aggr="mean")
        X_ = self.drop(self.act(X_hg))
        return X_


class HSM(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels):
        super(HSM, self).__init__()
        self.conv1 = HSMConv(in_channels, h_channels)
        self.conv2 = HSMConv(h_channels, out_channels)
        # self.fc = nn.Sequential(
        #     nn.Linear(out_channels, 1),
        #     nn.ReLU()
        # )
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )  # 请确保你的损失函数与此相匹配，例如使用二元交叉熵损失

    def encode(self, X, hg):
        x = self.conv1(X, hg).relu()
        return self.conv2(x, hg)

    def forward(self, X, link, hg):
        # 编码器, 混合编码
        x = self.encode(X, hg)

        out = []

        for edge in link.e[0]:
            # chatgpt推荐最大池化
            # edge_x = torch.stack([x[i] for i in edge], dim=0).max(dim=0).values
            # 对每条超边的顶点特征进行融合(平均池化),记得节点编号从0开始，如果从1开始，请修改x[i]->x[i-1]
            edge_x = torch.cat([x[i].unsqueeze(0) for i in edge], dim=0).mean(dim=0)
            out.append(edge_x)

        out = torch.stack(out)  # 转换为张量
        out = self.fc(out)  # 预测
        return out


class GCN_HGNNPLUSConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,  # 帮助模型更灵活地拟合数据
            drop_rate: float = 0.1,
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels * 2, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, g: dhg.Graph, hg: dhg.Hypergraph) -> torch.Tensor:
        X_1 = g.smoothing_with_GCN(X)  # gcn
        X_g = torch.cat([X, X_1], dim=1)
        Y = hg.v2e(X_g, aggr="mean")
        X_hg = hg.e2v(Y, aggr="mean")
        X1 = self.theta(X_hg)
        X_ = self.drop(self.act(X1))
        return X_


class GCN_HGNNPLUS(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels):
        super(GCN_HGNNPLUS, self).__init__()
        self.conv1 = GCN_HGNNPLUSConv(in_channels, h_channels)
        self.conv2 = GCN_HGNNPLUSConv(h_channels, out_channels)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )  # 请确保你的损失函数与此相匹配，例如使用二元交叉熵损失

    def encode(self, X, g, hg):
        x = self.conv1(X, g, hg).relu()
        return self.conv2(x, g, hg)

    def forward(self, X, g, hg):
        # 编码器, 混合编码
        x = self.encode(X, g, hg)
        out = []

        for edge in hg.e[0]:
            # chatgpt推荐最大池化
            #edge_x = torch.stack([x[i] for i in edge], dim=0).max(dim=0).values
            # 对每条超边的顶点特征进行融合(平均池化),记得节点编号从0开始，如果从1开始，请修改x[i]->x[i-1]
            edge_x = torch.cat([x[i].unsqueeze(0) for i in edge], dim=0).mean(dim=0)
            out.append(edge_x)

        out = torch.stack(out)  # 转换为张量
        out = self.fc(out)  # 预测

        return out


class GraphSAGE_HGNNConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            aggr: str = "mean",
            bias: bool = True,
            drop_rate: float = 0.1,
    ):
        super().__init__()
        self.aggr = aggr
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels * 2, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, g: dhg.Graph, hg: dhg.Hypergraph) -> torch.Tensor:
        X_nbr = g.v2v(X, aggr="mean")
        X_g = torch.cat([X, X_nbr], dim=1)
        X_hg = hg.smoothing_with_HGNN(X_g)  # hgnn
        X1 = self.theta(X_hg)
        X_ = self.drop(self.act(X1))
        return X_


class GraphSAGE_HGNN(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels):
        super(GraphSAGE_HGNN, self).__init__()
        self.conv1 = GraphSAGE_HGNNConv(in_channels, h_channels)
        self.conv2 = GraphSAGE_HGNNConv(h_channels, out_channels)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )  # 请确保你的损失函数与此相匹配，例如使用二元交叉熵损失

    def encode(self, X, g, hg):
        x = self.conv1(X, g, hg).relu()
        return self.conv2(x, g, hg)

    def forward(self, X, g, hg):
        # 编码器, 混合编码
        x = self.encode(X, g, hg)
        out = []

        for edge in hg.e[0]:
            # chatgpt推荐最大池化
            edge_x = torch.stack([x[i] for i in edge], dim=0).max(dim=0).values
            # 对每条超边的顶点特征进行融合(平均池化),记得节点编号从0开始，如果从1开始，请修改x[i]->x[i-1]
            #edge_x = torch.cat([x[i].unsqueeze(0) for i in edge], dim=0).mean(dim=0)
            out.append(edge_x)

        out = torch.stack(out)  # 转换为张量
        out = self.fc(out)  # 预测

        return out


class GraphSAGE_HGCNConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,  # 帮助模型更灵活地拟合数据
            drop_rate: float = 0.1,
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels * 2, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, g: dhg.Graph, hg: dhg.Hypergraph) -> torch.Tensor:
        X_nbr = g.v2v(X, aggr="mean")
        X_g = torch.cat([X, X_nbr], dim=1)
        g1 = Graph.from_hypergraph_hypergcn(
            hg, X_g, False #, device=X.device
        )
        X2 = g1.smoothing_with_GCN(X_g)
        X1 = self.theta(X2)
        X_ = self.drop(self.act(X1))
        return X_


class GraphSAGE_HGCN(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels):
        super(GraphSAGE_HGCN, self).__init__()
        self.conv1 = GraphSAGE_HGCNConv(in_channels, h_channels)
        self.conv2 = GraphSAGE_HGCNConv(h_channels, out_channels)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )  # 请确保你的损失函数与此相匹配，例如使用二元交叉熵损失

    def encode(self, X, g, hg):
        x = self.conv1(X, g, hg).relu()
        return self.conv2(x, g, hg)

    def forward(self, X, g, hg):
        # 编码器, 混合编码
        x = self.encode(X, g, hg)
        out = []

        for edge in hg.e[0]:
            # chatgpt推荐最大池化
            edge_x = torch.stack([x[i] for i in edge], dim=0).max(dim=0).values
            # 对每条超边的顶点特征进行融合(平均池化),记得节点编号从0开始，如果从1开始，请修改x[i]->x[i-1]
            #edge_x = torch.cat([x[i].unsqueeze(0) for i in edge], dim=0).mean(dim=0)
            out.append(edge_x)

        out = torch.stack(out)  # 转换为张量
        out = self.fc(out)  # 预测

        return out


# GAT + hgnn+  基于空域的混合模型
class GAT_HGNNConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            drop_rate: float = 0.1,
            atten_neg_slope: float = 0.2,
    ):
        super().__init__()
        self.atten_dropout = nn.Dropout(drop_rate)
        self.atten_act = nn.LeakyReLU(atten_neg_slope)
        self.act = nn.ELU(inplace=True)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.atten_src = nn.Linear(out_channels, 1, bias=False)
        self.atten_dst = nn.Linear(out_channels, 1, bias=False)


    def forward(self, X: torch.Tensor, g: dhg.Graph, hg: dhg.Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        x_for_src = self.atten_src(X)
        x_for_dst = self.atten_dst(X)
        e_atten_score = x_for_src[g.e_src] + x_for_dst[g.e_dst]
        e_atten_score = self.atten_dropout(self.atten_act(e_atten_score).squeeze())
        X_g = g.v2v(X, aggr="softmax_then_sum", e_weight=e_atten_score)
        X_hg = hg.smoothing_with_HGNN(X)  # hgnn
        X1 = (X_g + X_hg) / 2
        X_ = self.act(X1)
        return X_



class GAT_HGNN(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels):
        super(GAT_HGNN, self).__init__()
        self.conv1 = GAT_HGNNConv(in_channels, h_channels)
        self.conv2 = GAT_HGNNConv(h_channels, out_channels)
        # self.fc = nn.Sequential(
        #     nn.Linear(out_channels, 1),
        #     nn.ReLU()
        # )
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )  # 请确保你的损失函数与此相匹配，例如使用二元交叉熵损失

    def encode(self, X, g, hg):
        x = self.conv1(X, g, hg).relu()
        return self.conv2(x, g, hg)

    def forward(self, X, g, hg):
        # 编码器, 混合编码
        x = self.encode(X, g, hg)

        out = []

        for edge in hg.e[0]:
            # chatgpt推荐最大池化
            edge_x = torch.stack([x[i] for i in edge], dim=0).max(dim=0).values
            # 对每条超边的顶点特征进行融合(平均池化),记得节点编号从0开始，如果从1开始，请修改x[i]->x[i-1]
            # edge_x = torch.cat([x[i].unsqueeze(0) for i in edge], dim=0).mean(dim=0)
            out.append(edge_x)

        out = torch.stack(out)  # 转换为张量
        out = self.fc(out)  # 预测

        return out


class GAT_HGNNPLUSConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            drop_rate: float = 0.1,
            atten_neg_slope: float = 0.2,
    ):
        super().__init__()
        self.atten_dropout = nn.Dropout(drop_rate)
        self.atten_act = nn.LeakyReLU(atten_neg_slope)
        self.act = nn.ELU(inplace=True)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.atten_src = nn.Linear(out_channels, 1, bias=False)
        self.atten_dst = nn.Linear(out_channels, 1, bias=False)


    def forward(self, X: torch.Tensor, g: dhg.Graph, hg: dhg.Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        x_for_src = self.atten_src(X)
        x_for_dst = self.atten_dst(X)
        e_atten_score = x_for_src[g.e_src] + x_for_dst[g.e_dst]
        e_atten_score = self.atten_dropout(self.atten_act(e_atten_score).squeeze())
        X_g = g.v2v(X, aggr="softmax_then_sum", e_weight=e_atten_score)
        Y = hg.v2e(X, aggr="mean")
        X_hg = hg.e2v(Y, aggr="mean")
        X1 = (X_g + X_hg) / 2
        X_ = self.act(X1)
        return X_



class GAT_HGNNPLUS(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels):
        super(GAT_HGNNPLUS, self).__init__()
        self.conv1 = GAT_HGNNPLUSConv(in_channels, h_channels)
        self.conv2 = GAT_HGNNPLUSConv(h_channels, out_channels)
        # self.fc = nn.Sequential(
        #     nn.Linear(out_channels, 1),
        #     nn.ReLU()
        # )
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )  # 请确保你的损失函数与此相匹配，例如使用二元交叉熵损失

    def encode(self, X, g, hg):
        x = self.conv1(X, g, hg).relu()
        return self.conv2(x, g, hg)

    def forward(self, X, g, hg):
        # 编码器, 混合编码
        x = self.encode(X, g, hg)

        out = []

        for edge in hg.e[0]:
            # chatgpt推荐最大池化
            edge_x = torch.stack([x[i] for i in edge], dim=0).max(dim=0).values
            # 对每条超边的顶点特征进行融合(平均池化),记得节点编号从0开始，如果从1开始，请修改x[i]->x[i-1]
            # edge_x = torch.cat([x[i].unsqueeze(0) for i in edge], dim=0).mean(dim=0)
            out.append(edge_x)

        out = torch.stack(out)  # 转换为张量
        out = self.fc(out)  # 预测

        return out


class GAT_HGCNConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            drop_rate: float = 0.1,
            atten_neg_slope: float = 0.2,
    ):
        super().__init__()
        self.atten_dropout = nn.Dropout(drop_rate)
        self.atten_act = nn.LeakyReLU(atten_neg_slope)
        self.act = nn.ELU(inplace=True)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.atten_src = nn.Linear(out_channels, 1, bias=False)
        self.atten_dst = nn.Linear(out_channels, 1, bias=False)


    def forward(self, X: torch.Tensor, g: dhg.Graph, hg: dhg.Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        x_for_src = self.atten_src(X)
        x_for_dst = self.atten_dst(X)
        e_atten_score = x_for_src[g.e_src] + x_for_dst[g.e_dst]
        e_atten_score = self.atten_dropout(self.atten_act(e_atten_score).squeeze())
        X_g = g.v2v(X, aggr="softmax_then_sum", e_weight=e_atten_score)
        g1 = Graph.from_hypergraph_hypergcn(
            hg, X, False  # , device=X.device
        )
        X1 = g1.smoothing_with_GCN(X_g)
        X_ = self.act(X1)
        return X_



class GAT_HGCN(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels):
        super(GAT_HGCN, self).__init__()
        self.conv1 = GAT_HGCNConv(in_channels, h_channels)
        self.conv2 = GAT_HGCNConv(h_channels, out_channels)
        # self.fc = nn.Sequential(
        #     nn.Linear(out_channels, 1),
        #     nn.ReLU()
        # )
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )  # 请确保你的损失函数与此相匹配，例如使用二元交叉熵损失

    def encode(self, X, g, hg):
        x = self.conv1(X, g, hg).relu()
        return self.conv2(x, g, hg)

    def forward(self, X, g, hg):
        # 编码器, 混合编码
        x = self.encode(X, g, hg)

        out = []

        for edge in hg.e[0]:
            # chatgpt推荐最大池化
            edge_x = torch.stack([x[i] for i in edge], dim=0).max(dim=0).values
            # 对每条超边的顶点特征进行融合(平均池化),记得节点编号从0开始，如果从1开始，请修改x[i]->x[i-1]
            # edge_x = torch.cat([x[i].unsqueeze(0) for i in edge], dim=0).mean(dim=0)
            out.append(edge_x)

        out = torch.stack(out)  # 转换为张量
        out = self.fc(out)  # 预测

        return out