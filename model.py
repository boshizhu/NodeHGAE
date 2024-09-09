import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import dgl.nn.pytorch as graph
from functools import partial
import torch as th
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
import torch.nn as nn
import math
from torch.nn import Parameter,Linear, ModuleList
from readout import AvgReadout
from nd_encoder import Nd_encoder
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
class HeteroGraphSAGE(nn.Module):
    def __init__(self, g, in_feats, hidden_feats, out_feats, num_layers, aggregator_type_dict,featsdim,topo_dim,dataset,alpha,Init,rank,K):
        super(HeteroGraphSAGE, self).__init__()

        self.rank = rank
        self.alpha = alpha
        self.K = K

        if Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
            TEMP = torch.tensor(np.array([TEMP for i in range(rank)]))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
            TEMP = torch.tensor(np.array([TEMP for i in range(rank)]))
        elif Init == 'Fix':
            TEMP = np.ones(K + 1)
            TEMP = torch.tensor(np.array([TEMP for i in range(rank)]))
        elif Init == 'Mine':
            TEMP = []
            para = torch.ones([rank, K + 1])
            TEMP = torch.nn.init.xavier_normal_(para)
        elif Init == 'Mine_PPR':
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)  # 创建一个数组 其中每个元素的值根据公式 alpha*(1-alpha)**i 计算得出
            TEMP[-1] = (1 - alpha) ** K
            TEMP = torch.tensor(
                np.array([TEMP] * rank))  # 将 NumPy 数组 TEMP 转换为 PyTorch 张量，并使用 rank 参数指定张量的复制次数，以匹配模型中 gamma 参数的形状。
        self.gamma = Parameter(TEMP.float())
        proj_list = []
        for _ in range(K + 1):
            proj_list.append(Linear(hidden_feats, rank))
        self.proj_list = ModuleList(proj_list)
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=0.5)
        self.dataset=dataset
        self.decoder = InnerProductDecoder(0.5, act=lambda x: x)
        self.read = AvgReadout()
        # 定义每种节点类型的特征变换
        self.layers = nn.ModuleDict()
        for ntype in aggregator_type_dict:
            self.layers[ntype] = nn.Linear(in_feats[ntype], hidden_feats)
        self.nd = Nd_encoder(topo_dim, hidden_feats, 0.5)
        if dataset=='YELP':
            self.predict3 = nn.Linear(hidden_feats * 3, hidden_feats, bias=True)
        else:
            self.predict3 = nn.Linear(hidden_feats * 2, hidden_feats, bias=True)

        self.conv = HeteroGraphConv({rel: graph.GraphConv(hidden_feats,hidden_feats) for rel in g.etypes},aggregate='att')
        self.conv2 = HeteroGraphConv({rel: graph.GraphConv(hidden_feats, hidden_feats) for rel in g.etypes}, aggregate='att')
        self.conv3 = HeteroGraphConv({rel: graph.GraphConv(hidden_feats, hidden_feats) for rel in g.etypes}, aggregate='att')
        # self.conv4 = HeteroGraphConv({rel: graph.GraphConv(hidden_feats, hidden_feats) for rel in g.etypes},
        #                              aggregate='att')
        # self.conv5 = HeteroGraphConv({rel: graph.GraphConv(hidden_feats, hidden_feats) for rel in g.etypes},
        #                              aggregate='att')
        # self.conv6 = HeteroGraphConv({rel: graph.GraphConv(hidden_feats, hidden_feats) for rel in g.etypes},
        #                              aggregate='att')
        self.predict1 = nn.Sequential(
            nn.Linear(hidden_feats, 128),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(128, featsdim),
            nn.Tanh()
        )
        #最后一层输出
        self.out_layer = nn.Linear(hidden_feats, out_feats)
        self.predict4 = nn.Sequential(
            nn.Linear(hidden_feats, 128),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(128, topo_dim),
            nn.Tanh()
        )
    def forward(self, g,ADJ_TOPO,z_pre,features_v1,target):
        g.ndata['feature'][target]=features_v1
        h_dict={}
        x_list, eta_list = [], []
        for ntp in g.ntypes:
            h_dict[ntp]=self.dropout(self.layers[ntp](g.ndata['feature'][ntp]))
        Tx_0 = h_dict
        h_0= torch.tanh(self.proj_list[0](Tx_0[target]))
        print('第一层：')
        Tx_1 = self.conv(g, h_dict)
        h_1 = torch.tanh(self.proj_list[1](Tx_1[target]))
        gamma_0 = self.gamma[:, 0].unsqueeze(dim=-1)  # gamma_0=[3,1]
        # 选择self.gamma张量的第一列，然后将这个一维张量转换为一个二维张量，其中新增的维度大小为1。这样做通常是为了满足后续操作对输入张量形状的要求，
        gamma_1 = self.gamma[:, 1].unsqueeze(dim=-1)  # 形状（(rank, K+1）
        eta_0 = torch.matmul(h_0, gamma_0) / self.rank  # 低秩分解[2708, 1]
        eta_1 = torch.matmul(h_1, gamma_1) / self.rank
        hidden = torch.matmul(Tx_0[target].unsqueeze(dim=-1), eta_0.unsqueeze(dim=-1)).squeeze(
            dim=-1)  # Tx_0.unsqueeze(dim=-1).shape=[2708, 7, 1],eta_0.unsqueeze(dim=-1)=[2708, 1, 1]
        hidden = hidden + torch.matmul(Tx_1[target].unsqueeze(dim=-1), eta_1.unsqueeze(dim=-1)).squeeze(dim=-1)

        x_list.append(h_0)
        x_list.append(h_1)
        eta_list.append(eta_0)
        eta_list.append(eta_1)
        for k in range(1, self.K):
            Tx_2 =  self.conv2(g, Tx_1)
            Tx_2[target]=2*Tx_2[target]-Tx_0[target]
            Tx_0[target], Tx_1[target] = Tx_1[target], Tx_2[target]
            x_list.append(Tx_1[target])
            h_k     = torch.tanh(self.proj_list[k+1](Tx_1[target]))#线性变换

            gamma_k = self.gamma[:,k+1].unsqueeze(dim=-1)#每一层的低秩分解 k+1列
            eta_k   = torch.matmul(h_k, gamma_k)/self.rank
            hidden = hidden + torch.matmul(Tx_1[target].unsqueeze(dim=-1), eta_k.unsqueeze(dim=-1)).squeeze(dim=-1)
            eta_list.append(eta_k)
        h=hidden

        ADJ_P=self.decoder(h)
        h=F.tanh(h)
        z_pre=self.predict3(z_pre)
        h_all=self.read(h.unsqueeze(0),msk=None)
        z_all=self.read(z_pre.unsqueeze(0),msk=None)
        a=0.8
        loss_dis1=self.HSIC(h_all,z_all)
        loss_dis = self.cov(h, z_pre)
        x_pre = self.predict1(h)
        z_pre1 = self.predict1(z_pre)
        logits = self.out_layer(self.dropout(h))
        return logits, h,x_pre,z_pre1,ADJ_P,loss_dis+loss_dis1
    def cov(self,z_u, z_s):
        mean1 = torch.mean(z_u, dim=1, keepdim=True)
        mean2 = torch.mean(z_s, dim=1, keepdim=True)
        tensor1_centered = z_u - mean1
        tensor2_centered = z_s - mean2
        covariance = torch.mean(tensor1_centered * tensor2_centered, dim=1)
        loss_dis = torch.sum(covariance) / len(z_u)
        return loss_dis*loss_dis
    def reset_parameters(self):
        torch.nn.init.zeros_(self.gamma)
        for k in range(self.K+1):
            self.gamma.data[k] = self.alpha*(1-self.alpha)**k
        self.gamma.data[-1] = (1-self.alpha)**self.K

    def cov1(self,X, Y):
        """
        用协方差近似代替 HSIC 的计算
        X 和 Y 是 n × d 的样本矩阵
        """

        # 样本数量
        n = X.shape[0]

        # 中心化嵌入矩阵
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        Y_centered = Y - np.mean(Y, axis=0, keepdims=True)

        # 计算协方差矩阵
        Sigma_XY = (1 / n) * X_centered.T @ Y_centered

        # 计算 Frobenius 范数
        cov_frobenius = np.linalg.norm(Sigma_XY, 'fro') ** 2

        return cov_frobenius*cov_frobenius
    def HSIC(self,X, Y):
        """
        Calculate Hilbert-Schmidt Independence Criterion (HSIC) between two sets of data.

        Parameters:
            X : array-like, shape (n_samples_X, n_features)
                The first set of data.
            Y : array-like, shape (n_samples_Y, n_features)
                The second set of data.

        Returns:
            hsic : float
                The HSIC value.
        """
        n_samples_X = X.shape[0]
        n_samples_Y = Y.shape[0]

        K_X = pairwise_kernels(X.cpu().detach().numpy(), metric='rbf')  # Radial Basis Function (RBF) kernel
        K_Y = pairwise_kernels(Y.cpu().detach().numpy(), metric='rbf')

        # Center the kernel matrices
        H = np.eye(n_samples_X) - np.ones((n_samples_X, n_samples_X)) / n_samples_X
        K_X_centered = np.dot(np.dot(H, K_X), H)

        H = np.eye(n_samples_Y) - np.ones((n_samples_Y, n_samples_Y)) / n_samples_Y
        K_Y_centered = np.dot(np.dot(H, K_Y), H)

        # Compute HSIC
        hsic = np.trace(K_X_centered @ K_Y_centered) / (n_samples_X ** 2)

        return hsic
class InnerProductDecoder(nn.Module):

    def __init__(self, dropout, act=F.elu):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj  # /内积重建图

class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds,nty):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))

        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc,beta.data.cpu().numpy()


class HeteroGraphConv(nn.Module):
    def __init__(self, mods, aggregate='att'):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if aggregate == 'att':
            self.agg_fn = inter_att(64, 0)


    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty : [] for nty in g.dsttypes}
        outputs_tp = {nty : [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
                outputs_tp[dtype].append(stype)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
                outputs_tp[dtype].append(stype)
        rsts = {}
        for nty, alist in outputs.items():

            if len(alist) != 0:
                #这是注意力
                rsts[nty],att= self.agg_fn(alist,nty)

                print('源节点类型：',nty,'  邻居节点类型：',outputs_tp[nty],'  邻居节点类型分配的注意力：',att)
                #
                #rsts[nty] = self.agg_fn(alist, nty)
                #if nty == 'paper':
                    #print(alist[0].shape)

        return rsts
