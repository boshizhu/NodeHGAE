import argparse
import torch
from tools import evaluate_results_nc, EarlyStopping
from load_data import load_data_ACM
from model import HeteroGraphSAGE
import numpy as np
import random
import time
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import psutil
import os
import dgl
import warnings
#F.dtype = F.float64
import torch.nn.functional as F
def get_feat_mask(features, mask_rate):#这里是进行数据增强中的特征遮蔽
    feat_node = features.shape[1]#得到特征的维度
    mask = torch.zeros(features.shape)#构建一个和节点特征矩阵同样大小的空矩阵，后面用于存放特征遮蔽后的特征矩阵
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)#从特征维度中选择一定比率，不重复选择
    mask[:, samples] = 1#将选择的维度变成1（猜测原始特征就是0和1）
    return mask, samples
def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
def loss_function(preds, labels):
    pos_weight = float(labels.shape[0] * labels.shape[0] - labels.sum()) / labels.sum()
    norm = labels.shape[0] * labels.shape[0] / float((labels.shape[0] * labels.shape[0] - labels.sum()) * 2)
    zero = torch.zeros(preds.shape[0]).cuda()

    pos_weight = zero+pos_weight
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    return cost
def position_loss(pred_feats, true_feats):
    loss = 1 - (pred_feats * true_feats).sum() / (torch.norm(pred_feats) * torch.norm(true_feats))
    return loss
def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss
def main(args):
    warnings.filterwarnings("ignore")
    g, train_idx, val_idx, test_idx,ADJ_TOPO,ADJ = load_data_ACM()
    ADJ = [graph.to(args['device']) for graph in ADJ]
    topo_dim = ADJ_TOPO.shape[1]
    z_pre=np.load(r'D:\daima\NodeHGAE\ACM\metapath2vec_emb.npy')
    z_pre=torch.FloatTensor(z_pre).cuda()
    z_pre=z_pre[0:4019,:]
    ADJ_TOPO = ADJ_TOPO.to(args['device'])
    featsdim=g.ndata["feature"]['paper'].shape[1]
    in_dims = [g.ndata["feature"][tp].shape[1] for tp in g.ntypes]
    labels = g.ndata['label']['paper']
    labels = labels.to(args['device'])
    numclass, _ = torch.max(g.ndata['label']['paper'], dim=0)
    print(numclass.item() + 1)
    g = g.to(args['device'])
    print(g.canonical_etypes)
    x_ture=g.ndata["feature"]['paper']

    svm_macro_avg = np.zeros((7,), dtype=np.float64)
    svm_micro_avg = np.zeros((7,), dtype=np.float64)
    nmi_avg = 0
    ari_avg = 0
    print('start train with repeat = {}\n'.format(args['repeat']))

    #==========这里暂时设置超参数
    #in_feats = {'author': 7167, 'paper': 4000, 'subject': 60}
    in_feats = {'author': 4000, 'paper': 4000, 'subject': 4000}
    hidden_feats = 64
    out_feats = numclass.item() + 1
    num_layers = 2
    aggregator_type_dict = {'paper': 'mean', 'author': 'gcn','subject': 'gcn'}
    #==========
    target='paper'
    for cur_repeat in range(args['repeat']):
        set_random_seed(args['seed'] + cur_repeat)
        print('cur_repeat = {}   ==============================================================='.format(cur_repeat))

        model = HeteroGraphSAGE(g, in_feats, hidden_feats, out_feats, num_layers, aggregator_type_dict, featsdim,
                                topo_dim, args['dataset'], args['alpha'], args['Init'], args['rank'], args['K']).to(
            args['device'])
        loss_fcn = torch.nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=args['patience'], verbose=True,save_path='checkpoint_{}.pt'.format(args['dataset']))  # 提早停止，设置的耐心值为5
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],weight_decay=args['weight_decay'])

        for epoch in range(args['num_epochs']):
            mask_v1, _ = get_feat_mask(g.ndata['feature'][target], 0.1)  # 设置锚视角下的特征遮蔽
            features_v1 = g.ndata['feature'][target] * (1 - mask_v1).to(args['device'])

            model.train()
            optimizer.zero_grad()
            logits, h,x_pre,topo_pre,ADJ_P,loss_dis = model(g,ADJ_TOPO,z_pre,features_v1,target)
            #loss = F.cross_entropy(logits[output_nodes_dict['paper']], y[output_nodes_dict['paper']])
            w_High = args['w_High']
            w_Local = args['w_Local']
            w_ToPo = args['w_ToPo']
            vae_loss = 0
            for i in range(len(ADJ)):
                vae_loss = vae_loss + loss_function(preds=ADJ_P, labels=ADJ[i])
            vae_loss = vae_loss / len(ADJ)
            mse_loss = position_loss(x_pre, x_ture)
            topo_loss = position_loss(topo_pre, x_ture)
            loss = w_High * vae_loss + w_ToPo * topo_loss + w_Local * mse_loss+loss_dis
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            logits, h,x_pre,topo_pre,ADJ_P,loss_dis = model(g,ADJ_TOPO,z_pre,features_v1,target)
            vae_loss = 0
            for i in range(len(ADJ)):
                vae_loss = vae_loss + loss_function(preds=ADJ_P, labels=ADJ[i])
            vae_loss = vae_loss / len(ADJ)
            mse_loss = position_loss(x_pre, x_ture)
            topo_loss = sce_loss(topo_pre, x_ture,3)
            Val_loss = w_High * vae_loss + w_ToPo * topo_loss + w_Local * mse_loss+loss_dis
            print('Epoch{:d}| Train Loss{:.4f}| Val Loss{:.4f}'.format(epoch + 1, loss.item(), Val_loss.item(), ))
            early_stopping(Val_loss.data.item(), model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
        print('\ntesting...')
        model.load_state_dict(torch.load('checkpoint_{}.pt'.format(args['dataset'])))
        model.eval()
        logits, h,x_pre,topo_pre,ADJ_P,_ = model(g,ADJ_TOPO,z_pre,features_v1,target)
        svm_macro, svm_micro, nmi, ari = evaluate_results_nc(h[test_idx].detach().cpu().numpy(), labels[test_idx].cpu().numpy(),int(labels.max()) + 1)  # 使用SVM评估节点
        svm_macro_avg = svm_macro_avg + svm_macro
        svm_micro_avg = svm_micro_avg + svm_micro
        nmi_avg += nmi
        ari_avg += ari

        # # # ======================#这个作用是可视化
        # Y = labels[test_idx].numpy()
        # ml = TSNE(n_components=2)
        # node_pos = ml.fit_transform(h[test_idx].detach().cpu().numpy())
        # color_idx = {}
        # for i in range(len(h[test_idx].detach().cpu().numpy())):
        #     color_idx.setdefault(Y[i], [])
        #     color_idx[Y[i]].append(i)
        # for c, idx in color_idx.items():  # c是类型数，idx是索引
        #     if str(c) == '1':
        #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#DAA520', s=15, alpha=1)
        #     elif str(c) == '2':
        #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#8B0000', s=15, alpha=1)
        #     elif str(c) == '0':
        #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#6A5ACD', s=15, alpha=1)
        #     elif str(c) == '3':
        #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#006400', s=15, alpha=1)
        # plt.legend()
        # plt.savefig("OSGNN_323_" + str(args['dataset']) + "分类图" + str(cur_repeat) + ".png", dpi=1000,
        # 			bbox_inches='tight')
        # plt.show()



    svm_macro_avg = svm_macro_avg / args['repeat']
    svm_micro_avg = svm_micro_avg / args['repeat']
    nmi_avg /= args['repeat']
    ari_avg /= args['repeat']
    print('---\nThe average of {} results:'.format(args['repeat']))
    print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
    print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))
    print('NMI: {:.6f}'.format(nmi_avg))
    print('ARI: {:.6f}'.format(ari_avg))
    print('all finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='这是我们基于GAT所构建的HAN模型')
    parser.add_argument('--dataset', default='ACM', help='数据集')
    parser.add_argument('--lr', default=0.001, help='学习率')
    parser.add_argument('--num_heads', default=[8], help='多头注意力数及网络层数')
    parser.add_argument('--hidden_units', default=8, help='隐藏层数（实际隐藏层数：隐藏层数*注意力头数）')
    parser.add_argument('--dropout', default=0.5, help='丢弃率')
    parser.add_argument('--num_epochs', default=100000, help='最大迭代次数')
    parser.add_argument('--weight_decay', default=0.0003, help='权重衰减')
    parser.add_argument('--patience', type=int, default=20, help='耐心值')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    parser.add_argument('--w_Local', default=0.8, help='属性重构损失')  # 0.8
    parser.add_argument('--w_High', default=0.1, help='边缘重构损失')  # 0.1
    parser.add_argument('--w_ToPo', default=0.1, help='局部拓扑重构损失')  # 0.1
    parser.add_argument('--device', type=str, default='cuda:0', help='使用cuda:0或者cpu')
    parser.add_argument('--repeat', type=int, default=1, help='重复训练和测试次数')
    parser.add_argument('--alpha', type=int, default=0.25, help='')
    parser.add_argument('--Init', type=str, default='PPR', help='重复训练和测试次数')
    parser.add_argument('--rank', type=int, default=3, help='秩')
    parser.add_argument('--K', type=int, default=5, help='卷积层数')

    args = parser.parse_args().__dict__

    print(args)
    main(args)