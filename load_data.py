import torch as th
import dgl
from dgl import function as fn
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import scipy.sparse as sp


def load_data_ACM():
    datapath = r"D:\daima\NodeHGAE\ACM"
    #
    # PA = scipy.sparse.coo_matrix(np.load(datapath+'PA.npy',allow_pickle=True))
    # PS = scipy.sparse.coo_matrix(np.load(datapath + 'PS.npy', allow_pickle=True))
    # PP = scipy.sparse.coo_matrix(np.load(datapath + 'PP.npy', allow_pickle=True))
    PA = sp.load_npz(datapath + '//sp_p_a.npz')
    PS = sp.load_npz(datapath + '//sp_p_s.npz')
    PP = sp.load_npz(datapath + '//sp_p_p.npz')

    AP = PA.T
    SP = PS.T


    PP2 = scipy.sparse.coo_matrix(np.eye(4019))
    SS = scipy.sparse.coo_matrix(np.eye(60))
    AA = scipy.sparse.coo_matrix(np.eye(7167))

    g = dgl.heterograph({
        ('paper', 'paper-paper', 'paper'): (
            th.tensor(PP2.row.astype(np.int32), dtype=torch.int32),
            th.tensor(PP2.col.astype(np.int32), dtype=torch.int32)),
        ('subject', 'subject-subject', 'subject'): (
            th.tensor(SS.row.astype(np.int32), dtype=torch.int32),
            th.tensor(SS.col.astype(np.int32), dtype=torch.int32)),
        ('author', 'author-author', 'author'): (
            th.tensor(AA.row.astype(np.int32), dtype=torch.int32),
            th.tensor(AA.col.astype(np.int32), dtype=torch.int32)),
        ('paper', 'paper-subject', 'subject'): (
        th.tensor(PS.row.astype(np.int32), dtype=torch.int32), th.tensor(PS.col.astype(np.int32), dtype=torch.int32)),
        ('paper', 'paper-author', 'author'): (
        th.tensor(PA.row.astype(np.int32), dtype=torch.int32), th.tensor(PA.col.astype(np.int32), dtype=torch.int32)),
        ('author', 'author-paper', 'paper'): (
        th.tensor(AP.row.astype(np.int32), dtype=torch.int32), th.tensor(AP.col.astype(np.int32), dtype=torch.int32)),
        ('subject', 'subject-paper', 'paper'): (
        th.tensor(SP.row.astype(np.int32), dtype=torch.int32), th.tensor(SP.col.astype(np.int32), dtype=torch.int32))})


    labels = np.load(datapath + '/labels.npy')  # 加载标签，4019
    labels = torch.LongTensor(labels)
    g.ndata['label'] = {'paper': labels}

    features_0 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_0.npz').toarray())
    features_1 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_1.npz').toarray())
    features_2 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_2.npz').toarray())
    # features_1 = torch.FloatTensor(np.eye(7167))
    # features_2 = torch.FloatTensor(np.eye(60))

    g.ndata['h'] = {'paper': features_0, 'author': features_1, 'subject': features_2}
    g.srcdata['h'] = g.ndata['h']
    g.ndata["feature"] = g.ndata['h']
    #dgl.save_graphs('ACM.dgl',g)
    # g, _ = dgl.load_graphs('ACM.dgl')
    train_val_test_idx = np.load(datapath + '/train_val_test_idx.npz')  # 加载训练集，验证集，测试集的索引
    train_nodes = th.tensor(train_val_test_idx['train_idx'], dtype=th.int64)
    val_nodes = th.tensor(train_val_test_idx['val_idx'], dtype=th.int64)
    test_nodes = th.tensor(train_val_test_idx['test_idx'], dtype=th.int64)

    # # 单向边变双向
    # for etype in g.canonical_etypes:
    #     src, t, dst = etype
    #     if src != dst:
    #         continue
    #     u, v = g.edges(etype=etype)
    #     g.add_edges(v, u, etype=etype)
    PAP = scipy.sparse.load_npz(datapath + '/pap.npz').A
    PAP = torch.from_numpy(PAP).type(torch.FloatTensor)
    # PAP = PAP - torch.diag_embed(torch.diag(PAP))  # 去自环
    PAP = F.normalize(PAP, dim=1, p=2)  # 2范数按行归一化

    PSP = scipy.sparse.load_npz(datapath + '/psp.npz').A
    PSP = torch.from_numpy(PSP).type(torch.FloatTensor)
    # PSP = PSP - torch.diag_embed(torch.diag(PSP))  # 去自环
    PSP = F.normalize(PSP, dim=1, p=2)
    ADJ1 = scipy.sparse.load_npz(datapath + '/adjM.npz').A
    PP = ADJ1[0:4019, 0:4019]
    PP = torch.from_numpy(PP).type(torch.FloatTensor)
    # PP = F.normalize(PP, dim=1, p=2)
    ADJ1 = ADJ1[:4019, 4019:]
    ADJ1 = torch.FloatTensor(ADJ1)
    ADJ1 = F.normalize(ADJ1, dim=1, p=2)
    ADJ = [PAP, PSP,PP]  # 注意这里的图有语义强度

    return g, train_nodes, val_nodes, test_nodes,ADJ1,ADJ










def load_data_IMDB():

    datapath = "D:\工作\任务\IMDB"
    #
    MA = sp.load_npz(datapath+'//sp_m_a.npz')
    MD = sp.load_npz(datapath+'//sp_m_d.npz')

    AM = MA.T
    DM = MD.T


    MM = scipy.sparse.coo_matrix(np.eye(4278))
    AA = scipy.sparse.coo_matrix(np.eye(5257))
    DD = scipy.sparse.coo_matrix(np.eye(2081))

    g = dgl.heterograph({
        ('movie', 'movie-movie', 'movie'): (
            th.tensor(MM.row.astype(np.int32), dtype=torch.int32),
            th.tensor(MM.col.astype(np.int32), dtype=torch.int32)),
        ('director', 'director-director', 'director'): (
            th.tensor(DD.row.astype(np.int32), dtype=torch.int32),
            th.tensor(DD.col.astype(np.int32), dtype=torch.int32)),
        ('actor', 'actor-actor', 'actor'): (
            th.tensor(AA.row.astype(np.int32), dtype=torch.int32),
            th.tensor(AA.col.astype(np.int32), dtype=torch.int32)),
        ('movie', 'movie-actor', 'actor'): (
        th.tensor(MA.row.astype(np.int32), dtype=torch.int32), th.tensor(MA.col.astype(np.int32), dtype=torch.int32)),
        ('actor', 'actor-movie', 'movie'): (
        th.tensor(AM.row.astype(np.int32), dtype=torch.int32), th.tensor(AM.col.astype(np.int32), dtype=torch.int32)),
        ('movie', 'movie-director', 'director'): (
        th.tensor(MD.row.astype(np.int32), dtype=torch.int32), th.tensor(MD.col.astype(np.int32), dtype=torch.int32)),
        ('director', 'director-movie', 'movie'): (
        th.tensor(DM.row.astype(np.int32), dtype=torch.int32), th.tensor(DM.col.astype(np.int32), dtype=torch.int32))})


    labels = np.load(datapath + '/labels.npy')  # 加载标签，4019
    labels = torch.LongTensor(labels)
    g.ndata['label'] = {'movie': labels}

    features_0 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_0.npz').toarray())
    features_1 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_1.npz').toarray())
    features_2 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_2.npz').toarray())
    # features_1 = torch.FloatTensor(np.eye(7167))
    # features_2 = torch.FloatTensor(np.eye(60))
    #这里没改到============
    g.ndata['h'] = {'movie': features_0, 'director': features_1, 'actor': features_2}
    g.srcdata['h'] = g.ndata['h']
    g.ndata["feature"] = g.ndata['h']
    dgl.save_graphs('IMDB.dgl',g)
    # g, _ = dgl.load_graphs('ACM.dgl')
    train_val_test_idx = np.load(datapath + '/train_val_test_idx.npz')  # 加载训练集，验证集，测试集的索引
    train_nodes = th.tensor(train_val_test_idx['train_idx'], dtype=th.int64)
    val_nodes = th.tensor(train_val_test_idx['val_idx'], dtype=th.int64)
    test_nodes = th.tensor(train_val_test_idx['test_idx'], dtype=th.int64)
    MDM = np.load(datapath + '/mdm.npy')
    MDM = torch.from_numpy(MDM).type(torch.FloatTensor)
    # MDM = MDM - torch.diag_embed(torch.diag(MDM))  # 去自环
    MDM = F.normalize(MDM, dim=1, p=2)
    MAM = np.load(datapath + '/mam.npy')
    MAM = torch.from_numpy(MAM).type(torch.FloatTensor)
    # MAM = MAM - torch.diag_embed(torch.diag(MAM))  # 去自环
    MAM = F.normalize(MAM, dim=1, p=2)
    ADJ = [MDM, MAM]  # 注意这里的图有语义强度
    ADJ1 = scipy.sparse.load_npz(datapath + '/adjM.npz').A
    ADJ1 = ADJ1[:4278, 4278:]
    ADJ1 = torch.FloatTensor(ADJ1)
    ADJ1 = F.normalize(ADJ1, dim=1, p=1)
    # # 单向边变双向
    # for etype in g.canonical_etypes:
    #     src, t, dst = etype
    #     if src != dst:
    #         continue
    #     u, v = g.edges(etype=etype)
    #     g.add_edges(v, u, etype=etype)

    return g, train_nodes, val_nodes, test_nodes,ADJ1,ADJ
def load_data_DBLP():
    datapath = r"D:\daima\NodeHGAE\DBLP"
    # PA = scipy.sparse.coo_matrix(np.load(datapath+'PA.npy',allow_pickle=True))
    # PS = scipy.sparse.coo_matrix(np.load(datapath + 'PS.npy', allow_pickle=True))
    # PP = scipy.sparse.coo_matrix(np.load(datapath + 'PP.npy', allow_pickle=True))
    PA = sp.load_npz(datapath + '//sp_p_a.npz')
    PC = sp.load_npz(datapath + '//sp_p_c.npz')
    PT = sp.load_npz(datapath + '//sp_p_t.npz')
    AP = PA.T
    CP = PC.T
    TP = PT.T


    PP2 = scipy.sparse.coo_matrix(np.eye(14328))
    AA = scipy.sparse.coo_matrix(np.eye(4057))
    TT=scipy.sparse.coo_matrix(np.eye(7723))
    CC=scipy.sparse.coo_matrix(np.eye(20))

    g = dgl.heterograph({
        ('paper', 'paper-paper', 'paper'): (
            th.tensor(PP2.row.astype(np.int32), dtype=torch.int32),
            th.tensor(PP2.col.astype(np.int32), dtype=torch.int32)),
        ('conferance', 'conferance-conferance', 'conferance'): (
            th.tensor(CC.row.astype(np.int32), dtype=torch.int32),
            th.tensor(CC.col.astype(np.int32), dtype=torch.int32)),
        ('author', 'author-author', 'author'): (
            th.tensor(AA.row.astype(np.int32), dtype=torch.int32),
            th.tensor(AA.col.astype(np.int32), dtype=torch.int32)),
        ('term', 'term-term', 'term'): (
            th.tensor(TT.row.astype(np.int32), dtype=torch.int32),
            th.tensor(TT.col.astype(np.int32), dtype=torch.int32)),
        ('paper', 'paper-conferance', 'conferance'): (
        th.tensor(PC.row.astype(np.int32), dtype=torch.int32), th.tensor(PC.col.astype(np.int32), dtype=torch.int32)),
        ('paper', 'paper-author', 'author'): (
        th.tensor(PA.row.astype(np.int32), dtype=torch.int32), th.tensor(PA.col.astype(np.int32), dtype=torch.int32)),
        ('author', 'author-paper', 'paper'): (
        th.tensor(AP.row.astype(np.int32), dtype=torch.int32), th.tensor(AP.col.astype(np.int32), dtype=torch.int32)),
        ('conferance', 'conferance-paper', 'paper'): (
        th.tensor(CP.row.astype(np.int32), dtype=torch.int32), th.tensor(CP.col.astype(np.int32), dtype=torch.int32)),
        ('term', 'term-paper', 'paper'): (
            th.tensor(TP.row.astype(np.int32), dtype=torch.int32),
            th.tensor(TP.col.astype(np.int32), dtype=torch.int32)),
        ('paper', 'paper-term', 'term'): (
            th.tensor(PT.row.astype(np.int32), dtype=torch.int32),
            th.tensor(PT.col.astype(np.int32), dtype=torch.int32))
    })


    labels = np.load(datapath + '/labels.npy')  # 加载标签，4019
    labels = torch.LongTensor(labels)
    g.ndata['label'] = {'author': labels}

    features_0 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_0.npz').toarray())
    features_1 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_1.npz').toarray())
    features_2 = np.load(datapath + '/features_2.npy')#t
    features_2 = torch.FloatTensor(features_2)
    features_3 = np.eye(20)  # c
    features_3 = torch.FloatTensor(features_3)

    # features_1 = torch.FloatTensor(np.eye(7167))
    # features_2 = torch.FloatTensor(np.eye(60))

    g.ndata['h'] = {'author': features_0, 'paper': features_1,'term': features_2, 'conferance': features_3}
    g.srcdata['h'] = g.ndata['h']
    g.ndata["feature"] = g.ndata['h']
    #dgl.save_graphs('DBLP.dgl',g)
    # g, _ = dgl.load_graphs('ACM.dgl')
    train_val_test_idx = np.load(datapath + '/train_val_test_idx.npz')  # 加载训练集，验证集，测试集的索引
    train_nodes = th.tensor(train_val_test_idx['train_idx'], dtype=th.int64)
    val_nodes = th.tensor(train_val_test_idx['val_idx'], dtype=th.int64)
    test_nodes = th.tensor(train_val_test_idx['test_idx'], dtype=th.int64)
    APA = scipy.sparse.load_npz(datapath + '/apa.npz').A
    APA = torch.from_numpy(APA).type(torch.FloatTensor)
    APA = F.normalize(APA, dim=1, p=2)

    APTPA = scipy.sparse.load_npz(datapath + '/aptpa.npz').A
    APTPA = torch.from_numpy(APTPA).type(torch.FloatTensor)
    APTPA = F.normalize(APTPA, dim=1, p=2)

    APCPA = scipy.sparse.load_npz(datapath + '/apvpa.npz').A
    APCPA = torch.from_numpy(APCPA).type(torch.FloatTensor)
    APCPA = F.normalize(APCPA, dim=1, p=2)

    ADJ = [APA, APTPA, APCPA]  # 注意这里的图有语义强度
    ADJ1 = scipy.sparse.load_npz(datapath + '/adjM.npz').A

    ADJ1 = ADJ1[:4057, 4057:]
    ADJ1 = torch.FloatTensor(ADJ1)
    ADJ1 = F.normalize(ADJ1, dim=1, p=2)








    return g, train_nodes, val_nodes, test_nodes, ADJ1, ADJ
def load_data_YELP():
    datapath = "D:\工作\任务\YELP"
    # PA = scipy.sparse.coo_matrix(np.load(datapath+'PA.npy',allow_pickle=True))
    # PS = scipy.sparse.coo_matrix(np.load(datapath + 'PS.npy', allow_pickle=True))
    # PP = scipy.sparse.coo_matrix(np.load(datapath + 'PP.npy', allow_pickle=True))
    BL = sp.load_npz(datapath + '//sp_b_l.npz')
    BS = sp.load_npz(datapath + '//sp_b_s.npz')
    BU = sp.load_npz(datapath + '//sp_b_u.npz')
    LB = BL.T
    SB = BS.T
    UB = BU.T


    BB2 = scipy.sparse.coo_matrix(np.eye(2614))
    UU = scipy.sparse.coo_matrix(np.eye(1286))
    SS=scipy.sparse.coo_matrix(np.eye(4))
    LL=scipy.sparse.coo_matrix(np.eye(9))

    g = dgl.heterograph({
        ('business', 'business-business', 'business'): (
            th.tensor(BB2.row.astype(np.int32), dtype=torch.int32),
            th.tensor(BB2.col.astype(np.int32), dtype=torch.int32)),
        ('users', 'users-users', 'users'): (
            th.tensor(UU.row.astype(np.int32), dtype=torch.int32),
            th.tensor(UU.col.astype(np.int32), dtype=torch.int32)),
        ('sevice', 'sevice-sevice', 'sevice'): (
            th.tensor(SS.row.astype(np.int32), dtype=torch.int32),
            th.tensor(SS.col.astype(np.int32), dtype=torch.int32)),
        ('leval', 'leval-leval', 'leval'): (
            th.tensor(LL.row.astype(np.int32), dtype=torch.int32),
            th.tensor(LL.col.astype(np.int32), dtype=torch.int32)),
        ('business', 'business-users', 'users'): (
        th.tensor(BU.row.astype(np.int32), dtype=torch.int32), th.tensor(BU.col.astype(np.int32), dtype=torch.int32)),
        ('users', 'users-business', 'business'): (
        th.tensor(UB.row.astype(np.int32), dtype=torch.int32), th.tensor(UB.col.astype(np.int32), dtype=torch.int32)),
        ('business', 'business-sevice', 'sevice'): (
        th.tensor(BS.row.astype(np.int32), dtype=torch.int32), th.tensor(BS.col.astype(np.int32), dtype=torch.int32)),
        ('sevice', 'sevice-business', 'business'): (
        th.tensor(SB.row.astype(np.int32), dtype=torch.int32), th.tensor(SB.col.astype(np.int32), dtype=torch.int32)),
        ('business', 'business-leval', 'leval'): (
            th.tensor(BL.row.astype(np.int32), dtype=torch.int32),
            th.tensor(BL.col.astype(np.int32), dtype=torch.int32)),
        ('leval', 'leval-business', 'business'): (
            th.tensor(LB.row.astype(np.int32), dtype=torch.int32),
            th.tensor(LB.col.astype(np.int32), dtype=torch.int32))
    })


    labels = np.load(datapath + '/labels.npy')  # 加载标签，4019
    labels = torch.LongTensor(labels)
    g.ndata['label'] = {'business': labels}

    features_0 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_0_b.npz').toarray())
    features_1 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_1_u.npz').toarray())
    features_2 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_2_s.npz').toarray())
    features_3 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_3_l.npz').toarray())

    # features_1 = torch.FloatTensor(np.eye(7167))
    # features_2 = torch.FloatTensor(np.eye(60))

    g.ndata['h'] = {'business': features_0, 'users': features_1,'sevice': features_2, 'leval': features_3}
    g.srcdata['h'] = g.ndata['h']
    g.ndata["feature"] = g.ndata['h']
    dgl.save_graphs('YELP.dgl',g)
    # g, _ = dgl.load_graphs('ACM.dgl')
    train_val_test_idx = np.load(datapath + '/train_val_test_idx.npy', allow_pickle=True)
    train_val_test_idx = train_val_test_idx.item()
    train_nodes = th.tensor(train_val_test_idx['train_idx'], dtype=th.int64)
    val_nodes = th.tensor(train_val_test_idx['val_idx'], dtype=th.int64)
    test_nodes = th.tensor(train_val_test_idx['test_idx'], dtype=th.int64)
    BUB = scipy.sparse.load_npz(datapath + '/adj_bub.npz').A
    BUB = torch.from_numpy(BUB).type(torch.FloatTensor)
    # BUB = BUB - torch.diag_embed(torch.diag(BUB))  # 去自环
    BUB = F.normalize(BUB, dim=1, p=2)

    BSB = scipy.sparse.load_npz(datapath + '/adj_bsb.npz').A
    BSB = torch.from_numpy(BSB).type(torch.FloatTensor)
    # BSB = BSB - torch.diag_embed(torch.diag(BSB))  # 去自环
    BSB = F.normalize(BSB, dim=1, p=2)

    BLB = scipy.sparse.load_npz(datapath + '/adj_blb.npz').A
    BLB = torch.from_numpy(BLB).type(torch.FloatTensor)
    # BLB = BLB - torch.diag_embed(torch.diag(BLB))  # 去自环
    BLB = F.normalize(BLB, dim=1, p=2)

    ADJ = [BUB, BSB, BLB]  # 注意这里的图有语义强度
    ADJ1 = scipy.sparse.load_npz(datapath + '/adjM.npz').A

    ADJ1 = ADJ1[:2614, 2614:]
    ADJ1 = torch.FloatTensor(ADJ1)
    ADJ1 = F.normalize(ADJ1, dim=1, p=2)


    return g, train_nodes, val_nodes, test_nodes, ADJ1, ADJ
def load_data_MAG():
    datapath = "D:\工作\任务\MAG"
    # PA = scipy.sparse.coo_matrix(np.load(datapath+'PA.npy',allow_pickle=True))
    # PS = scipy.sparse.coo_matrix(np.load(datapath + 'PS.npy', allow_pickle=True))
    # PP = scipy.sparse.coo_matrix(np.load(datapath + 'PP.npy', allow_pickle=True))
    adj=scipy.sparse.load_npz('adjm.npz').toarray()#p4017 a 15383 i 1480 f 5454
    PF = adj[0:4017,20880:26334]
    PA=adj[0:4017,4017:19400]
    PI=adj[0:4017,19400:20880]
    FP=PF.T
    AP=PA.T
    IP=PI.T



    PP2 = scipy.sparse.coo_matrix(np.eye(4017))
    AA = scipy.sparse.coo_matrix(np.eye(15383))
    II=scipy.sparse.coo_matrix(np.eye(1480))
    FF=scipy.sparse.coo_matrix(np.eye(5454))

    g = dgl.heterograph({
        ('paper', 'paper-paper', 'paper'): (
            th.tensor(PP2.row.astype(np.int32), dtype=torch.int32),
            th.tensor(PP2.col.astype(np.int32), dtype=torch.int32)),
        ('author', 'author-author', 'author'): (
            th.tensor(AA.row.astype(np.int32), dtype=torch.int32),
            th.tensor(AA.col.astype(np.int32), dtype=torch.int32)),
        ('item', 'item-item', 'item'): (
            th.tensor(II.row.astype(np.int32), dtype=torch.int32),
            th.tensor(II.col.astype(np.int32), dtype=torch.int32)),
        ('filed', 'filed-filed', 'filed'): (
            th.tensor(FF.row.astype(np.int32), dtype=torch.int32),
            th.tensor(FF.col.astype(np.int32), dtype=torch.int32)),
        ('paper', 'paper-author', 'author'): (
        th.tensor(PA.row.astype(np.int32), dtype=torch.int32), th.tensor(PA.col.astype(np.int32), dtype=torch.int32)),
        ('author', 'author-paper', 'paper'): (
        th.tensor(AP.row.astype(np.int32), dtype=torch.int32), th.tensor(AP.col.astype(np.int32), dtype=torch.int32)),
        ('paper', 'paper-item', 'item'): (
        th.tensor(PI.row.astype(np.int32), dtype=torch.int32), th.tensor(PI.col.astype(np.int32), dtype=torch.int32)),
        ('item', 'item-paper', 'paper'): (
        th.tensor(IP.row.astype(np.int32), dtype=torch.int32), th.tensor(IP.col.astype(np.int32), dtype=torch.int32)),
        ('paper', 'paper-filed', 'filed'): (
            th.tensor(PF.row.astype(np.int32), dtype=torch.int32),
            th.tensor(PF.col.astype(np.int32), dtype=torch.int32)),
        ('filed', 'filed-paper', 'paper'): (
            th.tensor(FP.row.astype(np.int32), dtype=torch.int32),
            th.tensor(FP.col.astype(np.int32), dtype=torch.int32))
    })


    labels = np.load(datapath + '/p_label.npy')  # 加载标签，4019
    labels = torch.LongTensor(labels)
    g.ndata['label'] = {'paper': labels}

    features_0 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_0_a.npz').toarray())
    features_1 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_1_f.npz').toarray())
    features_2 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_2_i.npz').toarray())
    features_3 = torch.FloatTensor(scipy.sparse.load_npz(datapath + '/features_3_p.npz').toarray())

    # features_1 = torch.FloatTensor(np.eye(7167))
    # features_2 = torch.FloatTensor(np.eye(60))

    g.ndata['h'] = {'author': features_0, 'field': features_1,'item': features_2, 'paper': features_3}
    g.srcdata['h'] = g.ndata['h']
    g.ndata["feature"] = g.ndata['h']
    dgl.save_graphs('MAG.dgl',g)
    # g, _ = dgl.load_graphs('ACM.dgl')
    train_idx = np.load(datapath + '/train_idx1.npy')
    val_idx = np.load(datapath + '/val_idx1.npy')

    # test_idx = np.load(prefix+'lable_test.npy')

    test_idx = np.load(datapath + '/test_idx1.npy')
    train_nodes = th.tensor(train_idx, dtype=th.int64)
    val_nodes = th.tensor(val_idx, dtype=th.int64)
    test_nodes = th.tensor(test_idx, dtype=th.int64)
    PAP = scipy.sparse.load_npz(datapath + '/adj_pap.npz').A
    PAP = torch.from_numpy(PAP).type(torch.FloatTensor)
    PAP = PAP - torch.diag_embed(torch.diag(PAP))  # 去自环
    PAP = F.normalize(PAP, dim=1, p=1)  # 2范数按行归一化

    PFP = scipy.sparse.load_npz(datapath + '/adj_pfp.npz').A
    PFP = torch.from_numpy(PFP).type(torch.FloatTensor)
    PFP = PFP - torch.diag_embed(torch.diag(PFP))  # 去自环
    PFP = F.normalize(PFP, dim=1, p=2)
    PAIAP = scipy.sparse.load_npz(datapath + '/adj_paiap.npz').A
    PAIAP = torch.from_numpy(PAIAP).type(torch.FloatTensor)
    PAIAP = PAIAP - torch.diag_embed(torch.diag(PAIAP))  # 去自环
    PAIAP = F.normalize(PAIAP, dim=1, p=2)

    ADJ = [PAP,PFP,PAIAP] # 注意这里的图有语义强度
    ADJ1 = scipy.sparse.load_npz(datapath + '/adjm.npz').A

    ADJ1 = ADJ1[:4017, 4017:]
    ADJ1 = torch.FloatTensor(ADJ1)
    ADJ1 = F.normalize(ADJ1, dim=1, p=1)


    return g, train_nodes, val_nodes, test_nodes, ADJ1, ADJ