import numpy as np
import torch
import dgl
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss






def idx_to_one_hot(idx_arr):
    one_hot = np.zeros((idx_arr.shape[0], idx_arr.max() + 1))
    one_hot[np.arange(idx_arr.shape[0]), idx_arr] = 1
    return one_hot


def kmeans_test(X, y, n_clusters, repeat=10):#聚类数，也就是分成3类
    nmi_list = []
    ari_list = []
    for _ in range(repeat):#迭代进行训练次数
        kmeans = KMeans(n_clusters=n_clusters)#进行KMeans聚类
        y_pred = kmeans.fit_predict(X)#预测
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')#计算标准互信息值，NMI【0,1】
        ari_score = adjusted_rand_score(y, y_pred)#调整的兰德系数，ARI【-1,1】
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)#返回NMI和ARI的均值及标准差


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):#遍历重复次数
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])#按照test_sizes进行分割测试集
            svm = LinearSVC(dual=False)#线性分类支持向量机。dual为False解决原始问题，否则解决对偶问题
            svm.fit(X_train, y_train)#训练模型
            y_pred = svm.predict(X_test)#预测模型，返回预测值
            macro_f1 = f1_score(y_test, y_pred, average='macro')#测试集（这里指分割后的测试集）的损失与预测
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))#记录重复repeat次数下的均值和标准差
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list#返回的是均值和标准差的列表


def evaluate_results_nc(embeddings, labels, num_classes):#输入的是测试集的嵌入，测试集的标签，类别的数量
    repeat = 20
    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels, repeat=repeat)#经过SVM进行测试20次，返回每次的评价列表
    #打印训练集比率为？？下的结果
    print('Macro-F1: ' + ', '.join(['{:.4f}~{:.4f}({:.2f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])]))
    print('Micro-F1: ' + ', '.join(['{:.4f}~{:.4f}({:.2f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])]))
    print('\nK-means test')#得到的嵌入进行K-means测试
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes, repeat=repeat)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    macro_mean = [x for (x, y) in svm_macro_f1_list]
    micro_mean = [x for (x, y) in svm_micro_f1_list]
    return np.array(macro_mean), np.array(micro_mean), nmi_mean, ari_mean#返回各参数的均值