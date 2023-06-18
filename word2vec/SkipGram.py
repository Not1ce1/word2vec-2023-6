"""
■ 在一个合适的数据集上，实现torch dataset，loader，trainer，按照
word2vec的方式，训练word embedding (e.g., skip-gram)，最后进
行word embedding之间的visualization。
■ 加分项：
● 不使用Pytorch给定的loss函数，trainer等高级函数，自己实
现sgd(随机梯度下降)等优化算法
"""
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
from sklearn.manifold import TSNE
import heapq
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts


windowsize = 3  # 窗口大小
embedding_size = 32  # 词嵌入向量为32维
batch_size = 128  # DataLoader的样本量
alpha = 0.1  # 学习率η


class EmbeddingDataset(Dataset):  # 自定义Dataset类
    def __init__(self, dataURL: str):
        super(EmbeddingDataset, self).__init__()
        with open(file=dataURL, mode="r", encoding="utf-8") as f:
            txt = f.readlines()
        self.sentence = []  # 句子列表，每一行为一个互相有语义关系的句子/段落
        self.LenthOfSentence = []  # 句子长度列表
        self.word_dic = {}  # 词字典(词：次数)
        self.example_num_list = []  # 每个句子的样本数列表
        self.example_num = 0  # 总样本数
        for _ in txt:
            this_sentence = _.split()
            for x in this_sentence:
                if x not in self.word_dic.keys():
                    self.word_dic[x] = 1
                else:
                    self.word_dic[x] += 1
            self.LenthOfSentence.append(len(this_sentence))
            self.sentence.append(this_sentence)
        for _ in self.LenthOfSentence:  # 背景词为恒定2*windowsize
            if _ < 2 * windowsize + 1:
                self.example_num_list.append(0)
            else:
                thisnum = _ - 2 * windowsize
                self.example_num_list.append(thisnum)
                self.example_num += thisnum

    def __getitem__(self, index):
        """
        back_words = []  # 背景词为恒定2*windowsize
        if index < windowsize:
            st = 0
            ed = 2 * windowsize + 1
        elif index + windowsize + 1 > len(self.sentence):
            st = len(self.sentence) - 2 * windowsize - 1
            ed = len(self.sentence)
        else:
            st = index - windowsize
            ed = index + windowsize + 1

        for _ in range(st, index):
            back_words.append(self.sentence[_])
        for _ in range(index + 1, ed):
            back_words.append(self.sentence[_])
        return (self.sentence[index], back_words)  # 返回中心词，背景词
        # 负样本？
        """
        back_words = []  # 背景词列表大小为恒定2*windowsize
        this_sentence_index = 0  # 第几句
        this_index = index  # 的第几个样本(从0开始)
        while this_index + 1 > self.example_num_list[this_sentence_index]:
            this_index -= self.example_num_list[this_sentence_index]
            this_sentence_index += 1
        for _ in range(this_index, this_index + windowsize):
            back_words.append(self.sentence[this_sentence_index][_])
        center_word = self.sentence[this_sentence_index][this_index + windowsize]
        for _ in range(this_index + windowsize + 1, this_index + 2 * windowsize + 1):
            back_words.append(self.sentence[this_sentence_index][_])
        return (center_word, back_words)  # 返回中心词和背景词列表

    def __len__(self):
        return self.example_num


def word2one_hot(word: str, word_list):  # 输入word和word列表，输出one-hot编码
    tensor = torch.zeros(len(word_list), 1)
    for x, y in enumerate(word_list):
        if word == y:
            tensor[x] = 1
    return tensor


def tensor2csv(tensor, file_path):  # 将tensor保存为csv
    np.savetxt(fname=file_path, delimiter=",", X=tensor, encoding="utf-8")


class EmbeddingModel(nn.Module):
    # TORCH.NN.FUNCTIONAL.EMBEDDING:一个简单的查找表，用于在固定字典和大小中查找嵌入。输入是索引列表和嵌入矩阵，输出是相应的词嵌入。
    # torch.nn.functional.one_hot
    # TORCH.AUTOGRAD.BACKWARD
    def __init__(self, Embedding_Dataset: EmbeddingDataset):
        super(EmbeddingModel, self).__init__()
        self.dataloader = torch.utils.data.DataLoader(
            dataset=Embedding_Dataset, batch_size=batch_size, shuffle=True
        )
        self.wordlist = [_ for _ in Embedding_Dataset.word_dic.keys()]  # 词列表
        self.w1 = torch.rand(embedding_size, len(self.wordlist))  # 权重矩阵1
        self.w2 = torch.rand(len(self.wordlist), embedding_size)  # 权重矩阵2

    def forward(self, one_hot):
        # 矩阵乘法w2*(w1*input)=output，再softmax
        hidden_layer = torch.mm(self.w1, one_hot)
        return (
            torch.nn.functional.softmax(torch.mm(self.w2, hidden_layer), dim=0),
            hidden_layer,
        )  # 返回预测值和隐藏层

    def train(self):
        # print(self(word2one_hot("数学", self.wordlist)))
        # self(one_hot)就是调用forward函数
        """
        一、SkipGram:以中心词预测其上下文词汇
          1.获取样本并遍历
          2.生成样本中心词的one-hot编码,遍历背景词列表：#此时对背景词取平均概率不遍历？
            ①调用forward,计算出预测概率
            ②计算损失函数:预测概率-背景词one-hot编码
            ③反向传播更新参数矩阵w1,w2
          3.遍历完所有样本训练结束
        """
        for i, (centerword, backlist) in enumerate(self.dataloader):  # 为什么遍历4遍？
            # center为64个中心词，backlist为6*64个背景词
            # backlist[0-2][0] center[0] backlist[3-5][0]构成一句话
            # backword=self.wordlist[torch.nonzero(word2one_hot(backword, self.wordlist))[0][0]]
            for x in range(0, len(centerword)):
                for y in range(0, 2 * windowsize):
                    backword = backlist[y][x]  # backlist[y][x]为本次循环背景词
                    # print(
                    #     self(word2one_hot(centerword, self.wordlist))
                    #     - word2one_hot(backword, self.wordlist)
                    # )
                    # print(
                    #     self.wordlist[
                    #         torch.nonzero(word2one_hot(backword, self.wordlist))[0][0]
                    #     ]
                    # )
                    output_pred, hidden_layer = self(
                        word2one_hot(centerword[x], self.wordlist)
                    )  # output_pred预测值，hidden_layer隐藏层
                    output_fact = word2one_hot(backword, self.wordlist)  # 实际值
                    residual = output_pred - output_fact  # 残差=预测值-实际值
                    thisw1 = self.w1
                    thisw2 = self.w2
                    ######################
                    # 更新w2矩阵
                    # tensor2csv(
                    #     tensor=self.w2, file_path=".\\python\\2023-6-14\\pre_w2.csv"
                    # )  # 保存w2矩阵
                    for _ in range(0, len(self.wordlist)):
                        self.w2[_] -= (
                            alpha * residual[_][0].item() * hidden_layer.reshape(-1, 1)
                        )[0]
                    # tensor2csv(
                    #     tensor=self.w2, file_path=".\\python\\2023-6-14\\last_w2.csv"
                    # )  # 保存w2矩阵
                    ######################
                    # 更新w1矩阵
                    # tensor2csv(
                    #     tensor=self.w1, file_path=".\\python\\2023-6-14\\pre_w1.csv"
                    # )  # 保存w1矩阵
                    x_index_in_w1 = torch.nonzero(output_fact)[0][
                        0
                    ].item()  # x_index_in_w1为w1要更新行向量的行号
                    thisw1 = torch.transpose(thisw1, 0, 1)
                    thisw1[x_index_in_w1] -= torch.transpose(
                        alpha * torch.mm(torch.transpose(thisw2, 0, 1), residual), 0, 1
                    )[0]
                    self.w1 = torch.transpose(thisw1, 0, 1)
                    # tensor2csv(
                    #     tensor=self.w1, file_path=".\\python\\2023-6-14\\last_w1.csv"
                    # )  # 保存w1矩阵
                    ######################


def Embedding_Visualization(
    embeddingModel: EmbeddingModel, file_path1: str, file_path2: str
):  # 可视化
    word_embedding = []
    word_val = []
    for _ in embeddingModel.wordlist:
        word_embedding.append(
            torch.transpose(
                torch.mm(embeddingModel.w1, word2one_hot(_, embeddingModel.wordlist)),
                0,
                1,
            )[0].numpy(),
        )
        word_val.append(_)
    word_embedding = np.asarray(word_embedding)
    word_val = np.asarray(word_val)
    # 用t-SNE降维可视化(参考"https://zhuanlan.zhihu.com/p/511793980")
    tsne = TSNE(n_components=2, random_state=0)
    tsne_word_embedding = tsne.fit_transform(word_embedding)
    x_vals = np.asarray([v[0] for v in tsne_word_embedding])
    y_vals = np.asarray([v[1] for v in tsne_word_embedding])
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    random.seed(0)
    plt.figure(figsize=(12, 12), dpi=300)
    plt.scatter(x_vals, y_vals)
    indices = list(range(len(word_val)))
    for i in indices:
        plt.text(
            x_vals[i],
            y_vals[i],
            word_val[i],
            fontdict={
                "fontsize": 5,
                "color": "black",
            },
        )
    plt.show()
    plt.savefig(file_path1)
    # 用pyechart制作词向量相似词列表(皮尔逊相关系数)
    table_head = ["词", "相似度前五", "相似度列表"]  # 表头
    table_row = []  # 保存图中每一行的数据
    for i in range(0, len(embeddingModel.wordlist)):
        this_similarity_word = []
        this_similarity = []
        this_embedding = word_embedding[i]
        for j in range(0, len(embeddingModel.wordlist)):
            this_similarity.append(
                stats.pearsonr(this_embedding, word_embedding[j])[0]  # 皮尔逊相关系数
            )
        max_simi_list = heapq.nlargest(5, this_similarity)  # 前5个最大值
        for _ in max_simi_list:
            this_index = this_similarity.index(_)
            this_similarity_word.append(embeddingModel.wordlist[this_index])
        max_simi_list = [np.around(x, 3) for x in max_simi_list]
        table_row.append(
            [embeddingModel.wordlist[i], str(this_similarity_word), str(max_simi_list)]
        )
    # 作图
    table = Table()
    table.add(table_head, table_row)
    table.set_global_opts(title_opts=ComponentTitleOpts(title="SkipGram词相似度表"))
    table.render(file_path2)


def word2vec_pred(embeddingModel: EmbeddingModel, file_path: str):  # 遍历词列表输出预测最大值
    table_head = ["词", "预测词前五", "预测列表"]
    table_row = []
    for i in embeddingModel.wordlist:
        this_pred = embeddingModel(word2one_hot(i, embeddingModel.wordlist))[0]
        this_pred = torch.transpose(this_pred, 0, 1).tolist()[0]
        this_similarity_word = []
        this_similarity_list = []
        max_simi_list = heapq.nlargest(3, this_pred)  # 前3个最大值
        for _ in max_simi_list:
            if _ > 0.01:
                this_index = this_pred.index(_)
                this_similarity_list.append(_)
                this_similarity_word.append(embeddingModel.wordlist[this_index])
        table_row.append([i, str(this_similarity_word), str(this_similarity_list)])
    # 作图
    table = Table()
    table.add(table_head, table_row)
    table.set_global_opts(title_opts=ComponentTitleOpts(title="SkipGram预测词表"))
    table.render(file_path)


def main():
    dataset = EmbeddingDataset(".\\python\\test-6-14.txt")
    model = EmbeddingModel(dataset)
    model.train()
    torch.save(model, ".\\python\\2023-6-14\\SkipGram.pth")
    kk = torch.load(".\\python\\2023-6-14\\SkipGram.pth")
    Embedding_Visualization(
        kk,
        ".\\python\\2023-6-14\\output_SkipGram.png",
        ".\\python\\2023-6-14\\output_SkipGram.html",
    )
    word2vec_pred(kk, ".\\python\\2023-6-14\\word2vec_pred.html")
    """
    2.在大数据集上实现
    3.有空写CBOW、负采样
    4.写报告
    """


main()
