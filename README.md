# word2vec-2023-6  
## 一、test-6-19文件夹   
1.test-6-19.txt为数据集。  
2.SkipGram.py为python代码文件。  
3.test-6-19.pth为训练后的model文件。  
4.test-6-19_tSNE.png为对词向量用t-SNE方法降维并可视化后的文件。  
5.test-6-19_embedding.html为对每个词向量找前五个最相似向量的列表。  
6.test-6-19_pred.html为用word2vec进行词预测的结果。  
## 二、项目介绍  
### 1.实现功能  
#### (1)EmbeddingDataset类：  
继承torch中的Dataset类。输入数据集的地址，加载数据集。  
#### (2)word2one_hot函数：  
输入word和word列表，输出one-hot编码。  
#### (3)EmbeddingModel类：  
继承torch中的Module类。实现了forward函数和train函数。利用自己写的代码实现了word2vec中的sgd算法。  
#### (4)Embedding_Visualization函数：  
输入EmbeddingModel类和想输出的地址。先用t-SNE方法将词向量转换到二维并可视化。再计算词向量间的皮尔逊相关系数，用pyechart制作词向量相似词列表。
#### (5)word2vec_pred函数：  
输入EmbeddingModel类和想输出的地址。遍历EmbeddingModel中的词列表，在给定地址打印所有单词预测最大值表格。  
### 2.运行结果  
见test-6-19文件夹下的"test-6-19.pth"、"test-6-19_tSNE.png"、"test-6-19_embedding.html"、"test-6-19_pred.html"。  
### 3.使用技术  
(1)torch包  
(2)sklearn中的TSNE功能  
(3)利用pyecharts包打印表格  
### 4.难点及解决方法  
#### (1)在数据集过大时训练耗时较长
- [x] 利用torch.cuda函数将矩阵运算转移到GPU中。
- [ ] 使用负采样，word2vec反向更新矩阵时每次只部分更新w2矩阵。
- [ ] Hierarchical Softmax。
