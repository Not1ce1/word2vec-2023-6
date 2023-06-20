# word2vec-2023-6  
## 一、test-6-19文件夹   
1.[test-6-19.txt](/test-6-19/test-6-19.txt)数据集为wiki百科数据集中的第一行，共1047个词。   
2.[SkipGram.py](/test-6-19/SkipGram.py)为python代码文件。  
3.[test-6-19.pth](/test-6-19/test-6-19.pth)为训练后的model文件。  
4.[test-6-19_tSNE.png](/test-6-19/test-6-19_tSNE.png)为对词向量(word2vec中的w1矩阵乘以词的one-hot编码)用t-SNE方法降维并可视化后的文件。  
5.[test-6-19_embedding.html](/test-6-19/test-6-19_embedding.html)为对每个词向量找前五个最相似向量的列表。  
6.[test-6-19_pred.html](/test-6-19/test-6-19_pred.html)为用word2vec进行词预测的结果。  
## 二、output_SkipGram文件夹   
1.[test-6-14.txt](/output_SkipGram/test-6-14.txt)为测试数据集。   
2.[SkipGram.pth](/output_SkipGram/SkipGram.pth)为训练后的model文件。  
3.[output_SkipGram_tSNE.png](/output_SkipGram/output_SkipGram_tSNE.png)为对词向量(word2vec中的w1矩阵乘以词的one-hot编码)用t-SNE方法降维并可视化后的文件。  
4.[output_SkipGram_embedding.html](/output_SkipGram/output_SkipGram_embedding.html)为对每个词向量找前五个最相似向量的列表。  
5.[output_SkipGram_pred.html](/output_SkipGram/output_SkipGram_pred.html)为用word2vec进行词预测的结果。  
## 三、项目介绍  
### 1.实现功能  
(见[SkipGram.py](/test-6-19/SkipGram.py)文件)  
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
#### (1)[test-6-19_tSNE.png](/test-6-19/test-6-19_tSNE.png)：  
对词向量用t-SNE方法降维并可视化后的结果。可以看到词向量在二维空间中呈现规律的形状。原因是数据集为wiki百科中的一段话，词之间有较强的逻辑关系，所以词向量在高维空间中极有可能有较强的联系(如线性关系)，而在二维空间中表现为有规律的形状。  
#### (2)[test-6-19_embedding.html](/test-6-19/test-6-19_embedding.html)：  
对每个词向量找前五个最相似向量的列表。可以看到最相似的词中较少出现原文中的直接上下文，体现了word2vec产生的词向量捕获词关系的能力。  
#### (3)[test-6-19_pred.html](/test-6-19/test-6-19_pred.html)：  
对每个词向量找前五个最相似向量的列表。可以看到许多词的预测值只有一个词，概率为100%。经检验模型中的w1和w2矩阵均未出现nan的情况，可能是超参数选择不当导致过拟合。  
### 3.使用技术  
(1)torch包。  
(2)利用sklearn包中的TSNE实现词向量的降维。  
(3)利用pyecharts包打印表格。  
(4)利用scipy包计算词向量之间的皮尔逊相关系数。  
### 4.难点及解决方法  
#### (1)在数据集过大时训练耗时较长
- [x] 利用torch.cuda函数将矩阵运算转移到GPU中。
- [ ] 使用负采样，word2vec反向更新矩阵时每次只部分更新w2矩阵。
- [ ] Hierarchical Softmax。
#### (2)超参数的选择
