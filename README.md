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
(1)EmbeddingDataset类：  
继承torch中的Dataset类。输入数据集的地址，加载数据集。  
(2)word2one_hot函数：  
输入word和word列表，输出one-hot编码。  
(3)EmbeddingModel类：  

(4)Embedding_Visualization函数：  

(5)word2vec_pred函数：  

### 2.运行结果  

### 3.使用技术  

### 4.难点及解决方法  
(1)在数据集过大时训练耗时较长
- [x] 利用torch.cuda函数将矩阵运算转移到GPU中。
- [ ] word2vec过程中使用负采样，每次只部分更新w2矩阵。
- [ ] Hierarchical Softmax。
