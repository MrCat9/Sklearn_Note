# Sklearn_Note

sklearn 官网  https://scikit-learn.org/stable/index.html

sklearn 中文文档  http://sklearn.apachecn.org/#/





## 目录

#### 1_轻松看懂机器学习十大常用算法

https://www.jianshu.com/p/55a67c12d3e9

#### 2_《机器学习实战》学习笔记

https://blog.csdn.net/c406495762/article/details/75172850

#### 3_sklearn聚类算法评估方法

https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

https://blog.csdn.net/u010159842/article/details/78624135

#### 3_2_寻找KMeans的最佳K值

#### 4_K折交叉验证

StratifiedKFold 和 KFold 的比较 https://www.jianshu.com/p/c84818b56fa0

sklearn.model_selection.KFold https://blog.csdn.net/kancy110/article/details/74910185

#### 5_skimage

skimage-图像基本操作 https://blog.csdn.net/wc781708249/article/details/78368825

#### 6_马尔科夫链

https://blog.csdn.net/bitcarmanlee/article/details/82819860

https://www.cnblogs.com/skyme/p/4651331.html

https://www.jianshu.com/p/a3572391a42d

#### 7_机器学习中的Stacking模型融合

https://blog.csdn.net/xiao2cai3niao/article/details/80571021

#### 8_XGBoost

https://arxiv.org/pdf/1603.02754.pdf

https://xgboost.readthedocs.io/en/latest/

#### 9_sklearn模型评估（分类Classification、聚类Clustering、回归Regression）

https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation

#### 10_自动调参tpot

https://github.com/EpistasisLab/tpot

#### 11_数据预处理`sklearn.preprocessing`

#### 12_系统聚类/层次聚类

```
凝聚法--自下而上
分裂法--自上而下
```

[python实现层次聚类的方法](http://www.zzvips.com/article/225824.html)

[Python中的层次聚类，详细讲解](https://blog.csdn.net/weixin_46211269/article/details/127175675)

#### 13_DBSCAN密度聚类算法

[DBSCAN密度聚类算法（理论+图解+python代码）](https://mp.weixin.qq.com/s?__biz=MzAxMDcyOTQxNA==&mid=2649893103&idx=2&sn=d5066b90962c6c26142a46383d14cc6e&chksm=834d6a46b43ae3501b64806dcd192d6c0a1db55f66c568d47fa73821ba931633845f5d075cb3&scene=27)

[DBSCAN算法可视化](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)

```
DBSCAN（Density-Based Spatial Clustering of Applications with Noise，具有噪声的基于密度的聚类方法）
```

```
k-means聚类算法
优点：
1.算法简单迅速，处理大量数据的时候，效率较高。
缺点：
1.需要给出聚类的类数K。
2.对初始点、异常点敏感。
3.只能处理球形的簇，也就是一个聚成实心的团。

DBSCAN聚类算法
优点：
1. 基于密度定义，能够处理任意形状和大小的簇。
2. 可在聚类的同时发现异常点。
3. 不需要指定要划分的簇的个数。
缺点：
1. 对于输入参数ξ和MinPts敏感，确定参数困难。
2. ξ和MinPts是全局唯一的，当聚类的密度不均匀时，聚类距离相差很大时，聚类效果较差。
3. 当数据量大时，计算密度单元的计算复杂度大。
```

```
DBSCAN聚类算法效果评估：
可以使用轮廓系数。聚类结果的轮廓系数的取值在[-1,1]之间，值越大，说明同类样本相距约近，不同样本相距越远，则聚类效果越好。可以使用sklearn.metrics模块中的silhouette_score()函数计算所有点的平均轮廓系数。
```

```python
from sklearn import metrics  
score = metrics.silhouette_score(X, labels)
```

#### 14_常见聚类算法

```
常见聚类算法：K-means聚类算法，系统/层次聚类算法，DBSCAN密度聚类算法
```

[常见聚类模型](https://blog.csdn.net/qq_45857800/article/details/105126049)

[K-means聚类算法可视化](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

[DBSCAN算法可视化](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)

#### 15_常见的模型评估指标与方法

[模型评估：常见的模型评估指标与方法大全、汇总](https://blog.csdn.net/Bluemoon17/article/details/124323666)