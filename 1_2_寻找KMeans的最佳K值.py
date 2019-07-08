# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import metrics


# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False


plt.figure(figsize=(12, 12))


# 生成测试用的点
n_samples = 1500
random_state = 123
X, y = make_blobs(n_samples=n_samples, centers=10, random_state=random_state)

plt.subplot(221)
plt.title('生成测试用的点', fontsize=20)
plt.scatter(X[:, 0], X[:, 1], c=y)


n_list = []  # 保存 n 的值
s_list = []  # 保存 s 的值
best_dict = {  # 记录最佳值
    'n': 2,
    's': 0,
    'y_pred': None,
}
for n in range(2, 21):
    model = KMeans(n_clusters=n, random_state=random_state).fit(X)

    labels = model.labels_
    s = metrics.calinski_harabaz_score(X, labels)  # 评估
    print(n, s)

    # 保存值
    n_list.append(n)
    s_list.append(s)

    # 记录最佳值
    if s > best_dict['s']:
        y_pred = model.predict(X)
        best_dict['n'] = n
        best_dict['s'] = s
        best_dict['y_pred'] = y_pred

plt.subplot(222)
plt.title('KMeans聚类 最佳结果', fontsize=20)
plt.scatter(X[:, 0], X[:, 1], c=best_dict['y_pred'])

plt.subplot(223)
plt.title('得分(s) - k值(n)', fontsize=20)
plt.xlabel('n', fontsize=20)
plt.ylabel('s', fontsize=20)
plt.plot(n_list, s_list, 'b.-')
best_n = best_dict['n']
best_s = best_dict['s']
plt.scatter(best_n, best_s, c='r', marker='o', s=50)
plt.text(best_n, best_s, '({}, {})'.format(best_n, best_s), fontsize=14)

plt.show()

print('best n is', best_dict['n'])
