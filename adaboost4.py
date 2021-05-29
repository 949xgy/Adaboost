import time

import numpy as np
import pandas as pd


# 获得 特征矩阵 和 标签矩阵
def get_Mat(path):
    """
    函数功能:加载数据函数
    参数说明：
    path: txt文件路径
    返回:
    xMat:特征矩阵
    yMat:标签矩阵
    """
    dataSet = pd.read_table(path, header=None)
    xMat = np.mat(dataSet.iloc[:, :-1].values)  # 从开头获取到 倒数第二列数据   [:2, :2]->取两行两列
    yMat = np.mat(dataSet.iloc[:, -1].values).T  # 获取最后一列数据 并转置      [:3, -1]->取3行最后一列
    return xMat, yMat


# 单层决策树分类函数
def Classify0(xMat, i, Q, S):
    """
    函数功能：单层决策树分类函数
    参数说明:
    xMat: 数据矩阵
    i: 第i列，也就是第几个特征
    Q: 阈值
    S: 标志
    返回:
    re: 分类结果  (n行1列 的数组)
    """
    re = np.ones((xMat.shape[0], 1))  # 初始化re为1的数组

    if S == 'lt':
        re[xMat[:, i] <= Q] = -1    # 如果小于阈值,则赋值为-1
    else:
        re[xMat[:, i] > Q] = -1     # 如果大于阈值,则赋值为-1

    return re


# 找到数据集上最佳的单层决策树
def get_Stump(xMat, yMat, D):
    """
    函数功能：找到数据集上最佳的单层决策树
    参数说明:
        xMat：特征矩阵
        yMat：标签矩阵
        D：样本权重
    返回:
        bestStump：最佳单层决策树信息
        minE：最小 分类误差率
        bestClas：最佳的分类结果
    """
    m, n = xMat.shape  # m为样本个数，n为特征数
    Steps = 10  # 初始化一个步数
    bestStump = {}  # 用字典形式来储存树桩信息，最佳单层决策树信息
    bestClas = np.mat(np.zeros((m, 1)))  # 初始化分类结果为0 的矩阵
    minE = np.inf  # 最小分类误差率 初始化为正无穷大

    for i in range(n):                      # 遍历所有特征
        Min = xMat[:, i].min()              # 找到特征中最小值
        Max = xMat[:, i].max()              # 找到特征中最大值
        stepSize = (Max - Min) / Steps      # 计算步长
        for j in range(-1, int(Steps) + 1):
            for S in ['lt', 'gt']:          # 大于和小于的情况，均遍历。lt:less than，gt:greater than
                Q = (Min + j * stepSize)    # 计算阈值

                re = Classify0(xMat, i, Q, S)  # 计算分类结果  哪些+1 哪些-1

                err = np.mat(np.ones((m, 1)))  # 初始化 误差矩阵为 1

                err[re == yMat] = 0  # 分类正确的,赋值为0

                eca = D.T * err  # 计算分类误差率  (即为分类错误的样本数据的 权值 相加)

                if eca < minE:  # 找到误差最小的分类方式
                    minE = eca          # 保存 最小 分类误差率
                    bestClas = re.copy()  # 保存分类结果  哪些+1 哪些-1
                    bestStump['特征列'] = i
                    bestStump['阈值'] = Q
                    bestStump['标志'] = S
    return bestStump, minE, bestClas


# AdaBoost 训练函数
def Ada_train(xMat, yMat, maxC=40):
    """
    函数功能：训练弱分类器 函数
    参数说明:
        xMat：特征矩阵
        yMat：标签矩阵
        maxC：若分类器总数
    返回:
        weakClass:所有的最佳弱分类器的信息
    """

    weakClass = []                          # 用来存储最佳单层决策树，最佳弱分类器
    err = []                                # 存储权重误差，里面保存所有的最佳弱分类器的权重误差
    m = xMat.shape[0]                       # 样本个数
    D = np.mat(np.ones((m, 1)) / m)         # 初始化权重 都相同
    aggClass = np.mat(np.zeros((m, 1)))     # 更新累计分类器 类别估计值

    for i in range(maxC):                   # 循环训练 弱分类器

        # 寻找最佳的单层决策树 stump：最佳单层决策树信息 error:分类误差 bestClas:保存分类结果  哪些+1 哪些-1
        Stump, error, bestClas = get_Stump(xMat, yMat, D)

        alpha = float(0.5 * np.log((1 - error) / max(error, 1e-16)))   # 根据公式 计算弱分类器权重alpha  因为error当分子 所以做max处理

        Stump['alpha'] = np.round(alpha, 6)  # 存储弱分类器的权重,保留六位小数
        weakClass.append(Stump)  # 存储单层决策树，里面保存着所有的最佳弱分类器
        err.append(error)  # 存储权重误差，里面保存所有的最佳弱分类器的权重误差

        expon = np.multiply(-1 * alpha * yMat, bestClas)  # 根据公式计算e的指数项  np.multiply-->对应相乘
        D = np.multiply(D, np.exp(expon))  # 根据公式 计算D
        D = D / D.sum()  # 根据样本权重公式，更新样本权重，D.sum()规范化因子

        aggClass += alpha * bestClas  # 更新累计分类器 类别估计值 (意思就是 半成品强分类器)bestClas相当于公式的 G(x)分段函数

        # xx = np.sign(aggClass) != yMat   结果是[false,true] bool型矩阵
        aggErr = np.multiply(np.sign(aggClass) != yMat, np.ones((m, 1)))  # 计算误差    np.sign(aggClass)半成品分类器累加和

        errRate = aggErr.sum() / m

        print('\r分类器总数目{0}个  \t正在训练第{1}个 ---->当前进度:{2:3.3f}%'.format(maxC, i + 1, ((i + 1) * 100 / maxC)), end='\t')

        if errRate == 0:
            break  # 误差为0，退出循环

    for i in range(maxC):
        print('\n第{0}个 切分特征: {1}, \t阈值:{2:.2f},\t标志:{3}, \talpha: {4} \t 权重误差: {5},'.format(i + 1, weakClass[i]['特征列'],
                                                                                            weakClass[i]['阈值'],
                                                                                            weakClass[i]['标志'],
                                                                                            weakClass[i]['alpha'],
                                                                                            err[i]), end='')
    #print("")

    return weakClass        # 返回 保存的 所有的最佳弱分类器的信息


# AdaBoost分类函数
def AdaClassify(data, weakClass):
    """
    函数功能：AdaBoost分类函数
    参数说明：
    data: 待分类样例
    weakClass:训练好的分类器
    返回:
    分类结果
    """
    dataMat = np.mat(data)          # 转为矩阵类型
    m = dataMat.shape[0]            # 数据总数
    aggClass = np.mat(np.zeros((m, 1))) # 初始化累计分类器 类别估计值
    for i in range(len(weakClass)):  # 遍历所有分类器，进行分类
        classEst = Classify0(dataMat,
                             weakClass[i]['特征列'],
                             weakClass[i]['阈值'],
                             weakClass[i]['标志'])
        aggClass += weakClass[i]['alpha'] * classEst
        # print(aggClass)
    return np.sign(aggClass)


# 开始训练 预测
def calAcc(train_xMat, train_yMat, test_xMat, test_yMat, maxC=40):
    """
    函数功能：开始 训练、预测 函数
    参数说明：
    train_xMat,train_yMat: 训练数据集 和标签
    test_xMat, test_yMat:测试数据集 和 标签
    maxC :  弱分类器总数
    返回:
    train_acc, test_acc: 训练集准确率,测试集准确率
    """

    # 训练 弱分类器
    print('开始训练弱分类器...')
    time_1 = time.time()

    weakClass = Ada_train(train_xMat, train_yMat, maxC)  # 返回最佳单层决策树(最佳弱分类器)集合,

    time_2 = time.time()
    print('训练{0}个 弱分类器花费时间: {1:f} s'.format(maxC, (time_2 - time_1)))


    # 预测 训练集
    yhat = AdaClassify(train_xMat, weakClass)
    train_re = 0
    m = train_xMat.shape[0]
    for i in range(m):
        if yhat[i] == train_yMat[i]:
            train_re += 1
    train_acc = train_re / m


    # 预测 测试集
    print('开始分类 测试数据...')
    time_3 = time.time()
    yhat = AdaClassify(test_xMat, weakClass)     # 返回 yhat 预测的 分类 标签结果
    time_4 = time.time()
    print('测试花费时间: {0:f} s\n'.format((time_4 - time_3)))

    test_re = 0
    n = test_xMat.shape[0]
    for i in range(n):
        if yhat[i] == test_yMat[i]:
            test_re += 1
    test_acc = test_re / n

    # print(weakClass)   #弱分类器 集合 每次迭代 找到的 最小权重误差的 分类器 也就是在哪怎么切分
    return train_acc, test_acc



if __name__ == '__main__':

    # Cycles = [1, 10, 20, 50, 200, 500]  # 弱分类器个数
    Cycles = [1,10,55,200]
    train_acc = []  # 训练集准确率
    test_acc = []  # 测试集准确率

    print('读取数据...')
    time_1 = time.time()

    train_xMat, train_yMat = get_Mat('horseColicTraining2.txt')  # 读取 训练数据
    test_xMat, test_yMat = get_Mat('horseColicTest2.txt')  # 读取 测试数据

    time_2 = time.time()
    print('读取数据花费时间: %f s' % (time_2 - time_1))

    for maxC in Cycles:     # 循环测试多组分类器
        a, b = calAcc(train_xMat, train_yMat, test_xMat, test_yMat, maxC)  # 返回 a：训练集准确率  b： 测试集准确率
        train_acc.append(round(a * 100, 2))
        test_acc.append(round(b * 100, 2))

    df = pd.DataFrame({'分类器数目': Cycles,  # DataFrame() 是带标签的二维数组
                       '训练集准确率': train_acc,
                       '测试集准确率': test_acc})
    print(df)
