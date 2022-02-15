import os
import numpy as np
import pandas as pd
import xlwt
import xlrd
import xlutils.copy
import time

from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from numpy.compat import basestring
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoLars, BayesianRidge, TweedieRegressor, \
    RANSACRegressor, SGDRegressor, PassiveAggressiveRegressor, OrthogonalMatchingPursuit, ElasticNet
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, \
    ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor, XGBRFRegressor

''' 模型评估： 样本个数<10采用LeaveOneOut，否则10-fold 交叉验证法。评估指标为RMSE，MAPE，R-Squared，MAE，MSE。
    保存最优模型时所对应的RMSE，MAPE，R-Squared，MAE，MSE,，以及训练时间time, 超参数params'''

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def traverse(f):
    # print("正在获取数据文件..........")
    global filename
    fs = os.listdir(f)  # 获取当前目录所有文件列表
    for fitem in fs:
        tmp_path = os.path.join(f, fitem)
        if not os.path.isdir(tmp_path):
            filename.append(tmp_path)  # 如果不是文件夹则把路径加入filename
        else:
            traverse(tmp_path)  # 如果是文件夹则继续递归遍历


### 处理缺失值
def data_missing(source_data):
    # print("\n正在处理缺失值..........")
    # df.dropna()函数用于删除dataframe数据中的缺失数据，即删除NaN数据
    source_data.dropna(axis=0 )  # 删除至少缺少一个元素的行并修改原数据
    np_data = source_data.values  # 返回ndarray类型的对象,ndarray类型即numpy的 N 维数组对象,通常将DataFrame类型的数据转换为ndarray类型的比较方便操作。
    m = n = 0

    while m < np_data.shape[0]:  # 1、shape输出数组的形状；shape[]矩阵的长度，0行数 1列数；2、shape=(r,c)  改变数组的形状行c列
        while n < np_data.shape[1]:
            if np_data[m][n] == "？" or np_data[m][n] == '?' or np_data[m][n] == 'unknown' or np_data[m][n] == ' ':
                np_data = np.delete(np_data, m, axis=0)  # 如果存在空值，则删除所在空值的行，axis=0指行，axis=1指列
                # print("已删除含缺失值的一行",np_data)
                n = 0
                if m >= np_data.shape[0]:
                    break
            else:
                n = n + 1

        m = m + 1
        n = 0
    # print("缺失值处理完成..........\n")
    return np_data


### 处理类别型特征
'''
方法：sklearn 包的 OneHotEncoder
需要注意的是，sklearn库中的 OneHotEncoder() 方法只能处理数值型变量。
如果是字符型数据，需要先对其使用 LabelEncoder() 转换为数值数据，再使用 OneHotEncoder() 进行独热编码处理，
并且需要自行在原数据集中删去进行独热编码处理的原变量  
'''
def data_encoding(X, y):
    # print("正在处理类别型特征........")
    le = LabelEncoder()
    ohe = OneHotEncoder(sparse=False)
    columns = X.shape[1]  # 数据集列数
    k = 0
    t = columns
    while k < t:  # 对类别型特征进行onehot编码
        temp = X[:, k]  # 行向量
        str_flag = 0
        for j in range(X.shape[0]):
            if isinstance(X[j][k], basestring):  # 判断当前元素（数组中第j行k列的数据）是不是string类型
                str_flag = 1
                break
        if str_flag == 1:
            # print("原始某类别型特征列:", temp)
            temp = le.fit_transform(temp.astype(str))  # fit来学习编码；transform进行编码；fit_transform方法是先调用fit然后调用transform
            # print("LabelEncoder之后的某列:", temp)
            temp = temp.reshape(len(temp), 1)  # 改变输出形状：reshape((页数,)行数,列数)将矩阵或数组变成len(temp)行1列
            temp = ohe.fit_transform(temp)
            # print("OneHotEncoder之后的某列:", temp)
            X = np.delete(X, k, axis=1)  # 删除原X的第k列
            index = temp.shape[1] + t - 2  #
            for item in range(temp.shape[1]):  # 插入onehot编码位的列
                X = np.insert(X, item + t - 1, X[:, item], axis=1)  # numpy.insert()将X[:, item]插入到数组X的第“item + t - 1”列
            while index > k + temp.shape[1] - 1:  # 数据集后移onehot编码位
                X[:, index] = X[:, index - temp.shape[1]]
                index = index - 1
            for item in range(temp.shape[1]):
                X[:, k + item] = temp[:, item].astype('int')
            k = k + temp.shape[1]
            t = t + temp.shape[1] - 1
            # print("某类别型特征列编码后的X：", X)
        else:
            k = k + 1
    # print("所有类别型特征处理完成........\n")

    # 数据归一化
    '''StandarScaler，它可以在训练数据集上做了标准转换操作之后，把相同的转换应用到测试训练集中。
    这是相当好的一个功能。可以对训练数据，测试数据应用相同的转换，以后有新的数据进来也可以直接调用，不用再重新把数据放在一起再计算一次了
    调用fit方法，根据已有的训练数据创建一个标准化的转换器,使用上面这个转换器去调用transform方法转换训练数据x,对于新样本要想得到相同的转换只需直接调用这个训练好的transform方法
    标准化的过程为两步：去均值的中心化（均值变为0）；方差的规模化（方差变为1）
    '''
    # print(X, "\n正在进行数据归一化........\n")
    X = X.astype('float')  # 条件属性
    y = y.astype('float')  # 决策属性
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1)).ravel()
    # print(X, y)
    # print("数据归一化处理完成........\n")
    return X, y

'''二、超参数优化 训练eval 三、最小化目标函数 loss=score(即RMSE) '''
def hyperopt_train(params):
    global X, y
    '''在训练之前，构建每次eval对应超参数的自动回归模型'''
    algorithm = params['type']
    del params['type']
    if algorithm == 0:
        model = LinearRegression(**params)
    elif algorithm == 1:
        model = Lasso(**params)
    elif algorithm == 2:
        model = Ridge(**params)
    elif algorithm == 3:
        model = LassoLars(**params)
    elif algorithm == 4:
        model = BayesianRidge(**params)
    elif algorithm == 5:
        lr = LinearRegression()
        params['base_estimator'] = lr
        model = RANSACRegressor(**params)
    elif algorithm == 6:
        model = KNeighborsRegressor(**params)
    elif algorithm == 7:
        model = GaussianProcessRegressor(**params)
    elif algorithm == 8:
        model = DecisionTreeRegressor(**params)
    elif algorithm == 9:
        base = {}
        base['criterion'] = params['criterion']
        del params['criterion']
        base['splitter'] = params['splitter']
        del params['splitter']
        base['max_depth'] = params['max_depth']
        del params['max_depth']
        base['min_samples_split'] = params['min_samples_split']
        del params['min_samples_split']
        base['min_samples_leaf'] = params['min_samples_leaf']
        del params['min_samples_leaf']
        base['max_features'] = params['max_features']
        del params['max_features']
        base_estimator = DecisionTreeRegressor(**base)
        params['base_estimator'] = base_estimator
        model = AdaBoostRegressor(**params)
    elif algorithm == 10:
        model = RandomForestRegressor(**params)
    elif algorithm == 11:
        model = MLPRegressor(**params)
    elif algorithm == 12:
        model = SGDRegressor(**params)
    elif algorithm == 13:
        model = SVR(**params)
    elif algorithm == 14:
        model = LinearSVR(**params)
    elif algorithm == 15:
        model = KernelRidge(**params)
    elif algorithm == 16:
        model = ElasticNet(**params)
    elif algorithm == 17:
        model = PassiveAggressiveRegressor(**params)
    else:
        return 0
    '''使用sklearn中的各种交叉验证方法随机划分训练集和测试集（样本个数<10，采用LeaveOneOut法，否则Kfold法），
    +训练前定义好超参数的自动回归模型
    求测试集预测值与真实值之间的均方根误差RMSE 作为算法的评估结果score'''
    try:
        #交叉验证
        RMSE = 0
        MAE = 0
        MSE = 0
        R2 = 0
        MAPE = 0
        k = 0

        if X.shape[0] < 10:
            from sklearn.model_selection import LeaveOneOut
            loo = LeaveOneOut()
            for train_index, test_index in loo.split(X):
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]
                model.fit(X_train, y_train)      #训练模型
                y_pred = model.predict(X_test)

                # 评估回归性能
                from sklearn import metrics
                RMSE = RMSE + np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                MAE = MAE + metrics.mean_absolute_error(y_test, y_pred)
                MSE = MSE + metrics.mean_squared_error(y_test, y_pred)
                R2 = R2 + metrics.r2_score(y_test, y_pred)
                MAPE = MAPE + mape(y_test, y_pred)
                k = k + 1

        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=10)
            for train_index, test_index in kf.split(X):
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # 评估回归性能
                from sklearn import metrics
                # 评估回归性能
                RMSE = RMSE + np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                MAE = MAE + metrics.mean_absolute_error(y_test, y_pred)
                MSE = MSE + metrics.mean_squared_error(y_test, y_pred)
                R2 = R2 + metrics.r2_score(y_test, y_pred)
                MAPE = MAPE + mape(y_test, y_pred)
                k = k + 1

        RMSE = RMSE / k
        MAE = MAE / k
        MSE = MSE / k
        R2 = R2 / k
        MAPE = MAPE / k
    except Exception as e:
        print(e)
        RMSE = float("inf")
        MAE = float("inf")
        MSE = float("inf")
        R2 = float("-inf")
        MAPE = float("inf")
    return RMSE, MAE, MSE, R2, MAPE


'''二、超参数优化 训练eval 三、最小化目标函数 loss=score(即RMSE) 四、结果的历史数据'''
def f(params):
    global best_RMSE, best_MAE, best_MSE, best_R2, best_MAPE, best_time, best_params
    time_start = time.time()
    RMSE_score, MAE, MSE, R2, MAPE = hyperopt_train(params)
    time_end = time.time()
    if RMSE_score < best_RMSE:  # 因为本问题score=loss,所以score值越小越好
        # print('score: ', score, 'using ', params)
        best_time = time_end - time_start
        best_params = params
        best_RMSE = RMSE_score
        best_MAE = MAE
        best_MSE = MSE
        best_R2 = R2
        best_MAPE = MAPE
    # 对于STATUS_OK会统计它的loss值，而对于STATUS_FALL的返回则会忽略
    return {'loss': RMSE_score, 'status': STATUS_OK}


if __name__ == "__main__":
    filename = []         #全局变量，数据集的名称集合
    traverse("data/general_dataset/")
    # traverse("data/material_dataset/")

    '''总共n份数据集，每份数据集对应18个回归模型，每个模型对应着best_RMSE, best_MAE, best_MSE, best_R2, best_MAPE, best_time, best_params。'''
    # datasets_evaluation = []    # 总共n份数据集。datasets_evaluation.append(dataset_evaluation)

    # xlwt写入Excel
    workbook = xlwt.Workbook()
    worksheet_RMSE = workbook.add_sheet('RMSE')
    worksheet_MAE = workbook.add_sheet('MAE')
    worksheet_MSE = workbook.add_sheet('MSE')
    worksheet_R2 = workbook.add_sheet('R2')
    worksheet_MAPE = workbook.add_sheet('MAPE')
    worksheet_time = workbook.add_sheet('time')
    worksheet_params = workbook.add_sheet('params')
    # workbook.save("result_Loo_kFold_multi/model_evaluations_in_material.xls")
    workbook.save("result_Loo_kFold_multi/model_evaluations_in_general.xls")

    try:
        for row in range(len(filename)):
            # dataset_evaluation = []  # 每份数据集对应18个回归模型。dataset_evaluation.append(models_evaluation)
            '''性能评估结果 1、创建excel及其sheet 2、分别将第row个数据集的路径分别写入3个sheet的第row行第1列'''
            # print("当前正在处理的数据集是:"+filename[row])
            # excel_data = xlrd.open_workbook('result_Loo_kFold_multi/model_evaluations_in_material.xls')
            excel_data = xlrd.open_workbook('result_Loo_kFold_multi/model_evaluations_in_general.xls')

            ws = xlutils.copy.copy(excel_data)
            worksheet_RMSE = ws.get_sheet(0)
            worksheet_MAE = ws.get_sheet(1)
            worksheet_MSE = ws.get_sheet(2)
            worksheet_R2 = ws.get_sheet(3)
            worksheet_MAPE = ws.get_sheet(4)
            worksheet_time = ws.get_sheet(5)
            worksheet_params = ws.get_sheet(6)
            # 写入数据集名
            worksheet_RMSE.write(row, 0, filename[row])
            worksheet_MAE.write(row, 0, filename[row])
            worksheet_MSE.write(row, 0, filename[row])
            worksheet_R2.write(row, 0, filename[row])
            worksheet_MAPE.write(row, 0, filename[row])
            worksheet_time.write(row, 0, filename[row])
            worksheet_params.write(row, 0, filename[row])

            ####文件读取：读取原数据集
            source_data = pd.read_excel(filename[row])
            # print('原始数据：', len(source_data), source_data.shape[1])

            '''数据预处理之进行缺失值处理'''
            np_data = data_missing(source_data)

            '''数据预处理之进行类别型特征处理及数据归一化'''
            X = np_data[:, 0:np_data.shape[1] - 1]  # 条件属性  或者np_data[:, :-1]
            y = np_data[:, np_data.shape[1] - 1]  # 决策属性
            '''如果某数据集中的样本数量>10000或者特征数量>50则舍弃该数据集 '''
            # if X.shape[0] > 10000 or X.shape[1] > 50:
            #     continue    #终止本次循环
            X, y = data_encoding(X, y)
            # print("数据预处理后的条件属性：\n", X, "\n数据预处理后的决策属性：", y)
            # print("\n数据预处理完成............\n")

            # Hyperopt 自动化超参数调优——贝叶斯优化，有四个组成部分：域空间、优化算法、目标函数、结果的历史数据。'''
            '''1、定义域空间（待搜索的超参数值）。不同模型，模型并不是其所有参数都需要调优，因此采用不同的分布来定义每个参数的域空间
            hp.choice(lable,p_options)以一定的概率返回一个p_options的一个选项，这个选项使得函数在搜索过程中对每个选项的可能性不均匀.其中options应是 python 列表或元组。
            hp.uniform(lable,low,high)参数在low和high之间均匀分布。
            hp.normal(label, mu, sigma) 其中mu和sigma分别是均值和标准差。
            '''
            # print("定义参数空间.......")
            lr = {
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'normalize': hp.choice('normalize', [True, False]),
                'type': 0
            }
            lasso = {
                'alpha': hp.uniform('alpha', 0, 10),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'normalize': hp.choice('normalize', [True, False]),
                'positive': hp.choice('positive', [True, False]),
                'selection': hp.choice('selection',['cyclic', 'random']),
                'type': 1
            }
            ridge = {
                'alpha': hp.uniform('alpha', 0, 50),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'normalize': hp.choice('normalize', [True, False]),
                'tol': hp.uniform('tol', 1e-5, 1e-1),
                'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
                'type': 2
            }
            lassolars = {
                'alpha': hp.uniform('alpha', 0.0, 50.0),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'verbose': hp.choice('verbose', [True, False]),
                'normalize': hp.choice('normalize', [True, False]),
                'fit_path': hp.choice('fit_path', [True, False]),
                'positive': hp.choice('positive', [True, False]),
                'type': 3
            }
            bayesine = {
                'tol': hp.uniform('tol', 1e-6, 1e-2),
                'alpha_1': hp.uniform('alpha_1', 1e-8, 1e-3),
                'alpha_2': hp.uniform('alpha_2', 1e-8, 1e-3),
                'lambda_1': hp.uniform('lambda_1', 1e-8, 1e-3),
                'lambda_2': hp.uniform('lambda_2', 1e-8, 1e-3),
                'compute_score': hp.choice('compute_score', [True, False]),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'normalize': hp.choice('normalize', [True, False]),
                'verbose': hp.choice('verbose', [True, False]),
                'type': 4
            }
            # 需要设置基学习器
            ransac = {
                'max_trials': hp.uniform('max_trials', 50, 150),
                'min_samples': hp.uniform('min_samples', 0.0, 1.0),
                'loss': hp.choice('loss', ['absolute_loss', 'squared_loss']),
                'residual_threshold': hp.uniform('residual_threshold', 1, 10),
                'type': 5
            }
            nearest_neighbors = {
                'n_neighbors': hp.choice('n_neighbors', range(1, 20)),
                'weights': hp.choice('weights' ,['uniform', 'distance']),
                'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'leaf_size': hp.choice('leaf_size', range(20, 50)),
                'p': hp.choice('p', [1, 2]),
                'type': 6
            }
            gaussian_process = {
                'alpha': hp.uniform('alpha', 1e-12, 1e-6),
                'n_restarts_optimizer': hp.choice('n_restarts_optimizer', range(0, 10)),
                'normalize_y': hp.choice('normalize_y', [True, False]),
                'type': 7
            }
            if X.shape[1] > 15:
                max_features = 15
            else:
                max_features = X.shape[1]
            decision_tree = {
                'criterion': hp.choice('criterion', ["mse", "friedman_mse", "mae"]),
                'splitter': hp.choice('splitter', ["best", "random"]),
                'max_depth': hp.choice('max_depth', range(1, 30)),
                'min_samples_split': hp.choice('min_samples_split', range(2, 30)),
                'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 30)),
                'max_features': hp.uniform('max_features', 0, 1),
                'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 2),
                'type': 8
            }
            adaboost = {
                'criterion': hp.choice('criterion', ["mse", "friedman_mse", "mae"]),
                'splitter': hp.choice('splitter', ["best", "random"]),
                'max_depth': hp.choice('max_depth', range(1, 30)),
                'min_samples_split': hp.choice('min_samples_split', range(2, 30)),
                'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 30)),
                'max_features': hp.uniform('max_features', 0, 1),
                'n_estimators': hp.choice('n_estimators', range(2, 500)),
                'learning_rate': hp.uniform('learning_rate', 0.001, 5),
                'loss': hp.choice('loss', ['linear', 'square', 'exponential']),
                'type': 9
            }
            random_forest = {
                'n_estimators': hp.choice('n_estimators', range(2, 500)),
                'criterion': hp.choice('criterion', ["mse", "mae"]),
                'max_depth': hp.choice('max_depth', range(1, 30)),
                'min_samples_split': hp.choice('min_samples_split', range(2, 30)),
                'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 30)),
                'max_features': hp.uniform('max_features', 0, 1),
                'bootstrap': hp.choice('bootstrap', [True, False]),
                'oob_score': hp.choice('oob_score', [True, False]),
                'warm_start': hp.choice('warm_start', [True, False]),
                'type': 10
            }
            mlp = {
                'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
                'solver': hp.choice('solver', ['lbfgs', 'sgd', 'adam']),
                'alpha': hp.uniform('alpha', 1e-7, 0.1),
                'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
                'learning_rate_init': hp.uniform('learning_rate_init', 1e-5, 0.1),
                'power_t': hp.uniform('power_t', 0.0, 1.0),
                'momentum': hp.uniform('momentum', 0.0, 1.0),
                'validation_fraction': hp.uniform('validation_fraction', 0.1, 0.3),
                'beta_1': hp.uniform('beta_1', 0.0, 1.0),
                'beta_2': hp.uniform('beta_2', 0.0, 1.0),
                'epsilon': hp.uniform('epsilon', 1e-9, 1e-7),
                'hidden_layer_sizes': hp.choice('hidden_layer_sizes', range(1, 300)),
                'type': 11
            }
            sgd = {
                'loss': hp.choice('loss', ['squared_loss', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
                'penalty': hp.choice('penalty', ['l2', 'l1', 'elasticnet']),
                'alpha': hp.uniform('alpha', 1e-6, 1e-2),
                'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'tol': hp.uniform('tol', 1e-5, 1e-1),
                'epsilon': hp.uniform('epsion', 1e-5, 1e-1),
                'learning_rate': hp.choice('learning_rate', ['constant', 'optimal', 'invscaling']),
                'eta0': hp.uniform('eta0', 1e-6, 1),
                'power_t': hp.uniform('power_t', 1e-5, 1),
                'average': hp.choice('average', [1, 0]),
                'type': 12
            }
            svr = {
                'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                'degree': hp.choice('degree', [1, 2, 3, 4, 5]),
                'gamma': hp.uniform('gamma', 1e-5, 100),
                'C': hp.uniform('C', 1e-5, 200),
                'tol': hp.uniform('tol', 1e-5, 1e-1),
                'epsilon': hp.uniform('epsilon', 0, 1),
                'shrinking': hp.choice('shrinking', [True, False]),
                'max_iter': 10000,
                'type': 13
            }
            linearsvr = {
                'epsilon': hp.uniform('epsilon', 0, 1),
                'tol': hp.uniform('tol', 1e-5, 1e-1),
                'loss': hp.choice('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']),
                'dual': hp.choice('dual', [True, False]),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'C': hp.uniform('C', 1e-5, 200),
                'max_iter': 10000,
                'type': 14
            }
            kernel_ridge = {
                'alpha': hp.uniform('alpha', 1e-3, 50),
                'kernel': hp.choice('kernel', ["linear", "rbf", "sigmoid", "poly"]),
                'gamma': hp.uniform('gamma', 1e-5, 100),
                'degree': hp.uniform('degree', 1, 10),
                'coef0': hp.uniform('coef0', 1, 10),
                'type': 15
            }
            elasticnet = {
                'alpha': hp.uniform('alpha', 0, 5),
                'l1_ratio': hp.uniform('l1_ratio', 0, 1),
                'normalize': hp.choice('normalize', [True, False]),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'tol': hp.uniform('tol', 1e-6, 1e-2),
                'type': 16
            }
            pa = {
                'C': hp.uniform('C', 0, 100),
                'fit_intercept': hp.choice('fit_intercept', [True, False]),
                'tol': hp.uniform('tol', 1e-4, 1e-2),
                'loss': hp.choice('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']),
                'type': 17
            }
            '''二、优化算法：在Hyperopt中创建优化算法只需要一行代码，使用树形Parzen评估器，简称TPE'''
            '''三、目标函数：（带超参数的机器学习模型的验证误差），本问题中是将回归模型的预测损失RMSE作为算法性能之一，也作为超参数优化的loss'''
            '''四、Trials用来记录每次eval的时候，具体使用了什么参数以及相关的返回值'''
            '''函数fmin返回一个python字典
            1、首先接收一个函数fn来最小化，函数fn可以是任何有效的带有返回值的函数(回归评价指标MSE、RMSE、MAE、R-Squared)
            2、第二个参数space指定了超参数的搜索空间（search space）
            3、参数algo接收一个搜索算法（serach algorithm）
            4、第四个参数指定fmin函数执行的最大次数max_evals
            5、Trials允许我们存储每一时间步长（time step）所存储的信息。然后我们可以输出这些函数在给定时间步长上对给定参数的求值。 
            '''
            # models_evalution = []  # 每个回归算法都对应着best_RMSE, best_MAE, best_MSE, best_R2, best_time, best_params。
            for j in range(18):
                best_RMSE = float("inf")  # RMSE值越小越好，初始值设置为无穷大说明模型最差
                best_MAE = float("inf")  # MAE值越小越好
                best_MSE = float("inf")  # MSE值越小越好
                best_R2 = float("-inf")  # R2取值区间为[负无穷，1]。R2值越大越好.表示模型拟合效果越好，最优解为1
                best_MAPE = float("inf")  # MAPE值越小越好
                best_time = 0
                best_params = {}
                trials = Trials()
                try:
                    if j == 0:
                        #f(params)的返回值是dict,包括loss和status。max_evals最大优化
                        fmin(f, lr, algo=tpe.suggest, max_evals=10, trials=trials)
                    elif j == 1:
                        fmin(f, lasso, algo=tpe.suggest, max_evals=100, trials=trials)
                    elif j == 2:
                        fmin(f, ridge, algo=tpe.suggest, max_evals=100, trials=trials)
                    elif j == 3:
                        fmin(f, lassolars, algo=tpe.suggest, max_evals=150, trials=trials)
                    elif j == 4:
                        fmin(f, bayesine, algo=tpe.suggest, max_evals=200, trials=trials)
                    elif j == 5:
                        fmin(f, ransac, algo=tpe.suggest, max_evals=150, trials=trials)
                    elif j == 6:
                        fmin(f, nearest_neighbors, algo=tpe.suggest, max_evals=200, trials=trials)
                    elif j == 7:
                        fmin(f, gaussian_process, algo=tpe.suggest, max_evals=150, trials=trials)
                    elif j == 8:
                        fmin(f, decision_tree, algo=tpe.suggest, max_evals=200, trials=trials)
                    elif j == 9:
                        fmin(f, adaboost, algo=tpe.suggest, max_evals=150, trials=trials)
                    elif j == 10:
                        fmin(f, random_forest, algo=tpe.suggest, max_evals=250, trials=trials)
                    elif j == 11:
                        fmin(f, mlp, algo=tpe.suggest, max_evals=400, trials=trials)
                    elif j == 12:
                        fmin(f, sgd, algo=tpe.suggest, max_evals=250, trials=trials)
                    elif j == 13:
                        fmin(f, svr, algo=tpe.suggest, max_evals=200, trials=trials)
                    elif j == 14:
                        fmin(f, linearsvr, algo=tpe.suggest, max_evals=200, trials=trials)
                    elif j == 15:
                        fmin(f, kernel_ridge, algo=tpe.suggest, max_evals=100, trials=trials)
                    elif j == 16:
                        fmin(f, elasticnet, algo=tpe.suggest, max_evals=150, trials=trials)
                    elif j == 17:
                        fmin(f, pa, algo=tpe.suggest, max_evals=150, trials=trials)
                except Exception as e:
                    print(e)
                except Warning as w:
                    print(w)

                # 每个模型都对应着best_RMSE, best_MAE, best_MSE, best_R2, best_time, best_params
                # best_evalution = {"RMSE": best_RMSE, "MAE": best_MAE, "MSE": best_MSE, "R2": best_R2, "MAPE": best_MAPE,
                #                   "TIME": best_time,
                #                   "Params": best_params}
                # models_evalution.append(best_evalution)
                # print('第', j, '个回归算法的best_evalution:', best_evalution)
                worksheet_RMSE.write(row, j + 1, best_RMSE)
                worksheet_MAE.write(row, j + 1, best_MAE)
                worksheet_MSE.write(row, j + 1, best_MSE)
                worksheet_R2.write(row, j + 1, best_R2)
                worksheet_MAPE.write(row, j + 1, best_MAPE)
                worksheet_time.write(row, j + 1, best_time)
                worksheet_params.write(row, j + 1, str(best_params))
            # dataset_evaluation.append(models_evalution)
            # ws.save("result_Loo_kFold_multi/model_evaluations_in_material.xls")
            ws.save("result_Loo_kFold_multi/model_evaluations_in_general.xls")
        #     print("数据集:" + filename[row] + '已训练完毕')
        # datasets_evaluation.append(dataset_evaluation)
    except Exception as e:
        1==1
        # print(e)
    print("所有数据集训练完成：",'\n')
    # for k in range(len(datasets_evaluation)):
    #     print(datasets_evaluation[k],'\n')





