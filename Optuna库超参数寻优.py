import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_excel(r'D:\0文献整理\网络入侵检测\KDD99\vif_after_alpha0.04.xlsx')

X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1412)

'''XGBoost'''
def xgboost_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 1, 1000, 1)
    max_depth = trial.suggest_int('max_depth', 1, 100, 1)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.999, log=True)
    subsample = trial.suggest_float('subsample', 0.05, 0.99)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 300)
    gamma = trial.suggest_float('gamma', 0, 1)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
    reg_lambda = trial.suggest_float('reg_lambda', 0, 1)

    clf = xgb.XGBClassifier(n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            subsample=subsample,
                            min_child_weight=min_child_weight,
                            gamma=gamma,
                            colsample_bytree=colsample_bytree,
                            reg_alpha=reg_alpha,
                            reg_lambda=reg_lambda,
                            random_state=1412)

    # 使用划分好的训练集进行拟合
    clf.fit(X_train, y_train)

    # 使用划分好的验证集进行交叉验证评估
    cv = KFold(n_splits=10, shuffle=True, random_state=1412)
    validation_accuracy = cross_validate(clf, X_test, y_test,
                                         scoring='accuracy',
                                         cv=cv,
                                         verbose=True,
                                         n_jobs=-1,
                                         error_score='raise')

    return np.mean(validation_accuracy['test_score'])

'''贝叶斯优化器'''
def optimizer_optuna(n_trials, algo, objective):
    if algo == 'TPE':
        sampler = optuna.samplers.TPESampler(n_startup_trials=20, n_ei_candidates=30)
    elif algo == 'GP':
        from optuna.integration import SkoptSampler
        import skopt
        sampler = SkoptSampler(skopt_kwargs={'base_estimator': 'GP',
                                             'n_initial_points': 10,
                                             'acq_func': 'EI'})

    study = optuna.create_study(sampler=sampler, direction='maximize')

    scores = []

    for _ in range(n_trials):
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        best_score = study.best_value
        scores.append(best_score)


    best_params = study.best_params

    return best_params, scores

'''使用TPE过程进行贝叶斯优化'''
'''XGBoost迭代中'''
xgboost_best_params, xgboost_scores = optimizer_optuna(1500, 'TPE', xgboost_objective)

'''最优超参数组合'''
results = {
    'XGBoost': (xgboost_best_params, xgboost_scores),
}

for algorithm, (best_params, scores) in results.items():
    print(f'{algorithm} - Best Parameters: {best_params}, Best Scores: {scores}')

'''绘制迭代效果图'''
plt.plot(xgboost_scores, label='XGBoost', marker='.', linestyle='-', color='#2ca02c')

plt.xlabel('贝叶斯优化迭代次数', fontsize=12)
plt.ylabel('准确率', fontsize=12)
plt.title('XGBoost基于TPE过程贝叶斯优化迭代图', fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig(r'D:\0文献整理\网络入侵检测\KDD99\XGBoost基于TPE过程贝叶斯优化迭代图.png', dpi=600)
plt.show()
results = pd.DataFrame(results)
results.to_excel(r'D:\0文献整理\网络入侵检测\KDD99\XGBoost基于TPE过程贝叶斯优化最优超参数组合.xlsx',index=False)
