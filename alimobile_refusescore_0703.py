# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:12:29 2017
对支付宝+运营商特征进行评分 以is_pass为y变量
gbdt 具体调参步骤
@author: xiaolinli4
"""
import pandas as pd
import ks_iv
from sklearn  import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics,cross_validation
from sklearn.grid_search import GridSearchCV 
import numpy as np
from sklearn.learning_curve import learning_curve #c查看是否过拟合


def ks_iv_evalua(file_test,y_name):
    a = list(file_test.columns)
    ks = []
    iv = []
    for i in a[7:-3]:
        ks.append(ks_iv.ks_iv_evaluate(file_test[[y_name,i]])[0])
        iv.append(ks_iv.ks_iv_evaluate(file_test[[y_name,i]])[1])
    ks_iv_eva = pd.DataFrame([ks,iv]).T
    ks_iv_eva.columns = ['ks','iv']
    ks_iv_eva.index = a[7:-3]
    return ks_iv_eva 
    
def fea_corr(file_name):   
    fea = file_name[[ ]]      
    fea_corr = fea.corr()
    return fea_corr
  
###寻找最优参数
def gridsearch(estimator, parameters,x1,y1,test_size):
    x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size = test_size,random_state=6)     
    gridcv = GridSearchCV(estimator,param_grid = parameters, scoring ='roc_auc',cv= 3)
    gridcv.fit(x_train,y_train)
    return gridcv.best_params_ , gridcv.best_score_ , gridcv.grid_scores_

def fea_split(x1,y1,estimator,pre_poba):       
    x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size = 0.25,random_state=6)      
#    cv_score =  cross_validation.cross_val_score(gb, x_train, y_train,cv=4,scoring='roc_auc') #0.69 
    estimator.fit(x_train,y_train)
    y_test= pd.DataFrame(y_test)
    if pre_poba==True:
        y_test['y_pre'] = pd.DataFrame(estimator.predict_proba(x_test),index = y_test.index)[1]
        y_train['y_pre'] = pd.DataFrame(estimator.predict_proba(x_train),index = y_train.index)[1]
    else:
        y_test['y_pre']= estimator.predict(x_test)
        y_train['y_pre']  = estimator.predict(x_train)
#    ks,iv,mbr,a = ks_iv.ks_iv_evaluate(y_test)        
    feature_importance = estimator.feature_importances_     
    feature =  pd.DataFrame(feature_importance)
    feature.index = x1.columns
    feature= feature.sort_values(by =[0])
    feature_im = feature[feature[0]>0.005].dropna()   
    print feature_im
    feature_im.plot(kind='bar', title='Feature_im')
    plt.ylabel('feature importance') 
    return y_train,y_test,feature_im

def model_rocauc(y_testpre,r,y_pre):      
    print "------model evaluate------"
    print metrics.roc_auc_score(y_testpre['is_pass'], y_testpre[y_pre])   
    precision, recall, thresholds =metrics.precision_recall_curve(y_testpre['is_pass'],y_testpre[y_pre]) 
    answer = y_testpre[y_pre]>r
    print metrics.classification_report(y_testpre['is_pass'],answer,target_names=['refuse','lend']) 
    print metrics.confusion_matrix(y_testpre['is_pass'],answer)
    print "Test Accuracy : %.7g" % metrics.accuracy_score(y_testpre['is_pass'], answer)        

if(__name__=='__main__'):
    path = 'E:/fengchao_data/haf_dataans/feature&py/ali_data/'
    alimol_data1  =  pd.read_csv(path+'/alimobilefea_get1.csv',sep=',')
    ali_columns = pd.read_csv(path+'alimobilefea_col.csv')
    a = list(ali_columns.columns)
    alimol_data1.columns = a    
    ##数据清洗##
    ali_data = alimol_data1.drop(['ali.dt2','mol.limit_id','mol.create_time',
             'mol.usercode','mol.username','mol.apply_id','mol.category2','mol.dt','mol.dt2'],axis=1)        
    ali_data.rename(columns = lambda x:x.replace('ali.','') if  'ali.' in x else ( x.replace('mol.','') if  'mol.' in x else x ), inplace = True) #修改列明    
    ali_data = ali_data.drop_duplicates(['limit_id'],keep=False) 
    ali_data = ali_data.fillna(0)
    ali_data = ali_data [ali_data['tual.is_pass']!=0]
    ali_data['is_pass'] = ali_data['tual.is_pass'].map( lambda x: 1 if x=='T' else 0  )
    ali_data['is_lend'] = ali_data['tua.category2'].map( lambda x: 1 if x>0 else 0 )
    ali_data['is_overdue'] = ali_data['tua.category2'].map( lambda x: 1 if x>=3  else( 0 if x<3 and x>0 else -1) )
    
    ali_data = ali_data.drop(['dt'],axis=1)
#    ks = ks_iv_evalua(ali_data,'is_lend')
#    ks.to_csv(path+'alifea_ks.csv')
    x1  = ali_data.iloc[:,7:-8]
    x1['join_time'] = x1['join_time'].map(lambda x : 0 if x<0 else x)
   # x1 = ali_data[fea_gbdt]
   # x1 = x1.drop(['egb.score'],axis=1) #特征选择
    y1 = ali_data[['is_pass']]
    
#############GBDT模型调参########### 
    '''
   默认参数
   ''' 
    params = {'random_state':5}
    gb  = GradientBoostingRegressor(**params)
    ytrain_pre,ytest_pre = fea_split(x1,y1,gb,False)
    model_rocauc(ytest_pre,0.28,'y_pre') #准确率0.7523  auc0.743
    model_rocauc(ytrain_pre,0.28,'y_pre') #准确率0.76  auc0.7303   
    
    '''
   选定基模型
   '''
    params = {'random_state':5,'min_samples_split':1300, 'min_samples_leaf':80 ,'max_depth' : 6 ,'max_features' :'sqrt' ,'subsample' :0.8}
    gbt  = GradientBoostingRegressor(**params)    
    ytrain_pre,ytest_pre = fea_split(x1,y1,gbt,False)
    model_rocauc(ytest_pre,0.28,'y_pre') 
    model_rocauc(ytrain_pre,0.28,'y_pre')    
    
    '''
    调节树的棵树 
    '''   
    gbt  = GradientBoostingRegressor( random_state=5,min_samples_split=1300, min_samples_leaf=80 ,max_depth=6 ,max_features ='sqrt' ,subsample =0.8)   
    gb_params = { 'n_estimators':range(40,160,20)}
    gb_bestparams , gb_bestscore , gb_gridscores =  gridsearch(gbt, gb_params,x1,y1,0.25)

    '''
    固定树的棵树 为140 调节深度 
    '''                                              
    gbt  = GradientBoostingRegressor( n_estimators=140,learning_rate=0.08,random_state=5, min_samples_leaf=40 ,max_features ='sqrt' ,subsample =0.8)   
    gb_params2 = { 'max_depth':range(5,10,2), 'min_samples_split':range(300,1501,300)}
    gb_bestparams2 , gb_bestscore2 , gb_gridscores2 =  gridsearch(gbt, gb_params2,x1,y1,0.25)
    
    '''
    固定深度为上一步最优深度  调节 min_samples_split\min_sample_leaf 
    '''
    gbt  = GradientBoostingRegressor( max_depth=7, n_estimators=140,learning_rate=0.12,random_state=5,max_features ='sqrt' ,subsample =0.8)    
    gb_params2 = { 'min_samples_split':range(1200,1801,200),'min_samples_leaf':range(40,71,10 )} 
    gb_bestparams2 , gb_bestscore2 , gb_gridscores2 =  gridsearch(gbt, gb_params2,x1,y1,0.25)    
    
    '''
    上一步调参结果并不好 沿用之前的参数值来调整max_feature 参数 
    '''   
    gbt  = GradientBoostingRegressor( min_samples_split=1800,min_samples_leaf=40,max_depth=7, n_estimators=140,learning_rate=0.12,random_state=5,subsample =0.8)    
    gb_params2 = { 'max_features':range(10,23,2 )} 
    gb_bestparams2 , gb_bestscore2 , gb_gridscores2 =  gridsearch(gbt, gb_params2,x1,y1,0.25)
    
    
    
    gbt  = GradientBoostingRegressor( max_features ='sqrt',min_samples_split=1800,min_samples_leaf=40,max_depth=7, n_estimators=140,learning_rate=0.12,random_state=5)    
    gb_params2 = { 'subsample' :[0.7,0.75,0.8,0.85,0.9]} 
    gb_bestparams2 , gb_bestscore2 , gb_gridscores2 =  gridsearch(gbt, gb_params2,x1,y1,0.25)
    

    gbt  = GradientBoostingRegressor( subsample =0.8,max_features ='sqrt',min_samples_split=1800,min_samples_leaf=40,max_depth=7, n_estimators=140,random_state=5)    
    gb_params2 = { 'learning_rate' :[0.1,0.08]} 
    gb_bestparams2 , gb_bestscore2 , gb_gridscores2 =  gridsearch(gbt, gb_params2,x1,y1,0.25)

    '''
    调参后最优参数
    '''
    gbt  = GradientBoostingRegressor( learning_rate = 0.08, subsample =0.8,max_features ='sqrt',min_samples_split=1800,min_samples_leaf=40,max_depth=7, n_estimators=140,random_state=5)   
    ytrain_pre,ytest_pre,fea_im  = fea_split(x1,y1,gbt,False)
    model_rocauc(ytest_pre,0.28,'y_pre') #准确率0.7539  auc0.749
    model_rocauc(ytrain_pre,0.28,'y_pre') #准确率0.7811  auc0.8064



'''
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=2, train_sizes=np.linspace(.05, 1., 10), verbose=0, plot=True):
    """
    查看模型是否过拟合：画出data在某模型上的learning curve.参数解释
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"train_sample")
        plt.ylabel(u"score")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"train_score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"test_score")
        plt.legend(loc="best")
        plt.draw()
        plt.show()
#        plt.gca().invert_yaxis()
#        plt.savefig("learn_curve.jpg") 
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff
           
##测试learing_curve函数   
    params = {'n_estimators':100, 'max_depth':4}
    gb  = GradientBoostingRegressor(**params) 
    plot_learning_curve(gb, 'aaa', x1, y1, ylim=None, cv=None, n_jobs=2, train_sizes=np.linspace(.05, 1., 10), verbose=0, plot=False )    
 '''   
