# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:12:29 2017
对支付宝特征进行评分 以is_pass为y变量
多模型gbdt rf  rank融合
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

tao_feature_select=['ali_zhuanzhang_sucothpnt', 'ali_payback_otherside_count', 'exg.score', 
       'ali_payback1_sucpnt', 'succ_max_tradetime', 'avg_period_trade', 
       'ali_thirdpart_count', 'sub_order_count_msh_s_p', 'ali_money_count', 
       'ali_trans_count1', 'sub_orderamount_msh_s_p', 
       'ali_zhuanzhang_sucpnt', 'ali_trans_successed_pnt', 
       'sub_orderamount_zhx_s', 'ali_trans_worktime_count', 
       'ali_payback1_suc1pnt', 'ali_zhuanzhang_successed1_count', 
       'ali_zhuanzhang_successed_othercnt', 'ali_trans_noworktime_pnt', 
       'ali_zhuanzhnag_othercnt1', 'ali_suczhuanzhang_othercnt1', 
       'ali_worktime_pnt', 'ali_payback1_successed_absamt', 
       'ali_zhuanzhnag_othercnt2', 'sub_orderamount_msh_s', 
       'ali_money_pnt', 'ali_worktime1_pnt', 'ali_cshopping_charge_pnt', 
       'ali_zhuanzhang_sucamtpnt', 'ali_trans_close_pnt', 
       'ali_trans_successed_abz_pnt', 'ali_money_succount', 
       'ali_payback_successed_absamt', 'ali_trans_noclose_pnt']
       

fea_gbdt = ['ali_trans_lingchen_pnt1',
            'ali_payment_amt1pnt',
            'min_tradetime',
            'elr.y_pre',
            'ali_trans_small4_pnt',
            'ali_cshopping_jiaju_pnt',
            'period_2days_cnt',
            'ali_cshopping_book_pnt',
            'ali_trans_big2_pnt',
            'ali_cshopping_3c_pnt',
            'sub_orderamount_msh_f_p',
            'sub_orderamount_ych_s',
            'sub_orderamount_msh_s',
            'sub_order_count_msh_s_p',
            'ali_cshopping_charge_count',
            'sub_orderamount_jk_s',
            'ali_other_train_pnt',
            'ali_zhuanzhang_sucamtpnt',
            'ali_worktime_pnt',
            'ali_zhuanzhang_othercnt',
            'ali_money_pnt',
            'max_tradetime',
            'ali_cshopping_game_count',
            'ali_cshopping_chuxing_pnt',
            'ali_trans_close_count',
            'succ_period_2days_cnt',
            'ali_cshopping_clothes_pnt',
            'ali_trans_small3_pnt',
            'ali_payment_suc1amtpnt',
            'ali_trans_successed_pnt',
            'ali_trans_lingchen_pnt',
            'elr.score',
            'sub_orderamount_zhx_s',
            'succ_max_tradetime',
            'ali_payback_sucpnt',
            'ali_payment_sucamtpnt',
            'exg.score',
            'succ_avg_period_trade',
            'egb.score']
  

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
        y_train['y_pre'] = pd.DataFrame(estimator.predict_proba(y_train),index = y_train.index)[1]
    else:
        y_test['y_pre']= estimator.predict(x_test)
        y_train['y_pre']  = estimator.predict(x_train)
#    ks,iv,mbr,a = ks_iv.ks_iv_evaluate(y_test)        
    feature_importance = estimator.feature_importances_     
    feature =  pd.DataFrame(feature_importance)
    feature.index = x1.columns
    feature= feature.sort_values(by =[0])
    feature_im = feature[feature[0]>0.007].dropna()   
    print feature_im
    feature_im.plot(kind='bar', title='Feature_im')
    plt.ylabel('feature importance') 
    return y_train,y_test

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
    ali_data1  =  pd.read_csv(path+'/bankfea_getv2.csv',sep=',')
    ali_columns = pd.read_csv(path+'alifeav2col.csv')
    a = list(ali_columns.columns)
    ali_data1.columns = a    
    ##数据清洗##    
    ali_data1.rename(columns = lambda x:x.replace('ali.',''), inplace = True) #修改列明
    ali_data = ali_data1.drop_duplicates(['limit_id'],keep=False) 
        ali_data = ali_data.fillna(0)
    ali_data = ali_data [ali_data['tual.is_pass']!=0]
    ali_data['is_pass'] = ali_data['tual.is_pass'].map( lambda x: 1 if x=='T' else 0  )
    ali_data['is_lend'] = ali_data['tua.category2'].map( lambda x: 1 if x>0 else 0 )
    ali_data['is_overdue'] = ali_data['tua.category2'].map( lambda x: 1 if x>=3  else( 0 if x<3 and x>0 else -1) )
    ali_data = ali_data.drop(['dt','dt2'],axis=1)

#    ks = ks_iv_evalua(ali_data,'is_lend')
#    ks.to_csv(path+'alifea_ks.csv')
    x1  = ali_data.iloc[:,7:-8]
   # x1 = ali_data[fea_gbdt]
   # x1 = x1.drop(['egb.score'],axis=1) #特征选择
    y1 = ali_data[['is_pass']]
    
#############多个模型训练###########  
#------- GBDT -------------
    gbt  = GradientBoostingRegressor()    
    gb_params = { 'n_estimators':(70,100,200,150), 'max_depth':(2,4,8)} 
    gb_bestparams , gb_bestscore , gb_gridscores =  gridsearch(gbt, gb_params,x1,y1,0.25)
    params = {'n_estimators':100, 'max_depth':4,'loss':'ls'}
    gb  = GradientBoostingRegressor(**params)  
    feature,ytest_pre = fea_split(x1,y1,gb,False)
    ytest_pre['rf_rank']= ytest_pre['y_pre'].rank()
    ytest_pre['rf_rank1']= ytest_pre['rf_rank'].astype(float)/ytest_pre.shape[0]
    model_rocauc(ytest_pre,0.28,'y_pre') #准确率0.7523  auc0.7303
    model_rocauc(ytest_pre,0.8,'rf_rank1') #准确率0.76  auc0.7303
# ------ RF ----------
    rf = RandomForestClassifier(oob_score=True)
    rf_params = { "n_estimators"       : [130,200,300],
                   "criterion"         : ["gini"],
                   "max_depth"         : [ 4,10,15],
                   "min_sample_leaf"   : [5,10,40 ]}
                   
    rf_bestparams , rf_bestscore , rf_gridscores = gridsearch(rf,rf_params,x1,y1['is_pass'],0) 
    print rf_bestparams , rf_bestscore , rf_gridscores    
    rf2 = RandomForestClassifier(n_estimators= 200, max_depth=8, criterion="gini",oob_score=True)
    feature2,ytest_pre2 = fea_split(x1,y1,rf2,True)
    model_rocauc(ytest_pre2,0.2,'y_pre')  #准确率0.7406   auc0.705
    ytest_pre2['rf_rank']= ytest_pre2['y_pre'].rank()
    ytest_pre2['rf_rank1']= ytest_pre2['rf_rank'].astype(float)/ytest_pre2.shape[0]
    model_rocauc(ytest_pre2,0.8,'rf_rank1')     
    ytest_pre['eneank'] = (ytest_pre['rf_rank1']+ytest_pre2['rf_rank1'])/2
    model_rocauc(ytest_pre,0.86,'eneank')

##将数据分成交易类和资金类分别建模
    x2_1 = ali_data.iloc[:,7:34]
    x2_2 = ali_data.iloc[:,34:91]
    x2_3 = ali_data.iloc[:,91:120]
    x2_4 = ali_data.iloc[:,120:154]
    x2_5 = ali_data.iloc[:,155:190]
    x2_6 = ali_data.iloc[:,190:217]
    x2_7 = ali_data.iloc[:,217:265]    
    x3_1 = x2_1.join(x2_3).join(x2_6) #交易状态相关
    x3_1 = x3_1.drop(['ali_money_count'],axis = 1)
    x3_2 = x2_2.join(x2_4) #资金相关
    x3_2['ali_money_count'] =ali_data['ali_money_count']
    x3_3 = x2_5.join(x2_7) #分类相关
    
    ##待进行调参 ？？？
    params = {'n_estimators':100, 'max_depth':4,'loss':'ls'}
    gb  = GradientBoostingRegressor(**params)  
    ytrian_pre3_1,ytest_pre3_1 = fea_split(x3_1,y1,gb,False)
    
    ytrain_pre3_2,ytest_pre3_2 = fea_split(x3_2,y1,gb,False)
    ytrain_pre3_3,ytest_pre3_3 = fea_split(x3_3,y1,gb,False)
    
    ytest_pre3_1['rf_rank']= ytest_pre3_1['y_pre'].rank()
    ytest_pre3_1['rf_rank1']= ytest_pre3_1['rf_rank'].astype(float)/ytest_pre3_1.shape[0]
    
    ytest_pre3_2['rf_rank']= ytest_pre3_2['y_pre'].rank()
    ytest_pre3_2['rf_rank1']= ytest_pre3_2['rf_rank'].astype(float)/ytest_pre3_2.shape[0]    

    ytest_pre3_3['rf_rank']= ytest_pre3_3['y_pre'].rank()
    ytest_pre3_3['rf_rank1']= ytest_pre3_3['rf_rank'].astype(float)/ytest_pre3_3.shape[0]
 
   ##模型融合方案1
    ytest_pre3_1['rf_rank3']=(ytest_pre3_1['rf_rank1']+ytest_pre3_2['rf_rank1']+ytest_pre3_3['rf_rank1'])/3.00   
    model_rocauc(ytest_pre3_1,0.88,'rf_rank3') #准确率0.83  auc0.709   
   ##模型融合方案2
    from sklearn.linear_model import LogisticRegression  
    
    ytest_pre3 = ytest_pre3_1.concat(ytest_pre3_2)
    lr = LogisticRegression()
   
    lr_trainx = ytrian_pre3_1[['y_pre']]
    lr_trainx['y_pre2'] = ytrain_pre3_2['y_pre']
    lr_trainx['y_pre3'] = ytrain_pre3_3['y_pre']
    lr_trainy = ytrian_pre3_1[['is_pass']]    
    lr_testx = ytest_pre3_1[['y_pre']]  
    lr_testx['y_pre2'] = ytest_pre3_2['y_pre']
    lr_testx['y_pre3'] = ytest_pre3_3['y_pre']
    lr_testy = ytest_pre3_1[['is_pass']]
    
    lr.fit(lr_trainx,lr_trainy)
    print(u'模型的平均正确率为：%s' % lr.score(lr_trainx,lr_trainy))      
    lr_testy['lr_pre']= pd.DataFrame(lr.predict_proba(lr_testx) ,index = lr_testx.index)[1]
    model_rocauc(lr_testy,0.35,'lr_pre')   #准确率0.833  auc0.714 
     

    











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
