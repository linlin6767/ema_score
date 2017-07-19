# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:46:38 2017

对支付宝+运营商特征进行评分 以is_pass为y变量
gbdt 具体调参步骤
gbdt+lr融合
@author: xiaolinli4

"""
import pandas as pd
import ks_iv
from sklearn  import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn import metrics,cross_validation
from sklearn.grid_search import GridSearchCV 
import numpy as np
#from sklearn.learning_curve import learning_curve #c查看是否过拟合
from sklearn.linear_model import LogisticRegression


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
    
def rule_email(dat,y_name,feature,groups): 
    """
    评估规则效果1
    dat:数据集名称
    feature:特征名称
    groups: 按人数等组划分的组数
    """   
    cut_list = [0]*(groups+1)
    cut_list[0]=aliemail_data2[feature].min()-10
    for i in range(1,groups): #groups =10 分成10等分
        cut_list[i]=(np.percentile(aliemail_data2[feature],(100/groups)*i))
       #print cut_list
    cut_list[groups]= aliemail_data2[feature].max()+10
    cut_list2= sorted(set(cut_list),key=cut_list.index)
    hist,bin_edges = np.histogram(dat[feature],cut_list2)
    dat['bins']= pd.cut(dat[feature],bin_edges)
    a = pd.crosstab(dat['bins'],dat[y_name])
    a1 = a.reset_index()
    #print a1.head()
    a1.columns = ['bins','0','1']
    a1['refuse_pnt'] =  map(lambda x,y : 0 if x+y == 0 else  float(x)/(x+y) ,a1['0'],a1['1'])
    return a1
        
def rule_search (dat,y_name,groups,p_limit): 
    rule_df = pd.DataFrame(columns=['0','1','refuse_pnt','feature'])     
    for i in aliemail_data2.columns :
        if i in ['notoriginal_pnt','ali_trans_count','ali_trans_successed_count','ali_trans_worktime_count','max_tradetime','succ_max_tradetime','ali_thirdpart_count','workhour_call_count','join_time','top10callcnt']:
        #if i not in ['i_call_periodpct', 'i_called_periodpct','i_call_cntpct','i_called_cntpct','usercode', 'limit_id', 'is_refuse', 'create_time','egb.score', 'exg.score', 'is_pass', 'is_lend', 'is_overdue','bins'] :
            print '_________________________'
            print i
            feature = i 
            temp = rule_email(dat,y_name,feature,groups)
            temp_tt = temp[temp['refuse_pnt']>p_limit]
            temp_len =temp[temp['refuse_pnt']>p_limit].shape[0]
            print temp
            if temp_len >0 :
                temp_tt['feature']=feature
                temp_tt1 = temp_tt[['0','1','refuse_pnt','feature']]
                #print temp_tt 
                rule_df = rule_df.append(temp_tt1)               
    return rule_df 

def dat_eva(y_name,feature,thre,data,ineq):
    """
    对数据集进行每一步规则的过滤
    y_name： y变量
    feature:特征名称
    thre：特征拒贷临界值
    data：数据集名称
    ineq: '<'或‘>’
    """
    if ineq=='<' :
        a = data[data[feature]>=thre] 
        base_num = float(a.shape[0])    
        refuse_num = float(a[a[y_name]>0].shape[0])
        if base_num==0.0 :
            refuse_pnt =0
        else:
            refuse_pnt = refuse_num/base_num
    else :
        a = data[data[feature]<=thre]
        base_num = float(a.shape[0])
        refuse_num = float(a[a[y_name]>0].shape[0])
        if base_num == 0.0 :
            refuse_pnt = 0
        else:
            refuse_pnt = refuse_num/base_num
    return a,refuse_pnt
            
def feature_rule(fea_list,data,y_name): 
    """
    对数据集进行规则集的过滤
    y_name： y变量
    fea_list: 规则字典集
    data: 数据集
    """     
    for i in range(len(fea_list[0])):
        feature = fea_list[0][i]
        ineq = fea_list[1][i]
        thre = fea_list[2][i]
        print(feature)

        if i==0:
            a,refuse_pnt =  dat_eva(y_name,feature,thre,data,ineq)
            print refuse_pnt
            print len(a)
#各规格间相互平行 看一个规则的效果
#        else:
#            a,refuse_pnt = dat_eva(y_name,feature,thre,data,ineq)
#            print len(a),refuse_pnt
#前一个规则滤过的用户再进行下一个规则-看几个规则一起用的效果
        else:
            a,refuse_pnt = dat_eva(y_name,feature,thre,a,ineq)
            print refuse_pnt          
            print len(a)
    return a,refuse_pnt

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
    feature_im = feature[feature[0]>0.003].dropna()   
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
    ali_data = ali_data.fillna(0)    
    ali_data = ali_data.drop_duplicates(['limit_id'],keep=False) 
    ali_data = ali_data [ali_data['tual.is_pass']!=0]
    ali_data['join_time'] = ali_data['join_time'].map(lambda x : x if x>0 else 1160)

    ali_data['is_pass'] = ali_data['tual.is_pass'].map( lambda x: 1 if x=='T' else 0 )
    ali_data['is_lend'] = ali_data['tua.category2'].map( lambda x: 1 if x>0 else 0 )
    ali_data['is_overdue'] = ali_data['tua.category2'].map( lambda x: 1 if x>=3  else( 0 if x<3 and x>0 else -1) )   
    tt = [x for x in ali_data.columns if  x not in ['dt','username', 'apply_id','category2', 'tua.apply_id', 'tua.dapply_date','tua.s_audit_date','tua.category2','tual.is_pass'] ]
    ali_data2 = ali_data[tt]                      
    em1 = pd.read_csv(path+'emailfea_detail.csv')
    em2 = pd.read_csv(path+'emailfea_nodetail.csv')    
    em1_col = pd.read_csv(path+'emailfea_detailcol.csv')
    em2_col = pd.read_csv(path+'emailfea_nodetailcol.csv')    
    e1col = list(em1_col.columns)
    em1.columns = e1col
    em1 = em1.rename(columns = lambda x: x.replace('em1.',''))    
    e2col = list(em2_col.columns)
    em2.columns = e2col
    em2 = em2.rename(columns = lambda x: x.replace('em2.',''))    
    email=pd.merge(em1,em2,on = ['limit_id','emailaccount','usercode','create_time','is_refuse','dt'])
    email = email.drop(['dt','emailaccount'],axis=1)    
    aliemail_data2 = pd.merge(email,ali_data2,on=['limit_id','usercode','create_time','is_refuse'])
    aliemail_data2=aliemail_data2.fillna(0)          
    ###is_pass规则查找###
    #r2 = rule_search (aliemail_data2,'is_pass',60,0.99)
    #t1 = rule_email(aliemail_data2,'is_pass','notoriginal_pnt',60) 
    fea_list = [ [ 'notoriginal_pnt','ali_trans_count','succ_max_tradetime','max_tradetime','ali_trans_worktime_count','ali_trans_successed_count','join_time','workhour_call_count'],
    ['>','<','<','<','<','<','<','<'],[0.1,10,21,51,5,5,80,10]] 
    rule_data,rule_refusepnt = feature_rule(fea_list,aliemail_data2,'is_pass')

    x2 = rule_data.iloc[:,5:-3]
    #x2 = x2.drop (['join_time'],axis= 1)
    y2 = rule_data[['is_pass']]
    
    fea_select = [ 
                'hour_22_call_count_per',
                'hour_20_call_time_per',
                'ali_payment_suc1pnt',
                'hour_22_call_time_per',
                'sub_orderamount_msh_s',
                'hour_7_call_count',
                'xiaoe_cnt_pnt',
                'max_crelimit',
                'ali_cshopping_caipiao_pnt',
                'ali_zhuanzhang_othercnt',
                'ali_trans_close_pnt',
                'hour_23_call_count_per',
                'hour_18_call_count_per',
                'ali_zhuanzhang_suc1othpnt',
                'payment_avgamount',
                'ali_payback_suc1pnt',
                'ali_cshopping_bao_pnt',
                'maxcharges_pnt',
                'min_minpayment',
                'ali_trans_successed_blz_pnt',
                'sub_order_count_ych_s_p',
                'hour_13_call_count_per',
                'email_cnt',
                'sub_orderamount_zhx_f_p',
                'nocate_avgamount',
                'billdate_cnt',
                'banklist_call_count_call',
                'sub_orderamount_zhx_s',
                'nocate_cnt_pnt',
                'hour_15_call_count_per',
                'banklist_callperiod_sum_call',
                'hour_14_call_time_per',
                'ali_cshopping_accessories_pnt',
                'ali_cshopping_jiaju_pnt',
                'ali_zhuanzhang_successed_count',
                'ali_trans_small1_pnt',
                'else_avgamount',
                'hour_13_call_time_per',
                'ali_zhuanzhang_successed_othercnt',
                'avg_charge',
                'max_minpayment',
                'ali_money_pnt',
                'detail_maxamount',
                'min_tradetime',
                'sub_orderamount_yd_f',
                'sub_orderamount_yd_s_p',
                'sub_orderamount_yd_s',
                'ali_cshopping_child_pnt',
                'nocate_maxamount',
                'avg_period_trade',
                'ali_payback1_successed_absamt',
                'avgcharges_pnt',
                'ali_trans_small4_pnt',
                'payment_count',
                'ali_cshopping_3c_pnt',
                'hour_17_call_count_per',
                'max_tradetime',
                'else_maxamount',
                'max_period_trade',
                'succ_max_tradetime',
                'ali_other_train_pnt',
                'ali_trans_big1_pnt',
                'ali_trans_noclose_pnt',
                'nocate_avgamount_pnt',
                'avg_minpayment',
                'payment_maxamount',
                'maxbalance_pnt',
                'ali_cshopping_chuxing_pnt',
                'hour_10_call_count_per',
                'ali_trans_close_count',
                'ali_trans_lingchen_pnt',
                'sub_orderamount_zhx_s_p',
                'ali_cshopping_game_pnt',
                'ali_payback1_sucpnt',
                'ali_cshopping_3c_count',
                'ali_trans_small3_pnt',
                'ali_cshopping_game_count',
                'ali_trans_count1',
                'hour_9_call_count_per',
                'succ_avg_period_trade',
                'fetch_days',
                'billmonth_cnt',
                'ali_cshopping_charge_count',
                'ali_cshopping_clothes_pnt',
                'avgbalance_pnt',
                'ali_cshopping_charge_pnt',
                'ali_trans_lingchen_pnt1',
                'ali_trans_successed_pnt',
                'join_time',
                'notoriginal_bkcount',
                'notoriginal_pnt',
                'notoriginal_emcount'
                ]  
    x2 = x2[fea_select]
    #x2 = x2.drop(['hour_15_call_count_per'],axis=1)
                   
##########调参是支付宝+运营数据 还未修改############## 
#############GBDT模型调参########### 
    '''
    默认参数
    ''' 
    gbt  = GradientBoostingRegressor(random_state=5)   
    ytrain_pre,ytest_pre,fea_im  = fea_split(x2,y2,gbt,False)
    model_rocauc(ytest_pre,0.35,'y_pre') #准确率0.726  auc0.8318
    model_rocauc(ytrain_pre,0.35,'y_pre') #准确率0.737  auc0.8445
    
    '''
   选定基模型
   '''
    params = {'random_state':5,'min_samples_split':300, 'min_samples_leaf':30 ,'max_depth' : 6 ,'max_features' :'sqrt' ,'subsample' :0.8}
    gbt  = GradientBoostingRegressor(**params)    
    ytrain_pre,ytest_pre = fea_split(x2,y2,gbt,False)
    model_rocauc(ytest_pre,0.28,'y_pre') 
    model_rocauc(ytrain_pre,0.28,'y_pre')    
    
    '''
    调节树的棵树 
    '''   
    gbt  = GradientBoostingRegressor( random_state=5,min_samples_split=300, min_samples_leaf=30 ,max_depth=6 ,max_features ='sqrt' ,subsample =0.8)   
    gb_params = { 'n_estimators':range(40,160,20)}
    gb_bestparams , gb_bestscore , gb_gridscores =  gridsearch(gbt, gb_params,x2,y2,0.25)

    '''
    固定树的棵树 为140 调节深度 
    '''                                              
    gbt  = GradientBoostingRegressor( n_estimators=140,learning_rate=0.08,random_state=5, min_samples_leaf=30 ,max_features ='sqrt' ,subsample =0.8)   
    gb_params2 = { 'max_depth':range(5,10,2), 'min_samples_split':range(200,1401,200)}
    gb_bestparams2 , gb_bestscore2 , gb_gridscores2 =  gridsearch(gbt, gb_params2,x2,y2,0.25)
    
    '''
    固定深度为上一步最优深度  调节 min_samples_split\min_sample_leaf 
    '''
    gbt  = GradientBoostingRegressor( max_depth=9, n_estimators=140,learning_rate=0.1,random_state=5,max_features ='sqrt' ,subsample =0.8)    
    gb_params2 = { 'min_samples_split':range(1200,1801,200),'min_samples_leaf':range(20,71,10 )} 
    gb_bestparams2 , gb_bestscore2 , gb_gridscores2 =  gridsearch(gbt, gb_params2,x2,y2,0.25)    
    
    '''
    上一步调参结果并不好 沿用之前的参数值来调整max_feature 参数 
    '''   
    gbt  = GradientBoostingRegressor( min_samples_split=1200,min_samples_leaf=20,max_depth=9, n_estimators=140,learning_rate=0.1,random_state=5,subsample =0.8)    
    gb_params2 = { 'max_features':range(14,30,2 )} 
    gb_bestparams2 , gb_bestscore2 , gb_gridscores2 =  gridsearch(gbt, gb_params2,x2,y2,0.25)
    
    gbt  = GradientBoostingRegressor( max_features =22 ,min_samples_split=1200,min_samples_leaf=20,max_depth=9, n_estimators=140,learning_rate=0.1,random_state=5)    
    gb_params2 = { 'subsample' :[0.7,0.75,0.8,0.85,0.9]} 
    gb_bestparams2 , gb_bestscore2 , gb_gridscores2 =  gridsearch(gbt, gb_params2,x2,y2,0.25)   

    gbt  = GradientBoostingRegressor( subsample =0.8,max_features =22,min_samples_split=1200,min_samples_leaf=20,max_depth=9, n_estimators=140,random_state=5)    
    gb_params2 = { 'learning_rate' :[0.1,0.08,0.11,0.12]} 
    gb_bestparams2 , gb_bestscore2 , gb_gridscores2 =  gridsearch(gbt, gb_params2,x2,y2,0.25)

    '''
    全量特征调参后最优参数
    '''
    gbt  = GradientBoostingRegressor( learning_rate = 0.1, subsample =0.8,max_features =22,min_samples_split=1800,min_samples_leaf=40,max_depth=7, n_estimators=140,random_state=5)   
    ytrain_pre,ytest_pre,fea_im  = fea_split(x2,y2,gbt,False)
    model_rocauc(ytest_pre,0.32,'y_pre') #准确率0.7539  auc0.838
    model_rocauc(ytrain_pre,0.28,'y_pre') #准确率0.7811  auc0.878
    '''
    部分特征调参后最优参数
    '''    
    gbt  = GradientBoostingRegressor( learning_rate = 0.08, subsample =0.8,max_features =22,min_samples_split=1200,min_samples_leaf=20,max_depth=9, n_estimators=140,random_state=5)   
    ytrain_pre,ytest_pre,fea_im1  = fea_split(x2,y2,gbt,False)
    model_rocauc(ytest_pre,0.4,'y_pre') #准确率0.7539  auc0.837
    model_rocauc(ytrain_pre,0.4,'y_pre') #准确率0.7811  auc0.890



####尝试方法gbdt+lr模型
    gbt  = GradientBoostingClassifier( n_estimators=60,random_state=5)
    gbt = GradientBoostingRegressor( n_estimators=140,random_state=5)
    from sklearn.preprocessing import OneHotEncoder
    gbt_enc = OneHotEncoder()
    gbt_lr = LogisticRegression()    
    x_train,x_test,y_train,y_test = train_test_split(x2,y2,test_size = 0.25,random_state=6)     
    gbt.fit(x_train, y_train)
   # fit one-hot编码器
    gbt_enc.fit(gbt.apply(x_train))
#    gbt_enc.fit(gbt.apply(x_train)[:, :, 0])
    gbt_lr.fit(gbt_enc.transform(gbt.apply(x_train)), y_train)
    y_pred_gbt_lr = gbt_lr.predict_proba(gbt_enc.transform(gbt.apply(x_test)))[:, 1]
    y_test['y_pre']=y_pred_gbt_lr
    model_rocauc(y_test,0.4,'y_pre') 
  
    
          