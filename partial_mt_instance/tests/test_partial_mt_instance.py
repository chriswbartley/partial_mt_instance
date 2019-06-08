from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.testing as npt
import partial_mt_instance

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#from sklearn.datasets import load_boston
from sklearn.utils import resample
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

def my_func(X,mt_feat_types):
    a=0.95
    b=1.43005#1.2
    p=X.shape[1]
    #p_mt=p-#n
    #p_mt=np.int(np.ceil(p/2))
    f=0.
    for j in np.arange(p)[np.abs(mt_feat_types)==1]:
        f=f+X[:,j]
    for j in np.arange(p)[np.abs(mt_feat_types)==0]:#np.arange(p_mt,p):
        f=f - a * np.sin(X[:,j]*np.pi*b)
    #f[np.logical_and(X[:,0]>0,X[:,1]<-0.5)]=1
    return f

def my_func_safe_3(X):
    a=0.55
    b=1.43005#1.2
    p=X.shape[1]
    p_mt=p-1#n
    p_mt=np.int(np.ceil(p/2))
    f=0.
    for j in np.arange(p_mt):
        f=f+X[:,j]
    for j in np.arange(p_mt,p):
        f=f - a * np.sin(X[:,j]*np.pi*b)
    #f[np.logical_and(X[:,0]>0,X[:,1]<-0.5)]=1
    return f

def my_func_safe_2(X):
    a=0.55
    b=1.43005#1.2
    f=-a * np.sin(X[:,1]*np.pi*b) + X[:,0]
    #f[np.logical_and(X[:,0]>0,X[:,1]<-0.5)]=1
    return f

def calc_loss(pred_proba,y_true):
    return log_loss(y_true,pred_proba)

def bootstrap_ci(values, ci_percentage=0.95,iters=10000):
    
    n_size=values.shape[0]
    if len(values.shape)<2:
        cols=1
    else:
        cols=values.shape[1]
    stats=np.zeros([iters,cols])
    for i in np.arange(iters):
        sample=resample(values, n_samples=n_size)
        stats[i,:]=np.mean(sample,axis=0)
    ci_percentage = 0.95
    p_lower = ((1.0-ci_percentage)/2.0) * 100
    p_upper = (ci_percentage+((1.0-ci_percentage)/2.0)) * 100
    lowers=[]
    uppers=[]
    for icol in np.arange(cols):
        lowers =lowers+ [ np.percentile(stats[:,icol], p_lower)]
        uppers =uppers+ [np.percentile(stats[:,icol], p_upper)]

        
    return np.asarray([lowers,uppers])
    
noise_percentage=0.
num_pts=100
max_delta_N=250
#noise_seed=368 #361125
bound_linewidth=4
bound_color='m'
ylim=1.2
xlim=1.2
incr_feats=np.asarray([1,2])
decr_feats=np.asarray([])
mt_feats=np.asarray(list(incr_feats)+list(decr_feats))
n_feats=4
nmt_feats=np.asarray([i for i in np.arange(n_feats)+1 if i not in mt_feats])
mode='one_solve'#one_solve' #noise_plots log_regn
nmt_plane_type='independent'#independent'#joint'


if mode=='log_regn':
    n_expt=200
    coefs=np.zeros([n_expt,2])
    mt_feat_types=[1,0]
    for i in np.arange(n_expt):
        [X_train,y_train,xxx,y_train_true]=partial_mt_instance.generate_partial_mono_data_by_function(num_samples=num_pts,mt_feat_types=mt_feat_types, classes=[-1.5,+1.5], class_dist='uniform',feat_dist='uniform',feat_spec=[-1,1], sign_function=my_func,NMI_target=noise_percentage,random_state=noise_seed+i)
        clf_=LogisticRegression(C=1e4)
        clf_.fit(X_train,y_train)
        coefs[i,:]=clf_.coef_ 
        print(clf_.coef_ )
    print(np.mean(coefs,axis=0))
        #print(clf_.intercept_)
elif mode=='one_solve':
    # generate data
    noise_seed=123457
    mt_feat_types=[1,0]#
    #mt_feat_types=[1,1,0,0]
    [X_train,y_train,xxx,y_train_true]=partial_mt_instance.generate_partial_mono_data_by_function(num_samples=num_pts,mt_feat_types=mt_feat_types, classes=[-1.5,+1.5], class_dist='uniform',feat_dist='uniform',feat_spec=[-1,1], sign_function=my_func,NMI_target=noise_percentage,random_state=noise_seed)
    [X_test,y_test,xxx,y_test_true]=partial_mt_instance.generate_partial_mono_data_by_function(num_samples=np.min([300,num_pts*10]),mt_feat_types=mt_feat_types, classes=[-1.5,+1.5], class_dist='uniform',feat_dist='uniform',feat_spec=[-1,1], sign_function=my_func,NMI_target=noise_percentage,random_state=noise_seed+1)
    
    # fit model
    #mt_feat_types=[1,0]
    fit_type='none'#none' #linear
    #mt_feat_types=[1,0]
    #fit_type='none'#none' #linear
    #mt_feat_types=[1,-1]
    #fit_type='linear'#none' #linear
    clf=partial_mt_instance.PartialInstanceBinaryClassifier(mt_feat_types,
                     fit_type='linear',relabel=True,
                        scale_X=False,nmt_plane_type=nmt_plane_type)
    
    clf.fit(X_train,y_train,max_delta_N=max_delta_N)
    [pdf_pred_all,y_pred,counts]=clf.predict_proba(X_test,use_relabelled_if_avail=False)
    acc=np.sum(y_pred==y_test)/len(y_test)
    #print(acc)   
    print('class balance: ' + str(np.sum(y_train==1)/len(y_train)))
                     
    print(acc)    
    
    # PLOT
    
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h)) 
    Z_true=my_func(np.c_[xx.ravel(), yy.ravel()],mt_feat_types)
    Z_true = Z_true.reshape(xx.shape)        
    
    Z_remove_y=np.zeros(xx.ravel().shape)-1
    Z_remove_y[xx.ravel()>0.]=1.
    Z_remove_y = Z_remove_y.reshape(xx.shape)  
    
    Z_y_mt=np.zeros(xx.ravel().shape)-1
    Z_y_mt[(-xx.ravel()+0.9*yy.ravel())>0.]=1.
    Z_y_mt = Z_y_mt.reshape(xx.shape) 
               
    train_err=[]
    test_err=[]
    svm_vs=[0.01,0.1,0.25,0.5,0.75]#[0.001,0.1,1,5,10]#,0.25,0.5,1]#[0.01,0.1,0.2,0.5,1]
    y_std=y_train.copy()
    y_std[y_std==-1]=0
    
    plot_rows=1
    plot_iters=1
    f, ax = plt.subplots(1,2)#len(plot_rows), len(plot_iters))#, sharex='col', sharey='row')
    
    this_fig=ax[0]
    this_fig.scatter(X_train[y_train==-1,0], X_train[y_train==-1,1], c='r')
    this_fig.scatter(X_train[y_train==1,0], X_train[y_train==1,1], c='k',marker='+',s=50)
    this_fig.set_ylim(-ylim,ylim)
    this_fig.set_xlim(-xlim,xlim)
    
    
    
    #f, ax = plt.subplots(1,2)#len(plot_rows), len(plot_iters))#, sharex='col', sharey='row')
    
    this_fig=ax[1]
    y_=clf.y_relabelled
    this_fig.scatter(X_train[y_==-1,0], X_train[y_==-1,1], c='r')
    this_fig.scatter(X_train[y_==1,0], X_train[y_==1,1], c='k',marker='+',s=50)
    this_fig.set_ylim(-ylim,ylim)
    this_fig.set_xlim(-xlim,xlim)    
    
    
elif mode=='noise_plots':
    vs=[0.001,0.01,0.1,1.0,10.,100.,1000.,10000.]
    vs=[0.01,0.25,0.5,0.75]
#    runs=[['cone_nmt_filt_rem_k2',[1,0],'linear',[0.0001],'remove',[2]],
#            #['cone_nmt_filt_rem_k3',[1,0],'linear',[0.0001],'remove',[3]],
#            ['cone_nmt_filt_rem_k5',[1,0],'linear',[0.0001],'remove',[5]],
#            ['cone_nmt_filt_rem_k10',[1,0],'linear',[0.0001],'remove',[10]]
#            ]#,['cone_nmt',[1,0],'linear',vs],['monotonise_nmt',[1,-1],'none',[0.]],['elim_nmt',[1,0],'none',[0.]]]#['correct_cone',[1,0],'linear',vs]
#    runs=[['cone_nmt_filt_rel_k2',[1,0],'linear',[0.0001],'relabel',[2]],
#            #['cone_nmt_filt_rem_k3',[1,0],'linear',[0.0001],'remove',[3]],
#            ['cone_nmt_filt_rel_k5',[1,0],'linear',[0.0001],'relabel',[5]],
#            ['cone_nmt_filt_rel_k10',[1,0],'linear',[0.0001],'relabel',[10]]
#            ]
#noise_plots params
    noises=[0,0.02,0.05]#,0.1,0.15, 0.20]#0.05,0.10,0.15]
    expts_per_noise=20
    noise_seed=1231
    k=10 #
    num_feats=2#4 6
    num_pts=100#30*30# 200 300
    max_delta_N=250
    #p_mt=num_feats-1#
    p_mt=np.int(np.ceil(num_feats/2))
    incr_feats=np.arange(1,p_mt+1,dtype=np.int32)
    decr_feats=np.asarray([])
    mt_feat_types_correct=np.zeros(num_feats,dtype=np.int32)
    mt_feat_types_correct[incr_feats-1]=1
    runs=[['cone_nmt_filt_rem_k_loo',mt_feat_types_correct,'linear',[0.001],'remove',ks]
          ,['cone_nmt_filt_rem_v_loo',mt_feat_types_correct,'linear',vs,'none',[3,6,9,12]]#,6,9,12,15,18
          ,['monotonise_nmt',mt_feat_types_correct,'none',[0.],'none',[0]]
          ,['elim_nmt',mt_feat_types_correct,'none',[0.],'none',[0]]
          ]
    
    opt_type='loo' # cv loo
    relabel_acc=np.zeros([len(runs),len(noises),expts_per_noise])
    pred_acc=np.zeros([len(runs),len(noises),expts_per_noise])
    pred_acc_relabelled=np.zeros([len(runs),len(noises),expts_per_noise])
    optimal_vs=np.zeros([len(runs),len(noises),expts_per_noise])
    optimal_ks=np.zeros([len(runs),len(noises),expts_per_noise])
    for irun in np.arange(len(runs)):
        print ('STARTING RUN ' + str(irun))
        label=runs[irun][0]
        mt_feat_types=runs[irun][1].copy()
        fit_type=runs[irun][2]
        svm_vs=runs[irun][3]
        local_mt_filter=runs[irun][4]
        local_mt_filter_ks=runs[irun][5]
        for inoise in np.arange(len(noises)):
            noise =noises[inoise]
            for iexpt in np.arange(expts_per_noise):
                print (' - run ' + str(irun)+ ' - noise ' + str(noise) + ' expt ' + str(iexpt))
                mt_feat_types=runs[irun][1].copy()
                seed_=noise_seed+iexpt
                [X_train,y_train,xxx,y_train_true]=partial_mt_instance.generate_partial_mono_data_by_function(num_samples=num_pts,mt_feat_types=mt_feat_types, classes=[-1.5,+1.5], class_dist='uniform',feat_dist='uniform',feat_spec=[-1,1], sign_function=my_func,NMI_target=noise,random_state=seed_)
                [X_test,y_test,xxx,y_test_true]=partial_mt_instance.generate_partial_mono_data_by_function(num_samples=np.min([300,num_pts]),mt_feat_types=mt_feat_types, classes=[-1.5,+1.5], class_dist='uniform',feat_dist='uniform',feat_spec=[-1,1], sign_function=my_func,NMI_target=0,random_state=seed_+1)
                #print('class balance: ' + str(np.sum(y_train_true==1)/len(y_train_true)))
                if label=='monotonise_nmt':
                    for f in np.arange(len(mt_feat_types)):
                        if mt_feat_types[f]==0:        
                            clf_=LogisticRegression(C=1e4)
                            clf_.fit(X_train[:,f].reshape(-1, 1),y_train)
                            mt_feat_types[f]=1 if np.mean( clf_.coef_[0,0])>0 else -1

                    print(mt_feat_types )
                    #mt_feat_types[f]=1 if np.mean( coefs)>0 else -1
                if label=='correct_cone':
                    # get correct cone
                    clf=partial_mt_instance.PartialInstanceBinaryClassifier(mt_feat_types,
                        fit_type=fit_type,
                        relabel=True,
                        scale_X=False)
                    clf.fit(X_train,y_train_true,svm_v_empir_error=0.00001,max_delta_N=max_delta_N)
                    nmt_plane=clf.nmt_planes[0,:]
                    # bodgy a classifier with correct cone and noisy data
                    clf=partial_mt_instance.PartialInstanceBinaryClassifier(mt_feat_types,
                        fit_type='none',
                        relabel=True,
                        scale_X=False
                        )
                    clf.fit(X_train,y_train,max_delta_N=max_delta_N)
                    clf.nmt_planes[0,:]=nmt_plane
                    clf.nmt_type=2#NMT_CONE#'cone_nmt'
                else:
                    if len(svm_vs)==1:
                        optimal_v=svm_vs[0]
                    else: # use cross validation
                        clf=partial_mt_instance.PartialInstanceBinaryClassifier(mt_feat_types,
                            fit_type=fit_type,
                            relabel=False,
                            scale_X=False
                        )
                        n_svm_vs=len(svm_vs)
                        losses=np.zeros(n_svm_vs)
                        counts=np.zeros(n_svm_vs)
                        if opt_type=='cv':
                            skf = StratifiedKFold(n_splits=k)
                            for i_v in np.arange(n_svm_vs):
                                oos_pred_proba=np.zeros(len(y_train))
                                oos_counts=np.zeros(len(y_train))
                                for train_index, test_index in skf.split(X_train, y_train):
                                    clf.fit(X_train[train_index,:],y_train[train_index],svm_v_empir_error=svm_vs[i_v],max_delta_N=max_delta_N)
                                    [pred_proba,y_pred,counts_]=clf.predict_proba(X_train[test_index,:])
                                    oos_pred_proba[test_index]=pred_proba[:,1]
                                    oos_counts[test_index]=np.sum(counts_,axis=1)
                                losses[i_v]=calc_loss(oos_pred_proba,y_train)
                                counts[i_v]=np.mean(oos_counts)
                        elif opt_type=='loo':
                            for i_v in np.arange(n_svm_vs):
                                #oos_pred_proba=np.zeros(len(y_train))
                                #oos_counts=np.zeros(len(y_train))
                                clf.fit(X_train,y_train,svm_v_empir_error=svm_vs[i_v],max_delta_N=max_delta_N)
                                #X_train_offset=X_train+np.vstack([np.zeros(X_train.shape[0]),np.ones(X_train.shape[0])*0.001]).T
                                [pred_proba,y_pred,counts_]=clf.predict_proba_loo()
                                oos_pred_proba=pred_proba[:,1]
                                oos_counts=np.sum(counts_,axis=1)
                                losses[i_v]=calc_loss(oos_pred_proba,y_train)
                                counts[i_v]=np.mean(oos_counts)
                        best=[1e9,-1]
                        for i_v in np.arange(n_svm_vs):
                            if losses[i_v]<best[0] and counts[i_v]>=9:
                                best=[losses[i_v],i_v]
                        #print(counts)
                        #print(losses)
                        optimal_v=svm_vs[best[1]]
                        print('optimal v='+str(optimal_v) + ' for noise = ' + str(noise))
                    if local_mt_filter=='none':
                        optimal_local_mt_filter_k=-99
                    else:
                        if len(local_mt_filter_ks)==1:
                            optimal_local_mt_filter_k=local_mt_filter_ks[0]
                        else:
                            
                            n_svm_ks=len(local_mt_filter_ks)
                            losses=np.zeros(n_svm_ks)
                            counts=np.zeros(n_svm_ks)
                            for i_k in np.arange(n_svm_ks):
                                #oos_pred_proba=np.zeros(len(y_train))
                                #oos_counts=np.zeros(len(y_train))
                                clf=partial_mt_instance.PartialInstanceBinaryClassifier(mt_feat_types,
                                    fit_type=fit_type,
                                    relabel=False,
                                    local_mt_filter='none' if local_mt_filter_ks[i_k]==0 else local_mt_filter,
                                    local_mt_filter_k=local_mt_filter_ks[i_k]
                                    ,scale_X=False)
                                clf.fit(X_train,y_train,svm_v_empir_error=svm_vs[0],max_delta_N=max_delta_N)
                                #X_train_offset=X_train+np.vstack([np.zeros(X_train.shape[0]),np.ones(X_train.shape[0])*0.001]).T
                                [pred_proba,y_pred,counts_]=clf.predict_proba_loo(use_relabelled_if_avail=False)
                                oos_pred_proba=pred_proba[:,1]
                                oos_counts=np.sum(counts_,axis=1)
                                losses[i_k]=calc_loss(oos_pred_proba,y_train)
                                counts[i_k]=np.mean(oos_counts)
                            best=[1e9,-1]
                            for i_k in np.arange(n_svm_ks):
                                if losses[i_k]<best[0] and counts[i_k]>=9:
                                    best=[losses[i_k],i_k]
                            print(counts)
                            print(losses)
                            optimal_local_mt_filter_k=local_mt_filter_ks[best[1]]
                            print('optimal k='+str(optimal_local_mt_filter_k) + ' for noise = ' + str(noise))

                    # solve optimal model
                    clf=partial_mt_instance.PartialInstanceBinaryClassifier(mt_feat_types,
                        fit_type=fit_type,
                        relabel=True,
                        local_mt_filter=local_mt_filter,
                        local_mt_filter_k=optimal_local_mt_filter_k,
                        scale_X=False)
                    
                    clf.fit(X_train,y_train,svm_v_empir_error=optimal_v,max_delta_N=max_delta_N)
                    #print(-clf.nmt_planes[0,:][1]/clf.nmt_planes[0,:][0])
                    print(clf.nmt_planes[0,:])
                    
                [pdf_pred_all,y_pred,counts]=clf.predict_proba(X_test,use_relabelled_if_avail=False)
                acc=np.sum(np.abs(y_pred-y_test)<1e-5) /len(y_test)
                pred_acc[irun,inoise,iexpt]=acc
                
                [pdf_pred_all,y_pred,counts]=clf.predict_proba(X_test,use_relabelled_if_avail=True)
                acc=np.sum(np.abs(y_pred-y_test)<1e-5) /len(y_test)
                pred_acc_relabelled[irun,inoise,iexpt]=acc
                
                acc_relab=np.sum(np.abs(clf.y_relabelled-y_train_true)<1e-5)/len(y_train)
                relabel_acc[irun,inoise,iexpt]=acc_relab
                
                optimal_vs[irun,inoise,iexpt]=optimal_v
                optimal_ks[irun,inoise,iexpt]=optimal_local_mt_filter_k
    f, ax = plt.subplots(1,2)
    legend_=[]
    legend_acc_=[]
    title_typ='('+str(num_feats) + 'f' + str(p_mt) + 'm' +str(num_pts)+'N)'
    for irun in np.arange(len(runs)):
        label=runs[irun][0]
        legend_=legend_+[label]
        legend_acc_=legend_acc_+[label +' (OSDL)', label +' (MOCA)']
        mt_feat_types=runs[irun][1]
        fit_type=runs[irun][2]
        # plot noise vs accuracy
        acc_means=np.mean(pred_acc[irun,:,:],axis=1)
        acc_use_relabelled_means=np.mean(pred_acc_relabelled[irun,:,:],axis=1)
        acc_relabel_means=np.mean(relabel_acc[irun,:,:],axis=1)
        
        
        acc_err=np.abs(bootstrap_ci(pred_acc[irun,:,:].T)-acc_means)
        acc_use_relabelled_err=np.abs(bootstrap_ci(pred_acc_relabelled[irun,:,:].T)-acc_use_relabelled_means)
        acc_relabel_err=np.abs(bootstrap_ci(relabel_acc[irun,:,:].T)-acc_relabel_means)
        #len(plot_rows), len(plot_iters))#, sharex='col', sharey='row')   
        this_fig=ax[0]
        this_fig.errorbar(noises+irun*0.005,acc_relabel_means,yerr=acc_relabel_err)# , yerr=[yerr, 2*yerr], xerr=[xerr, 2*xerr], fmt='--o')
        this_fig.set_title( ' Isotonic Relabelling Accuracy')
        this_fig.set_xlim(-0.01,np.max(noises)+0.01)
        this_fig.set_ylim(0.80,1.1)
        this_fig=ax[1]
        this_fig.errorbar(noises+irun*0.005,acc_means,yerr=acc_err)# , yerr=[yerr, 2*yerr], xerr=[xerr, 2*xerr], fmt='--o')
        #this_fig.errorbar(noises+irun*0.005+0.0025,acc_use_relabelled_means,yerr=acc_use_relabelled_err)# , yerr=[yerr, 2*yerr], xerr=[xerr, 2*xerr], fmt='--o')
        this_fig.set_title(title_typ +'OOS Accuracy Lievens')
        this_fig.set_xlim(-0.01,np.max(noises)+0.01)
        this_fig.set_ylim(0.65,1.1)
        
      
    ax[0].legend(legend_,loc='upper left', prop={'size': 7})
    ax[1].legend(legend_ ,loc='upper left', prop={'size': 7})#legend_acc_
    print('pred acc: ' + str(np.mean(pred_acc,axis=2)))
    print('relab acc: ' + str(np.mean(relabel_acc,axis=2)))
    print('here')
#Z=Z_all[0][:,i_iter]
#Z = Z.reshape(xx.shape)
#if not plt_ in ['data_only','remove_y_model','y_monotone_model']:
#    try:
#        this_fig.contour(xx, yy, Z, 0, linewidths=bound_linewidth, colors=bound_color)
#    except:
#        pass


   
#def load_data_set():
#    # Load data
#
#    data = load_boston()
#    y = data['target']
#    X = data['data']
#    features = data['feature_names']
#    # Specify monotone features
#    incr_feat_names = ['RM']#['RM', 'RAD']
#    decr_feat_names = ['CRIM', 'LSTAT'] # ['CRIM', 'DIS', 'LSTAT']
#    # get 1 based indices of incr and decr feats
#    incr_feats = [i + 1 for i in np.arange(len(features)) if
#                  features[i] in incr_feat_names]
#    decr_feats = [i + 1 for i in np.arange(len(features)) if
#                  features[i] in decr_feat_names]
#    # Convert to classification problem
#    # Multi-class
#    y_multiclass = y.copy()
#    thresh1 = 15
#    thresh2 = 21
#    thresh3 = 27
#    y_multiclass[y > thresh3] = 3
#    y_multiclass[np.logical_and(y > thresh2, y <= thresh3)] = 2
#    y_multiclass[np.logical_and(y > thresh1, y <= thresh2)] = 1
#    y_multiclass[y <= thresh1] = 0
#    # Binary
#    y_binary = y.copy()
#    thresh = 21  # middle=21
#    y_binary[y_binary < thresh] = -1
#    y_binary[y_binary >= thresh] = +1
#    return X, y_binary, y_multiclass, incr_feats, decr_feats


## Load data
#max_N = 400
#np.random.seed(13) # comment out for changing random training set
#X, y_binary, y_multiclass, incr_feats, decr_feats = load_data_set()
#indx_train=np.random.permutation(np.arange(X.shape[0]))[0:max_N]
#inx_test=np.asarray([i for i in np.arange(max_N) if i not in indx_train ])
#X_train=X[indx_train,:]
#X_test=X[inx_test,:]
#
#
#y_train=dict()
#y_test=dict()
#n_classes=dict()
#y_train['binary']=y_binary[indx_train]
#y_train['multiclass']=y_multiclass[indx_train]
#y_test['binary']=y_binary[inx_test]
#y_test['multiclass']=y_multiclass[inx_test]
#n_classes['binary']=2
#n_classes['multiclass']=4
#
#
#def test_model_fit():
#    # Specify hyperparams for model solution
#    n_estimators = 100#200
#    mtry = 3
#    mt_type='ict'
#    require_abs_impurity_redn=True
#    feat_data_types='auto'
#    base_tree_algo='scikit' # isotree
#    normalise_nmt_nodes=2
#    min_split_weight=0.25
#    split_criterion='both_sides_have_min_sample_wgt'
#    split_class='parent_class'
#    split_weight='hybrid_prob_empirical'
#    min_split_weight_type='prop_N' #num_pts
#    simplify=False
#    acc_correct={'multiclass-nmt': 0.752, 
#                 'binary-nmt': 0.84799999999999998, 
#                 'multiclass-mt': 0.74399999999999999, 
#                 'binary-mt': 0.85599999999999998}
#    acc_correct_scikit={'multiclass-mt': 0.76800000000000002, 
#          'binary-nmt': 0.86399999999999999, 
#          'binary-mt': 0.872, 
#          'multiclass-nmt': 0.72799999999999998}
#    acc=dict()
#    oob_score=dict()
#    for response in ['multiclass']:#,binary'multiclass']: #'multiclass']:#
#        y_train_=y_train[response]
#        y_test_=y_test[response]
#        n_classes_=n_classes[response]
#        for constr in ['mt']:#,'nmt']:
#            clf = partial_mt_instance.PartialInstanceClassifier(n_estimators=n_estimators,
#                                          criterion='gini_l1',
#                                          random_state=11,
#                                          feat_data_types=feat_data_types,
#                                          max_features=mtry,
#                                          monotonicity_type=None if constr=='nmt' else mt_type,
#                                          normalise_nmt_nodes=normalise_nmt_nodes,
#                                          require_abs_impurity_redn=require_abs_impurity_redn,
#                                          incr_feats=incr_feats if constr =='mt' else None,
#                                          decr_feats=decr_feats if constr =='mt' else None,
#                                          oob_score=True,
#                                          base_tree_algo=base_tree_algo,
#                                          min_split_weight=min_split_weight,
#                                          min_split_weight_type=min_split_weight_type,
#                                          split_criterion=split_criterion,
#                                          split_class=split_class,
#                                          split_weight=split_weight,
#                                          simplify=simplify
#                                          )
#
#            # Assess fit
#            start=time.time()
#            clf.fit(X_train, y_train_)
#            solve_durn=time.time()-start
#            print('solve took: ' + str(solve_durn) + ' secs')
#            #
#            y_pred = clf.predict(X_test)
#            acc[response + '-' + constr] = np.sum(y_test_ == y_pred) / len(y_test_)
#            oob_score[response + '-' + constr]=clf.oob_score_ #[(clf_sk.tree_.node_count+1.)/2., len(clf_mydt.tree_.leaf_nodes), len(clf_iso.tree_.leaf_nodes), len(clf_oa.tree_.leaf_nodes)]
#                
#            #print(acc[response + '-' + constr])
#            print(np.mean(get_peak_leaves(clf)))
#            print(np.mean(get_leaf_counts(clf)))
#            # Measure monotonicity
#            # mcc[response + '-' + constr] = np.mean(clf.calc_mcc(X_test,incr_feats=incr_feats, decr_feats=decr_feats))
#    
#    print('acc: ' + str(acc))
#    print('n oob_score: ', str(oob_score))
#    # BENCHMARK binary MT acc: 0.864, time: 25.7secs
#    #for key in acc.keys():
#    #    npt.assert_almost_equal(acc[key],acc_correct_scikit[key])
#    # print('mcc: ' + str(mcc))
#    # npt.assert_almost_equal(clf.oob_score_, 0.85999999999)
#    # npt.assert_almost_equal(acc_mc, 0.944999999999)
#
#def benchmark_against_scikit():
#    # binary should match, nulti-class could be different because
#    # pmsvm  uses montone ensembling but scikit uses OVR.
#    #
#    # Specify hyperparams for model solution
#    n_estimators = 200
#    mtry = 3
#    require_abs_impurity_redn=True
#    feat_data_types='auto'
#    
#    acc=dict()
#    oob_score=dict()
#    solve_time=dict()
#    # Solve models
#    for response in ['multiclass','binary']:#,'multiclass']:
#        y_train_=y_train[response]
#        y_test_=y_test[response]
#        n_classes_=n_classes[response]
#        for model in ['isotree','scikit']:
#            if model=='isotree':
#                clf = partial_mt_instance.PartialInstanceClassifier(n_estimators=n_estimators,
#                                          criterion='gini',
#                                          random_state=11,
#                                          feat_data_types=feat_data_types,
#                                          max_features=mtry,
#                                          monotonicity_type=None,
#                                          normalise_nmt_nodes=0,
#                                          require_abs_impurity_redn=require_abs_impurity_redn,
#                                          incr_feats=None,
#                                          decr_feats=None,
#                                          oob_score=True
#                                          )
#                #clf_iso=clf
#                
#            else:
#                clf = RandomForestClassifier(n_estimators=n_estimators,
#                                          criterion='gini',
#                                          random_state=11,
#                                          max_features=mtry,
#                                          oob_score=True)
#            # Assess fit
#            start=time.time()
#            clf.fit(X_train, y_train_)
#            durn=time.time()-start
#
#            #
#            #test constraints are satisifed
#            #res=clf.predict(clf.constraints[0][0,:,1])-clf.predict(clf.constraints[0][0,:,0])
#    #            if model=='pmrf':
#    #                support_vectors[response + '-' + model]= clf.support_vectors_[0][0,:]
#    #                n_support_vectors[response + '-' + model]= np.mean(clf.n_support_)
#    #                dual_coef[response + '-' + model]=np.flip(np.sort(np.abs(clf.dual_coef_[0])),axis=0)
#    #            else:
#    #                support_vectors[response + '-' + model]= clf.support_vectors_[0]
#    #                n_support_vectors[response + '-' + model]= np.sum(clf.n_support_[0:n_classes[response]])
#    #                dual_coef[response + '-' + model]=np.flip(np.sort(np.abs(clf.dual_coef_[0])),axis=0)
#            y_pred = clf.predict(X_test)
#            #oob_scores[response + '-' + model] = clf.oob_score_
#            solve_time[response + '-' + model]=durn
#            acc[response + '-' + model] = np.sum(y_test_ == y_pred) / len(y_test_)
#            oob_score[response + '-' + model]=clf.oob_score_
#            
#            #oob_scores[response + '-' + model] = clf.oob_score_
#            
#            #print(acc[response + '-' + model])
#            
#            # Measure monotonicity
#            #mcc[response + '-' + constr] = np.mean(clf.calc_mcc(X,incr_feats=incr_feats, decr_feats=decr_feats))    
#    print('acc: ' + str(acc))
#    print('n oob_score: ', str(oob_score))
#        #print(n_support_vectors)
#        #print(solve_time)
##    pmsvm_coefs=dual_coef['binary-pmrf']
##    scikit_coefs=dual_coef['binary-scikit']
##    min_len=np.min([pmsvm_coefs.shape[0],scikit_coefs.shape[0]])
##    diff=np.sum(np.abs(scikit_coefs[0:min_len]-pmsvm_coefs[0:min_len]))/np.sum(np.abs(scikit_coefs[0:min_len]))
##    print('dual coef abs diff: ' + str(diff))
##print(support_vectors)
#test_model_fit()
#benchmark_against_scikit()
