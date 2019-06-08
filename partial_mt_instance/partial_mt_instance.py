"""partial_mt_instance

Cone based partially monotone instance classification and relabelling.

PartialInstanceBinaryClassifier() can be used to:
     - construct a binary classifier (using sklearn nomenclature), or
     - perform partially monotone relabelling of a dataset,using 
       clf =PartialInstanceBinaryClassifier(relabel=True,mt_feat_types=_) 
       then clf.fit(X,y) and then accessing clf.y_relabelled


"""

# Authors: Chris Bartley <chris@bartleys.net>
#
# License: BSD 3 clause


import partial_mt_instance
import numpy as np
from cvxopt import matrix as cvxmat, sparse, spmatrix
from cvxopt import solvers

NMT_EQUAL=0
NMT_IGNORE=1
NMT_CONE=2


def std_scale(X,X_means=None,X_stdevs=None):
    if X_means is None:
       X_means =np.mean(X,axis=0) 
    if X_stdevs is None:
       X_stdevs  =np.std(X,axis=0)  
    X_copy=X.copy()
    for i in np.arange(X.shape[1]):
        if np.abs(X_stdevs[i])>1e-8:
            X_copy[:,i]=(X_copy[:,i]-X_means[i])/X_stdevs[i]
    return [X_copy,X_means,X_stdevs]        
            
class PartialInstanceBinaryClassifier(object):
    """A partially monotone instance based classifier (and relabelling algorithm).
    
    The algorithms are descibed in my PhD thesis 'High Accuracy Partially 
    Monotone Ordinal Classification, UWA 2019' Chapter 7.
                 
    Parameters
    ----------
    mt_feat_types : array-like of length n_feats, with values -1 
    (monotone decreasing feature),0 (nonmonotone) or +1 (monotone increasing)

    fit_type : string, optional (default="linear") 'none' or 'linear'

    relabel : bool, (default=False). If True, fit(X,y) will also make
       clf.y_relabelled available.

    local_mt_filter : string, optional (default="none") 'none' or 
      'remove' or 'relabel'. Optionally removes or relabels nonmonotone points
      prior to fitting partially monotone cone.

    local_mt_filter_k : int, optional (default=3): if local_mt_filter='relabel' or 
      'remove', use this as value for kNN nonmonotonicity identification.
      
    scale_X : string (default='yes'): by default scale features of X to have 
       zero mean and unit std deviation so that cone calculation is not biased.

    nmt_plane_type : string (default='joint'), optional 'joint' or 'separate'. 
       If 'joint' a single cone plane is calculated including all features,
       if 'separate' a separate cone is calculated for each monotone feature.

    References
    ----------

    .. [1] C. Bartley, "High Accuracy Partially Monotone Ordinal 
        Classification", UWA 2019 Chapter 7

    """
    def __init__(self,
                 mt_feat_types,
                 fit_type='linear' , # 'none' or 'linear'
                 relabel=False,
                 local_mt_filter='none', #'none' or 'remove' or 'relabel'
                 local_mt_filter_k=3,
                 scale_X='yes',
                 nmt_plane_type='joint' # 'independent'
                 ): 
        self.X=None
        self.y=None
        self.n_feats=len(mt_feat_types)
        self.fit_type=fit_type
        self.mt_feat_types=np.asarray(mt_feat_types)#list(incr_feats)+list(decr_feats))
        self.num_nmt_feats=np.sum(np.abs(self.mt_feat_types==0))
        self.num_mt_feats=self.n_feats-self.num_nmt_feats
        self.nmt_plane_type=nmt_plane_type
        num_planes=1 if nmt_plane_type=='joint' else self.num_nmt_feats if nmt_plane_type=='independent' else 1
        default_hp=np.zeros([num_planes,self.n_feats])
        default_hp[0,self.mt_feat_types!=0]=1
        self.nmt_planes=default_hp
        self.fitted=False
        self.relabel=relabel
        self.nmt_type=NMT_EQUAL
        
        self.local_mt_filter=local_mt_filter
        self.local_mt_filter_k=local_mt_filter_k
        self.scale_X=scale_X
        self.delta_X_size=-1

    
    # note: sample weights not used at present!
    def fit(self, X, y, sample_weight=None,svm_v_empir_error=0.01,max_delta_N=500,presolved_nmt_planes=None):
        """Build the classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like or sparse cvxmat of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse cvxmat is provided, it will be
            converted into a sparse ``csc_cvxmat``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        Returns
        -------
        self : object
            Returns self.
        """
        if self.scale_X=='yes':
            [self.X, self.X_means, self.X_stdevs]= std_scale(X,X_means=None,X_stdevs=None)
            
        else:
           self.X=np.asarray(X,dtype=np.float64,order='c')#[indx,:]
        self.y=np.asarray(y,dtype=np.int32,order='c')#[indx]
        self.n_datapts=X.shape[0]
        self.classes=np.sort(np.unique(y))
        self.n_classes=len(self.classes)
        self.y_cdf=np.zeros([self.X.shape[0],self.n_classes],dtype=np.float64)
        self.y_pdf=np.zeros([self.X.shape[0],self.n_classes],dtype=np.float64)
        self.y_counts=np.zeros([self.X.shape[0],self.n_classes],dtype=np.int32)
        self.y_cdf_relabelled=None
        self.y_relabelled=y
        # this was basedon collapsing dataset into unique X values, since removed. Hence this could be substantially improved.
        for  row in np.arange(self.X.shape[0]):
            classes=np.asarray([self.y[row]])#[self.y[indx] for indx in np.arange(len(self.y)) if rev_indices [indx]==row])
            self.y_counts[row,:]=[np.sum(classes==cls) for cls in self.classes ]
            self.y_pdf[row,:]=self.y_counts[row,:]/np.sum(self.y_counts[row,:])
            self.y_cdf[row,:]=[np.sum(self.y_pdf[row,:cls+1]) for cls in np.arange(self.n_classes) ]

        if self.num_nmt_feats==0 or self.fit_type=='none': # no nmt feats, so solution is default planes (soft nmt makes no difference)
            self.fitted=True
            self.nmt_type=NMT_IGNORE
        else:
            # PREPARE DATA (reverse decr feats)
            self.nmt_type=NMT_CONE
            # APPLY NMT_FILTER IF REQUESTED
            if self.local_mt_filter!='none':
                pm_pairs=get_increasing_pairs(self.X,self.mt_feat_types,nmt_type=NMT_IGNORE)   
                pair_nmt_dists=calc_nmt_distances(self.X,pm_pairs,self.mt_feat_types)
                relabelled_nett_lmtc_increase=calc_nett_lmtc_incr(self.X,self.y_counts,pm_pairs,pair_nmt_dists,self.local_mt_filter_k)
                if self.local_mt_filter=='remove':                                               
                    y_mask=relabelled_nett_lmtc_increase<=0.
                    X_cone=self.X[y_mask,:]
                    y_cone_cdf=self.y_cdf[y_mask,:]
                else:
                    raise NotImplemented
                    
            else:
                X_cone=self.X
                y_cone_cdf=self.y_cdf
            
            pm_pairs=get_increasing_pairs(X_cone,self.mt_feat_types,nmt_type=NMT_IGNORE)   
            
            # BUILD TRAINING DATA
            # get comparable pairs & remove redundant (X only)
            pm_pairs_clean=pm_pairs
            # filter for incomparable edge points (using Y)
            pm_pairs_boundary=get_boundary_pts(pm_pairs_clean,y_cone_cdf)
            # extract training data
            delta_X=get_one_class_train_data(X_cone,self.mt_feat_types,pm_pairs_boundary)
            # FIT ONE CLASS SVM
            self.delta_X_size=delta_X.shape[0]
            if delta_X.shape[0]>max_delta_N:
                print('SUBSAMPLING ' + str(delta_X.shape[0]) + ' to ' + str(max_delta_N) )
                if svm_v_empir_error<=0.01:  # choose to retain points likely near the boundary
                    # calculate angle approximation
                    horizontal=np.sqrt(np.sum(delta_X[:,np.abs(self.mt_feat_types)==1]**2,axis=1))
                    vertical=np.sqrt(np.sum(delta_X[:,np.abs(self.mt_feat_types)==0]**2,axis=1))
                    angle=vertical/(horizontal+1e-7)
                    ids=np.argsort(angle)[0:max_delta_N]
                else:# use random sample
                    ids=np.arange(delta_X.shape[0],dtype=np.int32)
                    ids=np.random.permutation(ids)[0:max_delta_N]
                delta_X=delta_X[ids,:]
            if self.nmt_plane_type=='joint':
                if svm_v_empir_error==1.0: # cheat - we know the answer!
                    self.nmt_planes[ 0,self.mt_feat_types!=0]=1./np.sum(self.mt_feat_types!=0)
                    self.nmt_planes[ 0,self.mt_feat_types==0]=0.
                else:
                    self.nmt_planes[ 0,:]=fit_one_class_svm(delta_X,weights=np.ones(delta_X.shape[0]),v=svm_v_empir_error,mt_feat_types_=self.mt_feat_types)#delta_y=np.ones(delta_X.shape[0]),
            else: #fit  independent planes one per nmt feat
                delta_X_mt_feats=delta_X[:,self.mt_feat_types!=0]
                mt_feat_types_indep=self.mt_feat_types[self.mt_feat_types!=0]
                mt_feat_types_indep=np.hstack([mt_feat_types_indep,[0.]])
                nmt_feats=np.arange(len(self.mt_feat_types))[self.mt_feat_types==0]
                for i_nmt_feat in np.arange(self.num_nmt_feats):
                    if presolved_nmt_planes is None:
                        solve_this_feat=True
                    else:
                        solve_this_feat=np.sum(np.abs(presolved_nmt_planes[i_nmt_feat,:]))==0.
                    if solve_this_feat:
                        if svm_v_empir_error==1.0: # cheat - we know the answer!
                            self.nmt_planes[ i_nmt_feat,self.mt_feat_types!=0]=1./np.sum(self.mt_feat_types!=0)
                            self.nmt_planes[ i_nmt_feat,self.mt_feat_types==0]=0.
                        else:
                            nmt_feat=nmt_feats[i_nmt_feat]
                            delta_X_indep=np.hstack([delta_X_mt_feats,delta_X[:,nmt_feat].reshape([-1,1])])
                            plane=fit_one_class_svm(delta_X_indep,weights=np.ones(delta_X_indep.shape[0]),v=svm_v_empir_error,mt_feat_types_=mt_feat_types_indep)
                            self.nmt_planes[ i_nmt_feat,self.mt_feat_types!=0]=plane[0:-1]
                            self.nmt_planes[ i_nmt_feat,nmt_feat]=plane[-1]
                    else:
                        self.nmt_planes[ i_nmt_feat,:]=presolved_nmt_planes[i_nmt_feat,:]
        # relabel training set if requested.
        if self.relabel: # use isotonnic regression to recalculate the cdf
            # get increasing pairs
            pm_pairs=get_increasing_pairs(self.X,self.mt_feat_types, nmt_type=self.nmt_type, nmt_intercept=0., nmt_planes=self.nmt_planes)#[0,:])   
            pm_pairs_clean=eliminate_unnecessary_incr_pairs(pm_pairs)        
            # calculate isotonic regression
            cdf=self.y_cdf#self.cdf[leaf_ids_,:]   #get_cum_probabilities()
            weights=np.ones(len(self.y))#self.sample_weight[leaf_ids_]
            cdf_iso=np.ones(cdf.shape) 
            pdf_iso=np.zeros(cdf.shape) 
            cum_sse=0.          
            for i_class in np.arange(cdf.shape[1]):
                probs_class=cdf[:,i_class]
                gir=partial_mt_instance.GeneralisedIsotonicRegression()
                if i_class<cdf.shape[1]-1:
                    status,monotonised_cdf=gir.fit(probs_class,pm_pairs_clean,sample_weight=weights,increasing=False)
                    if status=='optimal':
                        cdf_iso[:,i_class]=np.round(monotonised_cdf,6)
                    else:
                        print('Relabelling failed, usually due to singular KKT matrix. No relabelling performed (' +status+').')
                
                if i_class==0:
                    pdf_iso[:,i_class]=cdf_iso[:,i_class]
                else:
                    pdf_iso[:,i_class]=cdf_iso[:,i_class]-cdf_iso[:,i_class-1]
            cum_sse=np.sum((cdf_iso-cdf)**2)
            
            # store result
            self.y_cdf_relabelled=cdf_iso
            self.y_cdf_relabelled=self.y_cdf_relabelled
            y_uniq_relabelled=self.classes.take(np.argmax(cdf_iso>=0.5, axis=1), axis=0)
            self.y_relabelled=y_uniq_relabelled#[rev_indices]
        return self    


    def predict_proba_loo(self, S=0.5,use_relabelled_if_avail=True):#,loo_y=None,X_to_use=None,y_to_use=None):
        return self.predict_proba(self.X,S=S,use_relabelled_if_avail=use_relabelled_if_avail,loo_=True,scale_X=False)
        
    def predict_proba(self, X_pred_,S=0.5,use_relabelled_if_avail=True,loo_=False,scale_X='auto'):#,loo_y=None,X_to_use=None,y_to_use=None):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : array-like or sparse cvxmat of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse cvxmat is provided
            to a sparse ``csr_cvxmat``.
        loo : boolean, (default=False)
            True to exclude one matchng datapoint from training set when doing prediction.
            Effectively Leave-One-Out cross validation.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        # Uses Lievens et al 2008 A probabilistic framework for the design of instance-based supervised ranking algorithms in an ordinal setting
        # to create an instance based classifier based on the partial order of the points
        #if X_to_use is None:
        X_=self.X
        if len(X_pred_.shape)<2:
            X_pred_=X_pred_.reshape((1,X_pred_.shape[0]))
        if scale_X=='auto':
            if self.scale_X=='yes':
                [X_pred, X_means, X_stdevs]= std_scale(X_pred_,X_means=self.X_means,X_stdevs=self.X_stdevs)
            else:
                X_pred=X_pred_.copy()
        else:
            if scale_X:
                [X_pred, X_means, X_stdevs]= std_scale(X_pred_,X_means=self.X_means,X_stdevs=self.X_stdevs)
            else:
                X_pred=X_pred_.copy()
        y_counts_=self.y_counts
        if use_relabelled_if_avail:
            if self.y_cdf_relabelled is None:
                y_cdf_=self.y_cdf
            else:
                y_cdf_=self.y_cdf_relabelled
        else: # use empirical cdfs
            y_cdf_=self.y_cdf
        y_pred=np.zeros(X_pred.shape[0])
        cdf_pred_all=np.ones([X_pred.shape[0],self.n_classes])
        pdf_pred_all=np.ones([X_pred.shape[0],self.n_classes])
        F_min_all=np.ones([X_pred.shape[0],self.n_classes])
        F_max_all=np.ones([X_pred.shape[0],self.n_classes])
        N_min_all=np.zeros([X_pred.shape[0],self.n_classes])
        N_max_all=np.zeros([X_pred.shape[0],self.n_classes])
        for i_pred in np.arange(X_pred.shape[0]):
            x_p=X_pred[i_pred,:]
            s_=np.zeros(self.n_classes)
            t_=np.zeros(self.n_classes)
            comps=compare(x_p,X_, self.mt_feat_types,check_nmt_feats=self.nmt_type,strict=False,nmt_plane_intercepts=[0.], nmt_plane_norms_=self.nmt_planes)#[0,:])

            if loo_:
                indx=np.arange(X_.shape[0])
                indx=indx[indx!=i_pred]
                comps_idx=np.logical_and(comps[0]!=i_pred,comps[1]!=i_pred)
                F_min,F_max,N_min,N_max=partial_mt_instance.get_F_N_c(y_cdf_[indx,:],y_counts_[indx,:],comps[comps_idx,:],self.n_classes)
            else:
                F_min,F_max,N_min,N_max=partial_mt_instance.get_F_N_c(y_cdf_,y_counts_,comps,self.n_classes)
            
            # the next secction uses Lievens et al. Example 2 'reversed 
            # preference' resolution, pp 134-136 'formula in (20) under Corollary 2
            for l in np.arange(self.n_classes):
                if (N_min[l]+N_max[l])==0:
                    s_[l]=S
                    t_[l]=S
                else:
                    s_[l]=N_max[l]/(N_min[l]+N_max[l])
                    t_[l]=N_min[l]/(N_min[l]+N_max[l])
            # calculate cdf for this training point
            cdf_pred=np.zeros(self.n_classes)
            pdf_pred=np.zeros(self.n_classes)
            for l in np.arange(self.n_classes):
                cdf_pred[l]=s_[l]*F_min[l] + (1-s_[l])*F_max[l] if F_min[l]>=F_max[l] else  t_[l]*F_min[l] + (1-t_[l])*F_max[l]                        
                if l==0:
                    pdf_pred[l]=cdf_pred[l]
                else:
                    pdf_pred[l]=  cdf_pred[l]-cdf_pred[l-1]
            # store calculations and prediction
            F_min_all[i_pred,:]=F_min
            F_max_all[i_pred,:]=F_max
            N_min_all[i_pred,:]=N_min
            N_max_all[i_pred,:]=N_max
            cdf_pred_all[i_pred,:]=cdf_pred
            if np.abs(np.sum(pdf_pred))<1e-6:
                print('what the?')
            pdf_pred_all[i_pred,:]=pdf_pred
            y_pred[i_pred]=self.classes[np.argmax(cdf_pred>=0.5)]
        counts=np.vstack([np.sum(N_min_all,axis=1),np.sum(N_max_all,axis=1)]).T
        return [pdf_pred_all,y_pred,counts]
    def predict(self, X_pred,S=0.5,use_relabelled_if_avail=True):#,loo=False,loo_y=None,X_to_use=None,y_to_use=None):
        [pdf_pred_all,y_pred,counts]=self.predict_proba( X_pred,S=S,use_relabelled_if_avail=use_relabelled_if_avail)#,loo_y=loo_y,X_to_use=X_to_use,y_to_use=y_to_use),loo=False
        return y_pred
    

def get_one_class_train_data(X,mt_feat_types,pm_pairs_boundary):
    delta_X=np.zeros([pm_pairs_boundary.shape[0],X.shape[1]],dtype=np.float64,order='c')
    for j in np.arange(pm_pairs_boundary.shape[0]):
        delta_X[j,:]=np.abs(X[pm_pairs_boundary[j,0],:]-X[pm_pairs_boundary[j,1],:])    
    
    return delta_X
       
def get_boundary_pts(pm_pairs_clean,y_cdf):
    pm_pairs_boundary=np.empty_like(pm_pairs_clean)
    n_classes=y_cdf.shape[1]
    k=0
    for j in np.arange(pm_pairs_clean.shape[0]):
        boundary_pt=False
        for c_ in np.arange(n_classes):
            if y_cdf[pm_pairs_clean[j,0],c_]<y_cdf[pm_pairs_clean[j,1],c_]:
                boundary_pt=True
                break
        if boundary_pt:
            pm_pairs_boundary[k,:]=pm_pairs_clean[j,:]
            k=k+1
    return pm_pairs_boundary[0:k,:]
               
    
    return pm_pairs_boundary

def compare(x1,X, mt_feat_types_,check_nmt_feats=0,strict=False,nmt_plane_intercepts=[0.], nmt_plane_norms_=None):
    res=np.ones(X.shape[0],dtype=np.int32)
    mt_feat_types=np.asarray(mt_feat_types_,dtype=np.float64,order='c')
    if nmt_plane_norms_ is None:
        nmt_plane_norms=np.zeros(X.shape[1],dtype=np.float64)
    else:
        nmt_plane_norms=nmt_plane_norms_
    if np.isscalar(nmt_plane_intercepts):
        nmt_plane_intercepts_=np.asarray([nmt_plane_intercepts],dtype=np.float64)
    else:
        nmt_plane_intercepts_=np.asarray(nmt_plane_intercepts,dtype=np.float64)
    partial_mt_instance.compare_pt_with_array(x1,X, mt_feat_types,check_nmt_feats,1 if strict else 0,nmt_plane_intercepts_, nmt_plane_norms,res)
    
    return res
    
def get_increasing_pairs(X,mt_feat_types, nmt_type=NMT_IGNORE, nmt_intercept=0., nmt_planes=None): #nmt_plane
    max_pairs=int(np.round(X.shape[0]*X.shape[0]))
    incr_pairs=np.zeros([max_pairs,2],dtype=np.int32)
    if nmt_planes is None:
        nmt_planes_=np.zeros(X.shape[1],dtype=np.float64)
    else:
        nmt_planes_=nmt_planes
    n_pairs_new=partial_mt_instance.get_increasing_pairs_array(X,mt_feat_types, nmt_type, nmt_intercept, nmt_planes_,incr_pairs)
    return incr_pairs[0:n_pairs_new,:]
    
def calc_nmt_distances(X,pm_pairs,mt_feat_types):
    nmt_feats=np.arange(len(mt_feat_types),dtype=np.int32)[mt_feat_types==0]
    sum_=np.zeros(pm_pairs.shape[0],dtype=np.float64)
    for i_feat in nmt_feats:
        sum_=sum_+(X[pm_pairs[:,0],i_feat]-X[pm_pairs[:,1],i_feat])**2
    sum_=np.sqrt(sum_)
    return sum_            



def calc_nett_lmtc_incr(X,y_counts,pm_pairs,pair_nmt_dists,local_mt_filter_k):
    # build descriptive matrices
    N=X.shape[0]
    n_classes=y_counts.shape[1]
    m_num_comparable=np.zeros(N,dtype=np.int32)
    m_distances=np.zeros([N,2*N],dtype=np.float64)
    m_dirn_to_point=np.zeros([N,2*N],dtype=np.int32)
    m_counts=np.zeros([N,2*N],dtype=np.int32)
    m_pt_ids=np.zeros([N,2*N],dtype=np.int32)
    for k in np.arange(pm_pairs.shape[0]):
        i_lesser=pm_pairs[k,0]
        i_greater=pm_pairs[k,1]
        if i_lesser!=i_greater:
            m_distances[i_lesser,m_num_comparable[i_lesser]]=pair_nmt_dists[k]
            m_distances[i_greater,m_num_comparable[i_greater]]=pair_nmt_dists[k]
            m_dirn_to_point[i_lesser,m_num_comparable[i_lesser]]=+1
            m_dirn_to_point[i_greater,m_num_comparable[i_greater]]=-1
            m_pt_ids[i_lesser,m_num_comparable[i_lesser]]=i_greater
            m_pt_ids[i_greater,m_num_comparable[i_greater]]=i_lesser
            m_counts[i_lesser,m_num_comparable[i_lesser]]=np.sum(y_counts[i_greater,:])
            m_counts[i_greater,m_num_comparable[i_greater]]=np.sum(y_counts[i_lesser,:])
            m_num_comparable[i_lesser]=m_num_comparable[i_lesser]+1
            m_num_comparable[i_greater]=m_num_comparable[i_greater]+1
        
    # sort k nearest neighbours and calculate nett relabelling improvement
    nett_relab_impr=np.zeros(N,dtype=np.float64)
    for i in np.arange(N):
        this_pt_counts= y_counts[i,:]
        this_pt_class=np.argmax(this_pt_counts)
        distances=m_distances[i,0:m_num_comparable[i]]
        num_pts=np.min([local_mt_filter_k,m_num_comparable[i]])
        sorted_indexes=np.argsort(distances)[0:num_pts]
        ids=m_pt_ids[i,sorted_indexes]
        dirns=m_dirn_to_point[i,sorted_indexes]
        counts=m_counts[i,sorted_indexes]
        ttl_count=np.sum(counts)
        indxs_len=num_pts
        # trim if necessary due to sample weights
        if ttl_count>local_mt_filter_k:
            cc=0
            count=0
            while count<local_mt_filter_k:
                count=count+counts[cc]
                cc=cc+1
            dirns=dirns[0:cc]
            counts=counts[0:cc]
            ttl_count=count
            indxs_len=cc
        if ttl_count>0:
            nmt_pts=np.zeros(n_classes,dtype=np.float64)
            for c_ in np.arange(n_classes):
                greater_counts=np.zeros(n_classes,dtype=np.float64)
                lesser_counts=np.zeros(n_classes,dtype=np.float64)
                for j in np.arange(indxs_len):
                    true_indx=ids[j]#sorted_indexes[j]
                    if dirns[j]==+1:
                        greater_counts=greater_counts+y_counts[true_indx,:]
                    else:
                        lesser_counts=lesser_counts+y_counts[true_indx,:]
                # calculate non-monotonicity
                greater_nmt_count=  0 if c_==0 else np.sum(greater_counts[0:c_])   
                lesser_nmt_count=  0 if c_==n_classes-1 else np.sum(lesser_counts[c_+1:])   
                nmt_pts[c_]=greater_nmt_count+lesser_nmt_count
            relab_class=np.argmin(nmt_pts)
            mean_pt_relab_impr= (nmt_pts[this_pt_class]-nmt_pts[relab_class])/ttl_count#y_prob_1[i]*pos_pt_relab_impr+(1-y_prob_1[i])*neg_pt_relab_impr
            nett_relab_impr[i]=mean_pt_relab_impr
        else:
            nett_relab_impr[i]=0.

    return nett_relab_impr

def eliminate_unnecessary_incr_pairs(pm_pairs):
    if len(pm_pairs)==0:
        return pm_pairs
    else:
        out_pm_pairs=np.zeros(pm_pairs.shape,dtype=np.int32)
        num_pairs=partial_mt_instance.calculate_transitive_reduction_c(pm_pairs,out_pm_pairs)
        out_pm_pairs=out_pm_pairs[0:num_pairs,:]
        
        out_pm_pairs_w=np.zeros(pm_pairs.shape,dtype=np.int32)
        num_pairs=partial_mt_instance.calculate_transitive_reduction_c_warshal(pm_pairs,out_pm_pairs_w)
        out_pm_pairs_w=out_pm_pairs_w[0:num_pairs,:]
        
        return out_pm_pairs
    
def generate_partial_mono_data_by_function(num_samples,mt_feat_types, classes=[-1,+1], class_dist='uniform',feat_dist='uniform',feat_spec=[-1,1], sign_function=None,random_state=None,NMI_target=None):
    # adatation of Potharst et al 2009 'Two algorithms for generating structured and unstructured monotone ordinal data sets'                
    # to generate random PARTIALLY monotone data
    #
    ## generate random Xs
    num_feats=len(mt_feat_types)
    X=np.zeros([num_samples,num_feats])
    if random_state!=None:
        np.random.seed(random_state)
    for i in np.arange(num_samples):
        if feat_dist=='uniform':
            X[i,:]=feat_spec[0]+np.random.rand(num_feats)*(feat_spec[1]-feat_spec[0])
        elif feat_dist=='normal':
            X[i,:]=np.random.normal(loc=feat_spec[0], scale=feat_spec[1], size=num_feats)
    # calculate class for each X from function
    y=np.sign(sign_function(X,mt_feat_types))   
    y[y==0]=-1

    # add noise if requested
    noise_pts=[]
    y_true=y.copy()
    if not NMI_target is None:
        noise_pts=np.random.permutation(np.arange(X.shape[0]))[0:np.int(np.floor(X.shape[0]*NMI_target))]
        y[noise_pts]=-y[noise_pts]
    return [X,y,noise_pts,y_true]

def fit_one_class_svm_pub(delta_X,delta_y,weights,v,mt_feat_types):
        
        N = delta_X.shape[0]
        p = delta_X.shape[1]
        mt_feats = np.arange(p)[mt_feat_types!=0]#np.asarray(list(incr_feats) + list(decr_feats))
        nmt_feats = np.arange(p)[mt_feat_types==0]#np.asarray(
        solvers.options['show_progress'] = False
        if N == 0:
            return np.zeros(delta_X.shape[1])#[-99]
        else:
            # Build QP matrices
            # Minimize     1/2 x^T P x + q^T x
            # Subject to   G x <= h
            #             A x = b
            if weights is None:
                weights = np.ones(N)
            P = np.zeros([p + N, p + N])
            for ip in nmt_feats :
                P[ip, ip] = 1
            #for ip in mt_feats :
            #    P[ip, ip] = 1
            q = 1 / (N * v) * np.ones((N + p, 1))
            q[0:p, 0] = 0
            q[p:, 0] = q[p:, 0] * weights
            G1a = np.zeros([p, p])
            for ip in np.arange(p):
                G1a[ip, ip] = -1 if ip in mt_feats  else 1
            G1 = np.hstack([G1a, np.zeros([p, N])])
            G2 = np.hstack([np.zeros([N, p]), -np.eye(N)])
            G3 = np.hstack([delta_X, -np.eye(N)])
            G = np.vstack([G1, G2, G3])
            h = np.zeros([p + 2 * N])
            A = np.zeros([1, p + N])
            for ip in np.arange(p):
                A[0, ip] = 1 if ip in mt_feats  else -1
            b = np.asarray([1.])
            #b = np.asarray([0.])
            P = cvxmat(P)
            q = cvxmat(q)
            A = cvxmat(A)
            b = cvxmat(b)
            # options['abstol']=1e-20 #(default: 1e-7).
            # options['reltol']=1e-11 #(default: 1e-6)
            sol = solvers.qp(P, q, cvxmat(G), cvxmat(h), A, b)
            if sol['status'] != 'optimal':
                print(
                    '****** NOT OPTIMAL ' +
                    sol['status'] +
                    ' ******* [N=' +
                    str(N) +
                    ', p=' +
                    str(p) +
                    ']')
                return np.zeros(delta_X.shape[1])#[-99]
            else:
                soln = sol['x']
                w = np.ravel(soln[0:p, :])
                # err = np.asarray(soln[-N:, :])
                return w

def fit_one_class_svm(delta_X,weights,v,mt_feat_types_):
        
        mt_feat_types=np.asarray(mt_feat_types_,dtype=np.int32)
        N = delta_X.shape[0]
        p = delta_X.shape[1]
        mt_feats = np.arange(p)[mt_feat_types!=0]#np.asarray(list(incr_feats) + list(decr_feats))
        nmt_feats = np.arange(p)[mt_feat_types==0]#np.asarray(
        solvers.options['show_progress'] = False
        if N == 0:
            return np.zeros(delta_X.shape[1])#[-99]
        else:
            # Build QP matrices
            # Minimize     1/2 x^T P x + q^T x
            # Subject to   G x <= h
            #             A x = b
            if weights is None:
                weights = np.ones(N)
            P = np.zeros([p + 2*N, p + 2*N])
            lambda_=N
            for ip in nmt_feats :
                P[ip, ip] = lambda_*0.01
            for ip in mt_feats :
                P[ip, ip] = lambda_*0.0001# 0.01
            q=np.zeros(p+2*N)
            q[p:p+N]=v
            q[p+N:]=(1.-v)
            G1a = np.zeros([p, p])
            for ip in np.arange(p):
                G1a[ip, ip] = -1 if ip in mt_feats  else 1
            G1 = np.hstack([G1a, np.zeros([p, 2*N])])
            G2 = np.hstack([np.zeros([2*N, p]), -np.eye(2*N)])
            G3 = np.hstack([-delta_X ,-np.eye(N),np.zeros([N,N])])
            G4 = np.hstack([delta_X ,np.zeros([N,N]),-np.eye(N)])
            G = np.vstack([G1, G2, G3,G4])
            h = np.zeros([p + 4 * N])
            A = np.zeros([1, p + 2*N])
            for ip in np.arange(p):
                A[0, ip] = 1 if ip in mt_feats  else -1
            b = np.asarray([1.])
            P = cvxmat(P)
            q = cvxmat(q)
            A = cvxmat(A)
            b = cvxmat(b)
            # options['abstol']=1e-20 #(default: 1e-7).
            # options['reltol']=1e-11 #(default: 1e-6)
            sol = solvers.qp(P, q, cvxmat(G), cvxmat(h), A, b)

            if sol['status'] != 'optimal':
                print(
                    '****** NOT OPTIMAL ' +
                    sol['status'] +
                    ' ******* [N=' +
                    str(N) +
                    ', p=' +
                    str(p) +
                    ']')
                return np.zeros(delta_X.shape[1])-99#[-99]
            else:
                soln = sol['x']
                w = np.ravel(soln[0:p, :])
                if np.sum(np.abs(w))==0.:
                    print('what the?')
                # err = np.asarray(soln[-N:, :])
                return w
            
