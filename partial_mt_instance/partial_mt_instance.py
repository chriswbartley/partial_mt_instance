"""Forest of trees-based ensemble methods

Those methods include random forests and extremely randomized trees.

The module structure is the following:

- The ``BaseForest`` base class implements a common ``fit`` method for all
  the estimators in the module. The ``fit`` method of the base ``Forest``
  class calls the ``fit`` method of each sub-estimator on random samples
  (with replacement, a.k.a. bootstrap) of the training set.

  The init of the sub-estimator is further delegated to the
  ``BaseEnsemble`` constructor.

- The ``ForestClassifier`` and ``ForestRegressor`` base classes further
  implement the prediction logic by computing an average of the predicted
  outcomes of the sub-estimators.

- The ``RandomForestClassifier`` and ``RandomForestRegressor`` derived
  classes provide the user with concrete implementations of
  the forest ensemble method using classical, deterministic
  ``DecisionTreeClassifier`` and ``DecisionTreeRegressor`` as
  sub-estimator implementations.

- The ``ExtraTreesClassifier`` and ``ExtraTreesRegressor`` derived
  classes provide the user with concrete implementations of the
  forest ensemble method using the extremely randomized trees
  ``ExtraTreeClassifier`` and ``ExtraTreeRegressor`` as
  sub-estimator implementations.

Single and multi-output problems are both handled.

"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#
# License: BSD 3 clause

#from sklearn.ensemble.forest import ForestClassifier
#from pmtree import PMDecisionTreeClassifier
#from partial_mt_instance import IsoDecisionTreeClassifier
import partial_mt_instance
#import warnings
import numpy as np
from cvxopt import matrix as cvxmat, sparse, spmatrix
#from cvxopt.solvers import qp, options
from cvxopt import solvers
#from sklearn.utils import check_random_state, check_array, compute_sample_weight
#from numpy import bincount #, parallel_helper
#from softcomp import SoftComp
NMT_EQUAL=0
NMT_IGNORE=1
NMT_CONE=2


def std_scale(X,X_means=None,X_stdevs=None):
    if X_means is None:
       X_means =np.mean(X,axis=0) 
    if X_stdevs is None:
       X_stdevs  =np.std(X,axis=0)  
       #X_stdevs[X_stdevs==0.]=1.
    X_copy=X.copy()
    for i in np.arange(X.shape[1]):
        if np.abs(X_stdevs[i])>1e-8:
            X_copy[:,i]=(X_copy[:,i]-X_means[i])/X_stdevs[i]#scaled_lower_lim+(scaled_upper_lim-scaled_lower_lim) * (X[:,i]-X_mins[i])/(X_maxs[i]-X_mins[i])
    return [X_copy,X_means,X_stdevs]        
            
class PartialInstanceBinaryClassifier(object):
    """A random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and use averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float, optional (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. versionadded:: 0.18

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    class_weight : dict, list of dicts, "balanced",
        "balanced_subsample" or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    See also
    --------
    DecisionTreeClassifier, ExtraTreesClassifier
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
        """Build a forest of trees from the training set (X, y).
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
        #indx=np.argsort(X[:,0])
        #indx=np.arange(X.shape[0])
        if self.scale_X=='yes':
            [self.X, self.X_means, self.X_stdevs]= std_scale(X,X_means=None,X_stdevs=None)
            
        else:
            #X_=X.copy()
            self.X=np.asarray(X,dtype=np.float64,order='c')#[indx,:]
        self.y=np.asarray(y,dtype=np.int32,order='c')#[indx]
        self.n_datapts=X.shape[0]
        self.classes=np.sort(np.unique(y))
        self.n_classes=len(self.classes)
        #n_comparisions=self.n_datapts**2-self.n_datapts
        # prepare unique (stochastic) version of data (needed for predict() using instance based method)
        #[self.X_unique,indices,rev_indices,counts]=np.unique(X, return_index=True, return_inverse=True, return_counts=True, axis=0)
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
            #return self
        else:
            # PREPARE DATA (reverse decr feats)
            self.nmt_type=NMT_CONE
#            X_ordered=X.copy()
#            for dec_feat in np.arange(self.n_feats)[self.mt_feat_types==-1]:
#                X_ordered[:,dec_feat]=-1*X_ordered[:,dec_feat]
            # APPLY NMT_FILTER IF REQUESTED
            if self.local_mt_filter!='none':
                pm_pairs=get_increasing_pairs(self.X,self.mt_feat_types,nmt_type=NMT_IGNORE)   
                pair_nmt_dists=calc_nmt_distances(self.X,pm_pairs,self.mt_feat_types)
                relabelled_nett_lmtc_increase=calc_nett_lmtc_incr(self.X,self.y_counts,pm_pairs,pair_nmt_dists,self.local_mt_filter_k)
                #relabelled_nett_lmtc_increase_old=calc_nett_lmtc_incr_old(self.X,self.y_pdf[:,1],np.sum(self.y_counts,axis=1),pm_pairs,pair_nmt_dists,self.local_mt_filter_k)
                if self.local_mt_filter=='remove':                                               
                    y_mask=relabelled_nett_lmtc_increase<=0.
                    X_cone=self.X[y_mask,:]
                    y_cone_cdf=self.y_cdf[y_mask,:]
                    #y_uniq_cone_prob1=self.y_pdf[y_mask,1]
                else:
                    raise NotImplemented
#                    y_relabelled=self.y_pdf[:,1].copy()
#                    y_mask_positive=np.logical_and(relabelled_nett_lmtc_increase>0.,self.y_pdf[:,1]>0.5)
#                    y_mask_negative=np.logical_and(relabelled_nett_lmtc_increase>0.,self.y_pdf[:,1]<=0.5)
#                    y_relabelled[y_mask_positive]=0.
#                    y_relabelled[y_mask_negative]=1.
#                    X_cone=self.X
#                    y_uniq_cone_prob1=y_relabelled
                    
            else:
                X_cone=self.X
                y_cone_cdf=self.y_cdf
                #y_uniq_cone_prob1=self.y_pdf[:,1]
            
            pm_pairs=get_increasing_pairs(X_cone,self.mt_feat_types,nmt_type=NMT_IGNORE)   
            
            # BUILD TRAINING DATA
            # get comparable pairs & remove redundant (X only)
            #pm_pairs_clean=eliminate_unnecessary_incr_pairs(pm_pairs)        
            pm_pairs_clean=pm_pairs
            # filter for incomparable edge points (using Y)
            pm_pairs_boundary=get_boundary_pts(pm_pairs_clean,y_cone_cdf)
            # extract training data
            delta_X=get_one_class_train_data(X_cone,self.mt_feat_types,pm_pairs_boundary)
            # FIT ONE CLASS SVM
            #max_delta_N=250
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
            #if delta_X.shape[0]<5:
            #    print('what the?')
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
            #dot_=np.dot(delta_X,self.nmt_planes[ 0,:])
            #print(dot_)
            # STORE RESLTING HYPERPLANE
            #self.X=X_ordered
        # relabel training set if requested.
        if self.relabel: # use isotonnic regression to recalculate the cdf
            # get increasing pairs
            #if 'pm_pairs' not in locals():
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
                    #cdf_iso[:,i_class]=gir.fit(probs_class,pm_pairs_clean,sample_weight=weights,increasing=False)
                    #print(probs_class)
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
            #y_uniq_relabelled=np.take(self.classes,np.where(cdf_iso[:,0]<0.5,1,0))
            #y_relabelled_orig=np.zeros(len(self.y),dtype=np.int32)
            self.y_relabelled=y_uniq_relabelled#[rev_indices]
        return self    
#    def compare(self,x1_in,x2_in, check_nmt_feats=True,strict=False):
#        # returns: -1 if decreasing, 0 if identical, +1 if increasing, -99 if incomparable
#        if self.num_mt_feats==0:
#            return -99
#        elif len(x1_in.shape)>1:
#            x1=np.ravel(x1_in)
#            x2=np.ravel(x2_in)
#        else:
#            x1=x1_in.copy()
#            x2=x2_in.copy()
#        # check for identical
#        if np.array_equal(x1,x2):
#            return 0
#        # reverse polarity of decreasing features
#        #for dec_feat in self.decr_feats:
#        for dec_feat in np.arange(self.n_feats)[self.mt_feat_types==-1]:
#            x1[dec_feat]=-1*x1[dec_feat]
#            x2[dec_feat]=-1*x2[dec_feat]
#        # check mt feats all increasing (or decreasing)
#        mt_feats_difference=np.zeros(self.n_feats)
#        if self.num_mt_feats>0: 
#            mt_feats_difference[self.mt_feat_types!=0]=x2[self.mt_feat_types!=0]-x1[self.mt_feat_types!=0]
#        mt_feats_same=np.sum(mt_feats_difference[self.mt_feat_types!=0]==0)
#        if strict:
#            mt_feats_incr=np.sum(mt_feats_difference[self.mt_feat_types!=0]>0) 
#            mt_feats_decr=np.sum(mt_feats_difference[self.mt_feat_types!=0]<0)            
#        else:
#            mt_feats_incr=np.sum(mt_feats_difference[self.mt_feat_types!=0]>=0) 
#            mt_feats_decr=np.sum(mt_feats_difference[self.mt_feat_types!=0]<=0) 
#        if mt_feats_same==self.num_mt_feats:
#            comp=0
#        elif mt_feats_incr==self.num_mt_feats: # increasing
#            comp=+1        
#        elif mt_feats_decr==self.num_mt_feats: # decreasing
#            comp=-1
#        else: # incomparale
#            comp=-99
#        # short exit if available
#        if comp==-99 or comp==0:
#            return -99
#        # if still going, check mt feats by weakened planes
#        if self.num_nmt_feats==0:# or not check_nmt_feats:
#            nmt_feat_compliance=True
#        else:
#            if self.fit_type=='none':
#                nmt_feat_compliance=True
#            else:
#                #if self.fit_algo=='one-sided' or self.fit_algo=='two-sided':
#                    # put all values in positive quadrant
#                x_diff=np.abs(x2-x1)
#                nmt_feat_compliance=True 
#                for i_nmt_feat in np.arange(1):#self.num_nmt_feats):
#                    nmt_feat=0#self.nmt_feats[i_nmt_feat]
#                    dot_prod=np.dot(self.nmt_planes[nmt_feat-1,:],x_diff)
#                    nmt_feat_compliance=nmt_feat_compliance and dot_prod>=0.#-TOL
#                    if not nmt_feat_compliance:
#                        break
##            else:
##                x_diff=np.abs(x2-x1)
##                dot_prod=np.dot(self.nmt_planes[0,:],x_diff)
##                nmt_feat_compliance=dot_prod>=-TOL
#        # return result
#        if nmt_feat_compliance:
#            return comp
#        else: # incomparable due to nmt features
#            return -99

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
#        else: # use passed X & y
#            X_=X_to_use
#            y_unique_counts_=np.zeros([len(y_to_use),self.n_classes])
#            y_cdf_=np.zeros([len(y_to_use),self.n_classes])
#            for kk in np.arange(len(y_to_use)):
#                class_id=list(self.classes).index(y_to_use[kk])
#                y_cdf_[kk,class_id:]=1
#                y_unique_counts_[kk,class_id]=1
        y_pred=np.zeros(X_pred.shape[0])
        cdf_pred_all=np.ones([X_pred.shape[0],self.n_classes])
        pdf_pred_all=np.ones([X_pred.shape[0],self.n_classes])
        F_min_all=np.ones([X_pred.shape[0],self.n_classes])
        F_max_all=np.ones([X_pred.shape[0],self.n_classes])
        N_min_all=np.zeros([X_pred.shape[0],self.n_classes])
        N_max_all=np.zeros([X_pred.shape[0],self.n_classes])
        for i_pred in np.arange(X_pred.shape[0]):
            x_p=X_pred[i_pred,:]
#            F_min=np.ones(self.n_classes)
#            F_max=np.zeros(self.n_classes)
#            F_max[-1]=1
#            N_min=np.zeros(self.n_classes)
#            N_max=np.zeros(self.n_classes)
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
    
#    def predict(self, X):
#
#        """Predict class for X.
#
#
#
#        The predicted class of an input sample is a vote by the trees in
#
#        the forest, weighted by their probability estimates. That is,
#
#        the predicted class is the one with highest mean probability
#
#        estimate across the trees.
#
#
#
#        Parameters
#
#        ----------
#
#        X : array-like or sparse cvxmat of shape = [n_samples, n_features]
#
#            The input samples. Internally, its dtype will be converted to
#
#            ``dtype=np.float32``. If a sparse cvxmat is provided, it will be
#
#            converted into a sparse ``csr_cvxmat``.
#
#
#
#        Returns
#
#        -------
#
#        y : array of shape = [n_samples] or [n_samples, n_outputs]
#
#            The predicted classes.
#
#        """
#        #proba=np.zeros([X.shape[0],len(self.estimators_)],dtype=np.float64)
#        proba = self.predict_proba(X)
#        cum_proba=proba.copy()
#        for i in np.arange(1,proba.shape[1]):
#            cum_proba[:,i]=cum_proba[:,i-1]+proba[:,i]
#
#
#        if self.n_outputs_ == 1:
#
#            return self.classes_.take(np.argmax(cum_proba>=0.5, axis=1), axis=0) #self.classes_.take(np.argmax(proba, axis=1), axis=0)
#
#
#
#        else:
#            raise NotImplemented
       
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
#       if y[pm_pairs_clean[j,0]]>0.5:#==1:
#           if y[pm_pairs_clean[j,1]]<=0.5:#==-1:
#               pm_pairs_boundary[k,:]=pm_pairs_clean[j,:]
#               k=k+1
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
    
#    for i in np.arange(X.shape[0]):
#        res_mt=-9999 # -9999 utested, -99 incomp, -1 less than x1, 0 equal, +1 greater than x1
#        j=0
#        while res_mt!=-99 and j<X.shape[1]:
#            sgn=np.sign(mt_feat_types[j]*(-x1[j]+X[i,j]))
#            if res_mt==-9999:
#                res_mt=sgn
#            elif res_mt==0:
#                res_mt=sgn
#            else:
#                if res_mt*sgn <0:
#                    res_mt=-99  
#            if check_nmt_feats==NMT_EQUAL:
#                if mt_feat_types[j]==0:
#                    if X[i,j]!=x1[j]:
#                        res_mt=-99
#            j=j+1
#        res[i]=res_mt
#    if check_nmt_feats==NMT_CONE:
#        for i in np.arange(X.shape[0]):
#            if np.abs(res[i])<=1:
#                sum_=0.
#                for j in np.arange(X.shape[1]):
#                    sum_=sum_+np.abs(X[i,j]-x1[j])*nmt_plane_norm[j]
#                if (sum_+nmt_plane_intercept)<0:
#                    res[i]=-99
    return res
#
#    # returns: -1 if decreasing, 0 if identical, +1 if increasing, -99 if incomparable
#    if self.num_mt_feats==0:
#        return -99
#    elif len(x1_in.shape)>1:
#        x1=np.ravel(x1_in)
#        x2=np.ravel(x2_in)
#    else:
#        x1=x1_in.copy()
#        x2=x2_in.copy()
#    # check for identical
#    if np.array_equal(x1,x2):
#        return 0
#    # reverse polarity of decreasing features
#    #for dec_feat in self.decr_feats:
#    for dec_feat in np.arange(self.n_feats)[self.mt_feat_types==-1]:
#        x1[dec_feat]=-1*x1[dec_feat]
#        x2[dec_feat]=-1*x2[dec_feat]
#    # check mt feats all increasing (or decreasing)
#    mt_feats_difference=np.zeros(self.n_feats)
#    if self.num_mt_feats>0: 
#        mt_feats_difference[self.mt_feat_types!=0]=x2[self.mt_feat_types!=0]-x1[self.mt_feat_types!=0]
#    mt_feats_same=np.sum(mt_feats_difference[self.mt_feat_types!=0]==0)
#    if strict:
#        mt_feats_incr=np.sum(mt_feats_difference[self.mt_feat_types!=0]>0) 
#        mt_feats_decr=np.sum(mt_feats_difference[self.mt_feat_types!=0]<0)            
#    else:
#        mt_feats_incr=np.sum(mt_feats_difference[self.mt_feat_types!=0]>=0) 
#        mt_feats_decr=np.sum(mt_feats_difference[self.mt_feat_types!=0]<=0) 
#    if mt_feats_same==self.num_mt_feats:
#        comp=0
#    elif mt_feats_incr==self.num_mt_feats: # increasing
#        comp=+1        
#    elif mt_feats_decr==self.num_mt_feats: # decreasing
#        comp=-1
#    else: # incomparale
#        comp=-99
#    # short exit if available
#    if comp==-99 or comp==0:
#        return -99
#    # if still going, check mt feats by weakened planes
#    if self.num_nmt_feats==0:# or not check_nmt_feats:
#        nmt_feat_compliance=True
#    else:
#        if self.fit_type=='none':
#            nmt_feat_compliance=True
#        else:
#            #if self.fit_algo=='one-sided' or self.fit_algo=='two-sided':
#                # put all values in positive quadrant
#            x_diff=np.abs(x2-x1)
#            nmt_feat_compliance=True 
#            for i_nmt_feat in np.arange(self.num_nmt_feats):
#                nmt_feat=self.nmt_feats[i_nmt_feat]
#                dot_prod=np.dot(self.nmt_planes[nmt_feat-1,:],x_diff)
#                nmt_feat_compliance=nmt_feat_compliance and dot_prod>=-TOL
#                if not nmt_feat_compliance:
#                    break
##            else:
##                x_diff=np.abs(x2-x1)
##                dot_prod=np.dot(self.nmt_planes[0,:],x_diff)
##                nmt_feat_compliance=dot_prod>=-TOL
#    # return result
#    if nmt_feat_compliance:
#        return comp
#    else: # incomparable due to nmt features
#        return -99
    
def get_increasing_pairs(X,mt_feat_types, nmt_type=NMT_IGNORE, nmt_intercept=0., nmt_planes=None): #nmt_plane
    max_pairs=int(np.round(X.shape[0]*X.shape[0]))
    incr_pairs=np.zeros([max_pairs,2],dtype=np.int32)
    if nmt_planes is None:
        nmt_planes_=np.zeros(X.shape[1],dtype=np.float64)
    else:
        nmt_planes_=nmt_planes
    n_pairs_new=partial_mt_instance.get_increasing_pairs_array(X,mt_feat_types, nmt_type, nmt_intercept, nmt_planes_,incr_pairs)
    #incr_pairs_old=self.get_increasing_leaf_node_pairs_simple()
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
#            if m_num_comparable[i_greater]==260:
#                print('sdfsd')
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
        #if i==0:this_pt_counts= 
        #    print('sdfs')
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
        # calculate net relabelling MT improvement
        #greater_y_pos=0
        #greater_y_neg=0
        #lesser_y_pos=0
        #lesser_y_neg=0
        if ttl_count>0:
            nmt_pts=np.zeros(n_classes,dtype=np.float64)
            for c_ in np.arange(n_classes):
                greater_counts=np.zeros(n_classes,dtype=np.float64)
                lesser_counts=np.zeros(n_classes,dtype=np.float64)
                for j in np.arange(indxs_len):
                    true_indx=ids[j]#sorted_indexes[j]
                    #num_pos=y_prob_1[true_indx]*counts[j]
                    #num_neg=counts[j]-num_pos
                    if dirns[j]==+1:
                        greater_counts=greater_counts+y_counts[true_indx,:]
                        #greater_y_pos=greater_y_pos+num_pos
                        #greater_y_neg=greater_y_neg+num_neg
                    else:
                        lesser_counts=lesser_counts+y_counts[true_indx,:]
                        #lesser_y_pos=lesser_y_pos+num_pos
                        #lesser_y_neg=lesser_y_neg+num_neg
                # calculate non-monotonicity
                greater_nmt_count=  0 if c_==0 else np.sum(greater_counts[0:c_])   
                lesser_nmt_count=  0 if c_==n_classes-1 else np.sum(lesser_counts[c_+1:])   
                nmt_pts[c_]=greater_nmt_count+lesser_nmt_count
            relab_class=np.argmin(nmt_pts)
            #pos_pt_nmt=greater_y_neg/ttl_count
            #neg_pt_nmt=lesser_y_pos/ttl_count
            #pos_pt_relab_impr=pos_pt_nmt-neg_pt_nmt
            #neg_pt_relab_impr=neg_pt_nmt-pos_pt_nmt
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
    #mt_feats=np.arange(num_feats,dtype=np.int32)[mt_feat_types!=0]   +1#np.asarray(list(incr_feats)+list(decr_feats))
    #nmt_feats=np.arange(num_feats,dtype=np.int32)[mt_feat_types==0] +1 #np.asarray([j for j in np.arange(num_feats)+1 if j not in mt_feats])
    # calculate class for each X from function
    y=np.sign(sign_function(X,mt_feat_types))   
    y[y==0]=-1

    # add noise if requested
    noise_pts=[]
    y_true=y.copy()
    if not NMI_target is None:
        noise_pts=np.random.permutation(np.arange(X.shape[0]))[0:np.int(np.floor(X.shape[0]*NMI_target))]
        y[noise_pts]=-y[noise_pts]
        #[X,y,noise_pts]=noisify_partial_mono_rand_data(X,y,incr_feats=incr_feats,decr_feats=decr_feats,feat_dist=feat_dist,feat_spec=feat_spec,NMI_target=NMI_target,  pm_hyperplanes=pm_hyperplanes,random_state=random_state)
    return [X,y,noise_pts,y_true]

def fit_one_class_svm_pub(delta_X,delta_y,weights,v,mt_feat_types):
        
        N = delta_X.shape[0]
        p = delta_X.shape[1]
        #print(N)
        #num_feats = p
        mt_feats = np.arange(p)[mt_feat_types!=0]#np.asarray(list(incr_feats) + list(decr_feats))
        nmt_feats = np.arange(p)[mt_feat_types==0]#np.asarray(
        #    [j for j in np.arange(num_feats) + 1 if j not in mt_feats])
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
        #print(N)
        #num_feats = p
        mt_feats = np.arange(p)[mt_feat_types!=0]#np.asarray(list(incr_feats) + list(decr_feats))
        nmt_feats = np.arange(p)[mt_feat_types==0]#np.asarray(
        #    [j for j in np.arange(num_feats) + 1 if j not in mt_feats])
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
#            P = np.asarray([],dtype=np.float64)
            P = np.zeros([p + 2*N, p + 2*N])
            lambda_=N
            for ip in nmt_feats :
                P[ip, ip] = lambda_*0.01
            for ip in mt_feats :
                P[ip, ip] = lambda_*0.0001# 0.01
            q=np.zeros(p+2*N)
            q[p:p+N]=v
            q[p+N:]=(1.-v)
            #q = 1 / (N * v) * np.ones((N + p, 1))
            #q[0:p, 0] = 0
            #q[p:, 0] = q[p:, 0] * weights
            G1a = np.zeros([p, p])
            for ip in np.arange(p):
                G1a[ip, ip] = -1 if ip in mt_feats  else 1
            G1 = np.hstack([G1a, np.zeros([p, 2*N])])
            G2 = np.hstack([np.zeros([2*N, p]), -np.eye(2*N)])
            #G2 = np.hstack([np.zeros([N, p]), -np.eye(N),np.zeros([N,N])])
            
            #G2 = np.hstack([np.zeros([N, p]),np.zeros([N,N]), np.eye(N)])
            G3 = np.hstack([-delta_X ,-np.eye(N),np.zeros([N,N])])
            G4 = np.hstack([delta_X ,np.zeros([N,N]),-np.eye(N)])
            G = np.vstack([G1, G2, G3,G4])
            h = np.zeros([p + 4 * N])
            A = np.zeros([1, p + 2*N])
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
                return np.zeros(delta_X.shape[1])-99#[-99]
            else:
                soln = sol['x']
                w = np.ravel(soln[0:p, :])
                if np.sum(np.abs(w))==0.:
                    print('what the?')
                # err = np.asarray(soln[-N:, :])
                return w
            
def fit_one_class_svm_OLD(delta_X,delta_y,weights,lambda_,mt_feat_types):
    
    nmt_overweight=1e6
    nmt_feats_mask=mt_feat_types==0
    c_=np.hstack([-delta_y*weights,np.zeros(delta_X.shape[1])])
    c_[c_>0]=nmt_overweight*c_[c_>0]
    # build constraint cvxmat by rows
    Gt_row_2=np.hstack([-np.eye(delta_X.shape[0]),np.zeros([delta_X.shape[0],delta_X.shape[1]])])                    
    deltax_mod=delta_X.copy()
    deltax_mod[:,nmt_feats_mask]=-1.*np.abs(deltax_mod[:,nmt_feats_mask])
    Gt_row_1=np.hstack([-np.eye(delta_X.shape[0]),deltax_mod])                    
    Gt_row_3=np.hstack([np.zeros([delta_X.shape[1],delta_X.shape[0]]),-np.eye(delta_X.shape[1])])                    
    Gt_row_4=np.hstack([np.zeros([1,delta_X.shape[0]]),np.ones([1,delta_X.shape[1]])])                    
    Gt_row_5=np.hstack([np.zeros([1,delta_X.shape[0]]),-np.ones([1,delta_X.shape[1]])])   
    Gt = np.vstack([Gt_row_1,Gt_row_2,Gt_row_3,Gt_row_4,Gt_row_5])
    Gt = np.vstack([Gt_row_1,Gt_row_2,Gt_row_3])
    h_=np.zeros(2*delta_X.shape[0]+delta_X.shape[1])#+2)
    #h_[-2]=1
    #h_[-1]=-1
    A_=np.zeros([delta_X.shape[0]+delta_X.shape[1],1])
    A_[delta_X.shape[0]:]=1
    b_=1.
    G = cvxmat(Gt)
    h = cvxmat(h_)
    c = cvxmat(c_)   
    A=cvxmat(A_.T)
    b=cvxmat(np.ones([1,1]))
    # min cT.x s.t. G.x<=h and A.x=b
    sol=solvers.lp(c=c,G=G,h=h,A=A,b=b)
    if sol['status']!='optimal':
        print('****** NOT OPTIMAL '+sol['status']+' ******* '  )
    soln=sol['x']
    err=soln[0:delta_X.shape[0],:]
    w=np.ravel(soln[-delta_X.shape[1]:,:])
    w[nmt_feats_mask]=-1*w[nmt_feats_mask]
    #self.nmt_planes[0,:]=w  
    #self.nmt_planes[0,self.nmt_feats-1]=self.nmt_planes[0,self.nmt_feats-1]*-1.
    
    return w