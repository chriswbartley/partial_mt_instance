# partial_mt_instance
Cone based partially monotone instance classification and relabelling.

PartialInstanceBinaryClassifier() can be used to:

     - construct a binary classifier (using sklearn nomenclature), or
     
     - perform partially monotone relabelling of a dataset,using 
       clf =PartialInstanceBinaryClassifier(relabel=True,mt_feat_types=_) 
       then clf.fit(X,y) and then accessing clf.y_relabelled


class PartialInstanceBinaryClassifier(object):

    A partially monotone instance based classifier (and relabelling algorithm).
    
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
