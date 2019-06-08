# cython: cdivision=True, boundscheck=False, wraparound=False, nonecheck=False
# Author: Peter Prettenhofer
#
# License: BSD 3 clause

cimport cython

from libc.stdlib cimport free
from libc.string cimport memset
from libcpp cimport bool
from libc.math cimport exp
from libc.math cimport log
import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csr_matrix

#from sklearn.tree._tree cimport Node
#from sklearn.tree._tree cimport Tree
#from sklearn.tree._tree cimport DTYPE_t
#from sklearn.tree._tree cimport SIZE_t
#from sklearn.tree._tree cimport INT32_t
#from sklearn.tree._utils cimport safe_realloc
ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

ctypedef np.int32_t int32
ctypedef np.float64_t float64
ctypedef np.float_t float
ctypedef np.uint8_t uint8

# no namespace lookup for numpy dtype and array creation
from numpy import zeros as np_zeros
from numpy import ones as np_ones
from numpy import bool as np_bool
from numpy import float32 as np_float32
from numpy import float64 as np_float64


# constant to mark tree leafs
cdef SIZE_t TREE_LEAF = -1
cdef float64 RULE_LOWER_CONST=-1e9
cdef float64 RULE_UPPER_CONST=1e9
cdef float64 INF=np.inf

cdef int32 NMT_EQUAL=0
cdef int32 NMT_IGNORE=1
cdef int32 NMT_CONE=2

@cython.boundscheck(False) 
cdef _traverse_node_c(int32 node_id,
                    int32 num_feats,
                       int32 *children_left,
                       int32 *children_right,
                       int32 *features,
                       float64 *thresholds, 
                       int32 *out_leaf_ids,
                       float64 *out_rule_upper_corners,
                       float64 *out_rule_lower_corners):
    cdef int32 feature
    cdef float64 threshold
    cdef int32 left_node_id
    cdef int32 right_node_id
    cdef int32 j
    # recurse on children 
    if children_left[node_id] != -1: #TREE_LEAF:
        feature = features[node_id]
        threshold = thresholds[node_id]
        left_node_id = children_left[node_id]
        right_node_id = children_right[node_id]
        # update limit arrays
        for j in range(num_feats):
            out_rule_upper_corners[left_node_id*num_feats+j] = out_rule_upper_corners[node_id*num_feats+j]
            out_rule_lower_corners[left_node_id*num_feats+j] = out_rule_lower_corners[node_id*num_feats+j]
            out_rule_upper_corners[ right_node_id*num_feats+j] = out_rule_upper_corners[node_id*num_feats+j]
            out_rule_lower_corners[right_node_id*num_feats+j] = out_rule_lower_corners[node_id*num_feats+j]
        out_rule_upper_corners[left_node_id*num_feats+feature] = threshold
        out_rule_lower_corners[right_node_id*num_feats+feature] = threshold
        # recurse
        _traverse_node_c(left_node_id, num_feats,
                       children_left,children_right,features,thresholds, out_leaf_ids,out_rule_upper_corners,out_rule_lower_corners)  # "<="
        _traverse_node_c(right_node_id,num_feats,
                       children_left,children_right,features,thresholds, out_leaf_ids,out_rule_upper_corners,out_rule_lower_corners)  # ">"
    else:  # a leaf node
        out_leaf_ids[node_id] = node_id
        if node_id == 0:# the base node (0) is the only node!
            pass
            #print('Warning: Tree only has one node! (i.e. the root node)')

def extract_rules_from_tree_c(np.ndarray[int32, ndim=1] children_left,
                              np.ndarray[int32, ndim=1] children_right,
                              np.ndarray[int32, ndim=1] features,
                              np.ndarray[float64, ndim=1] thresholds, 
                              int32 num_feats, 
                              np.ndarray[int32, ndim=1] out_leaf_ids,
                              np.ndarray[float64, ndim=2] out_rule_upper_corners,
                              np.ndarray[float64, ndim=2] out_rule_lower_corners):
    _traverse_node_c(np.int32(0),
                     num_feats,
                     <int32*> (<np.ndarray> children_left).data ,
                     <int32*> (<np.ndarray> children_right).data ,
                     <int32*> (<np.ndarray> features).data ,
                     <float64*> (<np.ndarray> thresholds).data,
                     <int32*> (<np.ndarray> out_leaf_ids).data ,
                     <float64*> (<np.ndarray> out_rule_upper_corners).data,
                     <float64*> (<np.ndarray> out_rule_lower_corners).data
                     )
cdef int32 _initialise_path_matrix(int32 *pm_pairs,
                    int32 rows,
                       int32 N,
                       int32 *out_init_path_matrix_template):
    cdef int32 i=0
    cdef int32 j=0
    
    for i in range(rows):
        out_init_path_matrix_template[pm_pairs[i*2 + 0]*N + pm_pairs[i*2 + 1]]=1
    return 0

# This is O(N^3). 
# An alternative would be to use the DFS O(N^2) algorithm detailed
# https://www.geeksforgeeks.org/transitive-closure-of-a-graph-using-dfs/
cdef int32 _calculate_transitive_closure_path_matrix(
                       int32 N,
                       int32 *out_closed_path_matrix):
    cdef int32 i=0
    cdef int32 j=0
    cdef int32 k=0
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # a[i][j] = a[i][j] or (a[i][k] and a[k][j])
                if out_closed_path_matrix[i*N +k]==1 :
                    if out_closed_path_matrix[k*N +j]==1:
                        out_closed_path_matrix[i*N +j]=1
                        break
    return 0

cdef int32 _calculate_transitive_closure_path_matrix_with_sparse(
                       int32 N,
                       int32 *out_closed_path_matrix,
                       int32 *out_closed_path_sparse_matrix):
    cdef int32 i=0
    cdef int32 j=0
    cdef int32 k=0
    cdef int32 l=0
    
    for i in range(N):
        for j in range(N):
            # a[i][j] = a[i][j] or (a[i][k] and a[k][j])
            if out_closed_path_matrix[i*N +j]==1 :
                out_closed_path_sparse_matrix[l*2+0]=i
                out_closed_path_sparse_matrix[l*2+1]=j
                l=l+1
            else:
                for k in range(N):
                    if out_closed_path_matrix[i*N +k]==1 :
                        if out_closed_path_matrix[k*N +j]==1:
                            out_closed_path_matrix[i*N +j]=1
                            out_closed_path_sparse_matrix[l*2+0]=i
                            out_closed_path_sparse_matrix[l*2+1]=j
                            l=l+1
                            break
    return l

cdef int32 _calculate_transitive_reduction_path_matrix_c(
                       int32 N,
                       int32 *out_closed_path_matrix):
    cdef int32 i=0
    cdef int32 j=0
    cdef int32 k=0
#    for (int j = 0; j < N; ++j)
#      for (int i = 0; i < N; ++i)
#        if (m[i][j])
#          for (int k = 0; k < N; ++k)
#            if (m[j][k])
#              m[i][k] = false;
          
    for j in range(N):
        for i in range(N):
            if out_closed_path_matrix[i*N +j]==1:
                for k in range(N):
                    # a[i][j] = a[i][j] or (a[i][k] and a[k][j])
                    if out_closed_path_matrix[j*N +k]==1:
                        out_closed_path_matrix[i*N +k]=0
    return 0
    
cdef int32 _calculate_transitive_reduction_path_matrix_with_sparse_c(
                        int32 *closed_path_sparse_matrix,
                        int32 num_pairs,
                       int32 N,
                       int32 *out_closed_path_matrix,
                       int32 *out_pm_pairs):
    cdef int32 i=0
    cdef int32 j=0
    cdef int32 k=0
    cdef int32 l =0
    
    for j in range(num_pairs):
        if out_closed_path_matrix[closed_path_sparse_matrix[j*2+0]*N +closed_path_sparse_matrix[j*2+1]]==1:
            for k in range(N):
                # a[i][j] = a[i][j] or (a[i][k] and a[k][j])
                if out_closed_path_matrix[closed_path_sparse_matrix[j*2+1]*N +k]==1:
                    out_closed_path_matrix[closed_path_sparse_matrix[j*2+0]*N +k]=0
                    
    return 0

cdef int32 _calculate_transitive_reduction_edges(
                        int32 *closed_path_matrix,
                       int32 N,
                       int32 *out_pm_pairs):
    cdef int32 i=0
    cdef int32 j=0
    cdef int32 k=0
    for i in range(N):
        for j in range(N):
            if closed_path_matrix[i*N +j]==1:#
                out_pm_pairs[k*2+0]=i
                out_pm_pairs[k*2+1]=j
                k=k+1 
    return k

cdef int32 _calculate_transitive_reduction_edges_with_sparse(
                        int32 *closed_path_sparse_matrix,
                        int32 num_pairs,
                        int32 *closed_path_matrix,
                       int32 N,
                       int32 *out_pm_pairs):
    cdef int32 i=0
    cdef int32 j=0
    cdef int32 k=0
    for j in range(num_pairs):
        if closed_path_matrix[closed_path_sparse_matrix[j*2+0]*N +closed_path_sparse_matrix[j*2+1]]==1:
            out_pm_pairs[k*2+0]=closed_path_sparse_matrix[j*2+0]
            out_pm_pairs[k*2+1]=closed_path_sparse_matrix[j*2+1]
            k=k+1 
    return k

cdef int32 _get_uniq_val_lookups(
                        int32 *uniq_leaves,
                        int32 N_uniq,
                        int32 *lkp ,
                        int32 *rev_lkp,
                        int32 *pm_pairs,
                        int32 *pm_pairs_sequenced,
                        int32 N_pairs
                        ):
    cdef int32 i=0
    for i in range(N_uniq):
        lkp[uniq_leaves[i]]=i
        rev_lkp[i]=uniq_leaves[i]
        
    for i in range(N_pairs):
         pm_pairs_sequenced[i*2+0]= lkp[pm_pairs[i*2+0]]  
         pm_pairs_sequenced[i*2+1]= lkp[pm_pairs[i*2+1]] 
    return 0

cdef int32 _translate_back_to_orig_ids(
                        int32 *rev_lkp,
                        int32 *out_pm_pairs,
                        int32 N_pairs
                        ):    
    cdef int32 i=0
    for i in range(N_pairs):
        out_pm_pairs[i*2+0]= rev_lkp[out_pm_pairs[i*2+0]]  
        out_pm_pairs[i*2+1]= rev_lkp[out_pm_pairs[i*2+1]] 
    return 0
     
# A recursive DFS traversal function that finds
# all reachable vertices for s
cdef void  _dfs_util(
        int32 *outward_edges,
        int32 *num_edges,
        int32 N,
        int32 *out_reduced_path_matrix ,
        int32 s,
        int32 v
        ):
    cdef int32 i=0
    cdef int32 i_=0
    # Mark reachability from s to v as true.
    if s!=v:
        out_reduced_path_matrix[s*N+v] = 1
 
    # Find all the vertices reachable through v
    for i_ in range(num_edges[v]):#self.graph[v]:
        i=outward_edges[v*N+i_]
        if out_reduced_path_matrix[s*N+i]==0 : #self.tc[s][i]==0:
            _dfs_util(outward_edges,num_edges,N,out_reduced_path_matrix,s,i)
    return 
            #self.DFSUtil(s,i)
 

cdef int32 _initialise_outward_edges(
            int32 *pm_pairs_sequenced,
            int32 num_pairs_seq,
            int32 N,
            int32 *outward_edges,
            int32 *num_edges
            )     :
    cdef int32 i=0
    cdef int32 v=0
    for i in range(num_pairs_seq):
        v=pm_pairs_sequenced[i*2+0]
        outward_edges[v*N+num_edges[v]]=pm_pairs_sequenced[i*2+1]
        num_edges[v]=num_edges[v]+1
    return 0

# The function to find transitive closure. It uses
# recursive DFSUtil(). from: 
# https://www.geeksforgeeks.org/transitive-closure-of-a-graph-using-dfs/
cdef int32  _calculate_transitive_closure_path_matrix_dfs(
            int32 *outward_edges,
            int32 *num_edges,
            int32 N,
            int32 *out_reduced_path_matrix 
            ):
    cdef int32 i =0
    for i in range(N):
        _dfs_util(outward_edges,
                  num_edges,
                  N,
                  out_reduced_path_matrix,
                  i,
                  i)
    return 0

def calculate_transitive_reduction_c_warshal(np.ndarray[int32, ndim=2] pm_pairs,
                              np.ndarray[int32, ndim=2] out_pm_pairs):
    rows=pm_pairs.shape[0]
    N_max=np.max(pm_pairs)+1
    uniq_leaves=np.unique(pm_pairs)
    N_uniq=len(uniq_leaves)
    lkp=np.zeros([N_max,2],dtype=np.int32)#dict()
    rev_lkp=np.zeros([N_uniq,2],dtype=np.int32)#dict()
    pm_pairs_sequenced=np.zeros([pm_pairs.shape[0],pm_pairs.shape[1]],dtype=np.int32)
    _get_uniq_val_lookups(
                    <int32*> (<np.ndarray> uniq_leaves).data ,
                    <int32> N_uniq,
                    <int32*> (<np.ndarray> lkp ).data ,
                    <int32*> (<np.ndarray> rev_lkp).data ,
                    <int32*> (<np.ndarray> pm_pairs).data ,
                    <int32*> (<np.ndarray> pm_pairs_sequenced).data ,
                    <int32> rows
                    )
 
             
    N=np.max(pm_pairs_sequenced)+1
    #num_pairs_seq=pm_pairs.shape[0]
    out_reduced_path_matrix=np.zeros([N,N],dtype=np.int32,order='C')
#    closed_path_sparse_matrix=np.zeros([N*N,2],dtype=np.int32,order='C')

    # TRANSITIVE CLOSURE TECHNIQUE #2: DFS O(N^2)
    num_edges=np.zeros(N,dtype=np.int32)
    outward_edges=np.zeros([N,N],dtype=np.int32)
#    _initialise_outward_edges(
#            <int32*> (<np.ndarray> pm_pairs_sequenced).data,
#            <int32> pm_pairs.shape[0],
#            <int32> N,
#            <int32*> (<np.ndarray> outward_edges).data ,
#            <int32*> (<np.ndarray> num_edges).data 
#            )
#    _calculate_transitive_closure_path_matrix_dfs(
#            <int32*> (<np.ndarray> outward_edges).data ,
#            <int32*> (<np.ndarray> num_edges).data ,
#            <int32> N,
#            <int32*> (<np.ndarray> out_reduced_path_matrix).data 
#            )
    
    
    # TRANSITIVE CLOSURE TECHNIQUE #1 (wARSHAL O(N^3))
    _initialise_path_matrix(
            <int32*> (<np.ndarray> pm_pairs_sequenced).data ,
            <int32> rows,
            <int32> N,
            <int32*> (<np.ndarray> out_reduced_path_matrix).data )
    _calculate_transitive_closure_path_matrix(
            <int32> N,
            <int32*> (<np.ndarray> out_reduced_path_matrix).data )

    out_reduced_path_matrix2=out_reduced_path_matrix.copy()
    # TRANS REDUCTION
    _calculate_transitive_reduction_path_matrix_c(
                     <int32> N,
                     <int32*> (<np.ndarray> out_reduced_path_matrix).data 
               )
    num_pairs= _calculate_transitive_reduction_edges(
                     <int32*> (<np.ndarray> out_reduced_path_matrix).data ,
                     <int32> N,
                     <int32*> (<np.ndarray> out_pm_pairs).data 
               )
    _translate_back_to_orig_ids(
                    <int32*> (<np.ndarray> rev_lkp).data ,
                     <int32*> (<np.ndarray> out_pm_pairs).data ,
                     <int32> num_pairs
                     )
    return num_pairs #np.int32(np.sum(out_reduced_path_matrix2>0))#num_pairs
    
def calculate_transitive_reduction_c(np.ndarray[int32, ndim=2] pm_pairs,
                              np.ndarray[int32, ndim=2] out_pm_pairs):
    rows=pm_pairs.shape[0]
    N_max=np.max(pm_pairs)+1
    uniq_leaves=np.unique(pm_pairs)
    N_uniq=len(uniq_leaves)
    lkp=np.zeros([N_max,2],dtype=np.int32)#dict()
    rev_lkp=np.zeros([N_uniq,2],dtype=np.int32)#dict()
    pm_pairs_sequenced=np.zeros([pm_pairs.shape[0],pm_pairs.shape[1]],dtype=np.int32)
    _get_uniq_val_lookups(
                    <int32*> (<np.ndarray> uniq_leaves).data ,
                    <int32> N_uniq,
                    <int32*> (<np.ndarray> lkp ).data ,
                    <int32*> (<np.ndarray> rev_lkp).data ,
                    <int32*> (<np.ndarray> pm_pairs).data ,
                    <int32*> (<np.ndarray> pm_pairs_sequenced).data ,
                    <int32> rows
                    )
 
             
    N=np.max(pm_pairs_sequenced)+1
    #num_pairs_seq=pm_pairs.shape[0]
    out_reduced_path_matrix=np.zeros([N,N],dtype=np.int32,order='C')
#    closed_path_sparse_matrix=np.zeros([N*N,2],dtype=np.int32,order='C')

    # TRANSITIVE CLOSURE TECHNIQUE #2: DFS O(N^2)
    num_edges=np.zeros(N,dtype=np.int32)
    outward_edges=np.zeros([N,N],dtype=np.int32)
    _initialise_outward_edges(
            <int32*> (<np.ndarray> pm_pairs_sequenced).data,
            <int32> pm_pairs_sequenced.shape[0],
            <int32> N,
            <int32*> (<np.ndarray> outward_edges).data ,
            <int32*> (<np.ndarray> num_edges).data 
            )
    _calculate_transitive_closure_path_matrix_dfs(
            <int32*> (<np.ndarray> outward_edges).data ,
            <int32*> (<np.ndarray> num_edges).data ,
            <int32> N,
            <int32*> (<np.ndarray> out_reduced_path_matrix).data 
            )
    
    out_reduced_path_matrix2=out_reduced_path_matrix.copy()
    
#    # TRANSITIVE CLOSURE TECHNIQUE #1 (wARSHAL O(N^3))
#    _initialise_path_matrix(
#            <int32*> (<np.ndarray> pm_pairs_sequenced).data ,
#            <int32> rows,
#            <int32> N,
#            <int32*> (<np.ndarray> out_reduced_path_matrix).data )
#    _calculate_transitive_closure_path_matrix(
#            <int32> N,
#            <int32*> (<np.ndarray> out_reduced_path_matrix).data )

    
    # TRANS REDUCTION
    _calculate_transitive_reduction_path_matrix_c(
                     <int32> N,
                     <int32*> (<np.ndarray> out_reduced_path_matrix).data 
               )
    num_pairs= _calculate_transitive_reduction_edges(
                     <int32*> (<np.ndarray> out_reduced_path_matrix).data ,
                     <int32> N,
                     <int32*> (<np.ndarray> out_pm_pairs).data 
               )
    _translate_back_to_orig_ids(
                    <int32*> (<np.ndarray> rev_lkp).data ,
                     <int32*> (<np.ndarray> out_pm_pairs).data ,
                     <int32> num_pairs
                     )
    return num_pairs# np.int32(np.sum(out_reduced_path_matrix2>0))#num_pairs

def get_leaf_id_pairs_c(np.ndarray[int32, ndim=2] pm_pairs,
                        np.ndarray[int32, ndim=1] uniq_leaves,
                              np.ndarray[int32, ndim=2] out_pm_pairs):
    rows=pm_pairs.shape[0]
    N_max=np.max(uniq_leaves)+1
    #uniq_leaves=np.unique(pm_pairs)
    #uniq_leaves=np.arange()
    N_uniq=len(uniq_leaves)
    lkp=np.zeros([N_max,2],dtype=np.int32)#dict()
    rev_lkp=np.zeros([N_uniq,2],dtype=np.int32)#dict()
    #pm_pairs_sequenced=np.zeros([pm_pairs.shape[0],pm_pairs.shape[1]],dtype=np.int32)
    _get_uniq_val_lookups(
                    <int32*> (<np.ndarray> uniq_leaves).data ,
                    <int32> N_uniq,
                    <int32*> (<np.ndarray> lkp ).data ,
                    <int32*> (<np.ndarray> rev_lkp).data ,
                    <int32*> (<np.ndarray> pm_pairs).data ,
                    <int32*> (<np.ndarray> out_pm_pairs).data ,
                    <int32> rows
                    )
    return
#        lookup=np.zeros(np.max(self.leaf_ids_obj.get_idx_array())+1,dtype=np.int32)
#        
#        l_=0
#        for l in self.leaf_ids_obj.get_idx_array():
#            lookup[l]=l_
#            l_=l_+1
#        leaf_id_pairs=[]
#        for pair in cleaned_pairs:
#            leaf_id_pairs=leaf_id_pairs+[(lookup[pair[0]],lookup[pair[1]])]        
#        return leaf_id_pairs
 
#  
#    num_pairs=_calculate_transitive_closure_path_matrix_with_sparse(
#            <int32> N,
#            <int32*> (<np.ndarray> out_reduced_path_matrix).data,
#            <int32*> (<np.ndarray> closed_path_sparse_matrix).data )
#
#    _calculate_transitive_reduction_path_matrix_with_sparse_c(
#                <int32*> (<np.ndarray> closed_path_sparse_matrix).data,
#                <int32> num_pairs,
#                     <int32> N,
#                     <int32*> (<np.ndarray> out_reduced_path_matrix).data,
#                     <int32*> (<np.ndarray> out_pm_pairs).data 
#               )
#    return _calculate_transitive_reduction_edges_with_sparse(
#            <int32*> (<np.ndarray> closed_path_sparse_matrix).data,
#                <int32> num_pairs,
#                     <int32*> (<np.ndarray> out_reduced_path_matrix).data ,
#                     <int32> N,
#                     <int32*> (<np.ndarray> out_pm_pairs).data 
#               )

@cython.boundscheck(False)
cdef int32 _get_increasing_leaf_node_pairs(float64 *rule_lower_corners,
                           float64 *rule_upper_corners,
                           int32 *feat_types,
                           Py_ssize_t n_rules,
                           Py_ssize_t n_features,
                           int32 *out):
    cdef int32 n_pairs
    cdef int32 res 
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    
    n_pairs=0
    for i in range(n_rules):
        for j in range(n_rules):
            if i!=j:
                res=1
                k=0
                while res==1 and k<n_features:
                    if feat_types[k]==+1:
                        if rule_lower_corners[k + n_features * i] >=rule_upper_corners[k + n_features * j]:
                            res=0
                    elif feat_types[k]==-1:
                        if rule_upper_corners[k + n_features * i] <=rule_lower_corners[k + n_features * j]:
                            res=0
                    else: # non-monotone feature 
                        if rule_upper_corners[k + n_features * i] <=rule_lower_corners[k + n_features * j]:
                            res=0
                        if res==1:
                            if rule_upper_corners[k + n_features * j] <=rule_lower_corners[k + n_features * i]:
                                res=0                    
                    k=k+1
                if res==1:
                    out[n_pairs * 2 + 0]=i
                    out[n_pairs * 2 + 1]=j
                    n_pairs=n_pairs+1
    return n_pairs

def get_increasing_leaf_node_pairs(object rule_lower_corners, 
                                   object rule_upper_corners,
                                   object  feat_types,
                                   np.ndarray[int32, ndim=2] out):

    n_pairs_=_get_increasing_leaf_node_pairs(<float64*> (<np.ndarray> rule_lower_corners).data, 
                                    <float64*> (<np.ndarray> rule_upper_corners).data,
                                    <int32*> (<np.ndarray> feat_types).data,
                                    rule_lower_corners.shape[0],
                                    rule_lower_corners.shape[1],
                                    <int32*> (<np.ndarray> out).data)
    return n_pairs_

cdef int32 _populate_nodes_c(int32 *tree_features,
                       float64 *tree_thresholds,
                       float64 *tree_values,
                       int32 *tree_left_children,
                       int32 *tree_right_children,
                       float64 *train_X,
                       int32 *train_y,
                       int32 n_samples,
                       int32 n_feats,
                       int32 n_out_cols,
                       int32 *out_node_train_idxs,
                       int32 *out_node_train_nums):
    cdef int32 k_feat=0
    cdef int32 i=0
    cdef int32 j_node=0

    #n_samples=train_X.shape[0]
    #n_nodes=out_node_train_idxs.shape[0]
    #n_pts_per_node=np.zeros(n_nodes,dtype=np.int32)
    for i in range(n_samples):
        j_node=0
        out_node_train_idxs[j_node*n_out_cols+out_node_train_nums[j_node]]=i
        out_node_train_nums[j_node]=out_node_train_nums[j_node]+1
        while tree_left_children[j_node] != TREE_LEAF: 
            k_feat=tree_features[j_node]
            if train_X[i*n_feats+k_feat] <= tree_thresholds[j_node]: 
                j_node=tree_left_children[j_node]
            else:
                j_node=tree_right_children[j_node]
            out_node_train_idxs[j_node*n_out_cols+out_node_train_nums[j_node]]=i
            out_node_train_nums[j_node]=out_node_train_nums[j_node]+1
    return 0
    
def populate_nodes_c(np.ndarray[int32, ndim=1] tree_features, 
                      np.ndarray[float64, ndim=1]    tree_thresholds, 
                      np.ndarray[float64, ndim=2]   tree_values,
                      np.ndarray[int32, ndim=1]   tree_left_children, 
                     np.ndarray[int32, ndim=1]   tree_right_children, 
                     np.ndarray[float64, ndim=2]   train_X,
                      np.ndarray[int32, ndim=2]  train_y,
                      np.ndarray[int32, ndim=2]  out_node_train_idxs, 
                      np.ndarray[int32, ndim=1]  out_node_train_nums)       :
    return _populate_nodes_c(<int32*> (<np.ndarray> tree_features).data ,
                       <float64*> (<np.ndarray> tree_thresholds).data ,
                       <float64*> (<np.ndarray> tree_values).data ,
                       <int32*> (<np.ndarray> tree_left_children).data ,
                       <int32*> (<np.ndarray> tree_right_children).data ,
                       <float64*> (<np.ndarray> train_X).data ,
                       <int32*> (<np.ndarray> train_y).data ,
                       train_X.shape[0],
                       train_X.shape[1],
                       out_node_train_idxs.shape[1],
                       <int32*> (<np.ndarray> out_node_train_idxs).data ,
                       <int32*> (<np.ndarray> out_node_train_nums).data )

cdef int32 _apply_c(int32 *tree_features,
                       float64 *tree_thresholds,
                       float64 *tree_values,
                       int32 *tree_left_children,
                       int32 *tree_right_children,
                       float64 *train_X,
                       int32 n_samples,
                       int32 n_feats,
                       int32 *out_node_idxs):

    cdef int32 k_feat=0
    cdef int32 i=0
    cdef int32 j_node=0
    
    for i in range(n_samples):
        j_node=0
        while tree_left_children[j_node] != TREE_LEAF: 
            k_feat=tree_features[j_node]
            if train_X[i*n_feats+k_feat] <= tree_thresholds[j_node]: 
                j_node=tree_left_children[j_node]
            else:
                j_node=tree_right_children[j_node]
        out_node_idxs[i]=j_node
    return 0
    
def apply_c(np.ndarray[int32, ndim=1] tree_features, 
                      np.ndarray[float64, ndim=1]    tree_thresholds, 
                      np.ndarray[float64, ndim=2]   tree_values,
                      np.ndarray[int32, ndim=1]   tree_left_children, 
                     np.ndarray[int32, ndim=1]   tree_right_children, 
                     np.ndarray[float64, ndim=2]   train_X,
                      np.ndarray[int32, ndim=1]  out_node_idxs)       :
    return _apply_c(<int32*> (<np.ndarray> tree_features).data ,
                       <float64*> (<np.ndarray> tree_thresholds).data ,
                       <float64*> (<np.ndarray> tree_values).data ,
                       <int32*> (<np.ndarray> tree_left_children).data ,
                       <int32*> (<np.ndarray> tree_right_children).data ,
                       <float64*> (<np.ndarray> train_X).data ,
                       train_X.shape[0],
                       train_X.shape[1],
                       <int32*> (<np.ndarray> out_node_idxs).data )

cdef float64 _abs_float(float64 in_):
    if in_<0.:
        return -in_
    else:
        return in_
        
@cython.boundscheck(False)
cdef int32 _get_increasing_pairs_array(float64 *X,
                           int32 *feat_types,
                           Py_ssize_t n_samples,
                           Py_ssize_t n_features,
                           int32 nmt_type,
                           float64 nmt_intercept,
                           float64 *nmt_plane,
                           int32 num_nmt_planes,
                           int32 *out):
    cdef int32 n_pairs
    cdef int32 res 
    cdef float64 sum_=0.
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t i_
    cdef Py_ssize_t j_
    cdef Py_ssize_t j_nmt_plane
    n_pairs=0
    for i in range(n_samples):
        #i=leaf_ids[i_]
        for j in range(n_samples):
            #j=leaf_ids[j_]
            if i!=j:
                res=1
                k=0
                while res==1 and k<n_features:
                    if feat_types[k]==+1:
                        if X[k + n_features * i] >X[k + n_features * j]:
                            res=0
                    elif feat_types[k]==-1:
                        if X[k + n_features * i] <X[k + n_features * j]:
                            res=0
                    else: # non-monotone feature, ignore
                        if nmt_type==NMT_EQUAL:
                            if X[k + n_features * i] !=X[k + n_features * j]:
                                res=0
                    k=k+1
                if nmt_type==NMT_CONE:
                    j_nmt_plane=0
                    while res>0 and j_nmt_plane<num_nmt_planes:
                        k=0
                        sum_=0.
                        for k in range(n_features) :
                             sum_=sum_+_abs_float(X[k + n_features * i] -X[k + n_features * j])*nmt_plane[j_nmt_plane*n_features + k]
                        if (sum_+nmt_intercept)<0.:
                            res=0
                        j_nmt_plane=j_nmt_plane+1
#                        if X[k + n_features * i] <=X[k + n_features * j]:
#                            res=0
#                        if res==1:
#                            if rule_upper_corners[k + n_features * j] <=rule_lower_corners[k + n_features * i]:
#                                res=0                    
                    
                if res==1:
                    out[n_pairs * 2 + 0]=i
                    out[n_pairs * 2 + 1]=j
                    n_pairs=n_pairs+1
    return n_pairs

def get_increasing_pairs_array(object X, 
                                   object  feat_types,
                                   int32 nmt_type, 
                                   float64 nmt_intercept, 
                                   object nmt_planes,#nmt_plane
                                   np.ndarray[int32, ndim=2] out):
    if len(nmt_planes.shape)==1:
        num_nmt_planes=1
    else:
        num_nmt_planes=nmt_planes.shape[0]
    n_pairs_=_get_increasing_pairs_array(<float64*> (<np.ndarray> X).data, 
                                    <int32*> (<np.ndarray> feat_types).data,
                                    X.shape[0],
                                    X.shape[1],
                                    nmt_type,
                                    nmt_intercept,
                                    <float64*> (<np.ndarray> nmt_planes).data,
                                    num_nmt_planes,
                                    <int32*> (<np.ndarray> out).data)
    return n_pairs_

cdef int32 _sign(float64 val):
    if val<0:
        return -1
    elif val>0:
        return +1
    else:
        return 0
  
cdef int32 _abs_int(int32 val):
    if val<0:
        return -val
    else:
        return val
    
cdef float64 _abs_float_(float64 val):
    if val<0:
        return -val
    else:
        return val
#cdef float64 _abs_float(float64 val):
#    if val<0:
#        return -1.0*val
#    else:
#        return val
@cython.boundscheck(False)
cdef int32 _compare_pt_with_array(float64 *x1,
                                  float64 *X,
                           float64 *mt_feat_types,
                           Py_ssize_t n_samples,
                           Py_ssize_t n_features,
                           int32 nmt_type,
                           int32 strict,
                           float64 *nmt_intercepts,
                           float64 *nmt_plane_norms,
                           int32 num_nmt_planes,
                           int32 *res):
    cdef int32 sgn
    cdef int32 res_mt 
    cdef float64 sum_=0.
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_nmt_plane
#    cdef Py_ssize_t k
#    cdef Py_ssize_t i_
#    cdef Py_ssize_t j_
    
    for i in range(n_samples):
        res_mt=-9999 # -9999 utested, -99 incomp, -1 less than x1, 0 equal, +1 greater than x1
        j=0
        while res_mt!=-99 and j<n_features:
            sgn=_sign(mt_feat_types[j]*(-x1[j]+X[i*n_features+j]))
            if res_mt==-9999:
                res_mt=sgn
            elif res_mt==0:
                res_mt=sgn
            else:
                if res_mt*sgn <0:
                    res_mt=-99  
            if nmt_type==NMT_EQUAL:
                if mt_feat_types[j]==0.:
                    if X[i*n_features+j]!=x1[j]:
                        res_mt=-99
            j=j+1
        res[i]=res_mt
    if nmt_type==NMT_CONE:
        for i in range(n_samples):
            j_nmt_plane=0
            while _abs_int(res[i])<=1 and j_nmt_plane<num_nmt_planes:
            #if _abs_int(res[i])<=1:
                sum_=0.
                for j in range(n_features):
                    sum_=sum_+_abs_float_(X[i*n_features+j]-x1[j])*nmt_plane_norms[j_nmt_plane*n_features + j]
                if (sum_+nmt_intercepts[j_nmt_plane])<0:
                    res[i]=-99
                j_nmt_plane=j_nmt_plane+1
                    
    return 0
    
def compare_pt_with_array(object x1,
                          object X, 
                                   object  mt_feat_types,
                                   int32 nmt_type, 
                                   int32 strict,
                                   object nmt_intercepts, 
                                   object nmt_planes,
                                   np.ndarray[int32, ndim=1] out):
    if len(nmt_planes.shape)==1:
        num_nmt_planes=1
    else:
        num_nmt_planes=nmt_planes.shape[0]
    _compare_pt_with_array(<float64*> (<np.ndarray> x1).data, 
                           <float64*> (<np.ndarray> X).data, 
                            <float64*> (<np.ndarray> np.asarray(mt_feat_types,dtype=np.float64) ).data,
                            X.shape[0],
                            X.shape[1],
                            nmt_type,
                            strict ,
                            <float64*> (<np.ndarray> nmt_intercepts).data ,
                            <float64*> (<np.ndarray> nmt_planes).data,
                            num_nmt_planes,
                            <int32*> (<np.ndarray> out).data)
    return 0
   

cdef int32 _get_F_N_c(float64 *y_cdf_  ,
               int32 *y_unique_counts_  ,
               int32 *comps ,
               int32 n_classes,
               int32 n_samples,
               float64 *F_min ,
               float64 *F_max ,
               int32 *N_min ,
               int32 *N_max ):
    cdef int32 i_train=0
    #cdef int32 counts_t=0
    cdef int32 comp=0
    cdef int32 j_class=0
    cdef int32 cum_sum =0
    cdef int32 cum_sum_rev =0
    for i_train in range(n_samples):
        #counts_t=y_unique_counts_[i_train,:] 
        #cdf_t=y_cdf_[i_train,:] 
        comp=comps[i_train]#
        if comp ==0 or comp==1: # ie GREATER or EQ than the x we want to predict
            cum_sum=0
            for j_class in range(n_classes):
                #F_max=np.asarray([np.max([F_max[l],cdf_t[l]]) for l in np.arange(self.n_classes)])
                #N_max=np.asarray([N_max[l]+np.sum(counts_t[:l+1]) for l in np.arange(self.n_classes) ])
                if F_max[j_class]>y_cdf_[i_train*n_classes+j_class] :
                    F_max[j_class]=F_max[j_class] 
                else :
                    F_max[j_class]=y_cdf_[i_train*n_classes+j_class]            
                cum_sum=cum_sum+y_unique_counts_[i_train*n_classes+j_class]
                N_max[j_class]=N_max[j_class]+cum_sum
        if comp ==0 or comp==-1:  # ie LESSER or EQ than the x we want to predict
            #F_min=np.asarray([np.min([F_min[l],cdf_t[l]]) for l in np.arange(self.n_classes)])
            #N_min=np.asarray([N_min[l]+np.sum(counts_t[l+1:]) for l in np.arange(self.n_classes) ])
            cum_sum_rev=0
            for j_class in range(n_classes-1,-1,-1):
                #F_max=np.asarray([np.max([F_max[l],cdf_t[l]]) for l in np.arange(self.n_classes)])
                if F_min[j_class]<y_cdf_[i_train*n_classes+j_class] :
                    F_min[j_class]=F_min[j_class] 
                else :
                    F_min[j_class]=y_cdf_[i_train*n_classes+j_class]            
                #N_max=np.asarray([N_max[l]+np.sum(counts_t[:l+1]) for l in np.arange(self.n_classes) ])
                
                N_min[j_class]=N_min[j_class]+cum_sum_rev  
                cum_sum_rev=cum_sum_rev+y_unique_counts_[i_train*n_classes+j_class]
    return 0

def get_F_N_c(object y_cdf_,
              object y_unique_counts_,
              object comps,
              int32 n_classes):
    F_min=np.ones(n_classes,np.float64)
    F_max=np.zeros(n_classes,np.float64)
    F_max[n_classes-1]=1.
    N_min=np.zeros(n_classes,np.int32)
    N_max=np.zeros(n_classes,np.int32)
    n_samples=y_unique_counts_.shape[0]
    
    _get_F_N_c(<float64*> (<np.ndarray> y_cdf_).data  ,
               <int32*> (<np.ndarray> y_unique_counts_).data  ,
               <int32*> (<np.ndarray> comps).data ,
               n_classes,
               n_samples,
               <float64*> (<np.ndarray> F_min).data ,
               <float64*> (<np.ndarray> F_max).data ,
               <int32*> (<np.ndarray> N_min).data ,
               <int32*> (<np.ndarray> N_max).data )    
    return F_min,F_max,N_min,N_max
                    
cdef int32 get_next_free_node_id(int32 *free_node_ids, int32 *free_node_ids_num,int32 *num_nodes):
    #cdef idx=free_node_ids_num[0]-1
    cdef int32 res=0
    if free_node_ids_num[0]>0:
        res=free_node_ids[free_node_ids_num[0]-1]
        free_node_ids_num[0]=free_node_ids_num[0]-1
    else:
        res=num_nodes[0]
        num_nodes[0]=num_nodes[0]+1
    return res
#    results=np.zeros(number,dtype=np.int32)
#    #num_to_add=0
#    for i in np.arange(number):
#        idx=self.free_node_ids_num-1
#        if idx>=0:
#            results[i]=self.free_node_ids[idx]
#            self.free_node_ids_num=self.free_node_ids_num-1
#        else:
#            results[i]=self.num_nodes#num_to_add
#            #num_to_add=num_to_add+1
#            self.num_nodes=self.num_nodes+1
#    return results

cdef int32 return_free_node_id(int32 *free_node_ids, int32 *free_node_ids_num, int32 node_id):
    free_node_ids[free_node_ids_num[0]]=node_id
    free_node_ids_num[0]=free_node_ids_num[0]+1
    return 0

cdef int32 _replace_leaf_with_chn(int32 leaf_to_replace,
                                  int32 new_leaf_1,
                                  int32 new_leaf_2,
                                  int32 *leaf_index,
                                  int32 *leaf_array,
                                  int32 *leaf_curr_size):
    leaf_array[leaf_index[leaf_to_replace]]=new_leaf_1
    leaf_index[new_leaf_1]=leaf_index[leaf_to_replace]
    leaf_index[leaf_to_replace]=-1
    leaf_array[leaf_curr_size[0]]=new_leaf_2
    leaf_index[new_leaf_2]=leaf_curr_size[0]
    leaf_curr_size[0]=leaf_curr_size[0]+1
    return 0

cdef float64 calc_probability_(float64 *univar_vals,
                             float64 *univar_probs,
                             int32 num_max_probs,
                             int32 *univar_vals_num,
                             int32 i_feat,
                             float64 min_val,
                             float64 max_val):
    cdef int32 j=0
    cdef start_ttl=0
    cdef float64 ttl_prob=0.
    cdef int32 max_j=0
    max_j=univar_vals_num[i_feat]
    if min_val<RULE_LOWER_CONST:#==-INF:
        j=0
        while univar_vals[i_feat*num_max_probs+j]<=max_val and j<max_j:
            ttl_prob=ttl_prob+univar_probs[i_feat*num_max_probs+j]
            j=j+1
        #return np.sum(dist_probs[dist_vals<=max_val])
    elif max_val>RULE_UPPER_CONST:#==INF:
        j=univar_vals_num[i_feat]-1
        while univar_vals[i_feat*num_max_probs+j]>min_val and j>=0:
            ttl_prob=ttl_prob+univar_probs[i_feat*num_max_probs+j]
            j=j-1
        #return np.sum(dist_probs[dist_vals>min_val])
    else:
        j=0
        while univar_vals[i_feat*num_max_probs+j]<=min_val and j<max_j:
            j=j+1
        while univar_vals[i_feat*num_max_probs+j]<=max_val and j<max_j:
            ttl_prob=ttl_prob+univar_probs[i_feat*num_max_probs+j]
            j=j+1    
        #return np.sum(dist_probs[np.logical_and(dist_vals>min_val,dist_vals<=max_val)])
    return ttl_prob
    
    
cdef int32 _grow_segregated_nodes_c(int32 node_to_grow,
                    int32 node_to_intersect_with,
                    int32 *free_node_ids, 
                    int32 *free_node_ids_num,
                    int32 *num_nodes, 
                    int32 split_criterion,
                    float64 *sample_weight,
                    float64 min_split_weight,
                    int32 num_feats,
                    int32 split_weight,
                    float64 *lower_corners,
                    float64 *upper_corners,
                    int32 *node_train_num,
                    int32 *node_train_idx,
                    int32 assumed_max_nodes,
                    float64 *train_X,
                    int32 *train_y,
                    int32 num_classes,
                    int32 num_samples,
                    float64 *train_sample_weight,
                    int32 *features,
                    float64 *thresholds,
                    int32 *children_left,
                    int32 *children_right,
                    float64 *values,
                    float64 *cdf_data,
                    float64 *cdf,
                    int32 *pred_class,
                    int32 *leaf_index,
                    int32 *leaf_array,
                    int32 *leaf_curr_size,
                    float64 *univar_vals,
                    float64 *univar_probs,
                    int32 num_max_probs,
                    int32 *univar_vals_num
                    ):
    cdef int32 l1=node_to_grow
    cdef int32 l2=node_to_intersect_with
    cdef int32 change_made=0
    cdef int32 temp_split_left_node_id=0#get_next_free_node_id(free_node_ids, free_node_ids_num,num_nodes)
    cdef int32 temp_split_right_node_id=0#get_next_free_node_id(free_node_ids, free_node_ids_num,num_nodes)
    #[temp_split_left_node_id,temp_split_right_node_id]=self.get_next_free_node_ids(2)
    cdef int32 split_decision=0
    cdef int32 dirn=0
    cdef float64 split_val=0.
    cdef int32 i_=0
    cdef int32 i =0
    cdef int32 i_feat=0
    cdef int32 i_class=0
    cdef float64 cum_sum=0.
    cdef int32 i_feat_inner=0
    # split_criterion values:
    cdef int32 sc_both_sides_have_min_sample_wgt=3
    cdef int32 sc_both_sides_have_pts=1
    cdef int32 sc_incomp_side_has_pts=2
    
    
    # 
    # split_weight values:
    cdef int32 sw_univar_prob_distn =3
    cdef int32 sw_parent_weight = 0
    cdef int32 sw_contained_pts_weight = 1
    cdef int32 sw_hybrid_prob =2
    cdef int32 sw_hybrid_prob_empirical = 4
    cdef float64 prob_left =0.
    cdef float64 prob_right =0.
    temp_split_left_node_id=get_next_free_node_id(free_node_ids, free_node_ids_num,num_nodes)
    temp_split_right_node_id=get_next_free_node_id(free_node_ids, free_node_ids_num,num_nodes)
    
    if split_criterion==sc_both_sides_have_min_sample_wgt and sample_weight[l1]<2.0*min_split_weight: # there is no way to split this node, stop 'both_sides_have_min_sample_wgt'
        pass
    else:
        for i_feat in range(num_feats): #feats: # np.arange(len(l1.corner_lower)): #self.mt_feats:#  np.arange(len(l1.corner_lower)):
            dirn=-1
            while dirn<2:
            #for dirn in [-1,+1]:#'left','right']:
                split_val=-99e9
                if split_weight!=sw_univar_prob_distn or (split_weight==sw_univar_prob_distn and sample_weight[l1]>0.000005): # don't split when it gets too small!!
                    if dirn==+1:
                        if lower_corners[l1*num_feats+i_feat]<lower_corners[l2*num_feats+i_feat] and upper_corners[l1*num_feats+i_feat]>lower_corners[l2*num_feats+i_feat] : # slice off bottom bit
                            split_val=lower_corners[l2*num_feats+i_feat]
                    else: # left
                        if upper_corners[l1*num_feats+i_feat]>upper_corners[l2*num_feats+i_feat] and lower_corners[l1*num_feats+i_feat]<upper_corners[l2*num_feats+i_feat] :
                            split_val=upper_corners[l2*num_feats+i_feat]
                if split_val!=-99e9: # need to split on this feat value
                    # work out which points go where for this proposed split
                    node_train_num[temp_split_left_node_id]=0
                    node_train_num[temp_split_right_node_id]=0
                    sample_weight[temp_split_left_node_id]=0.
                    sample_weight[temp_split_right_node_id]=0.
                    for i_ in range(node_train_num[l1]):
                        i=node_train_idx[l1*num_samples+i_]
                        if train_X[i*num_feats+i_feat]<=split_val:
                            node_train_idx[temp_split_left_node_id*num_samples+node_train_num[temp_split_left_node_id]]=i
                            node_train_num[temp_split_left_node_id]=node_train_num[temp_split_left_node_id]+1
                            sample_weight[temp_split_left_node_id]=sample_weight[temp_split_left_node_id]+train_sample_weight[i]
                        else:
                            node_train_idx[temp_split_right_node_id*num_samples+node_train_num[temp_split_right_node_id]]=i
                            node_train_num[temp_split_right_node_id]=node_train_num[temp_split_right_node_id]+1
                            sample_weight[temp_split_right_node_id]=sample_weight[temp_split_right_node_id]+train_sample_weight[i]
                    # adjust child sample weights if required
                    if split_weight==sw_parent_weight:
                        sample_weight[temp_split_left_node_id]=sample_weight[l1]
                        sample_weight[temp_split_right_node_id]=sample_weight[l1]
                    elif split_weight==sw_contained_pts_weight:
                        pass # sample weights already correctly set
                        #self.sample_weight[temp_split_left_node_id]=self.sample_weight[temp_split_left_node_id]#np.max([0.5,self.sample_weight[temp_split_left_node_id]])
                        #self.sample_weight[temp_split_right_node_id]=self.sample_weight[temp_split_right_node_id]#np.max([0.5,self.sample_weight[temp_split_right_node_id]])
                    elif split_weight==sw_hybrid_prob or split_weight==sw_hybrid_prob_empirical:# or self.split_weight=='prob_empirical_cond' or self.split_weight=='hybrid_prob_empirical_orig_train' :
                        if sample_weight[temp_split_left_node_id]==0. or sample_weight[temp_split_right_node_id]==0.:
                            prob_left=calc_probability_(univar_vals,univar_probs,num_max_probs,univar_vals_num,i_feat,lower_corners[l1*num_feats+i_feat],split_val)
                            prob_right=calc_probability_(univar_vals,univar_probs,num_max_probs,univar_vals_num,i_feat,split_val,upper_corners[l1*num_feats+i_feat])
                            sample_weight[temp_split_left_node_id]=sample_weight[l1]*prob_left#/(prob_left+prob_right)
                            sample_weight[temp_split_right_node_id]=sample_weight[l1]*prob_right#/(prob_left+prob_right)
                    elif split_weight==sw_univar_prob_distn:
                        raise NotImplemented
                    # make decision to split or not
                    split_decision=0
                    if split_criterion==sc_both_sides_have_pts:
                        if sample_weight[temp_split_left_node_id]>0:
                            if sample_weight[temp_split_right_node_id]>0:
                                split_decision=1
                    elif split_criterion==sc_incomp_side_has_pts : #len(indx_left)>0 and len(indx_right)>0  : #CHANGE FROM SOLVED VERSION 10/4/17: (len(indx_left)>0 and dirn=='right') or  (len(indx_right)>0 and dirn=='left') :
                        if dirn==+1:
                            if sample_weight[temp_split_left_node_id]>0 :
                                split_decision=1
                        else: # left
                            if sample_weight[temp_split_right_node_id]>0 :
                                split_decision=1 
                    elif split_criterion==sc_both_sides_have_min_sample_wgt : #len(indx_left)>0 and len(indx_right)>0  : #CHANGE FROM SOLVED VERSION 10/4/17: (len(indx_left)>0 and dirn=='right') or  (len(indx_right)>0 and dirn=='left') :
                        if sample_weight[temp_split_left_node_id]>=min_split_weight :
                            if sample_weight[temp_split_right_node_id]>=min_split_weight:
                                split_decision=1
                    else: #if self.split_criterion=='all_splits_ok' : 
                        split_decision=1
                    # split if so decided
                    if split_decision==1:
                        change_made=1
                        features[l1]=i_feat
                        thresholds[l1]=split_val
                        children_left[l1]=temp_split_left_node_id
                        children_right[l1]=temp_split_right_node_id
                        children_left[temp_split_left_node_id]=TREE_LEAF
                        children_right[temp_split_left_node_id]=TREE_LEAF
                        children_left[temp_split_right_node_id]=TREE_LEAF
                        children_right[temp_split_right_node_id]=TREE_LEAF
                        
                        for i_ in range(node_train_num[temp_split_left_node_id]):
                            i=node_train_idx[temp_split_left_node_id*num_samples+i_]
                            values[temp_split_left_node_id*num_classes+train_y[i]]=values[temp_split_left_node_id*num_classes+train_y[i]]+train_sample_weight[i]
                        
                        for i_ in range(node_train_num[temp_split_right_node_id]):
                            i=node_train_idx[temp_split_right_node_id*num_samples+i_]
                            values[temp_split_right_node_id*num_classes+train_y[i]]=values[temp_split_right_node_id*num_classes+train_y[i]]+train_sample_weight[i]
                        
                        cum_sum=0.
                        for i_class in range(num_classes):
                            cum_sum=cum_sum+values[temp_split_left_node_id*num_classes+i_class]
                            cdf_data[temp_split_left_node_id*num_classes+i_class]=cum_sum   
                        if cum_sum>0.:
                            for i_class in range(num_classes):
                                cdf_data[temp_split_left_node_id*num_classes+i_class]=cdf_data[temp_split_left_node_id*num_classes+i_class]/cum_sum

                        cum_sum=0.
                        for i_class in range(num_classes):
                            cum_sum=cum_sum+values[temp_split_right_node_id*num_classes+i_class]
                            cdf_data[temp_split_right_node_id*num_classes+i_class]=cum_sum   
                        if cum_sum>0.:
                            for i_class in range(num_classes):
                                cdf_data[temp_split_right_node_id*num_classes+i_class]=cdf_data[temp_split_right_node_id*num_classes+i_class]/cum_sum

                        for i_class in range(num_classes):
                            cdf[temp_split_left_node_id*num_classes+i_class]=cdf[l1*num_classes+i_class]
                            cdf[temp_split_right_node_id*num_classes+i_class]=cdf[l1*num_classes+i_class]
                            
                        pred_class[temp_split_left_node_id]=pred_class[l1]
                        pred_class[temp_split_right_node_id]=pred_class[l1]
                        
                        for i_feat_inner in range(num_feats):
                            lower_corners[temp_split_left_node_id*num_feats+i_feat_inner]=lower_corners[l1*num_feats+i_feat_inner]
                            lower_corners[temp_split_right_node_id*num_feats+i_feat_inner]=lower_corners[l1*num_feats+i_feat_inner]
                            upper_corners[temp_split_left_node_id*num_feats+i_feat_inner]=upper_corners[l1*num_feats+i_feat_inner]
                            upper_corners[temp_split_right_node_id*num_feats+i_feat_inner]=upper_corners[l1*num_feats+i_feat_inner]
                        upper_corners[temp_split_left_node_id*num_feats+i_feat]=split_val
                        lower_corners[temp_split_right_node_id*num_feats+i_feat]=split_val
                        #self.leaf_ids_obj.replace_leaf_with_chn(l1,temp_split_left_node_id,temp_split_right_node_id)
                        _replace_leaf_with_chn(l1,temp_split_left_node_id,temp_split_right_node_id,leaf_index,leaf_array,leaf_curr_size)
                        # move to child to follow
                        temp_split_left_node_id=get_next_free_node_id(free_node_ids, free_node_ids_num,num_nodes)
                        temp_split_right_node_id=get_next_free_node_id(free_node_ids, free_node_ids_num,num_nodes)
                        l1=children_left[l1] if dirn==-1 else children_right[l1] 
                dirn=dirn+2
    return_free_node_id(free_node_ids, free_node_ids_num, temp_split_left_node_id)
    return_free_node_id(free_node_ids, free_node_ids_num, temp_split_right_node_id)
    
#    num_leaves=self.leaf_ids_obj.curr_size
#    if num_leaves>self.peak_leaves:
#        self.peak_leaves=num_leaves
    return change_made

def grow_segregated_nodes_c(int32 node_to_grow,
                    int32 node_to_intersect_with,
                    np.ndarray[int32, ndim=1] free_node_ids, 
                    np.ndarray[int32, ndim=1]  free_node_ids_num_arr,
                    np.ndarray[int32, ndim=1]  num_nodes_arr, 
                    int32 split_criterion,
                    np.ndarray[float64, ndim=1] sample_weight,
                    float64 min_split_weight,
                    int32 split_weight,
                    np.ndarray[float64, ndim=2] lower_corners,
                    np.ndarray[float64, ndim=2] upper_corners,
                    np.ndarray[int32, ndim=1] node_train_num,
                    np.ndarray[int32, ndim=2] node_train_idx,
                    int32 assumed_max_nodes,
                    np.ndarray[float64, ndim=2] train_X,
                    np.ndarray[int32, ndim=2] train_y,
                    int32 num_classes,
                    np.ndarray[float64, ndim=1] train_sample_weight,
                    np.ndarray[int32, ndim=1] features,
                    np.ndarray[float64, ndim=1] thresholds,
                    np.ndarray[int32, ndim=1] children_left,
                    np.ndarray[int32, ndim=1] children_right,
                    np.ndarray[float64, ndim=2] values,
                    np.ndarray[float64, ndim=2] cdf_data,
                    np.ndarray[float64, ndim=2] cdf,
                    np.ndarray[int32, ndim=1] pred_class,
                    np.ndarray[int32, ndim=1] leaf_index,
                    np.ndarray[int32, ndim=1] leaf_array,
                    np.ndarray[int32, ndim=1] leaf_curr_size,
                    np.ndarray[float64, ndim=2] univar_vals,
                    np.ndarray[float64, ndim=2] univar_probs,
                    np.ndarray[int32, ndim=1] univar_vals_num
                    ):
#    free_node_ids_num_arr=np.zeros(1,dtype=np.int32)
#    free_node_ids_num_arr[0]=free_node_ids_num
#    num_nodes_arr=np.zeros(1,dtype=np.int32)
#    num_nodes_arr[0]=num_nodes
    
    change_made=_grow_segregated_nodes_c(node_to_grow,
                    node_to_intersect_with,
                    <int32*> (<np.ndarray> free_node_ids).data , 
                    <int32*> (<np.ndarray> free_node_ids_num_arr).data ,
                    <int32*> (<np.ndarray> num_nodes_arr).data , 
                    split_criterion,
                    <float64*> (<np.ndarray> sample_weight).data ,
                    min_split_weight,
                    train_X.shape[1],
                    split_weight,
                    <float64*> (<np.ndarray> lower_corners).data ,
                    <float64*> (<np.ndarray> upper_corners).data ,
                    <int32*> (<np.ndarray> node_train_num).data ,
                    <int32*> (<np.ndarray> node_train_idx).data ,
                    assumed_max_nodes,
                    <float64*> (<np.ndarray> train_X).data ,
                    <int32*> (<np.ndarray> train_y).data ,
                    num_classes,
                    train_X.shape[0],
                    <float64*> (<np.ndarray> train_sample_weight).data ,
                    <int32*> (<np.ndarray> features).data ,
                    <float64*> (<np.ndarray> thresholds).data ,
                    <int32*> (<np.ndarray> children_left).data ,
                    <int32*> (<np.ndarray> children_right).data ,
                    <float64*> (<np.ndarray> values).data ,
                    <float64*> (<np.ndarray> cdf_data).data ,
                    <float64*> (<np.ndarray> cdf).data ,
                    <int32*> (<np.ndarray> pred_class).data ,
                    <int32*> (<np.ndarray> leaf_index).data ,
                    <int32*> (<np.ndarray> leaf_array).data ,
                    <int32*> (<np.ndarray> leaf_curr_size).data ,
                    <float64*> (<np.ndarray> univar_vals).data ,
                    <float64*> (<np.ndarray> univar_probs).data ,
                    univar_vals.shape[1],
                    <int32*> (<np.ndarray> univar_vals_num).data 
                    )
    return change_made#[free_node_ids_num_arr[0],num_nodes_arr[0]]
    
    
    # post: update leaf ids, and peak leaves if required