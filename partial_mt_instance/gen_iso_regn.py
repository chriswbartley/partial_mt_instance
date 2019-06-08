# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:09:20 2016

@author: 19514733
"""
import numpy as np
# from quadprog_cvx import quadprog
from cvxopt import matrix as cvxmat, sparse, spmatrix
from cvxopt.solvers import qp, options

##################### START QUAD PROG CODE #####################

def quadprog(H, f, A, b, Aeq, beq, lb, ub):
    """
    minimize:
            (1/2)*x'*H*x + f'*x
    subject to:
            A*x <= b
            Aeq*x = beq 
            lb <= x <= ub
    """
    P, q, G, h, A_, b_ = _convert(H, f, A,b,Aeq, beq, lb, ub)
    options['show_progress'] = False # make verbose=t
    #options['abstol']=1e-20 #(default: 1e-7).
    #options['reltol']=1e-11 #(default: 1e-6)


    results = qp(P, q, G, h, A_, b_)

    # Convert back to NumPy matrix
    # and return solution
    xstar = results['x']
    return results['status'],np.matrix(xstar)

def _convert(H, f, A, b, Aeq, beq, lb, ub):                                                                                  
    """
    Convert everything to                                                                                              
    cvxopt-style matrices                                                                                              
    """ 
    P = cvxmat(H)                                                                                                      
    q = cvxmat(f)
    if Aeq is None:                                                                                                    
        A_ = None                                                                                                       
    else: 
        A_ = cvxmat(Aeq)                                                                                                
    if beq is None:                                                                                                    
        b_ = None                                                                                                       
    else: 
        b_ = cvxmat(beq)                                                                                                
    
    if lb is None and ub is None:
        if A is None:
            G=None
            h=None
        else:
            G = cvxmat(A)
            h = cvxmat(b)
    else:
        n=len(lb)
        if A is None:
            G = sparse([-speye(n), speye(n)])     
            h = cvxmat(np.vstack([-lb, ub])) 
        else:
            G = sparse([cvxmat(A), -speye(n), speye(n)]) 
            h = cvxmat(np.vstack([b,-lb, ub])) 

    return P, q, G, h, A_, b_ 

def speye(n):
    """Create a sparse identity matrix"""
    r = range(n)
    return spmatrix(1.0, r, r)

##################### END QUAD PROG CODE #####################
    
class GeneralisedIsotonicRegression():
    def __init__(self):
        self.X=None
        self.y=None
        self.y_pred=None
        pass
    
    def fit(self,  y, constraint_pairs_by_index=None,sample_weight=None,increasing=True):
        options['show_progress'] = False
        # initialise
        n=y.shape[0]
        self.y=np.asarray(y,dtype='float').copy().reshape([n,1])
        # if no cosntraints passed, default to monotone increasing
        if constraint_pairs_by_index is None:
            constraint_pairs_by_index=[]
            for i in np.arange(len(y)):
                if i<len(y)-1:
                    constraint_pairs_by_index.append([i,i+1])
        # if no sample weights, assumed equal            
        if sample_weight is None:
            sample_weight=np.ones(n,dtype='float').reshape([n,1])
        else:
            sample_weight=sample_weight.reshape([n,1])
        # eliminate y values with no constraints attached
        indexes_with_constraints=list(np.sort(np.unique(np.ravel(np.asarray(constraint_pairs_by_index)))))
        if len(indexes_with_constraints)<n:
            #indexes_without_constraints=[i not in indexes_with_constraints for i in np.arange(n) ]
            #indexes_without_constraints=set(np.arange(n))
            #indexes_without_constraints.difference_update(indexes_with_constraints)
            y_reduced=self.y[indexes_with_constraints].copy()
            constraints_reduced=constraint_pairs_by_index.copy()
            for i in np.arange(len(constraint_pairs_by_index)):
                constraints_reduced[i]=(indexes_with_constraints.index(constraint_pairs_by_index[i][0]),indexes_with_constraints.index(constraint_pairs_by_index[i][1]))
            n_reduced=len(y_reduced)
            sample_weight_reduced=sample_weight[indexes_with_constraints]
        else:
            y_reduced=self.y.copy()
            constraints_reduced=constraint_pairs_by_index
            n_reduced=n
            sample_weight_reduced=sample_weight
        # solve
        K=len(constraints_reduced)
        
        
        # set up qp matrices
        """
        minimize:
                (1/2)*x'*H*x + f'*x
        subject to:
                A*x <= b
                Aeq*x = beq 
                lb <= x <= ub
        """
        H=np.eye(n_reduced)
        for i in np.arange(n_reduced):
            H[i,i]=sample_weight_reduced[i,0]
        f=-1.0*y_reduced*sample_weight_reduced
        b=np.zeros([K,1])
        A=np.zeros([K,n_reduced],dtype='float')
        for k in np.arange(K):
            A[k,constraints_reduced[k][0]]=1.0 if increasing else -1.0
            A[k,constraints_reduced[k][1]]=-1.0 if increasing else 1.0
        Aeq=None
        beq=None
        ub=None
        lb=None
        # solve
        status, alphas=quadprog(H, f, A, b, Aeq, beq, lb, ub)
        alphas=np.ravel(alphas)
        # get results
        y_pred_reduced=alphas
        if n_reduced==n:
            self.y_pred=y_pred_reduced.copy()
        else:
            self.y_pred=self.y.copy()
            self.y_pred[indexes_with_constraints,0]=y_pred_reduced #.reshape([len(y_pred_reduced),1])
        return status,np.ravel(self.y_pred)
        
    def predict(self,X=None):
        if X is None:
            return self.y_pred
        else: # lookup results
            raise NotImplemented('not implemented for arbitrary X yet')
            return
        