import numpy as np
from scipy.stats import bernoulli
from scipy.linalg import sqrtm
from sklearn.utils.extmath import randomized_svd
#from numpy import random
import math
#import igraph
import random
import time
import functorch
import torch
import pandas as pd
from numpy.linalg import inv
from numpy.linalg import det
from numpy import log, prod, absolute, exp, pi, trace, dot, cumsum, \
                  hstack, ix_, sqrt, eye, diag, array, sum

def sample_graph(adj_A, sel_nodes):
    A11 = torch.index_select(adj_A, dim = 0, index = sel_nodes)
    A12 = torch.index_select(A11, dim = 1, index = sel_nodes)
    return A12

def generate_adj(pm):
    nodes = pm.shape[0]
    A = torch.bernoulli(pm)
    i, j = torch.triu_indices(nodes, nodes)
    vals = A[i,j]
    A.T[i, j] = vals
    return A


def get_emded_gpu_origin(A,d):
    d = d
    w, lamb = torch.linalg.eigh(A)
    sortedEig, indices=torch.sort(w, dim=0, descending=True, out=None)
    indices1 = indices[0:d]
    lambSort = torch.index_select(lamb, dim = 1, index = indices1)
    diag_eigval = torch.diag(torch.sqrt(sortedEig[0:d]))
    lpvs = torch.matmul(lambSort, diag_eigval)
    phat = torch.matmul(lpvs, lpvs.T)
    return phat

def get_emded_gpu_fast(lamb,sortedEig,d):
    d = d
    lambSort = lamb[:,0:d]
    diag_eigval = torch.diag(torch.sqrt(sortedEig[0:d]))
    lpvs = torch.matmul(lambSort, diag_eigval)
    phat = torch.matmul(lpvs, lpvs.T)
    return phat

def pd_transformation(A):
    A1 = torch.matmul(A.T,A)
    A2 = torch.Tensor(sqrtm(A1))
    return A2

def generate_M_univariate(nodes,cov_val,M):
    mean = torch.zeros(M)
    cov = torch.zeros(M,M)
    i, j = torch.triu_indices(M,M)
    cov[i, j] = cov_val
    cov.T[i,j] = cov_val
    cov.fill_diagonal_(1)
    x = np.random.multivariate_normal(mean, cov, nodes).T
    return x

def generate_probability_matrix(x,nodes):
    pm_tensor = torch.ones(nodes, nodes, dtype=torch.float64)
    i, j = torch.triu_indices(nodes,nodes)
    pm_var = torch.exp(-torch.pow(x[i]-x[j],2))
    pm_tensor[i, j] = pm_var
    pm_tensor.T[i,j] = pm_var
    #pm_tensor.fill_diagonal_(0)
    return pm_tensor

def get_approx_svd(matrix,d0):
    U, Sigma, VT = randomized_svd(np.array(matrix), 
                              n_components=d0,
                              n_iter=5)
    return U, Sigma


def get_opt_embed_fast(Sigma,nodes):
    decay_rate = np.log(Sigma[0]/Sigma[1])
    opt_embed = np.log(nodes)/(3*decay_rate)
    return round(opt_embed.item())


def get_opt_embed(matrix,nodes):
    w, lamb = torch.linalg.eigh(matrix/nodes)
    sortedEig, indices=torch.sort(w, dim=0, descending=True, out=None)
    decay_rate = np.log(sortedEig[0]/sortedEig[1])
    opt_embed = np.log(nodes)/(3*decay_rate)
    return round(opt_embed.item())

def cal_product_k(tensor):
    M = tensor.shape[0]
    nodes = tensor.shape[1]
    k0 = torch.ones(nodes,nodes)
    for i in range(M):
        k0 = torch.mul(k0,tensor[i])
    return k0

def hat_GHSIC(tensor):
    k = cal_product_k(tensor)
    n = nodes
    vec1 = torch.ones(n,dtype=torch.float32)
    r1 = torch.matmul(torch.matmul(vec1.T,k),vec1)/(n*n)
    M = tensor.shape[0]
    r2 = 1
    r3 = 1
    for i in range(M):
        r2 = r2*(torch.matmul(torch.matmul(vec1.T,tensor[i]),vec1)/(n*n))
        r3 = r3*(torch.matmul(tensor[i],vec1)/n)
    r3 = torch.matmul(vec1.T,r3)/n
    est = r1 + r2 - 2*r3
    return est

def get_pval(bootsam,tensor,stat):
    sum_gre = 0
    sel_n = nodes
    for k in range(bootsam.shape[1]):
        embed_tensor1 = torch.zeros(M,nodes, nodes)
        for i in range(M):
            embed_tensor1[i] = sample_graph(tensor[i], bootsam[i,k,:])
        est_GHSIC1 = hat_GHSIC(embed_tensor1)
        sum_gre = sum_gre + np.greater_equal(est_GHSIC1, stat)
    pval = (1+sum_gre)/(1+n_boot)
    return pval




#nodes_list = [50,70,90,110,130,150,170,190,210,230,250]
#nodes_list = [300,350,400,450,500]
#nodes_list = [50,100,150,200,250,300,350,400,450,500]
nodes_list = [50,100,150,200,250,300]

fix_pow = torch.zeros(len(nodes_list),20)
unfix_pow = torch.zeros(len(nodes_list),20)
unfix_fast_pow = torch.zeros(len(nodes_list),20)

unfix_time = torch.zeros(len(nodes_list),20)
unfix_fast_time = torch.zeros(len(nodes_list),20)


alpha = 0.05
pt = 200
n_boot = 500

cov_val = 0.3
M = 2

for k in range(len(nodes_list)):
    nodes = nodes_list[k]

    for l in range(20):
        x = generate_M_univariate(nodes,cov_val,M)
        x_gpu = torch.from_numpy(x)

        gcorr_pow1 = 0
        gcorr_pow2 = 0
        gcorr_pow3 = 0

        pm_tensor = torch.ones(M,nodes, nodes)
        embed_para1 = torch.zeros(M)
        embed_para2 = torch.zeros(M)
        embed_para3 = torch.zeros(M)

        adj_tensor = torch.zeros(M,nodes, nodes)
        trans_adj_tensor = torch.zeros(M,nodes, nodes)

        embed_tensor1 = torch.zeros(M,nodes, nodes)
        embed_tensor2 = torch.zeros(M,nodes, nodes)
        embed_tensor3 = torch.zeros(M,nodes, nodes)

        time1_step1 = 0
        time2_step1 = 0

        time1_step2 = 0
        time2_step2 = 0

        time1_step3 = 0
        time2_step3 = 0

        for i in range(M):
            pm_tensor[i] = generate_probability_matrix(x_gpu[i],nodes)
            embed_para1[i] = get_opt_embed(pm_tensor[i],nodes)
            adj_tensor[i] = generate_adj(pm_tensor[i].fill_diagonal_(0))

            s0 = time.time()
            trans_adj_tensor[i] = pd_transformation(adj_tensor[i])
            embed_para2[i] = get_opt_embed(trans_adj_tensor[i],nodes)
            embed_tensor1[i] = get_emded_gpu_origin(trans_adj_tensor[i],int(embed_para1[i].item()))
            e0 = time.time()
            time1_step1 = time1_step1+(e0-s0)

            #embed_tensor2[i] = get_emded_gpu_origin(trans_adj_tensor[i],int(embed_para1[i].item()))


        s0 = time.time()
        est_GHSIC1 = hat_GHSIC(embed_tensor1)
        e0 = time.time()
        time1_step2 = e0-s0


        for h in range(pt):
            sel_n = nodes
            bootsam1 = np.zeros((M,n_boot, sel_n), int)
            for i in range(M):
                for m in range(n_boot):
                    bootsam1[i,m,:] = random.sample(range(0,nodes),sel_n)
            bootsam1 = torch.from_numpy(bootsam1)
            
            s0 = time.time()
            gcorr_pval1 = get_pval(bootsam1,embed_tensor1,est_GHSIC1)
            e0 = time.time()
            time1_step3 = time1_step3+(e0-s0)
            gcorr_pow1 = gcorr_pow1 + np.greater_equal(alpha, gcorr_pval1)


            print(k,h,gcorr_pval1)
        
        unfix_pow[k,l] = gcorr_pow1/pt

        unfix_time[k,l] = time1_step1 + time1_step2 + time1_step3/pt

    
    print(torch.mean(unfix_pow[k,:]),torch.mean(unfix_time[k,:]),)


d1 = unfix_pow.numpy() 


df1 = pd.DataFrame(d1)


df1.to_csv('pow_dhat_M2_cov4.csv')


d4 = unfix_time.numpy() 

df4 = pd.DataFrame(d4)

df4.to_csv('ptime_dhat_M2_cov4.csv')


########M=3#########

nodes_list = [50,100,150,200,250,300]

fix_pow = torch.zeros(len(nodes_list),20)
unfix_pow = torch.zeros(len(nodes_list),20)
unfix_fast_pow = torch.zeros(len(nodes_list),20)

unfix_time = torch.zeros(len(nodes_list),20)
unfix_fast_time = torch.zeros(len(nodes_list),20)


alpha = 0.05
pt = 200
n_boot = 500

cov_val = 0.3
M = 3

for k in range(len(nodes_list)):
    nodes = nodes_list[k]

    for l in range(20):
        x = generate_M_univariate(nodes,cov_val,M)
        x_gpu = torch.from_numpy(x)

        gcorr_pow1 = 0
        gcorr_pow2 = 0
        gcorr_pow3 = 0

        pm_tensor = torch.ones(M,nodes, nodes)
        embed_para1 = torch.zeros(M)
        embed_para2 = torch.zeros(M)
        embed_para3 = torch.zeros(M)

        adj_tensor = torch.zeros(M,nodes, nodes)
        trans_adj_tensor = torch.zeros(M,nodes, nodes)

        embed_tensor1 = torch.zeros(M,nodes, nodes)
        embed_tensor2 = torch.zeros(M,nodes, nodes)
        embed_tensor3 = torch.zeros(M,nodes, nodes)

        time1_step1 = 0
        time2_step1 = 0

        time1_step2 = 0
        time2_step2 = 0

        time1_step3 = 0
        time2_step3 = 0

        for i in range(M):
            pm_tensor[i] = generate_probability_matrix(x_gpu[i],nodes)
            embed_para1[i] = get_opt_embed(pm_tensor[i],nodes)
            adj_tensor[i] = generate_adj(pm_tensor[i].fill_diagonal_(0))

            s0 = time.time()
            trans_adj_tensor[i] = pd_transformation(adj_tensor[i])
            embed_para2[i] = get_opt_embed(trans_adj_tensor[i],nodes)
            embed_tensor1[i] = get_emded_gpu_origin(trans_adj_tensor[i],int(embed_para1[i].item()))
            e0 = time.time()
            time1_step1 = time1_step1+(e0-s0)



        s0 = time.time()
        est_GHSIC1 = hat_GHSIC(embed_tensor1)
        e0 = time.time()
        time1_step2 = e0-s0


        for h in range(pt):
            sel_n = nodes
            bootsam1 = np.zeros((M,n_boot, sel_n), int)
            for i in range(M):
                for m in range(n_boot):
                    bootsam1[i,m,:] = random.sample(range(0,nodes),sel_n)
            bootsam1 = torch.from_numpy(bootsam1)
            
            s0 = time.time()
            gcorr_pval1 = get_pval(bootsam1,embed_tensor1,est_GHSIC1)
            e0 = time.time()
            time1_step3 = time1_step3+(e0-s0)
            gcorr_pow1 = gcorr_pow1 + np.greater_equal(alpha, gcorr_pval1)


            print(k,h,gcorr_pval1)
        
        unfix_pow[k,l] = gcorr_pow1/pt

        unfix_time[k,l] = time1_step1 + time1_step2 + time1_step3/pt

    
    print(torch.mean(unfix_pow[k,:]),torch.mean(unfix_time[k,:]),)


d1 = unfix_pow.numpy() 


df1 = pd.DataFrame(d1)


df1.to_csv('pow_dhat_M3_cov4.csv')


d4 = unfix_time.numpy() 

df4 = pd.DataFrame(d4)

df4.to_csv('ptime_dhat_M3_cov4.csv')


######M=5##########

#######M=3#########

nodes_list = [50,100,150,200,250,300]

fix_pow = torch.zeros(len(nodes_list),20)
unfix_pow = torch.zeros(len(nodes_list),20)
unfix_fast_pow = torch.zeros(len(nodes_list),20)

unfix_time = torch.zeros(len(nodes_list),20)
unfix_fast_time = torch.zeros(len(nodes_list),20)


alpha = 0.05
pt = 200
n_boot = 500

cov_val = 0.3
M = 5

for k in range(len(nodes_list)):
    nodes = nodes_list[k]

    for l in range(20):
        x = generate_M_univariate(nodes,cov_val,M)
        x_gpu = torch.from_numpy(x)

        gcorr_pow1 = 0
        gcorr_pow2 = 0
        gcorr_pow3 = 0

        pm_tensor = torch.ones(M,nodes, nodes)
        embed_para1 = torch.zeros(M)
        embed_para2 = torch.zeros(M)
        embed_para3 = torch.zeros(M)

        adj_tensor = torch.zeros(M,nodes, nodes)
        trans_adj_tensor = torch.zeros(M,nodes, nodes)

        embed_tensor1 = torch.zeros(M,nodes, nodes)
        embed_tensor2 = torch.zeros(M,nodes, nodes)
        embed_tensor3 = torch.zeros(M,nodes, nodes)

        time1_step1 = 0
        time2_step1 = 0

        time1_step2 = 0
        time2_step2 = 0

        time1_step3 = 0
        time2_step3 = 0

        for i in range(M):
            pm_tensor[i] = generate_probability_matrix(x_gpu[i],nodes)
            embed_para1[i] = get_opt_embed(pm_tensor[i],nodes)
            adj_tensor[i] = generate_adj(pm_tensor[i].fill_diagonal_(0))

            s0 = time.time()
            trans_adj_tensor[i] = pd_transformation(adj_tensor[i])
            embed_para2[i] = get_opt_embed(trans_adj_tensor[i],nodes)
            embed_tensor1[i] = get_emded_gpu_origin(trans_adj_tensor[i],int(embed_para1[i].item()))
            e0 = time.time()
            time1_step1 = time1_step1+(e0-s0)



        s0 = time.time()
        est_GHSIC1 = hat_GHSIC(embed_tensor1)
        e0 = time.time()
        time1_step2 = e0-s0


        for h in range(pt):
            sel_n = nodes
            bootsam1 = np.zeros((M,n_boot, sel_n), int)
            for i in range(M):
                for m in range(n_boot):
                    bootsam1[i,m,:] = random.sample(range(0,nodes),sel_n)
            bootsam1 = torch.from_numpy(bootsam1)
            
            s0 = time.time()
            gcorr_pval1 = get_pval(bootsam1,embed_tensor1,est_GHSIC1)
            e0 = time.time()
            time1_step3 = time1_step3+(e0-s0)
            gcorr_pow1 = gcorr_pow1 + np.greater_equal(alpha, gcorr_pval1)


            print(k,h,gcorr_pval1)
        
        unfix_pow[k,l] = gcorr_pow1/pt

        unfix_time[k,l] = time1_step1 + time1_step2 + time1_step3/pt

    
    print(torch.mean(unfix_pow[k,:]),torch.mean(unfix_time[k,:]),)


d1 = unfix_pow.numpy() 


df1 = pd.DataFrame(d1)


df1.to_csv('pow_dhat_M5_cov4.csv')


d4 = unfix_time.numpy() 

df4 = pd.DataFrame(d4)

df4.to_csv('ptime_dhat_M5_cov4.csv')






    