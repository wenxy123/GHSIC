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


def generate_adj(pm):
    nodes = pm.shape[0]
    A = torch.bernoulli(pm)
    i, j = torch.triu_indices(nodes, nodes)
    vals = A[i,j]
    A.T[i, j] = vals
    return A


def get_emded_gpu_fast(lamb,sortedEig,d):
    d = d
    lambSort = lamb[:,0:d]
    diag_eigval = torch.diag(torch.sqrt(sortedEig[0:d]))
    lpvs = torch.matmul(lambSort, diag_eigval)
    phat = torch.matmul(lpvs, lpvs.T)
    return phat

def get_emded_gpu_origin(A,d):
    w, lamb = torch.linalg.eigh(A)
    sortedEig, indices=torch.sort(w, dim=0, descending=True, out=None)
    indices1 = indices[0:d]
    lambSort = torch.index_select(lamb, dim = 1, index = indices1)
    diag_eigval = torch.diag(torch.sqrt(sortedEig[0:d]))
    lpvs = torch.matmul(lambSort, diag_eigval)
    phat = torch.matmul(lpvs, lpvs.T)
    return phat

def pd_transformation(A):
    A1 = torch.matmul(A.T,A)
    A2 = torch.Tensor(sqrtm(A1))
    return A2


def generate_mean_cov(M):
    m1 = torch.zeros(M)
    cov1 = torch.zeros(M,M)
    i, j = torch.triu_indices(M,M)
    cov1[i, j] = cov_val
    cov1.T[i,j] = cov_val
    cov1.fill_diagonal_(1)
    
    m2 = torch.zeros(M)
    cov2 = torch.zeros(M,M)
    cov2.fill_diagonal_(1)
    return m1,cov1,m2,cov2

def cal_HSIC(m1,m2,cov1,cov2):
    dim = len(m1)
    gam = 2
    k3 = 1 / sqrt(absolute(det(gam * cov1 + gam * cov2 + eye(dim))))
    k1 = 1 / sqrt(absolute(det(gam * cov1 + gam * cov1 + eye(dim))))
    k2 = 1 / sqrt(absolute(det(gam * cov2 + gam * cov2 + eye(dim))))
    d = k1 + k2 - 2 * k3
    return d 

def generate_M_univariate(nodes,cov_val,M):
    mean = torch.zeros(M)
    cov = torch.zeros(M,M)
    i, j = torch.triu_indices(M,M)
    cov[i, j] = cov_val
    cov.T[i,j] = cov_val
    cov.fill_diagonal_(1)
    x = np.random.multivariate_normal(mean, cov, nodes).T
    return x

def generate_probability_matrix(x,gamma,nodes):
    pm_tensor = torch.ones(nodes, nodes, dtype=torch.float64)
    i, j = torch.triu_indices(nodes,nodes)
    pm_var = torch.exp(-(2/gamma)*torch.pow(x[i]-x[j],2))
    pm_tensor[i, j] = pm_var
    pm_tensor.T[i,j] = pm_var
    #pm_tensor.fill_diagonal_(0)
    return pm_tensor

def get_approx_svd(matrix,d0):
    U, Sigma, VT = randomized_svd(np.array(matrix), 
                              n_components=d0,
                              n_iter=5)
    return U, Sigma

def cal_tau(sigma,omega):
    beta = (2*sigma)/omega
    lambda0 = np.sqrt(2/(1+beta+np.sqrt(1+2*beta)))
    lambda1 = np.sqrt(2/(1+beta+np.sqrt(1+2*beta)))*(beta/(1+beta+np.sqrt(1+2*beta)))
    tau = np.log(lambda0/lambda1)
    return tau


def get_real_embed(tau,nodes):
    opt_embed = np.log(nodes)/(3*tau)
    return round(opt_embed.item())


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


#######################gamma=1######################################

nodes_list = [100,300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500,2700,3000]

re_time = 50
M = 2
cov_val = 0.9
d0 = 10

gamma = 1
omega = gamma/4
tau_m = cal_tau(1,omega)

gcor0_list = torch.zeros(len(nodes_list),re_time)
gcor1_list = torch.zeros(len(nodes_list),re_time)
gcor2_list = torch.zeros(len(nodes_list),re_time)
gcor3_list = torch.zeros(len(nodes_list),re_time)

hcor_list = torch.zeros(len(nodes_list),re_time)

gcor_time0 = torch.zeros(len(nodes_list),re_time)
gcor_time1 = torch.zeros(len(nodes_list),re_time)
gcor_time2 = torch.zeros(len(nodes_list),re_time)
gcor_time3 = torch.zeros(len(nodes_list),re_time)

hcor_time = torch.zeros(len(nodes_list),re_time)

for k in range(len(nodes_list)):
    nodes = nodes_list[k]

    for l in range(re_time):
        np.random.seed(l)
        x = generate_M_univariate(nodes,cov_val,M)
        x_gpu = torch.from_numpy(x)

        pm_tensor = torch.ones(M,nodes, nodes)

        embed_para0 = torch.zeros(M)
        embed_para1 = torch.zeros(M)
        embed_para2 = torch.zeros(M)
        embed_para3 = torch.zeros(M)

        adj_tensor = torch.zeros(M,nodes, nodes)
        trans_adj_tensor = torch.zeros(M,nodes, nodes)

        embed_tensor0 = torch.zeros(M,nodes, nodes)
        embed_tensor1 = torch.zeros(M,nodes, nodes)
        embed_tensor2 = torch.zeros(M,nodes, nodes)
        embed_tensor3 = torch.zeros(M,nodes, nodes)


        for i in range(M):
            embed_para0[i] = get_real_embed(tau_m,nodes)

            pm_tensor[i] = generate_probability_matrix(x_gpu[i],gamma,nodes)
            embed_para1[i] = get_opt_embed(pm_tensor[i],nodes)

            adj_tensor[i] = generate_adj(pm_tensor[i].fill_diagonal_(0))
            A1 = torch.matmul(adj_tensor[i].T,adj_tensor[i])
            
            s0 = time.time()
            trans_adj_tensor[i] = pd_transformation(adj_tensor[i])
            embed_para2[i] = get_opt_embed(trans_adj_tensor[i],nodes)
            e0 = time.time()
            gcor_time0[k,l] = e0-s0 

            embed_tensor0[i] = get_emded_gpu_origin(trans_adj_tensor[i],int(embed_para0[i].item()))
            embed_tensor1[i] = get_emded_gpu_origin(trans_adj_tensor[i],int(embed_para1[i].item()))
            embed_tensor2[i] = get_emded_gpu_origin(trans_adj_tensor[i],int(embed_para2[i].item()))
            
            s1 = time.time()
            U, Sigma = get_approx_svd(A1,d0)
            Sigma0 = np.sqrt(Sigma/(nodes*nodes))
            Sigma1 = np.sqrt(Sigma)
            embed_para3[i] = get_opt_embed_fast(Sigma0,nodes)
            e1 = time.time()
            gcor_time1[k,l] = e1-s1 
            
            
            s2 = time.time()
            embed_tensor3[i] = get_emded_gpu_fast(torch.Tensor(U),torch.Tensor(Sigma1),int(embed_para3[i].item()))
            e2 = time.time()
            gcor_time2[k,l] = e2-s2 
            
       
        
        est_GHSIC0 = hat_GHSIC(embed_tensor0)
        est_GHSIC1 = hat_GHSIC(embed_tensor1)
        est_GHSIC2 = hat_GHSIC(embed_tensor2)

        s3 = time.time()
        est_GHSIC3 = hat_GHSIC(embed_tensor3)
        e3 = time.time()
        gcor_time3[k,l] = e3-s3   

        s4 = time.time()
        emp_HSIC = hat_GHSIC(pm_tensor)
        e3 = time.time()
        hcor_time[k,l] = e3-s3  

        gcor0_list[k,l] = est_GHSIC0
        gcor1_list[k,l] = est_GHSIC1
        gcor2_list[k,l] = est_GHSIC2
        gcor3_list[k,l] = est_GHSIC3

        hcor_list[k,l] = emp_HSIC

        print(k,l)

d0 = pd.DataFrame(gcor0_list.numpy()).to_csv('gcor_gamma1_reald.csv')
d1 = pd.DataFrame(gcor1_list.numpy()).to_csv('gcor_gamma1_hatd.csv')
d2 = pd.DataFrame(gcor2_list.numpy()).to_csv('gcor_gamma1_hathatd.csv')
d3 = pd.DataFrame(gcor3_list.numpy()).to_csv('gcor_gamma1_hathatd_fast.csv')

d4 = pd.DataFrame(hcor_list.numpy()).to_csv('hcor_gamma1.csv')

d5 = pd.DataFrame(gcor_time0.numpy()).to_csv('gcor_gamma1_hathatd_time.csv')
d6 = pd.DataFrame(gcor_time1.numpy()).to_csv('gcor_gamma1_hathatd_fast_time.csv')
d7 = pd.DataFrame(gcor_time2.numpy()).to_csv('gcor_gamma1_hatK_time.csv')
d8 = pd.DataFrame(gcor_time3.numpy()).to_csv('gcor_gamma1_GHSIC_time.csv')

d9 = pd.DataFrame(hcor_time.numpy()).to_csv('hcor_gamma1_time.csv')


#######################gamma=2######################################

nodes_list = [100,300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500,2700,3000]

re_time = 50
M = 2
cov_val = 0.9
d0 = 10

gamma = 2
omega = gamma/4
tau_m = cal_tau(1,omega)

gcor0_list = torch.zeros(len(nodes_list),re_time)
gcor1_list = torch.zeros(len(nodes_list),re_time)
gcor2_list = torch.zeros(len(nodes_list),re_time)
gcor3_list = torch.zeros(len(nodes_list),re_time)

hcor_list = torch.zeros(len(nodes_list),re_time)

gcor_time0 = torch.zeros(len(nodes_list),re_time)
gcor_time1 = torch.zeros(len(nodes_list),re_time)
gcor_time2 = torch.zeros(len(nodes_list),re_time)
gcor_time3 = torch.zeros(len(nodes_list),re_time)

hcor_time = torch.zeros(len(nodes_list),re_time)

for k in range(len(nodes_list)):
    nodes = nodes_list[k]

    for l in range(re_time):
        np.random.seed(l)
        x = generate_M_univariate(nodes,cov_val,M)
        x_gpu = torch.from_numpy(x)

        pm_tensor = torch.ones(M,nodes, nodes)

        embed_para0 = torch.zeros(M)
        embed_para1 = torch.zeros(M)
        embed_para2 = torch.zeros(M)
        embed_para3 = torch.zeros(M)

        adj_tensor = torch.zeros(M,nodes, nodes)
        trans_adj_tensor = torch.zeros(M,nodes, nodes)

        embed_tensor0 = torch.zeros(M,nodes, nodes)
        embed_tensor1 = torch.zeros(M,nodes, nodes)
        embed_tensor2 = torch.zeros(M,nodes, nodes)
        embed_tensor3 = torch.zeros(M,nodes, nodes)


        for i in range(M):
            embed_para0[i] = get_real_embed(tau_m,nodes)

            pm_tensor[i] = generate_probability_matrix(x_gpu[i],gamma,nodes)
            embed_para1[i] = get_opt_embed(pm_tensor[i],nodes)

            adj_tensor[i] = generate_adj(pm_tensor[i].fill_diagonal_(0))
            A1 = torch.matmul(adj_tensor[i].T,adj_tensor[i])
            
            s0 = time.time()
            trans_adj_tensor[i] = pd_transformation(adj_tensor[i])
            embed_para2[i] = get_opt_embed(trans_adj_tensor[i],nodes)
            e0 = time.time()
            gcor_time0[k,l] = e0-s0 

            embed_tensor0[i] = get_emded_gpu_origin(trans_adj_tensor[i],int(embed_para0[i].item()))
            embed_tensor1[i] = get_emded_gpu_origin(trans_adj_tensor[i],int(embed_para1[i].item()))
            embed_tensor2[i] = get_emded_gpu_origin(trans_adj_tensor[i],int(embed_para2[i].item()))
            
            s1 = time.time()
            U, Sigma = get_approx_svd(A1,d0)
            Sigma0 = np.sqrt(Sigma/(nodes*nodes))
            Sigma1 = np.sqrt(Sigma)
            embed_para3[i] = get_opt_embed_fast(Sigma0,nodes)
            e1 = time.time()
            gcor_time1[k,l] = e1-s1 
            
            
            s2 = time.time()
            embed_tensor3[i] = get_emded_gpu_fast(torch.Tensor(U),torch.Tensor(Sigma1),int(embed_para3[i].item()))
            e2 = time.time()
            gcor_time2[k,l] = e2-s2 
            
       
        
        est_GHSIC0 = hat_GHSIC(embed_tensor0)
        est_GHSIC1 = hat_GHSIC(embed_tensor1)
        est_GHSIC2 = hat_GHSIC(embed_tensor2)

        s3 = time.time()
        est_GHSIC3 = hat_GHSIC(embed_tensor3)
        e3 = time.time()
        gcor_time3[k,l] = e3-s3   

        s4 = time.time()
        emp_HSIC = hat_GHSIC(pm_tensor)
        e3 = time.time()
        hcor_time[k,l] = e3-s3  

        gcor0_list[k,l] = est_GHSIC0
        gcor1_list[k,l] = est_GHSIC1
        gcor2_list[k,l] = est_GHSIC2
        gcor3_list[k,l] = est_GHSIC3

        hcor_list[k,l] = emp_HSIC

        print(k,l)

d0 = pd.DataFrame(gcor0_list.numpy()).to_csv('gcor_gamma2_reald.csv')
d1 = pd.DataFrame(gcor1_list.numpy()).to_csv('gcor_gamma2_hatd.csv')
d2 = pd.DataFrame(gcor2_list.numpy()).to_csv('gcor_gamma2_hathatd.csv')
d3 = pd.DataFrame(gcor3_list.numpy()).to_csv('gcor_gamma2_hathatd_fast.csv')

d4 = pd.DataFrame(hcor_list.numpy()).to_csv('hcor_gamma2.csv')

d5 = pd.DataFrame(gcor_time0.numpy()).to_csv('gcor_gamma2_hathatd_time.csv')
d6 = pd.DataFrame(gcor_time1.numpy()).to_csv('gcor_gamma2_hathatd_fast_time.csv')
d7 = pd.DataFrame(gcor_time2.numpy()).to_csv('gcor_gamma2_hatK_time.csv')
d8 = pd.DataFrame(gcor_time3.numpy()).to_csv('gcor_gamma2_GHSIC_time.csv')

d9 = pd.DataFrame(hcor_time.numpy()).to_csv('hcor_gamma2_time.csv')


#######################gamma=4######################################

nodes_list = [100,300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500,2700,3000]

re_time = 50
M = 2
cov_val = 0.9
d0 = 10

gamma = 4
omega = gamma/4
tau_m = cal_tau(1,omega)

gcor0_list = torch.zeros(len(nodes_list),re_time)
gcor1_list = torch.zeros(len(nodes_list),re_time)
gcor2_list = torch.zeros(len(nodes_list),re_time)
gcor3_list = torch.zeros(len(nodes_list),re_time)

hcor_list = torch.zeros(len(nodes_list),re_time)

gcor_time0 = torch.zeros(len(nodes_list),re_time)
gcor_time1 = torch.zeros(len(nodes_list),re_time)
gcor_time2 = torch.zeros(len(nodes_list),re_time)
gcor_time3 = torch.zeros(len(nodes_list),re_time)

hcor_time = torch.zeros(len(nodes_list),re_time)

for k in range(len(nodes_list)):
    nodes = nodes_list[k]

    for l in range(re_time):
        np.random.seed(l)
        x = generate_M_univariate(nodes,cov_val,M)
        x_gpu = torch.from_numpy(x)

        pm_tensor = torch.ones(M,nodes, nodes)

        embed_para0 = torch.zeros(M)
        embed_para1 = torch.zeros(M)
        embed_para2 = torch.zeros(M)
        embed_para3 = torch.zeros(M)

        adj_tensor = torch.zeros(M,nodes, nodes)
        trans_adj_tensor = torch.zeros(M,nodes, nodes)

        embed_tensor0 = torch.zeros(M,nodes, nodes)
        embed_tensor1 = torch.zeros(M,nodes, nodes)
        embed_tensor2 = torch.zeros(M,nodes, nodes)
        embed_tensor3 = torch.zeros(M,nodes, nodes)


        for i in range(M):
            embed_para0[i] = get_real_embed(tau_m,nodes)

            pm_tensor[i] = generate_probability_matrix(x_gpu[i],gamma,nodes)
            embed_para1[i] = get_opt_embed(pm_tensor[i],nodes) 

            adj_tensor[i] = generate_adj(pm_tensor[i].fill_diagonal_(0))
            A1 = torch.matmul(adj_tensor[i].T,adj_tensor[i])
            
            s0 = time.time()
            trans_adj_tensor[i] = pd_transformation(adj_tensor[i])
            embed_para2[i] = get_opt_embed(trans_adj_tensor[i],nodes)
            e0 = time.time()
            gcor_time0[k,l] = e0-s0 

            embed_tensor0[i] = get_emded_gpu_origin(trans_adj_tensor[i],int(embed_para0[i].item()))
            embed_tensor1[i] = get_emded_gpu_origin(trans_adj_tensor[i],int(embed_para1[i].item()))
            embed_tensor2[i] = get_emded_gpu_origin(trans_adj_tensor[i],int(embed_para2[i].item()))
            
            s1 = time.time()
            U, Sigma = get_approx_svd(A1,d0)
            Sigma0 = np.sqrt(Sigma/(nodes*nodes))
            Sigma1 = np.sqrt(Sigma)
            embed_para3[i] = get_opt_embed_fast(Sigma0,nodes) 
            e1 = time.time()
            gcor_time1[k,l] = e1-s1 
            
            
            s2 = time.time()
            embed_tensor3[i] = get_emded_gpu_fast(torch.Tensor(U),torch.Tensor(Sigma1),int(embed_para3[i].item()))
            e2 = time.time()
            gcor_time2[k,l] = e2-s2 
            
       
        
        est_GHSIC0 = hat_GHSIC(embed_tensor0)
        est_GHSIC1 = hat_GHSIC(embed_tensor1)
        est_GHSIC2 = hat_GHSIC(embed_tensor2)

        s3 = time.time()
        est_GHSIC3 = hat_GHSIC(embed_tensor3)
        e3 = time.time()
        gcor_time3[k,l] = e3-s3   

        s4 = time.time()
        emp_HSIC = hat_GHSIC(pm_tensor)
        e3 = time.time()
        hcor_time[k,l] = e3-s3  

        gcor0_list[k,l] = est_GHSIC0
        gcor1_list[k,l] = est_GHSIC1
        gcor2_list[k,l] = est_GHSIC2
        gcor3_list[k,l] = est_GHSIC3

        hcor_list[k,l] = emp_HSIC

        print(k,l)

d0 = pd.DataFrame(gcor0_list.numpy()).to_csv('gcor_gamma4_reald.csv')
d1 = pd.DataFrame(gcor1_list.numpy()).to_csv('gcor_gamma4_hatd.csv')
d2 = pd.DataFrame(gcor2_list.numpy()).to_csv('gcor_gamma4_hathatd.csv')
d3 = pd.DataFrame(gcor3_list.numpy()).to_csv('gcor_gamma4_hathatd_fast.csv')

d4 = pd.DataFrame(hcor_list.numpy()).to_csv('hcor_gamma4.csv')

d5 = pd.DataFrame(gcor_time0.numpy()).to_csv('gcor_gamma4_hathatd_time.csv')
d6 = pd.DataFrame(gcor_time1.numpy()).to_csv('gcor_gamma4_hathatd_fast_time.csv')
d7 = pd.DataFrame(gcor_time2.numpy()).to_csv('gcor_gamma4_hatK_time.csv')
d8 = pd.DataFrame(gcor_time3.numpy()).to_csv('gcor_gamma4_GHSIC_time.csv')

d9 = pd.DataFrame(hcor_time.numpy()).to_csv('hcor_gamma4_time.csv')