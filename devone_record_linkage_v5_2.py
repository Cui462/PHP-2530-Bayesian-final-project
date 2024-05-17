### DEVONE IMPLEMENTATION V5: Power prior and l-instances strucutre implementation
import numpy as np

import pandas as pd

from scipy.stats import dirichlet
from scipy.stats import beta
from scipy.special import logit, expit
import math

# pip install jaro-winkler
import jaro

import time
start_time = time.time()

ariable_choose = ["_STATE","FMONTH","IMONTH","IDAY","CELLFON3","SEX","MARITAL","_RACEG21","_PAINDX1"]

#A_temp = pd.read_csv(r"./2015 Shortened.csv")[variable_choose] #[0:100]
#B_temp = pd.read_csv(r"./2015 Shortened w Errors.csv")[variable_choose] #[0:100]

A_temp = pd.read_csv(r"./2015_selected.csv")[0:50]
B_temp = pd.read_csv(r"./2015_selected_Errors.csv")[0:50]

## Global Variables:

#A is the larger file 
if len(A_temp.index) >= len(B_temp.index):
    A = A_temp
    B = B_temp
    N_a = len(A.index) # Equivalent to N_a
    N_b = len(B.index) # Equivalent to N_b
else:
    A = B_temp
    B = A_temp
    N_a = len(A.index) # Equivalent to N_a
    N_b = len(B.index) # Equivalent to N_b

  
X_a = A[np.sort(A.columns.intersection(B.columns))]
X_b = B[np.sort(B.columns.intersection(A.columns))]

K = len(X_a.columns)

L_k_n = 5 # Levels of disagreement (100 for 2 decimal place values of Jaro-Winkler Distance)

comparison_arrays = np.full((K, (N_a*N_b)), fill_value = 0, dtype= float)
# Order of stored l instances: known matches (0), known non-matches (1), unknown matches (2), unknown non-matches (3)
base_l_instances = np.full((K, L_k_n, 4), fill_value=0, dtype=int)

C_prior = np.full(((N_a * N_b), 2), fill_value=0, dtype=int)
from scipy.stats import bernoulli
np.random.seed(seed=1)
known_pairs = bernoulli.rvs(size=N_a * N_b, p=0.15)
C_prior[:,1] = known_pairs
C_prior[:,0] = (known_pairs.reshape(N_a , N_b) * np.diag(np.ones(N_a))).reshape(N_a * N_b)

#np.fill_diagonal(C_prior[:,0].reshape(N_a,N_b)[0:int(N_a/10), 0:int(N_b/10)], 1)
#C_prior[0:int(N_a*N_b/10),1] = 1
#C_prior[int(N_a*N_b/10):,0] = np.random.randint(2, size=len(C_prior[int(N_a*N_b/10):,0]))

####C_prior[:,0] = 1

## Functions:
# Returns jaro_winkler_distance of two strings
def jaro_winkler_distance(s1, s2):
    jaro_winkler = round(jaro.jaro_winkler_metric(s1,s2),1)

    if jaro_winkler > 1: 
        jaro_winkler = 1.0
    
    return jaro_winkler

# Filling in Comparison Vectors (Gamma Vectors):
def fill_comparison_arrays():
    # Filling comparison vectors:
    for a in range(N_a):
        print(a)
        for b in range(N_b):
            for k in range(K):
                if str(X_a.iat[a,k]) != "" and str(X_b.iat[b,k]) != "":
                    distance = jaro_winkler_distance(str(X_a.iat[a,k]), str(X_b.iat[b,k]))
                    comparison_arrays[k, ((N_b*a) + b)] = distance
                    # Known match counter
                    if (C_prior[N_b*a + b, 0] == 1) and (C_prior[N_b*a + b, 1] == 1):
                        base_l_instances[k,int(((L_k_n - 1)*distance)),0] += 1
                    # Known non-match counter
                    elif (C_prior[N_b*a + b, 0] == 0) and (C_prior[N_b*a + b, 1] == 1):
                        base_l_instances[k,int(((L_k_n - 1)*distance)),1] += 1
                    # Unknown match counter  
                    elif (C_prior[N_b*a + b, 0] == 1) and (C_prior[N_b*a + b, 1] == 0):
                        base_l_instances[k,int(((L_k_n - 1)*distance)),2] += 1
                    # Unknown non-match counter  
                    elif (C_prior[N_b*a + b, 0] == 0) and (C_prior[N_b*a + b, 1] == 0):
                        base_l_instances[k,int(((L_k_n - 1)*distance)),3] += 1
                else:
                    comparison_arrays[k, ((N_b*a) + b)] = None


fill_comparison_arrays()

# Gibbs Sampler 
def theta_and_c_sampler(T:int, Burn):
    C = np.full(((N_a * N_b), 2), fill_value=0, dtype=int)
    C[:,:] = C_prior[:,:]
    #Establishing initial parameters for the Dirchlet Distributions from which we're sampling:
    M_alpha_priors = np.full(L_k_n, 1, dtype=int)
    U_alpha_priors = np.full(L_k_n, 1, dtype=int)
    ## Gibbs Sampler for Theta Values:
    theta_values = np.full((T, K, 2, L_k_n), 0.00, dtype=float) # Array with K rows (for number of iterations)
                                                                # F columns (one for each comparison variable), and 
                                                                # two theta values vectors in each cell (Theta_M and Theta_U 
                                                                # vectors of length L_f)
    temp_l_instances = np.full((K, L_k_n, 4), fill_value=0, dtype=int)
    temp_l_instances[:,:,:] = base_l_instances[:,:,:]

    accuarcy_all = np.full(T, 0.00, dtype=float)

    # alpha zero
    alpha_zero = np.full(T, 0.00, dtype=float)
    log_posteria_alpha = np.full(T, 0.00, dtype=float)

    #fills dirichlet parameters for theta_M  or theta_U depending on if theta_M == True or False
    def alpha_fill(k: int, theta_type: bool) -> np.ndarray:
        a_lst = []
        for l in range(L_k_n): 
            if theta_type:
                a_lst.append((temp_l_instances[k,l,0]*alpha + temp_l_instances[k,l,2] + M_alpha_priors[l]))
            else: 
                a_lst.append((temp_l_instances[k,l,1]*alpha + temp_l_instances[k,l,3] + U_alpha_priors[l]))
        alpha_params = np.array(a_lst)
        return alpha_params
    
    def likelihood_ratio(a, b) -> float: 
        m_lh = 1
        u_lh = 1
        for k in range(K): 
            lvl = comparison_arrays[k, int(N_b* a + b)]

            if pd.notna(lvl):
                theta_mkl = theta_values[t, k, 0, int((L_k_n-1)*lvl)]
                theta_ukl = theta_values[t, k, 1, int((L_k_n-1)*lvl)]
            else:
                theta_mkl = 1
                theta_ukl = 1
            
            m_lh *= theta_mkl
            u_lh *= theta_ukl
        
        lr = m_lh/u_lh 
        return lr

    def C_matrix_to_df(C, N_a, N_b):
        C_dataframe = pd.DataFrame(index=range(N_a), columns=range(N_b))
        for a in range(N_a):
            for b in range(N_b):
                C_dataframe.iat[a, b] = C[N_b * a + b, 0]

        counter = 0

        for a in range(N_a):
            if comparison_df.iat[a, a] == 1 and C_dataframe.iat[a, a] == 1:
                counter += 1
        correct_percentage = counter / N_a

        return correct_percentage

    Average_Accuracy = 0

    comparison_df = pd.DataFrame(index=range(N_a), columns=range(N_a))
    for a in range(N_a):
        for b in range(N_b):
            if a == b:
                comparison_df.iat[a, b] = 1
            else:
                comparison_df.iat[a, b] = 0

    for t in range(T):
        print(t)
        # step 0:
        if t == 0:
            alpha_zero[t] = 0.5
            log_posteria_alpha[t] = -1000000
        else:
            alpha_zero[t] = expit(np.random.normal(logit(alpha_zero[t - 1]), 1/100))  #20,50,200,300

            log_posteria_alpha[t] = alpha_zero[t] * \
                                    (np.sum(np.log(theta_values[t-1, :, 0, :]) * base_l_instances[:, :, 0]) +
                                     np.sum(np.log(theta_values[t-1, :, 1, :]) * base_l_instances[:, :, 1])) +\
                                     beta.logpdf(alpha_zero[t], 6*50*50*0.15, 1)
                # prior_theta + beta.logpdf(alpha_zero[t], 0.5, 0.5) # prior 8, 10
        ###
        #print("log_posteria---------")
        #print(log_posteria_alpha[t])
        if (t == 0):
            accept_ratio = 1
        elif(log_posteria_alpha[t] - log_posteria_alpha[t-1] > 1):
            accept_ratio = 1
        else:
            accept_ratio = math.exp(log_posteria_alpha[t] - log_posteria_alpha[t-1])
            #print("accept_ratio-----------")
            #print(accept_ratio)

        alpha_accept_ratio = min(1, accept_ratio)

        if (np.random.uniform() >= alpha_accept_ratio):
            alpha_zero[t] = alpha_zero[t-1]
            log_posteria_alpha[t] = log_posteria_alpha[t-1]

        alpha = alpha_zero[t]
        #print("--------alpha")
        #print(alpha)


        #Step 1: sampling thetas
        for k in range(K):
            ## Sampling for Theta_M Values:
            M_alpha_vec = alpha_fill(k, True)
            theta_values[t,k, 0] = np.random.dirichlet(M_alpha_vec)
            ## Sampling for Theta_U Values:
            U_alpha_vec = alpha_fill(k, False)
            theta_values[t,k, 1] = np.random.dirichlet(U_alpha_vec)

        #Step 2: sampling C
        #C[t+1]: for all unknown (ie unfixed) pairs, set link value to 0 
        C[:,0]= np.where(C[:,1] == 0, 0, C[:,0]) 
        temp_l_instances[:,:,:] = base_l_instances[:,:,:]

        row_order_list = ([a for a in range(N_a)])
        np.random.shuffle(row_order_list)
        #print(row_order_list)
        for a in row_order_list: 
            # indices of C where C[i, 0] == 0 (nonlink) and C[i, 1] == 0 (unknown)
            #unlinked_unknown_pairs = np.nonzero((C[:, 0] == 0) & (C[:, 1] == 0))[0]

            unlinked_unknown_pairs = np.nonzero((C[a*N_b:(a+1)*N_b,0] == 0) & (C[a*N_b:(a+1)*N_b,0] == 0))[0]
            # indices of C where C[i, 0] == 0 (nonlink) and C[i, 1] == 1 (unknown)
            #unlinked_known_pairs = np.nonzero((C[:, 0] == 0) & (C[:, 1] == 1))[0]

            unlinked_known_pairs = np.nonzero((C[a*N_b:(a+1)*N_b,0] == 0) & (C[a*N_b:(a+1)*N_b,0] == 1))[0]

            #print(unlinked_unknown_pairs, "---", unlinked_known_pairs)
            
            # (N_b*a + b) mod N_b returns b index of pair
            b_unlinked_unknown = list(set(unlinked_unknown_pairs % N_b))
            b_unlinked_known = list(set(unlinked_known_pairs % N_b))

            #print(b_unlinked_unknown, "---", b_unlinked_known)

            num_links = N_b - len(b_unlinked_unknown) - len(b_unlinked_known)
            
            #if there are no more unlinked bs, we just go on to next iteration of the sampler 
            if(b_unlinked_unknown == []): 
                break
            
            prob_no_link = (N_a - num_links)*(N_b - num_links)/(num_links + 1)
            num = [likelihood_ratio(a, b) for b in b_unlinked_unknown]
            num.append(prob_no_link)
            
            #TODO: CHECK: power prior implementation 
            denom = [sum(num)] * len(num)
            link_probs = [i / j for i, j in zip(num, denom)]
            link_probs_sum = sum(link_probs)
            link_probs_normalized = link_probs/link_probs_sum
            #print(link_probs,link_probs_sum)
            # samples b_unlinked index from the , creates a new link at that b with probability associated with that  b 
            new_link_index = (np.random.choice([i for i in range(len(link_probs))], 1, True, link_probs_normalized))[0]   
            
            # last index in index list == no_link. if it selected a valid index, we want 
            if(new_link_index != len(b_unlinked_unknown)):   
                C[N_b*a + b_unlinked_unknown[new_link_index], 0] = 1
                for k in range(K):
                    if pd.notna(comparison_arrays[k,(N_b*a + b_unlinked_unknown[new_link_index])]):
                        temp_l_instances[k,int((L_k_n - 1)*comparison_arrays[k,(N_b*a + b_unlinked_unknown[new_link_index])]),3] -= 1
                        temp_l_instances[k,int((L_k_n - 1)*comparison_arrays[k,(N_b*a + b_unlinked_unknown[new_link_index])]),2] += 1

        Accuracy = C_matrix_to_df(C, N_a, N_b)
        accuarcy_all[t] = Accuracy
        if (t >= Burn):
            Average_Accuracy += Accuracy
        # return(C, theta_values)
    return alpha_zero, theta_values, Average_Accuracy / (T - Burn), accuarcy_all

    #return(C, theta_values)


alpha_sampling, theta_sampling, results_acc,accuarcy_all = theta_and_c_sampler(1000,750)

rj_rate = alpha_sampling[750:1000]-alpha_sampling[749:999]

print(sum(rj_rate == 0))

import matplotlib.pyplot as plt

"""
plt.figure(figsize=(6.5, 4))
for chain in range(K):
    plt.plot(theta_sampling[750:1000,chain,0,4], label=f'$\\theta_M$ k = {chain + 1}')
plt.xlabel('Iteration')
plt.ylabel('Sample Value')
plt.title('Markov Chain Monte Carlo Samples $\\theta_{M k}, l_k = 5$ ')
plt.legend()
plt.grid(True)
#plt.savefig("M_5.png")
plt.show()
"""

"""
plt.figure(figsize=(6.5, 4))
plt.plot(alpha_sampling)
plt.xlabel('Iteration')
plt.ylabel('Sample Value')
plt.title('Markov Chain Monte Carlo Samples $\\alpha_{0}$')
plt.grid(True)
#plt.savefig("alpha_30.png")
plt.show()
"""


