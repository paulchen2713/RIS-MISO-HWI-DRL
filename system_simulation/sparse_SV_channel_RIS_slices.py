# -*- coding: utf-8 -*-
"""
Created on Tue May 23 16:05:07 2023

@author: Benson
"""

import numpy as np
import math
import scipy.io as sio
# from array_response import array_response as AR
import array_response 

USPA = 'USPA' # USPA or ULA
ULA = 'ULA'

## System parameter
Ns = 2 # of streams
Nt = 64 # of transmit antennas
Nr = 32 # of receive antennas
N_phi = 64 # the RIS dimension

## Channel parameter
los = 1
K = 64 # of subcarrier (OFDM)
Nc = 5 # of clusters
Nray = 10 # of rays in each cluster 
L = Nc*Nray + 1
angles_sigma = (10/180)*math.pi # standard deviation of the angles in azimuth and elevation both of Rx and Tx 


D = 64 # length of cyclic prefix
Tau = D # maxinum time delay
channel_num = 2
realization = 5000

## Power distribution ratio (the ratio of the LOS and NLOS)
Mu = 10 # unit_dB
# Mu = 0
Mu_trans = 10**(-Mu/10)

## location setting
BS_loc = np.array([2, 0, 10])
RIS_loc = np.array([0, 148, 10])
UE_loc = np.array([5, 150, 1.8])

d_H1 = np.linalg.norm(RIS_loc-BS_loc) # distance between BS and RIS
d_H2 = np.linalg.norm(RIS_loc-UE_loc) # distance between RIS and UE

angles_sigma = (10/180)*math.pi # standard deviation of the angles in azimuth and elevation both of Rx and Tx
b = angles_sigma / np.sqrt(2) # Laplace parameter : scale(lambda or b)

gamma1 = math.sqrt((Nt*N_phi)/L) # normalization factor (BS-RIS)
gamma2 = math.sqrt((N_phi*Nr)/L) # normalization factor (RIS-UE)


# # Path Loss
# PL_0 = 61.4 # (dB)
# PL_g = 2 # path loss exponent
# PL_H1 = PL_0 + 10*PL_g*np.log10(d_H1)
# PL_H2 = PL_0 + 10*PL_g*np.log10(d_H2)

# Without Path Loss
PL_H1 = 0
PL_H2 = 0

# Var_H1 = 10**(-PL_H1/10)
# Var_H2 = 10**(-PL_H1/10)

Var_H1 = 1 # channel_gain_H1
Var_H2 = 1 # channel_gain_H2

Tx_dBi = 0
Tx = 10**(Tx_dBi/10)
Rx_dBi = 0
Rx = 10**(Rx_dBi/10)


## initial setting
H1 = np.zeros((N_phi,Nt,K),dtype = complex)
H2 = np.zeros((Nr,N_phi,K),dtype = complex)

at_BS = np.zeros((Nt,Nc*Nray+1),dtype = complex)
ar_RIS = np.zeros((N_phi,Nc*Nray+1),dtype = complex)
at_RIS = np.zeros((N_phi,Nc*Nray+1),dtype = complex)
ar_UE = np.zeros((Nr,Nc*Nray+1),dtype = complex)

alpha = np.zeros((Nc*Nray+1),dtype = complex)
beta = np.zeros((Nc*Nray+1),dtype = complex)


for i in range(realization):

    AoD = np.zeros((Nc,2,Nray)) 
    AoA = np.zeros((Nc,2,Nray))
    AoD_new = np.zeros((Nc*Nray+1,2))
    AoA_new = np.zeros((Nc*Nray+1,2))
    # temp00 = np.zeros((Nt,Nc*Nray+1),dtype = complex)
    # temp01 = np.zeros((N_phi,Nc*Nray+1),dtype = complex)
    # temp_alpha =  np.zeros((Nc*Nray+1),dtype = complex)
    
    AoA_new[0,0] = np.arcsin((RIS_loc[0]-BS_loc[0])/np.sqrt((RIS_loc[0]-BS_loc[0])**2+(RIS_loc[1]-BS_loc[1])**2))  
    AoA_new[0,1] = np.arccos((RIS_loc[2]-BS_loc[2])/d_H1)
    AoD_new[0,0] = math.pi/2-AoA_new[0,0];

    for c in range(Nc):
        AoD_m = math.pi*np.random.rand(2,1)-math.pi/2 # cluster mean (-pi/2~pi/2)
        AoA_m = math.pi*np.random.rand(2,1)-math.pi/2 # cluster mean (-pi/2~pi/2)
    
        AoD[c,0,:] = np.random.laplace(AoD_m[0],b,(1,Nray))
        AoD[c,1,:] = np.random.laplace(AoD_m[1],b,(1,Nray))
        AoA[c,0,:] = np.random.laplace(AoA_m[0],b,(1,Nray))
        AoA[c,1,:] = np.random.laplace(AoA_m[1],b,(1,Nray))
        AoD_new[c*Nray+1:(c+1)*Nray+1,0] = np.reshape(AoD[c,0,:],Nray)
        AoD_new[c*Nray+1:(c+1)*Nray+1,1] = np.reshape(AoD[c,1,:],Nray)
        AoA_new[c*Nray+1:(c+1)*Nray+1,0] = np.reshape(AoA[c,0,:],Nray)
        AoA_new[c*Nray+1:(c+1)*Nray+1,1] = np.reshape(AoA[c,1,:],Nray)
    
    A_T1 = np.zeros((Nt,Nc*Nray+1), dtype=complex)   
    A_R1 = np.zeros((N_phi,Nc*Nray+1), dtype=complex) 

            
    for l in range(Nc*Nray+1):
        at = array_response(AoD_new[l, 0], AoD_new[l,1], Nt, ULA)
        ar = array_response(AoA_new[l, 0], AoA_new[l,1], N_phi, USPA)
        A_T1[:,l] = np.reshape(at,Nt)
        A_R1[:,l] = np.reshape(ar,N_phi)
    

    
    alpha_i = np.zeros((Nc*Nray+1))
    gain_H1_domi = np.sqrt(Var_H1/2)*(np.random.randn(1,1)+1j*np.random.randn(1,1))
    gain_H1_ndomi = np.sqrt(Mu_trans*Var_H1/2)*(np.random.randn(Nc*Nray,1)+1j*np.random.randn(Nc*Nray,1))
    alpha_i = Tx*gamma1*np.append(gain_H1_domi,gain_H1_ndomi)
    
    # test_H1 = A_R@np.diag(alpha)@A_T.T
    
    alpha_ii = np.reshape(alpha_i,[Nc*Nray+1,1])
    
    tau = Tau*np.random.rand(Nc*Nray+1,1)
    
    for k in range(K):
        P_1 = alpha_ii * np.exp(-1j*2*np.pi*tau*k/K)
        alpha_i_k = np.reshape(P_1,Nc*Nray+1)
        H1[:,:,k] = A_R1@np.diag(alpha_i_k)@A_T1.T
        

# ---------------------------------------------------------------- #



    AoD = np.zeros((Nc,2,Nray)) 
    AoA = np.zeros((Nc,2,Nray))
    AoD_new = np.zeros((Nc*Nray+1,2))
    AoA_new = np.zeros((Nc*Nray+1,2))
    # temp10 = np.zeros((N_phi,Nc*Nray+1),dtype = complex)
    # temp11 = np.zeros((Nr,Nc*Nray+1),dtype = complex)
    temp_beta =  np.zeros((Nc*Nray+1),dtype = complex)
    
    AoA_new[0,0] = np.arcsin((UE_loc[0]-RIS_loc[0])/np.sqrt((UE_loc[0]-RIS_loc[0])**2+(UE_loc[1]-RIS_loc[1])**2))  
    AoA_new[0,1] = np.arccos((UE_loc[2]-RIS_loc[2])/d_H2)
    AoD_new[0,0] = math.pi/2-AoA_new[0,0];

    for c in range(Nc):
        AoD_m = math.pi*np.random.rand(2,1)-math.pi/2 # cluster mean (-pi/2~pi/2)
        AoA_m = math.pi*np.random.rand(2,1)-math.pi/2 # cluster mean (-pi/2~pi/2)
    
        AoD[c,0,:] = np.random.laplace(AoD_m[0],b,(1,Nray))
        AoD[c,1,:] = np.random.laplace(AoD_m[1],b,(1,Nray))
        AoA[c,0,:] = np.random.laplace(AoA_m[0],b,(1,Nray))
        AoA[c,1,:] = np.random.laplace(AoA_m[1],b,(1,Nray))
        AoD_new[c*Nray+1:(c+1)*Nray+1,0] = np.reshape(AoD[c,0,:],Nray)
        AoD_new[c*Nray+1:(c+1)*Nray+1,1] = np.reshape(AoD[c,1,:],Nray)
        AoA_new[c*Nray+1:(c+1)*Nray+1,0] = np.reshape(AoA[c,0,:],Nray)
        AoA_new[c*Nray+1:(c+1)*Nray+1,1] = np.reshape(AoA[c,1,:],Nray)

    A_T2 = np.zeros((N_phi,Nc*Nray+1),dtype = complex)   
    A_R2 = np.zeros((Nr,Nc*Nray+1),dtype = complex) 

    for l in range(Nc*Nray+1):
        at = array_response(AoD_new[l,0],AoD_new[l,1],N_phi,USPA)
        ar = array_response(AoA_new[l,0],AoA_new[l,1],Nr,ULA)
        A_T2[:,l] = np.reshape(at,N_phi)
        A_R2[:,l] = np.reshape(ar,Nr)



    beta_i = np.zeros((Nc*Nray+1))
    gain_H2_domi = np.sqrt(Var_H2/2)*(np.random.randn(1,1)+1j*np.random.randn(1,1))
    gain_H2_ndomi = np.sqrt(Mu_trans*Var_H2/2)*(np.random.randn(Nc*Nray,1)+1j*np.random.randn(Nc*Nray,1))
    beta_i = Rx*gamma2*np.append(gain_H2_domi,gain_H2_ndomi)
    # test_H1 = A_R@np.diag(alpha)@A_T.T
    
    beta_ii = np.reshape(beta_i,[Nc*Nray+1,1])
    
    
    tau = Tau*np.random.rand(Nc*Nray+1,1)
    
    for k in range(K):
        P_2 = beta_ii * np.exp(-1j*2*np.pi*tau*k/K)
        beta_i_k = np.reshape(P_2,Nc*Nray+1)
        H2[:,:,k] = A_R2@np.diag(beta_i_k)@A_T2.T


    print('\r channel_H1_H2: No. %d realization '%(i+1) ,end="",flush = True)        
    

    data_H1_H2 = {'H1' : H1, 'H2' : H2}
    sio.savemat('D:\\code\\sparse_SV_channel_RIS\\training_data\\training_data_test\\training_data_s\\DU_H1_H2_Nt_%d_N_phi_%d_Nr_%d_Ns_%d_Training_data_H_%d.mat'%(Nt,N_phi,Nr,Ns,i+1001),data_H1_H2)




# # test
# Norm_H1 = np.zeros((K,realization),dtype = complex) 
# Norm_H2 = np.zeros((K,realization),dtype = complex) 

# for i in range(realization): 
#     for k in range(K):
#         Norm_H1[k,i] = np.linalg.norm(H1[:,:,k,i],'fro')
#         Norm_H2[k,i] = np.linalg.norm(H2[:,:,k,i],'fro')



















