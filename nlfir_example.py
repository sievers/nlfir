import numpy as np
import nlfir
import time
from matplotlib import pyplot as plt
plt.ion()

t1=time.time()
if True:
    adc,pileup,sig,pulse=nlfir.read_dir('claire/OF5_EMB_EMMiddle_eta0.5125_phi0.0125_mu140_BT1_rdGapLowE_bunchLength5_size22')
    to_cut=int(1e6)
    thresh=1.5 #seems to work well for the selected directory.  
else:
    adc=np.loadtxt('adc_out.txt')
    sig=np.loadtxt('signal.txt')
    pileup=np.loadtxt('pileup.txt')
    npad=len(adc)-len(pileup)    
    sig=np.hstack([sig,np.zeros(npad)])
    pileup=np.hstack([pileup,np.zeros(npad)])
    to_cut=int(1e5)
    thresh=3.0
t2=time.time()
print('read data in ',t2-t1)

s=sig+pileup
k1=to_cut  #k1/k2 define the stretch of data we'll use
k2=-k1

jl=30  #define the neighbors we're allowed to use while doing fits
jr=8

ones=0.0*adc+1.0
mat=nlfir.FIRmat(ones,0,0,k1,k2) #initialize the matrix with the constant offset term
mat.add_vec(adc,jl,jr)  #include the linear part of the response to the ADCs
mat.get_fts() #do this once you've added everything you want
mat.get_coeffs(s)
pred=mat.get_pred()
mat.set_adjust(s,thresh)
pred_adjust=mat.get_pred(adjust=True)

nlmat=nlfir.FIRmat(ones,0,0,k1,k2) #initialize the matrix with the constant offset term
nlmat.add_vec(adc,jl,jr)  #include the linear part of the response to the ADCs
nlmat.add_vec(adc**2,jl,jr)  #we can add the nonlinear part now
nlmat.add_vec(adc**3,jl,jr) 
nlmat.get_fts() #do this once you've added everything you want
nlmat.get_coeffs(s)
nlpred=nlmat.get_pred()
nlmat.set_adjust(s,thresh)
nlpred_adjust=nlmat.get_pred(adjust=True)

kk1=k1+int(1e5)
kk2=kk1+1000
plt.figure(1)
plt.clf()
plt.plot(s[kk1:kk2])
plt.plot(pred_adjust[kk1:kk2])
plt.plot(nlpred_adjust[kk1:kk2])
plt.title('Reconstructed Adjusted Signals')
plt.legend(['Signal','Linear Adjust','Non-linear Adjust'])
plt.show()

plt.figure(2)
kk2=kk1+int(6e4)
plt.clf()
plt.plot(s[kk1:kk2],(pred-s)[kk1:kk2],'.')
plt.plot(s[kk1:kk2],(pred_adjust-s)[kk1:kk2],'.')
plt.plot(s[kk1:kk2],(nlpred-s)[kk1:kk2],'.')
plt.plot(s[kk1:kk2],(nlpred_adjust-s)[kk1:kk2],'.')
plt.legend(['Linear err','Linear adjust err','Nonlinear err','Nonlinar adjust err'])
plt.show()

