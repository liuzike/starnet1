
# coding: utf-8

# # Preprocess training and test set for StarNet
# This notebook takes you through the steps of how to pre-process the training data necessary for training StarNet and separate out a high S/N test set.
# 
# Requirements:
# - python packages: `numpy h5py vos`
# * required data files: apStar_visits_main.h5

# In[5]:


import numpy as np
import h5py
import os
import vos

datadir=""


# ** If you have not downloaded apStar_visits_main.h5 uncomment the below code to copy the file **
# 
# Note: This file requires 38.6GB. It is necessary to download this file to run this particular notebook, although this notebook can be skipped by downloading the files created here seperately. See $1\_Download\_Data.ipynb$ for instructions on how to do so.

# In[7]:



def starnet_download_file(filename):
    vclient = vos.Client()
    vclient.copy('vos:starnet/public/'+filename, datadir+filename)
    print(filename+' downloaded')
    
starnet_download_file('apStar_visits_main.h5')


# **Load the file that contains individual visit spectra along with APOGEE data associated with each star**

# In[3]:


filename = datadir + 'apStar_visits_main.h5'

F = h5py.File(filename,'r')
print('Dataset keys in file: \n')
print(list(F.keys()))


# ** Load the APOGEE data set into memory**
# 
# For the training of StarNet, it is only necessary to obtain the spectra and labels, but we need to set restrictions on the training set to obtain the labels of highest validity so we will first include APOGEE_IDs, the spectra, the S/N of the combined spectra, $T_{\mathrm{eff}}$, $\log(g)$,  [Fe/H],  $V_{scatter}$,  STARFLAGs, and ASPCAPFLAGs

# In[4]:


ap_id = F['IDs'][:,0]
combined_snr = F['stacked_snr']
starflag = F['star_flag']
aspcapflag = F['aspcap_flag']
teff = F['TEFF'][:]
logg = F['LOGG'][:]
fe_h = F['FE_H'][:]
vscatter = F['VSCATTER']

print('Obtained data for '+str(len(ap_id))+' individual visits from '+str(len(list(set(list(ap_id)))))+' stars.')


# **Separate out a dataset with good labels**
# - combined spectral S/N $\geq$ 200
# - STARFLAG = 0
# - ASPCAPFLAG = 0
# - 4000K < $T_{\mathrm{eff}}$ < 5500K
# - -3.0 dex < [Fe/H]
# - $\log(g)$ $\neq$ -9999. (value defined by ASPCAP when no ASPCAP labels are given)
# - $V_{scatter}$ < 1.0 km/s

# In[5]:


snr_min = 200.
teff_min = 4000.
teff_max = 5500.
vscatter_max = 1.
fe_h_min = -3.


# In[6]:


indices, cols = np.where((aspcapflag[:]==0.)&(starflag[:]==0.)&(combined_snr[:]>=snr_min)&(vscatter[:]<vscatter_max)&(fe_h[:]>fe_h_min)&(teff[:]>teff_min)&(teff[:]<teff_max)&(logg[:]!=-9999.).reshape(len(ap_id),1))

ap_id_high_snr = ap_id[indices]
print(str(len(ap_id_high_snr))+' individual visits from '+str(len(set(ap_id_high_snr)))+' stars remain.')


# **Select the first **$num\_ref$** visits for the reference set**
# 
# We shuffle around the data to avoid local effects.
# Later on, it will be be split into training and cross-validation sets.
# The remaining high S/N spectra will be used in the test set

# In[4]:


num_ref = 44784 # number of reference spectra

indices_ref = indices[0:num_ref]
np.random.shuffle(indices_ref)

ap_id_ref = ap_id[indices_ref]
teff = teff[indices_ref]
logg = logg[indices_ref]
fe_h = fe_h[indices_ref]

print('Reference set includes '+str(len(ap_id_ref))+' individual visits from '+str(len(set(ap_id_ref)))+' stars.')


# **Now collect individual visit spectra, normalize each spectrum, and save data**
# 
# **Normalize spectra**
# 1. separate into three chips
# 2. divide by median value in each chip
# 3. recombine each spectrum into a vector of 7214 flux values

# In[8]:


# Define edges of detectors
blue_chip_begin = 322
blue_chip_end = 3242
green_chip_begin = 3648
green_chip_end = 6048   
red_chip_begin = 6412
red_chip_end = 8306 


# In[9]:


savename = 'training_data.h5'

with h5py.File(datadir + savename, "w") as f:
    
    # Create datasets for your reference data file 
    spectra_ds = f.create_dataset('spectrum', (1,7214), maxshape=(None,7214), dtype="f", chunks=(1,7214))
    teff_ds = f.create_dataset('TEFF', teff.shape, dtype="f")
    logg_ds = f.create_dataset('LOGG', logg.shape, dtype="f")
    fe_h_ds = f.create_dataset('FE_H', fe_h.shape, dtype="f")
    ap_id_ds = f.create_dataset('Ap_ID', ap_id_ref.shape, dtype="S18")
    
    teff_ds[:] = teff
    logg_ds[:] = logg
    fe_h_ds[:] = fe_h
    ap_id_ds[:] = ap_id_ref.tolist()
        
    first_entry=True
    
    for i in indices_ref:

        spectrum = F['spectrum'][i:i+1]
        spectrum[np.isnan(spectrum)]=0.

        # NORMALIZE SPECTRUM
        # Separate spectra into chips
        blue_sp = spectrum[0:1,blue_chip_begin:blue_chip_end]
        green_sp = spectrum[0:1,green_chip_begin:green_chip_end]
        red_sp = spectrum[0:1,red_chip_begin:red_chip_end]

        # Normalize spectra by chips

        blue_sp = (blue_sp.T/np.median(blue_sp, axis=1)).T
        green_sp = (green_sp.T/np.median(green_sp, axis=1)).T
        red_sp = (red_sp.T/np.median(red_sp, axis=1)).T 

        # Recombine spectra

        spectrum = np.column_stack((blue_sp,green_sp,red_sp))
        if first_entry:
            spectra_ds[0] = spectrum
            first_entry=False
        else:
            spectra_ds.resize(spectra_ds.shape[0]+1, axis=0)

            spectra_ds[-1] = spectrum

print(savename+' has been saved as the reference set to be used in 4_Train_Model.ipynb')  

