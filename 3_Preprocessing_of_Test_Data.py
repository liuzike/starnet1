
# coding: utf-8

# # Pre-process test data
# 
# This notebook takes you through the steps of how to preprocess a high S/N and low S/N test set
# * required packages: numpy, h5py, vos
# * required data files: apStar_combined_main.h5 and training_data.h5

# In[1]:


import numpy as np
import h5py
import os
import vos

datadir=""


# ** If you have not downloaded apStar_combined_main.h5 uncomment the below code to copy the file **
# 
# Note: This file requires 10.3GB. It is necessary to download this file to run  particular notebook, although this notebook can be skipped by downloading the files created here seperately. See $1\_Download\_Data.ipynb$ for instructions on how to do so.

# In[2]:


'''
def starnet_download_file(filename):
    vclient = vos.Client()
    vclient.copy('vos:starnet/public/'+filename, datadir+filename)
    print(filename+' downloaded')

starnet_download_file('apStar_combined_main.h5')
'''


# In[3]:


filename = datadir + 'apStar_combined_main.h5'
F = h5py.File(filename,'r')
print('Dataset keys in file: \n')
print(list(F.keys()))


# **Load the data into memory**
# 
# For the testing of StarNet, it is necessary to obtain the spectra, error spectra, combined S/N, and labels, but we need to make eliminations to the test set to obtain the labels of highest validity to compare with, so we will first include the APOGEE_IDs, the S/N of the combined spectra, $T_{\mathrm{eff}}$, $\log(g)$, [Fe/H], $V_{scatter}$, STARFLAGs, and ASPCAPFLAGs to make certain eliminations. Once the stars for the test sets have been collected we will then gather the spectra and error spectra and save the two test sets to an h5 file.

# In[4]:


ap_id = F['IDs'][:,0]
combined_snr = F['stacked_snr'][:]
starflag = F['star_flag']
aspcapflag = F['aspcap_flag']
teff = F['TEFF'][:]
logg = F['LOGG'][:]
fe_h = F['FE_H'][:]
vscatter = F['VSCATTER']

print('Obtainined data for '+str(len(list(set(list(ap_id)))))+' stars.')


# **Collect label normalization data**
# 
# Create a file that contains the mean and standard deviation for $T_{\mathrm{eff}}$, $\log(g)$, and  $[Fe/H]$ in order to normalize labels during training and testing. Ignore values equal to -9999.

# In[5]:


mean = np.array([np.mean(teff[teff!=-9999.]),np.mean(logg[logg!=-9999.]),np.mean(fe_h[fe_h!=-9999.])])
std = np.array([np.std(teff[teff!=-9999.]),np.std(logg[logg!=-9999.]),np.std(fe_h[fe_h!=-9999.])])
mean_and_std = np.row_stack((mean,std))
np.save(datadir+'mean_and_std', mean_and_std)

print('mean_and_std.npy saved')


# **Separate out a dataset with good labels**. 
# - STARFLAGs = 0
# - ASPCAPFLAGs = 0
# - 4000K < $T_{\mathrm{eff}}$ < 5500K
# - -3.0 < [Fe/H]
# - $\log(g)$ $\neq$ -9999. (value defined by ASPCAP when no ASPCAP labels are given)
# - $V_{scatter}$ < 1.0 km/s

# In[6]:


teff_min = 4000.
teff_max = 5500.
vscatter_max = 1.
fe_h_min = -3.


# In[7]:


indices, cols = np.where((aspcapflag[:]==0.)&(starflag[:]==0.)&(vscatter[:]<vscatter_max)&(fe_h[:]>fe_h_min)&(teff[:]>teff_min)&(teff[:]<teff_max)&(logg[:]!=-9999.).reshape(len(ap_id),1))

ap_id = ap_id[indices]
teff = teff[indices]
logg = logg[indices]
fe_h = fe_h[indices]
combined_snr = combined_snr[indices]

print(str(len(list(set(list(ap_id)))))+' stars remain.')


# **Load test set APOGEE IDs**
# 
# Load previously created file that contains the training data. We do not want to include any of the APOGEE IDs used in the training set in our test set. This file was created in 2_Preprocessing_of_Training_Data.ipynb

# In[8]:


savename = 'training_data.h5'

with h5py.File(datadir + savename, "r") as f:
    train_ap_id = f['Ap_ID'][:]


# **Separate data for High S/N test set**

# In[9]:


indices_test = [i for i, item in enumerate(ap_id) if item not in train_ap_id]

test_ap_id = ap_id[indices_test]
test_teff = teff[indices_test]
test_logg = logg[indices_test]
test_fe_h = fe_h[indices_test]
test_combined_snr = combined_snr[indices_test]

indices_test_set = indices[indices_test] # These indices will be used to index through the spectra

print('Test set includes '+str(len(test_ap_id))+' combined spectra')


# **Now collect spectra and error spectra. Then normalize each spectrum and save the data**
# 
# **Steps taken to normalize spectra:**
# 1. separate into three chips
# 2. divide by median value in each chip
# 3. recombine each spectrum into a vector of 7214 flux values
# 4. Error spectra must also be normalized with the same median values for use in the error propagation

# In[10]:


# Define edges of detectors
blue_chip_begin = 322
blue_chip_end = 3242
green_chip_begin = 3648
green_chip_end = 6048   
red_chip_begin = 6412
red_chip_end = 8306 


# In[11]:


savename = 'test_data.h5'

with h5py.File(datadir + savename, "w") as f:
    
    # Create datasets for your test data file 
    spectra_ds = f.create_dataset('spectrum', (1,7214), maxshape=(None,7214), dtype="f", chunks=(1,7214))
    error_spectra_ds = f.create_dataset('error_spectrum', (1,7214), maxshape=(None,7214), dtype="f", chunks=(1,7214))
    teff_ds = f.create_dataset('TEFF', test_teff.shape, dtype="f")
    logg_ds = f.create_dataset('LOGG', test_logg.shape, dtype="f")
    fe_h_ds = f.create_dataset('FE_H', test_fe_h.shape, dtype="f")
    combined_snr_ds = f.create_dataset('combined_snr', test_combined_snr.shape, dtype="f")
    ap_id_ds = f.create_dataset('Ap_ID', test_ap_id.shape, dtype="S18")
    
    # Save data to data file
    teff_ds[:] = test_teff
    logg_ds[:] = test_logg
    fe_h_ds[:] = test_fe_h
    combined_snr_ds[:] = test_combined_snr
    ap_id_ds[:] = test_ap_id.tolist()
        
    # Collect spectra
    first_entry=True
    
    for i in indices_test_set:

        spectrum = F['spectrum'][i:i+1]
        spectrum[np.isnan(spectrum)]=0.
        
        err_spectrum = F['error_spectrum'][i:i+1]

        # NORMALIZE SPECTRUM
        # Separate spectra into chips
        blue_sp = spectrum[0:1,blue_chip_begin:blue_chip_end]
        green_sp = spectrum[0:1,green_chip_begin:green_chip_end]
        red_sp = spectrum[0:1,red_chip_begin:red_chip_end]
        
        blue_sp_med = np.median(blue_sp, axis=1)
        green_sp_med = np.median(green_sp, axis=1)
        red_sp_med = np.median(red_sp, axis=1)

        # Normalize spectra by chips
        blue_sp = (blue_sp.T/blue_sp_med).T
        green_sp = (green_sp.T/green_sp_med).T
        red_sp = (red_sp.T/red_sp_med).T

        # Recombine spectra
        spectrum = np.column_stack((blue_sp,green_sp,red_sp))
        
        # Normalize error spectrum using the same method
        # Separate error spectra into chips

        blue_sp = err_spectrum[0:1,blue_chip_begin:blue_chip_end]
        green_sp = err_spectrum[0:1,green_chip_begin:green_chip_end]
        red_sp = err_spectrum[0:1,red_chip_begin:red_chip_end]

        # Normalize error spectra by chips
        blue_sp = (blue_sp.T/blue_sp_med).T
        green_sp = (green_sp.T/green_sp_med).T
        red_sp = (red_sp.T/red_sp_med).T

        # Recombine error spectra
        err_spectrum = np.column_stack((blue_sp,green_sp,red_sp))
        
        
        if first_entry:
            spectra_ds[0] = spectrum
            error_spectra_ds[0] = err_spectrum
            
            first_entry=False
        else:
            spectra_ds.resize(spectra_ds.shape[0]+1, axis=0)
            error_spectra_ds.resize(error_spectra_ds.shape[0]+1, axis=0)

            spectra_ds[-1] = spectrum
            error_spectra_ds[-1] = err_spectrum

print(savename+' has been saved as the test set to be used in 5_Test_Model.ipynb')  

