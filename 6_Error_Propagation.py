
# coding: utf-8

# # Propogate Errors
# 
# This notebook takes you through the steps of how to propogate errors for through the neural network model
# 
# * required packages: `numpy h5py keras`
# * data files: 
#     - starnet_cnn.h5
#     - mean_and_std.npy
#     - test_data.h5
#     - apStar_combined_main.h5

# In[1]:


import numpy as np
from keras.models import load_model
import h5py
import tensorflow as tf
import time
import keras.backend as K
import subprocess


datadir= ""


# Define path variables for your keras model, denormalization data, and test data

# In[2]:


model_path = datadir + 'starnet_cnn.h5'
denormalization_path = datadir + 'mean_and_std.npy'
test_data_path = datadir + 'test_data.h5'


# **Define functions to:**
# 
# 1. compute the jacobian matrix
# 2. compute the covariance
# 3. compute the variance
# 
# Note: these functions can be combined into one, but they are separated here to allow users to extract intermediate results for analysis

# In[9]:


def calc_jacobian(model,spectra,denormalize=None):
        
    if denormalize==None:
        y_list = tf.unstack(model.output)
    else:
        y_list = tf.unstack(denormalize(model.output[0]))

    J = [tf.gradients(y, model.input) for y in y_list]


    jacobian_func = [K.function([model.input, K.learning_phase()], j_) for j_ in J]

    jacobian = np.array([jf([spectra,False]) for jf in jacobian_func])[:,0,0,:]
    '''
    for i in range(len(spectra)):
        jacobian = np.array([jf([spectra,False]) for jf in jacobian_func])[:,:,0,:,0]
        np.save('temp/temp_jacobian_'+str(i)+'.npy',jacobian)
        if i%int(0.1*len(spectra))==0:
            print('Jacobians completed: '+str(i))
    
    for i in range(len(spectra)):
        if i==0:
            jacobian = np.load('temp/temp_jacobian_'+str(i)+'.npy')
        else:
            jacobian = np.concatenate((jacobian,np.load('temp/temp_jacobian_'+str(i)+'.npy')))
        subprocess.check_output(['rm','temp/temp_jacobian_'+str(i)+'.npy'])
    '''
    return jacobian

def calc_covariance(model,spectra,err_spectra,denormalize=None):
    jac_matrix = calc_jacobian(model,spectra,denormalize)
    err_spectra[err_spectra > 6] = 0
    jac_matrix = np.nan_to_num(jac_matrix)
    covariance = np.einsum('ij,jl->il',(jac_matrix*(err_spectra**2)),jac_matrix.T)
    return covariance

def calc_variance(model,spectra,err_spectra,denormalize=None):
    covariance = calc_covariance(model,spectra,err_spectra,denormalize)
    return np.diagonal(covariance, offset=0)


# ** Create a denormalization function **

# In[10]:


mean_and_std = np.load(denormalization_path)
mean_labels = mean_and_std[0]
std_labels = mean_and_std[1]
num_labels = mean_and_std.shape[1]

def denormalize(lb_norm):
    return ((lb_norm*std_labels)+mean_labels)


# **Load the StarNet model**

# In[11]:


model = load_model(model_path)


# ** Load Test Data **
# 
# The error propagation technique takes some time, so for the purpose of example, we will only use the first 100 spectra in the test set

# In[12]:


num_test = 300

f = h5py.File(test_data_path, 'r')
test_spectra = f['spectrum']
test_err_spectra = f['error_spectrum']
test_ap_ids = f['Ap_ID'][0:num_test]
test_labels = np.column_stack((f['TEFF'][0:num_test],f['LOGG'][0:num_test],f['FE_H'][0:num_test]))

print('Test set contains '  + str(len(test_ap_ids))+' stars')


# ** Compute predictions and errors for the test set **
# 
# **Steps:**
# 1. compute predictions
# 
#     \begin{equation}
#     h_(\textbf{x},\textbf{W}) =  h_{1}(\textbf{x},\textbf{W}),...,h_{j}(\textbf{x},\textbf{W}))
#     \end{equation} 
# 
#         j = 3
# 
# 2. compute jacobian matrix
# 
#     \begin{equation}
#     Jac = \frac{\partial h_{j}(\textbf{x},\textbf{W})}{\partial \textbf{x}} =  (\frac{\partial h_{j}(\textbf{x},\textbf{W})}{\partial x_{1}},...,\frac{\partial h_{j}(\textbf{x},\textbf{W})}{\partial x_{n}})
#     \end{equation} 
# 
#         j = 1,...,3
# 
#         n = 7214
# 
# 3. compute covariance matrix
# 
#     \begin{equation}
#     Cov = Jac \times \Delta \textbf{x}^2 \times Jac^T
#     \end{equation}
#     
# 
# 4. obtain propagated variance due to error spectrum from the diagonal of the covariance matrix
# 
#     \begin{equation}
#     \sigma_{\mathrm{prop}}^2 \approx diag(Cov)
#     \end{equation}
#     
# 
# 5. determine which region of the label-space the labels are within to obtain the intrinsic scatter in the corresponding bin. These values have been predetermined from training StarNet on synthetic data and applying it to a test set of synthetic data
# 
#     \begin{equation}
#     \sigma_{\mathrm{int}}
#     \end{equation}
#     
# 6. combine propagated error with the intrinsic scatter term
# 
#     \begin{equation}
#     \Delta h_{j} = \sqrt{\sigma_{\mathrm{prop}}^2  + \sigma_{\mathrm{int}}^2}
#     \end{equation}

# In[13]:


variance = np.zeros((len(test_labels),3))
predictions = np.zeros(test_labels.shape)
print('Making predictions and computing propagated variance for '+str(len(test_labels))+' spectra')
time_start = time.time()
for i in range(len(test_labels)):
    spectrum = test_spectra[i:i+1]
    err_spectrum = test_err_spectra[i:i+1]
    variance[i] = calc_variance(model,spectrum,err_spectrum,denormalize)
    predictions[i] = denormalize(model.predict(spectrum))
    if i%int(0.1*len(test_labels))==0:
        print('\n'+str(i+1)+' completed.\n'+str(time.time()-time_start)+' seconds elapsed.')
print('\nAll '+str(i+1)+' completed.\n'+str(time.time()-time_start)+' seconds elapsed.')
f.close()


# ** Create intrinsic scatter arrays (predetermined) **

# In[14]:


scatter_terms = np.array([[  2.85209088e+01,   2.30193645e+01,   2.10676180e+01,
          1.91357425e+01,   1.72090644e+01,   1.58693655e+01,
          1.52684102e+01,   1.42387830e+01,   1.64239293e+01,
          2.18981017e+01],
       [  3.86073715e-02,   3.04916170e-02,   2.44161726e-02,
          2.25093310e-02,   2.35929675e-02,   2.36922221e-02,
          2.58764773e-02,   2.80946934e-02,   3.34534390e-02,
          3.56641714e-02],
       [  3.90793092e-02,   2.43149947e-02,   2.25292707e-02,
          1.81974298e-02,   1.58638867e-02,   1.46142515e-02,
          1.36038125e-02,   1.25392930e-02,   1.24740228e-02,
          1.53680421e-02]])
scatter_ranges = np.array([[  3.50000000e+03,   3.95000000e+03,   4.40000000e+03,
          4.85000000e+03,   5.30000000e+03,   5.75000000e+03,
          6.20000000e+03,   6.65000000e+03,   7.10000000e+03,
          7.55000000e+03,   8.00000000e+03],
       [  0.00000000e+00,   5.00000000e-01,   1.00000000e+00,
          1.50000000e+00,   2.00000000e+00,   2.50000000e+00,
          3.00000000e+00,   3.50000000e+00,   4.00000000e+00,
          4.50000000e+00,   5.00000000e+00],
       [ -2.50000000e+00,  -2.20000000e+00,  -1.90000000e+00,
         -1.60000000e+00,  -1.30000000e+00,  -1.00000000e+00,
         -7.00000000e-01,  -4.00000000e-01,  -1.00000000e-01,
          2.00000000e-01,   5.00000000e-01]])


# ** assign each spectrum an intrinsic scatter term depending on which region of the parameter-space the prediction lies **

# In[15]:


scatter_errs = np.empty(test_labels.shape)

for i in range(scatter_terms.shape[0]):
    for j in range(scatter_terms.shape[1]):
        current_min = scatter_ranges[i,j]
        current_max = scatter_ranges[i,j+1]
        current_scatter = scatter_terms[i,j]
        index = np.where((test_labels[:,i]>current_min)&(test_labels[:,i]<current_max))[0]
        scatter_errs[index,i]=current_scatter


# ** combine the propagated error (or the square root of the variance) and intrinsic error in quadrature **

# In[16]:


total_errors = np.sqrt(variance+np.square(scatter_errs))


# In[17]:


# label names
label_names = ['Teff  ','log(g)','[Fe/H]']
units = ['K','dex','dex']

mean_err_total = np.mean(total_errors, axis=0)
print('Mean total statistical errors: \n')
for i, err in enumerate(mean_err_total):
      print(label_names[i]+':  '+"{0:.3f}".format(err)+' '+units[i])

