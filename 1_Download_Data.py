
# coding: utf-8

# # Download APOGEE data files and StarNet model
# 
# In order to go through the notebooks you will need a few files. Depending on the intended usage of the following notebooks, not all of these files are necessary.
# Below is a description of each file and which notebooks it is necessary for.
# 
# | file | size | content |
# | :---- | :----: | :----|
# | apStar_visits_main.h5 | 35.9GB | apStar individual visit spectra for training StarNet, created from the APOGEE DR13. Used in 2_Preprocessing_of_Training_Data.ipynb |
# | apStar_combined_main.h5 | 9.6GB | apStar combined spectra for preprocessing StarNet, created by pulling apStar combined spectra from the APOGEE DR13. Used in 3_Preprocessing_of_Test_Data.ipynb |
# | training_data.h5 | 1.2GB | apStar individual visit spectra for training, useful to skip preprocessing. Used in 3_Preprocessing_of_Test_Data.ipynb and 4_Train_Model.ipynb |
# | mean_and_std.npy | 104B |  mean and standard deviations for the stellar labels used during preprocessing. Used in 4_Train_Model.ipynb and 5_Test_Model.ipynb |
# | test_data.h5 | 1.13GB | apStar combined spectra test set. Used in 5_Test_Model.ipynb |
# | starnet_cnn.h5 | 85MB | pretrained StarNet model with keras (tensorflow as the backend) on APOGEE DR13. Used in 5_Test_Model.ipynb | 
# 
# 
# You can download the data from the StarNet public VOSpace at CADC. You can either:
# * browse and download from [this URL](http://apps.canfar.net/storage/list/starnet/public)
# * or download the files from the python VOSpace command line client, installed with pip:
# ```
# pip install vos
# ```
# ```
# getcert
# ```
# You can choose a directory with enough space (~50GB) and download all files into a directory:
# ```
# vcp vos:starnet/public /path/to/my/starnet/directory
# ```
# Or you can copy each file within an IPython session with a function such as the one below:

# In[1]:


import vos
datadir=""

def starnet_download_file(filename):
    vclient = vos.Client()
    vclient.copy('vos:starnet/public/'+filename, datadir+filename)
    print(filename+' downloaded')


# In[2]:


# for example:
starnet_download_file('apStar_visits_main.h5')

