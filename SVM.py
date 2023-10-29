#!/usr/bin/env python
# coding: utf-8

# ### Import

# In[5]:


import pandas as pd
import librosa
import numpy as np
from collections import Counter
get_ipython().system('pip install imblearn')
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
get_ipython().system('pip install tensorflow')
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


# ### Reading development csv file

# In[6]:


FileName = pd.read_csv(f'Project/development.csv')


# In[7]:


FileName.path


#  
#  # Data Pre-Processing Section

# # We Have used Trimming and Padding to extract more accurate data 
# 

# In[10]:


melspectograms = []

for i in range(0, len(FileName.count(axis=1))):
    path_address = FileName.path[i]
    y, sr = librosa.load(path_address)
    yy, _ = librosa.effects.trim(y,top_db = 20)  # trim silence from the beginning and end of the audio file
    p=235000-len(yy)
    padding=np.pad(yy,(0,p))
    mel_spect = librosa.feature.mfcc(padding, sr=sr)
    melspectograms.append(mel_spect)


# In[ ]:





# ## Converting List of MFCC to Array 
# 

# In[11]:


arrayofmel = np.array(melspectograms)
FinalMFCC = arrayofmel.reshape(arrayofmel.shape[0], -1)


# In[12]:


FinalMFCC


# ## Merging Action and Object as our label and Tranfsorm them to label Encoder function 

# In[13]:


# This function is for mixing two columns of labels 
labels = [FileName.action[i] + FileName.object[i] for i in range(len(FileName))]


# In[14]:


encoder = LabelEncoder()
labelencoded = encoder.fit_transform(labels)


# ### Oversampling Technique for Balancing Labels 

# In[16]:


# Count the number of elements for each label
labelcounts = Counter(labelencoded)

max_count = max(labelcounts.values())
melspectograms_resampled = []
integer_encoded_labels_resampled = []

for label, count in labelcounts.items():
    label_indices = [i for i, x in enumerate(labelencoded) if x == label]

    random_indices = np.random.choice(label_indices, size=(max_count-count), replace=True)
    
    for index in label_indices + list(random_indices):
        melspectograms_resampled.append(FinalMFCC[index])
        integer_encoded_labels_resampled.append(labelencoded[index])
        
melspectograms_resampled = np.array(melspectograms_resampled)
melspectograms_resampled = melspectograms_resampled.reshape(melspectograms_resampled.shape[0], -1)


# In[17]:


### Train Test Split
from sklearn.model_selection import train_test_split
train_data2,test_data2,y_train,y_test=train_test_split(melspectograms_resampled,integer_encoded_labels_resampled,test_size=0.25,random_state=0)


# ## Using Standard Scalar for Normalization 

# In[18]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_data2 = sc.fit_transform(train_data2)
test_data2 = sc.transform(test_data2)


# In[19]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(train_data2, y_train)

# Predicting the Test set results
y_pred = classifier.predict(test_data2)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# ## Now This Section is for Evaluating New DataSet to predict there labels 

# In[20]:


# fetchin new data to be predicted for labeling 
evaluation_raw_data = pd.read_csv(f'Project/evaluation.csv')


# In[21]:


melspectograms_test = []

for i in range(0, len(evaluation_raw_data.count(axis=1))):
    path_address = evaluation_raw_data.path[i]
    y, sr = librosa.load(path_address)
    yy, _ = librosa.effects.trim(y,top_db = 20)  # trim silence from the beginning and end of the audio file
    p=235000-len(yy)
    padding=np.pad(yy,(0,p))
    mel_spect = librosa.feature.mfcc(padding, sr=sr)
    melspectograms_test.append(mel_spect)


# In[23]:


melspectograms_test = np.array(melspectograms_test)
arrayed_melspectograms = melspectograms_test.reshape(melspectograms_test.shape[0], -1)


# In[24]:


arrayed_melspectograms_test=sc.transform(arrayed_melspectograms)


# In[25]:


predictions = classifier.predict(arrayed_melspectograms_test)
predictions


# In[26]:


predictions.shape


# In[28]:


predicted_classes = encoder.inverse_transform(predictions)


# In[29]:


predicted_classes


# In[30]:


num_samples = len(predicted_classes)
result = pd.DataFrame({'Id': range(num_samples), 'Predicted': predicted_classes})


# In[31]:


result.to_csv(r'E:/SVM_Result.csv', index=False, sep=',')


# In[ ]:




