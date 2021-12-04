import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

data = pd.read_csv("D.csv")

data.columns = ['0','time','pm2d5','lat','lon']


# In[13]:


data = data.drop(['0'],axis = 1)


# In[14]:


from datetime import datetime
import time
data


# In[15]:


import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary
X = data.iloc[:,[0,2,3]]
Y = data.iloc[:,1]
k = gpflow.kernels.RBF()
m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)


# In[16]:


opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=2))
print_summary(m)



