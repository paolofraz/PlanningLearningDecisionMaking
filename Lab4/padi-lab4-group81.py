#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 4: MNIST
# 
# In the end of the lab, you should save the notebook as `padi-lab4-groupXX.ipynb`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure that the subject is of the form `[<group n.>] LAB <lab n.>`.

# ### 1. The MNIST dataset
# 
# The National Institute of Standards and Technology (NIST) published in 1995 a corpus for handprinted document and character recognition. The corpus originally contained 810,000 character images from 3,600 different writers. The MNIST ("Modified NIST") dataset was created from the original NIST dataset and contains a total of 70,000 normalized images ($28\times28$ pixels) containing handwritten digits. All images are grayscale and anti-aliased. 
# 
# ---
# 
# In this lab, we work with a simplified version of the MNIST dataset, in order to have the algorithms run in a manageable amount of time. In such modified dataset, digit images have been pre-processed to $8\times 8$ images, where each pixel takes values between 0 and 16. The modified dataset is available in `scikit-learn` through its `datasets` module. We thus start by loading the digits dataset.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as data

# Load dataset and print its description
digits = data.load_digits()
print(digits.DESCR)

# Get dimensions 
nP = digits.data.shape[0]
nF = digits.data.shape[1]

fig = plt.figure()

# Print sample digits
for i in range(10): 
    plt.subplot(2, 5, i + 1)
    idx = list(digits.target).index(i)
    plt.imshow(digits.images[idx], cmap='Greys')
    plt.axis('off')

fig.tight_layout()
plt.show()


# In the first activities, you will prepare the dataset, before running any learning algorithms.
# 
# ---
# 
# #### Activity 1.        
# 
# From the MNIST dataset, construct the training and test sets. The input data can be accessed as the attribute `data` in the dataset `digits`; the corresponding output data can be accessed as the attribute `target` in `digits`. To build the train and test sets, you can use the function `train_test_split` from the module `model_selection` of `scikit-learn`. Make sure that the test set corresponds to $1/7$th of the total number of samples. 
# 
# **Note:** Don't forget to import the necessary modules from `scikit-learn`. Also, for reproducibility, initialize the seed of the `train_test_split` function to a fixed number (e.g., 42).
# 
# ---

# In[3]:


from sklearn.model_selection import train_test_split

# Dataset split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=1/7, random_state=42)

print("Total Number of samples:", nP)
print("Test Set size:", X_test.shape[0])


# ### 2. Principal component analysis (PCA)
# 
# Right now, each point in the dataset is represented by the pixel information, which roughly corresponds to $64$ features. In this activity, you will determine a small number of alternative features that manage to capture most of the relevant information contained in each picture but which provide a much more compact representation thereto. Such features correspond to the _principal components_ that you will compute next. PCA can be performed through the function `PCA`, in the `decomposition` module of `scikit-learn`. 
# 
# ---
# 
# #### Activity 2.        
# 
# * Run PCA on the training set. To do this, you should first fit the PCA model to the train data and then use the resulting model to transform the data. For details, check the documentation for the function `PCA`.
# 
# * To grasp how much of the information in the data is contained in the different components, plot the _cumulative explained variance_ (in percentage) as a function of the number of components. The explained variance can be accessed via the attribute `explained_variance_` of your model.
# 
# **Note:** In general, before running PCA on some training set, you should _standardize_ the data to make sure that all inputs are centered and lie in the same range. To do this, you can use the function `StandardScaler` of the module `preprocessing` of `scikit-learn`.
# 
# ---

# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardize 
scaler  = StandardScaler()
X_train = scaler.fit_transform( X_train )
X_test  = scaler.transform( X_test )


# In[5]:


# Run PCA
pca = PCA(random_state=42)
pca.fit(X_train)
var = np.cumsum(pca.explained_variance_) / np.sum(pca.explained_variance_)

# Plot PCA
plt.figure()
plt.plot(var)
plt.title("PCA - Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance [%]")
plt.show()


# Note how a small number of components explain around 90\% of the variance in the data. As such, it seems reasonable that we may rely only on those components as features to represent our data.
# 
# ### 3. Impact of number of features on a Logistic Regression classifier
# 
# To clearly understand the implications of the adopted representation, you will now run an extensive test to investigate how the number of components may impact the performance of the classifier. 
# 
# ---
# 
# #### Activity 3.        
# 
# Take the data in your training set and further split it in two sets, $D_T$ and $D_V$, where $D_T$ corresponds to $85\%$ of the training data and $D_V$ to the remaining $15\%$. You will use $D_T$ for training, and $D_V$ for validation. 
# 
# For $k\in\{5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64\}$,
# 
# * Run PCA with $k$ components on the data in $D_T$
# * Transform the data in $D_T$ using the computed PCA
# * Train a logistic regression classifier on the transformed data. Use $C=100$, the `'newton-cg'` solver, and set the multi_class option to `'auto'`
# * Compute the error in $D_T$ and in $D_V$
# 
# Repeat the _whole process_ (including the split of $D_T$ and $D_V$) 40 times.
# 
# **Note 1:** Don't forget that, in order to run PCA, you should standardize the data once again; you should not use the standardized data from Activity 2, since it has seen the whole data in $D_T$ and $D_V$. 
# 
# **Note 2:** Also, don't forget that, in order to run your classifier with the data in $D_V$, you must transform it with the PCA fit to $D_T$.
# 
# **Note 3:** The whole process may take a while, so don't despair. The logistic regression classifier can be accessed by importing `LogisticRegression` from `sklearn.linear_model`. To compute the error of a classifier, you can use the `accuracy_score` function from `sklearn.metrics`.
# 
# ---

# In[6]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
np.random.seed(42)

K = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64]
tra_error_l = np.empty( (40,len(K)) )
val_error_l = np.empty( (40,len(K)) )

for t in range(0,40):
    
    D_T, D_V, y_T, y_V = train_test_split(X_train, y_train, test_size=0.15)

    scaler = StandardScaler()
    D_T    = scaler.fit_transform( D_T )
    D_V    = scaler.transform( D_V )
    
    for i,k in enumerate(K):

        pca     = PCA(n_components=k)
        D_T_pca = pca.fit_transform(D_T)
        D_V_pca = pca.transform(D_V)

        logistic = LogisticRegression(C = 100, solver='newton-cg', multi_class='auto').fit(D_T_pca, y_T)
        pred_T   = logistic.predict(D_T_pca)
        pred_V   = logistic.predict(D_V_pca)

        tra_error_l[t,i] = accuracy_score(y_T, pred_T)
        val_error_l[t,i] = accuracy_score(y_V, pred_V)
    print(t, end = " ")


# ---
# 
# #### Activity 4.
# 
# Plot the average training and validation error from Activity 3 as a function of $k$. Explain the differences observed between the two curves.
# 
# ---

# In[7]:


plt.figure()
plt.scatter(K, 1 - tra_error_l.mean(axis=0), label="Training")
plt.scatter(K, 1 - val_error_l.mean(axis=0), label="Validation")
plt.title("Logistic Regression")
plt.xlabel("$k$")
plt.ylabel("Average Error")
plt.legend()
plt.show()


# In[8]:


print("Differences in accuracy:\n",tra_error_l.mean(axis=0) - val_error_l.mean(axis=0))


# <span style="color:blue"> As expected, the training error is always smaller than the validation error. Besides, we notice that by increasing the complexity $k$ the accuracies get closer but without showing meaningful evidence of overfitting, since the absolute difference does not increase. </span>

# ### 4. Comparison of different classifiers
# 
# In Activity 4 you investigated the impact of the number of features on the performance of the Logistic Regression algorithm. You will now compare the performance of the best logistic regression algorithm with another algorithm from the literature.

# ---
# 
# #### Activity 5.        
# 
# * Repeat Activity 3 but now using a 5-Nearest Neighbors classifier instead of a Logistic Regression. 
# * Plot the average training and validation error as a function of 𝑘.
# 
# **Note:** Again, the whole process may take a while, so don't despair. The kNN classifier can be accessed by importing `KNeighborsClassifier` from `sklearn.neighbors`.
# 
# ---

# In[9]:


from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)

K = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64]
tra_error_k = np.empty( (40,len(K)) )
val_error_k = np.empty( (40,len(K)) )

for t in range(0,40):
    
    D_T, D_V, y_T, y_V = train_test_split(X_train, y_train, test_size=0.15)

    scaler = StandardScaler()
    D_T    = scaler.fit_transform( D_T )
    D_V    = scaler.transform( D_V )
    
    for i,k in enumerate(K):

        pca     = PCA(n_components=k)
        D_T_pca = pca.fit_transform(D_T)
        D_V_pca = pca.transform(D_V)

        knn      = KNeighborsClassifier().fit(D_T_pca, y_T)
        pred_T   = knn.predict(D_T_pca)
        pred_V   = knn.predict(D_V_pca)

        tra_error_k[t,i] = accuracy_score(y_T, pred_T)
        val_error_k[t,i] = accuracy_score(y_V, pred_V)
    print(t, end = " ")

plt.figure()
plt.scatter(K, 1 - tra_error_k.mean(axis=0), label="Training")
plt.scatter(K, 1 - val_error_k.mean(axis=0), label="Validation")
plt.title("kNN")
plt.xlabel("$k$")
plt.ylabel("Average Error")
plt.legend()
plt.show()


# ---
# 
# #### Activity 6.        
# 
# Taking into consideration the results from Activities 3 and 5, select the classifier and number of features that you believe is best and
# 
# * Compute the performance of your selected classifier on the test data. 
# * Comment whether the performance on the test data matches what you expected, based on the results from activities 3 and 5.
# 
# **Note:** When computing the performance of your selected classifier, you should re-train it using the whole training data.
# 
# ---

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=1/7, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform( X_train )
X_test  = scaler.transform( X_test )

pca         = PCA(n_components=20) # 20 features
X_train_pca = pca.fit_transform( X_train )
X_test_pca  = pca.transform( X_test )

knn        = KNeighborsClassifier().fit(X_train_pca, y_train)
pred_train = knn.predict( X_train_pca )
pred_test  = knn.predict( X_test_pca )

tra_error_final = accuracy_score(y_train, pred_train)
tst_error_final = accuracy_score(y_test, pred_test)

print("Training Error: ", 1 - tra_error_final)
print("Validation Error for k=20: ", 1 - val_error_k.mean(axis=0)[3])
print("Test Error: ", 1 - tst_error_final)


# <span style="color:blue">We choose the 5-Nearest Neighbors classifier with $k=20$ PCA-features for two reasons: the first one is that kNN has higher validation errors than logistic regression, meaning that supposedly it will perform better on unseen data. Secondly, for $k>20$ the validation accuracy does not improve significantly by adding new features. In conclusion, we see that our choice was rational as the error on the test set is $<4\%$. </span>

# In[ ]:




