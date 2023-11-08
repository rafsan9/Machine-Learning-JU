"""
# Iris Data set
"""

import matplotlib.pyplot as plt
 
from sklearn import datasets
 

# import some data to play with
iris = datasets.load_iris()
print(iris)

"""## Scatter plot"""

X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

 

"""# Importing all the necessary packages for the classification problem"""

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
 
from sklearn.naive_bayes import GaussianNB
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
 
# Data Loading
url='https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
df = pd.read_csv(url)
df.head()



"""## Naïve Bayes Classifier
--Defining Target Variable

--Split the data as Test/Train

---Accuracy Measurement of Naïve Bayes Classifier

"""

x = df[['sepal_length',	'sepal_width',	'petal_length',	'petal_width']]
y = df['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state =3)
nb=GaussianNB()
nb.fit(x_train,y_train)
GaussianNB(priors=None, var_smoothing=1e-09)
predictionsNB = nb.predict(x_test)
print('Classification Report:\n',classification_report(y_test, predictionsNB))
print('Confusion Matrix:\n',confusion_matrix(y_test, predictionsNB))
print('Accuracy Score:',accuracy_score(y_test, predictionsNB))

# plotting both distibutions on the same figure
import seaborn as sns
fig, axs = plt.subplots(2, 4, figsize=(14, 7))
fig = sns.kdeplot(df['sepal_length'], shade=True, color="r", ax=axs[0, 0])
fig = sns.kdeplot(df['sepal_width'],  shade=True, color="r", ax=axs[0, 1])
fig = sns.kdeplot(df['petal_length'],  shade=True, color="r", ax=axs[0, 2])
fig = sns.kdeplot(df['petal_width'],  shade=True, color="r", ax=axs[0, 3])

sns.boxplot(x=df["species"], y=df["sepal_length"], palette = 'magma', ax=axs[1, 0])
sns.boxplot(x=df["species"], y=df["sepal_width"], palette = 'magma',ax=axs[1, 1])
sns.boxplot(x=df["species"], y=df["petal_length"], palette = 'magma',ax=axs[1, 2])
sns.boxplot(x=df["species"], y=df["petal_width"], palette = 'magma',ax=axs[1, 3])

plt.show()


# Alternatively 
#Exploring dataset:
sns.pairplot(df, kind="scatter", hue="species")
plt.show()