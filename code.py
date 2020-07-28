#We have a data which classified if patients have heart disease or 
#not according to features in it. We will try to use this data to 
#create a model which tries predict if a patient has this disease 
#or not. We will model our data on different algorithm and we will 
#the model that gives the best accuracy!

# for basic operations
import numpy as np
import pandas as pd

# for basic visualizations
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the dataset 
df = pd.read_csv("heart.csv")
#prints the top 5 rows of data
df.head()

#Data contains;
# 1. age - age in years
# 2. sex - (1 = male; 0 = female)
# 3. cp - chest pain type
# 4. trestbps - resting blood pressure (in mm Hg on admission to the hospital)
# 5. chol - serum cholestoral in mg/dl
# 6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 7. restecg - resting electrocardiographic results
# 8. thalach - maximum heart rate achieved
# 9. exang - exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak - ST depression induced by exercise relative to rest
# 11. slope - the slope of the peak exercise ST segment
# 12. ca - number of major vessels (0-3) colored by flourosopy
# 13. thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
# 14. target - have disease or not (1=yes, 0=no)

#DATA EXPLORATION

#counts the value of the target category wise
df.target.value_counts()
#showing total count of values of different target values on graph
sns.countplot(x="target", data=df, palette="bwr")
plt.show()

#count percentage of patients who had or not the heart diesease

countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))

#shows the count of female and male in the dataset 

sns.countplot(x='sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

#percentage of male and female patients in the dataset 

countFemale = len(df[df.sex == 0])
countMale = len(df[df.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))

#calculates the mean of every column w.r.t. diff target values(0,1 here)
df.groupby('target').mean()

#graph of frequency of heart diesease w.r.t. age

pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()

#graph of frequency of heart diesease w.r.t. sex
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()

#Scatter plot for maximum heart rate for heart diesease patients and 
#healthy patients w.r.t. age

plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()

#graph of heart diesease frequency as per slope 

pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['#DAF7A6','#FF5733' ])
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.show()

#graph of heart diesease frequency according to fbs 

pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()

#graph of frequency of diesease or not w.r.t. chest pain type

pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()

#Creating dummy variables
#Since 'cp', 'thal' and 'slope' are categorical variables,
#we'll turn them into dummy variables.

a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c]

#now concat this list to our df dataframe

df = pd.concat(frames, axis = 1)
df.head()

#drop the column of cp,thal,slope beacuse it is not needed now

df = df.drop(columns = ['cp', 'thal', 'slope'])
df.head()

#now, form independent variable(features/factors) dataframe(X) and o/p y
y = df.target.values
X = df.drop(['target'], axis = 1)

#we have normalized our data before split, because their are chances that out data may contain nan (infinity value)
#because of max-min=0 in denominator

# Normalize
X_norm = (X - np.min(X)) / (np.max(X) - np.min(X)).values

#split our dataset into training set and test set
#We will split our data. 80% of our data will be train data and 20% of it will be test data.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.20, random_state = 0)

#OUR MANUAL LOGISTIC REGRESSION 

#transpose matrices
X_train = X_train.T
y_train = y_train.T
X_test = X_test.T
y_test = y_test.T

#Let's say weight = 0.01 and bias = 0.0
#initialize
def initialize(dimension):
    
    weight = np.full((dimension,1),0.01)
    bias = 0.0
    return weight,bias

#because we use sigmoid function in logistic regression
def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head

#here, this forward backward is basically computation of cost function
#with the help of gradient descent 

#forward in computation of cost function
#backward is computation of derivative of cost function

def forwardBackward(weight,bias,x_train,y_train):
    # Forward
    
    y_head = sigmoid(np.dot(weight.T,x_train) + bias)
    loss = -(y_train*np.log(y_head) + (1-y_train)*np.log(1-y_head))
    cost = np.sum(loss) / x_train.shape[1]
    
    # Backward
    derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"Derivative Weight" : derivative_weight, "Derivative Bias" : derivative_bias}
    
    return cost,gradients

#update is computation of gradient descent
#we'll also plot the value of gradient descent on the graph after
#every iteration
    
def update(weight,bias,x_train,y_train,learningRate,iteration) :
    costList = []
    index = []
    
    #for each iteration, update weight and bias values
    for i in range(iteration):
        cost,gradients = forwardBackward(weight,bias,x_train,y_train)
        weight = weight - learningRate * gradients["Derivative Weight"]
        bias = bias - learningRate * gradients["Derivative Bias"]
        
        costList.append(cost)
        index.append(i)

    parameters = {"weight": weight,"bias": bias}
    
    print("iteration:",iteration)
    print("cost:",cost)

    plt.plot(index,costList)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients

#this function is used for predicting the class of our O/P
#here, if y_pred<=0.5, it belongs to class 0(non-cancer)
#else, it'll belong to class 1(cancer class)
    
def predict(weight,bias,x_test):
    z = np.dot(weight.T,x_test) + bias
    y_head = sigmoid(z)

    y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(y_head.shape[1]):
        if y_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction

#final, logistic regression main funciton
#this function will call all other function
    
def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration):
    dimension = x_train.shape[0]
    weight,bias = initialize(dimension)
    
    parameters, gradients = update(weight,bias,x_train,y_train,learningRate,iteration)

    y_prediction = predict(parameters["weight"],parameters["bias"],x_test)
    
    print("Manual Test Accuracy: {:.2f}%".format((100 - np.mean(np.abs(y_prediction - y_test))*100)))

logistic_regression(X_train,y_train,X_test,y_test,1,100)

#HENCE, MANUAL LOGISTIC REGRESSION COMPUTATION COMPLETED!!!

#now, again take transpose of the matrix(because transpose was taken previously)
#i.e. revert it back to original form
#transpose matrices
X_train = X_train.T
y_train = y_train.T
X_test = X_test.T
y_test = y_test.T

#make a list of accuracies to store all the accuracy for different algorithm
accuracies={}

#MODEL-1
# Training the Logistic Regression model on the Training set

X_train1=X_train
y_train1=y_train
X_test1=X_test

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred1 = classifier_lr.predict(X_test1)

print("Training Accuracy :", classifier_lr.score(X_train1, y_train1))
print("Testing Accuracy :", classifier_lr.score(X_test1, y_test))

#enter it's value of in the list 
accuracies['Logistic Regression']=classifier_lr.score(X_test1, y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
print(cm1)

#MODEL-2

#FITTING K_NEAREST_NEIGHBORS MODEL TO OUR TRAINING SET

X_train2=X_train
y_train2=y_train
X_test2=X_test

from sklearn.neighbors import KNeighborsClassifier
# try ro find best k value
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(X_train2, y_train2)
    scoreList.append(knn2.score(X_test2, y_test))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

# Training the K-NN model on the Training set
cknn = KNeighborsClassifier(n_neighbors = 8, metric = 'minkowski', p = 2)
cknn.fit(X_train2, y_train2)

# Predicting the Test set results
y_pred2 = cknn.predict(X_test2)

print("Training Accuracy :", cknn.score(X_train2, y_train2))
print("Testing Accuracy :", cknn.score(X_test2, y_test))

accuracies['k_nearset_neighbors']=cknn.score(X_test2, y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

#MODEL-3
#FITTING SUPPORT VECTOR MACHINE CLASSIFIER TO OUR TRAINING SET 

X_train3=X_train
y_train3=y_train
X_test3=X_test


# Training the SVM model on the Training set
from sklearn.svm import SVC
csvm = SVC(kernel = 'linear', random_state = 0)
csvm.fit(X_train3, y_train3)

# Predicting the Test set results
y_pred3 = csvm.predict(X_test3)

print("Training Accuracy :", csvm.score(X_train3, y_train3))
print("Testing Accuracy :", csvm.score(X_test3, y_test))

accuracies['Support vector machine']=csvm.score(X_test3, y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred3)
print(cm3)

#MODEL-4
#FITTING KERNEL SVM TO OUR TRAINING SET 
X_train4=X_train
y_train4=y_train
X_test4=X_test


# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
cksvm = SVC(kernel = 'rbf', random_state = 0)
cksvm.fit(X_train4, y_train4)

# Predicting the Test set results
y_pred4 = cksvm.predict(X_test4)

print("Training Accuracy :", cksvm.score(X_train4, y_train4))
print("Testing Accuracy :", cksvm.score(X_test4, y_test))

accuracies['kernel SVM']=cksvm.score(X_test4, y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test, y_pred4)
print(cm4)

#MODEL-5
#FITTING NAIVE BAYES TO TRAINING SET 

X_train5=X_train
y_train5=y_train
X_test5=X_test

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
cnb = GaussianNB()
cnb.fit(X_train5, y_train5)

# Predicting the Test set results
y_pred5 = cnb.predict(X_test5)

print("Training Accuracy :", cnb.score(X_train5, y_train5))
print("Testing Accuracy :", cnb.score(X_test5, y_test))

accuracies['Naive Bayes']=cnb.score(X_test5, y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm5 = confusion_matrix(y_test, y_pred5)
print(cm5)

#MODEL-6
#FITTING DECISION TREE CLASSIFIER 

X_train6=X_train
y_train6=y_train
X_test6=X_test

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
cdt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
cdt.fit(X_train6, y_train6)

# Predicting the Test set results
y_pred6 = cdt.predict(X_test6)

print("Training Accuracy :", cdt.score(X_train6, y_train6))
print("Testing Accuracy :", cdt.score(X_test6, y_test))

accuracies['Decision Tree']=cdt.score(X_test6, y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm6 = confusion_matrix(y_test, y_pred6)
print(cm6)

#MODEL-7
# Fitting Random Forest classifier with 100 trees to the Training set

X_train7=X_train
y_train7=y_train
X_test7=X_test

from sklearn.ensemble import RandomForestClassifier
crf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
crf.fit(X_train7, y_train7)

y_pred7 = crf.predict(X_test7)

print("Training Accuracy :", crf.score(X_train7, y_train7))
print("Testing Accuracy :", crf.score(X_test7, y_test))

accuracies['Random Forest']=crf.score(X_test7, y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm7 = confusion_matrix(y_test, y_pred7)
print(cm7)

#Comparing models

colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE","yellow"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,6))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()

print("Logistic Regression\n",cm1)
print("KNN\n",cm2)
print("SVM\n",cm3)
print("kSVM\n",cm4)
print("Naive Bayes",cm5)
print("Decision tree\n",cm6)
print("Random Forest\n",cm7)

#ACCURACY IS SIMILAR IN RANDOM FORSET AND K-NEAREST-NEIGHBORS
#FALSE NEGATIVES ARE ALSO SAME, INFACT COMPLETE CONFUSION MATRIX IS SAME

#NOTE:
#IF we apply a deep neural network we can achieve a mush better accuracy

