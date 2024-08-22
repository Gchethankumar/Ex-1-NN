<H3>ENTER YOUR NAME : G Chethan kumar</H3>
<H3>ENTER YOUR REGISTER NO. : 212222240022</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 22/08/2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

### IMPORT LIBRARIES : 

```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```

### READ THE DATA: 
```py
df=pd.read_csv("Churn_Modelling.csv")
```

### CHECK DATA: 
```py
df.head()
df.tail()
df.columns
```

### CHECK THE MISSING DATA:
```py
df.isnull().sum()
```

### ASSIGNING X:
```py
X = df.iloc[:,:-1].values
X
```

### ASSIGNING Y:
```py
Y = df.iloc[:,-1].values
Y
```

### CHECK FOR OUTLIERS:
```py
df.describe()
```

### DROPPING STRING VALUES DATA FROM DATASET:
```py
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```

### CHECKING DATASETS AFTER DROPPING STRING VALUES DATA FROM DATASET:
```py
data.head()
```

### NORMALIE THE DATASET USING (MinMax Scaler):
```py
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```

### SPLIT THE DATASET:
```py
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
print(X)
print(Y)
```

### TRAINING AND TESTING MODEL:
```py
X_train ,X_test ,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT:

### DATA CHECKING:
![Screenshot 2024-08-22 093243](https://github.com/user-attachments/assets/0ae5026d-66a8-419f-90de-851909eddabc)


### MISSING DATA:
![Screenshot 2024-08-22 093253](https://github.com/user-attachments/assets/4e993135-35aa-4996-9460-b414c8cd296c)


### DUPLICATES IDENTIFICATION:
![Screenshot 2024-08-22 093302](https://github.com/user-attachments/assets/7abaeff4-ac74-402a-a688-45c02da040bb)



### VALUE OF Y:
![Screenshot 2024-08-22 093617](https://github.com/user-attachments/assets/05b58a9c-df8d-4d6f-b894-79e39b9bbdf9)


### OUTLIERS:
![Screenshot 2024-08-22 093355](https://github.com/user-attachments/assets/a589c25b-a686-4aec-9e49-2bb9ef993e11)



### CHECKING DATASET AFTER DROPPING STRING VALUES DATA FROM DATASET:
![Screenshot 2024-08-22 093405](https://github.com/user-attachments/assets/1120c7c4-00f9-4c5b-8eb8-8e6acd999cfc)


### NORMALIZE THE DATASET:
![Screenshot 2024-08-22 093423](https://github.com/user-attachments/assets/50281b77-c58e-4314-b42f-3f6220c4452c)


### SPLIT THE DATASET:
![Screenshot 2024-08-22 093430](https://github.com/user-attachments/assets/1e649261-448e-48ab-bae4-1623e3a12c1b)


### TRAINING AND TESTING MODEL:
![Screenshot 2024-08-22 093439](https://github.com/user-attachments/assets/044d4ba5-8eb0-46e6-8b8e-106a90164cee)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
