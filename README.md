## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
NAME: SANTHOSE AROCKIARAJ J

REG NO: 21224230248
### Feature Encoding
```
import pandas as pd
df=pd.read_csv('/content/Encoding Data (2).csv')
df
```
![image](https://github.com/user-attachments/assets/8ced66ec-f746-421c-a81a-a904789afb5a)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pn=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pn])
e1.fit_transform(df[['ord_2']])
```
![image](https://github.com/user-attachments/assets/b732517f-56f8-4de2-afa2-6fe95c610a0f)
```
df['bo2']=e1.fit_transform(df[['ord_2']])
df
```
![image](https://github.com/user-attachments/assets/459f8c7d-33c6-42b0-9104-f098bb17fda4)

### Label Encoder
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/1e85e55e-2ce7-4ca0-8c78-1508b34f89b8)

### One Hot Encoder
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/2731ebfb-7392-42f3-9156-60a1cba70aa1)
```
pd.get_dummies(df2,columns=['nom_0'])
```
![image](https://github.com/user-attachments/assets/a05e756e-be75-4ad4-b681-c761c31467ce)

### Binary Encoder
```
from category_encoders import BinaryEncoder
df=pd.read_csv('/content/data (2).csv')
df
```
![image](https://github.com/user-attachments/assets/1c2c3729-a5d3-4fc0-b208-eb7529e4dcaa)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/52ff14cf-5f29-437d-8502-68c4ddecfa17)

### Target Encoder

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc['City'],y=cc['Target'])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/f8d6c7a6-d68e-4f71-a331-ebba1455bd00)

### Feature Transformation
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform (1).csv")
df
```
![image](https://github.com/user-attachments/assets/a88690b8-ad34-4197-acc7-a85f71baca48)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/ae6b1cbc-c63f-4fbe-bfca-55eaf6b4907c)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/8b85212d-d7e4-4132-a82b-75205bfa9905)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/edfb26c5-713a-46f1-8d93-5180a95a5038)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/77f543c3-1b51-494a-8087-9b978606e2b5)
```
df["Highly Positive Skew_bocox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/33a57ce5-3f84-469a-b8aa-3d3cd2784778)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/722dbd52-3d4e-4d93-a874-0dbbdb9817b0)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/e0717d83-4420-4304-a674-3e11c9039dc0)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution="normal")
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/c764061e-8fe5-4955-9ee1-592d9d10303b)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/4a4f9ddc-6fc1-43af-a5ae-a025ad75adc6)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c7bf84a1-59a2-41bb-b932-c071808c5ddf)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution="normal",n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/802a86de-bca7-4033-bce7-4429923b8431)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/36cacafc-b6df-410d-9350-a1ee431aa16d)
```
sm.qqplot(df["Highly Negative Skew_1"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/7b85f4b0-a6d0-452f-befc-134d5a30ae74)
```
dt=pd.read_csv("/content/titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution="normal",n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt["Age"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/62d5c540-3e78-4209-842e-7053a88e67ad)
```
sm.qqplot(dt["Age_1"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/5bc76c23-f8c0-4678-9f14-a011c8421a6f)


# RESULT:
  Thus, read the given data and perform Feature Encoding and Transformation process and save the data to a file.

       
