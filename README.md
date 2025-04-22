### EXNO-3-DS

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

## NAME: MOHAN S
## REGISTER NUMBER: 212223240094

```python
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![Screenshot 2025-04-22 103853](https://github.com/user-attachments/assets/042632d1-d875-4082-834d-d906b81830ae)

```python
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2025-04-22 103947](https://github.com/user-attachments/assets/423811cf-6483-40d9-8cf6-1306e7217f74)

```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2025-04-22 104018](https://github.com/user-attachments/assets/6b96b51b-ce6b-4bac-8267-7a598f9003c7)

```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2025-04-22 104052](https://github.com/user-attachments/assets/231693cc-7f31-4414-99d1-37ecee744bc3)

```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2025-04-22 104127](https://github.com/user-attachments/assets/e86977ed-1e98-4c29-9e63-54c885d2aefe)

```python
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2025-04-22 104157](https://github.com/user-attachments/assets/dab272e6-d06a-4568-b79a-1ef4dacb9f71)


```python
pip install --upgrade category_encoders
```

```python
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![Screenshot 2025-04-22 104416](https://github.com/user-attachments/assets/a27a42e0-e9d6-4f60-b222-c6923a833de9)

```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![Screenshot 2025-04-22 104447](https://github.com/user-attachments/assets/3825ec89-f1fc-47a2-ae60-d623899a35b3)

```python
dfb=pd.concat([df,nd],axis=1)
dfb
```
![Screenshot 2025-04-22 104520](https://github.com/user-attachments/assets/adae2e58-3c09-4f55-be76-9c99ca1ba9cc)

```python
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![Screenshot 2025-04-22 104555](https://github.com/user-attachments/assets/1c6e56e6-a07c-483c-b0d6-7a5459a0b3d2)

```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![Screenshot 2025-04-22 104625](https://github.com/user-attachments/assets/943c614d-c8de-4ccb-a515-d921498a8d28)

```python
df.skew()
```
![Screenshot 2025-04-22 104700](https://github.com/user-attachments/assets/404c8d11-e29d-4a9c-8888-f176b50c5f51)

```python
np.log(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 104729](https://github.com/user-attachments/assets/038c362b-c56e-46f8-8ad1-fb7c6f5a92bf)

```python
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2025-04-22 104756](https://github.com/user-attachments/assets/5815950c-0bf3-43ca-b4c1-d998a3cbb170)

```python
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 104829](https://github.com/user-attachments/assets/25c7d9f2-80b2-46be-a970-671d3ca39d25)


```python
np.square(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 104930](https://github.com/user-attachments/assets/da5a1079-3287-4e6d-b462-6ce33cea9211)

```python
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2025-04-22 105008](https://github.com/user-attachments/assets/b973fb32-bad3-488b-9f98-3a6eb631f8c0)

```python
df.skew()
```
![Screenshot 2025-04-22 105037](https://github.com/user-attachments/assets/691eef13-8b63-4c06-ab57-e9981df7fcee)

```python
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![Screenshot 2025-04-22 105107](https://github.com/user-attachments/assets/a106a127-3552-4478-85ff-4bff313c473f)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![Screenshot 2025-04-22 105140](https://github.com/user-attachments/assets/8e40f7b2-59b3-4399-8cfc-64d47c3290d1)

```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-22 105210](https://github.com/user-attachments/assets/86e63c1d-a097-4010-84eb-7a0b5b6e53d6)

```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2025-04-22 105238](https://github.com/user-attachments/assets/8a649c41-abfe-4504-9ce8-bb1b1d4e5324)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-22 105309](https://github.com/user-attachments/assets/1f57c227-eb5d-40dd-a93f-8b63facebb68)

```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-22 105336](https://github.com/user-attachments/assets/107b87e4-7413-4de7-9481-132c3d49ce87)

```python
dt=pd.read_csv("/content/titanic_dataset (1).csv")
dt
```
![Screenshot 2025-04-22 105412](https://github.com/user-attachments/assets/214c375a-4b47-40e9-8102-2fcfcae09974)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![Screenshot 2025-04-22 105449](https://github.com/user-attachments/assets/81b8452f-a020-4e73-a137-93528880cde6)

```python
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![Screenshot 2025-04-22 105531](https://github.com/user-attachments/assets/ce862f54-9338-49f5-bcab-26adbb3723e2)

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully

       
