import streamlit as st
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from matplotlib.pylab import rcParams
from sklearn.impute import SimpleImputer
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score
df = pd.read_csv('housing.csv')
df = pd.get_dummies(df)
df.rename(columns = {'ocean_proximity_<1H OCEAN':'<1H OCEAN','ocean_proximity_INLAND':'INLAND','ocean_proximity_ISLAND':'ISLAND','ocean_proximity_NEAR BAY':'NEAR BAY','ocean_proximity_NEAR OCEAN':'NEAR OCEAN'}, inplace = True)
df.head()
# display(df)
df.describe()
df.shape
df.isna().sum()
df.info()
imp = SimpleImputer(strategy='mean')
X = imp.fit_transform(df)
new_df = pd.DataFrame(X, columns = df.columns)
new_df.describe()
new_df.isna().sum()
# %matplotlib inline
new_df.hist(bins=50, figsize=(20,15))
plt.show()
corr_matrix = new_df.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
new_df['bedroom_ratio'] = new_df['total_bedrooms']/new_df['total_rooms'] #Total number of bedrooms per room
new_df['household_rooms'] = new_df['total_rooms']/new_df['households'] #Total number of rooms per household
# df
new_df.drop(['total_bedrooms','households','population'],axis = 1,inplace=True)
rcParams['figure.figsize'] = (10,5)
sns.heatmap(corr_matrix,annot=True,cmap='Greens')
attributes = ['median_income','bedroom_ratio','housing_median_age','<1H OCEAN']
scatter_matrix(new_df[attributes],figsize=(20,10))
new_df.plot(kind='scatter',x='median_house_value',y='median_income',alpha=0.8)
new_df.plot(kind='scatter',x='median_house_value',y='total_rooms',alpha=0.8)
pd.reset_option("display.max_rows")
import numpy as np
from scipy.spatial import distance
mean = np.mean(new_df, axis=0)
covariance = np.cov(new_df, rowvar=False)
mahalanobis_dist = distance.cdist(new_df, [mean], 'mahalanobis', VI=np.linalg.inv(covariance)).ravel()
df1 = pd.DataFrame(mahalanobis_dist, columns=['Mahalanobis distance'])
df1
column = df1['Mahalanobis distance']
n = column[column > 2.5].count()
print(n)
outliers = np.where(mahalanobis_dist > 2.5)[0]
outliers
# min(mahalanobis_dist)
new_df = new_df.drop(outliers, axis=0)
new_df.shape
new_df.plot(kind='scatter',x='median_house_value',y='median_income',alpha=0.8)
new_df.plot(kind='scatter',x='median_house_value',y='total_rooms',alpha=0.8)
x = new_df.drop("median_house_value",axis=1)
y = new_df["median_house_value"]
x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=100,test_size=0.2)
len(x_train),len(x_test)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
from sklearn.svm import SVR
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor() # Best model
# model = SVR(kernel='poly', C=1.0, epsilon=0.1)
model.fit(x_train,y_train)
x_test = scaler.fit_transform(x_test)
y_pred = model.predict(x_test)
len(y_pred)
MSE = mean_squared_error(y_test , y_pred)
MAE = mean_absolute_error(y_test , y_pred)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test , y_pred)
MSE , MAE , RMSE, R2


st.title('House price prediction')
 
st.write('---')

longitute = st.number_input('Enter Longitute of house')
latitude = st.number_input('Enter Latitude of house')

age = st.number_input('How old is the house (in years)?', min_value=0, step=1)

rooms = st.number_input('Total number of rooms')
 
median_income = st.number_input('Median income of house')

ocean_proximity = st.selectbox(
    'Specify Ocean proximity ?',
    ('Choose your Option', '<1H OCEAN', 'INLAND','ISLAND','NEAR BAY','NEAR OCEAN'))

if ocean_proximity == '<1H OCEAN':
  OCEAN = 1.0
  INLAND = 0.0
  ISLAND = 0.0
  NEAR_BAY = 0.0
  NEAR_OCEAN = 0.0
elif ocean_proximity == 'INLAND':
  OCEAN = 0.0
  INLAND = 1.0
  ISLAND = 0.0
  NEAR_BAY = 0.0
  NEAR_OCEAN = 0.0
elif ocean_proximity == 'ISLAND':
  OCEAN = 0.0
  INLAND = 0.0
  ISLAND = 1.0
  NEAR_BAY = 0.0
  NEAR_OCEAN = 0.0
elif ocean_proximity == 'NEAR BAY':
  OCEAN = 0.0
  INLAND = 0.0
  ISLAND = 0.0
  NEAR_BAY = 1.0
  NEAR_OCEAN = 0.0
elif ocean_proximity == 'NEAR OCEAN':
  OCEAN = 0.0
  INLAND = 0.0
  ISLAND = 0.0
  NEAR_BAY = 0.0
  NEAR_OCEAN = 1.0

bedroom_ratio = st.number_input('Total number of bedrooms per room')

household_rooms = st.number_input('Total number of rooms per household')
 
if st.button('Predict House Price'):
    cost = predict(np.array([[longitute, latitude, age, rooms, median_income, OCEAN, INLAND, ISLAND, NEAR_BAY, NEAR_OCEAN, bedroom_ratio, household_rooms]]))
    st.metric(label = "Predicted house price in $ ", value = cost[0])
if st.button('Check Accuracy'):
    accuracy = str(R2*100) + "%"
    st.metric(label="Accuracy", value=accuracy)
