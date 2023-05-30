# California-House-Price-Prediction-with-streamlit
Initially, I've imported all the required librarires which are mandatory for this project including pandas, numpy, matplotlib, sklearn.model_selection etc. After that, I proceed with the basic steps of building ML model which are as follows :- 
1) Data importing :- 
I used California housinsg dataset for this project.
3) Exploratory Data Analysis [EDA] :- 
i) Since there is a categorical variable named "ocean_proximity" in the imported dataset. Hence, I make use of pd.get_dummies(df) which divides that variable or feature into five variables(All are binary features).
ii) In EDA, I simply print the first five rows of imported dataset.
iii) Also, I renamed the features of variables which are complicated or large.
iv) In addition to above, I also determined total number of null values of particular feature and I got the total 207 null values only for total_bedrooms feature.
4) Data Preprocessing :- 
In this i make use of SimpleImputer, it is especially useful for treating null values. Here we make use of mean strategy to fill missing values with mean of featue[total_bedrooms].
5) Data Visualization :- 
I visualize the data using different visualization technique such as Histogram, Scatterplot etc.
6) Feature Engineering :- 
i) This is the important step in building ML model because it increases accuracy of your model by increasing number of features in your imported dataset.
ii) I basically added two extra features viz bedroom_ratio(total number of bedrooms per room) with the help of total_bedrooms & total_rooms features. Second one is household_rooms(total number of rooms per household) with the help of total_rooms & household features.
iii) After that I make use of correlation matrix to determine relationship between each variable. Based on that i deleted or dropped three variables from dataset.
7) Outlier Detection :- 
i) Outlier detection is the technique or method to detect outliers in the dataset because it affects the accuracy of our ML model.
ii) I used one of the popular method called mahalanobis distance to detect outliers in the dataset which calculates the mahalanobis distance of all feature values from the mean.
iii) Then based on the threshold the outliers are detected. Here I set the threshold value 2.5 but you may set it to 3 according to your dataset.
iv) After that, I dropped or deleted all that outlier rows by using new_df = new_df.drop(outliers, axis=0). <br />
Here you can see the diff :- 
Before removing outliers

<img width="576" alt="image" src="https://github.com/sohelkh1211/California-House-Price-Prediction-with-streamlit/assets/125993375/e9c5674d-09f5-45dc-97cf-b87a692a7174"> <br />
After removing outliers
<img width="562" alt="image" src="https://github.com/sohelkh1211/California-House-Price-Prediction-with-streamlit/assets/125993375/b65e63b3-4e99-4b09-b95d-3a8610625662"> <br />
8) Train Test split :- 
We simply split the dataset into train dataset & test dataset using train_test_split provided by sklearn.model_selection library.
9) Model Selection :- 
Model Selection plays an important role in building ML model which gives higher acuracy. I found RandomForestRegressor() which was giving the best accuracy of 86% then the rest ones. Actually, it was giving 65% accuracy before removal of outliers.
10) Hyperparameter tuning :- 
It is the technique to reduce RMSE or increase the accuracy of model by adjusting the values of parameters. But this did not gave the intended results.
11) Building ML app(Streamlit app):- 
Finally, I built an app with streamlit which takes input from user and displays the output which is the predicted price of house(in $) based on specified features. I also displayed the accuracy of the model.
