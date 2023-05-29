# California-House-Price-Prediction-with-streamlit
Initially, I've imported all the required librarires which are mandatory for this project including pandas, numpy, matplotlib, sklearn.model_selection etc. After that, I proceed with the basic steps of building ML model which are as follows :- 
1) Data importing  
2) Exploratory Data Analysis [EDA] :- 
i) Since there is a categorical variable named "ocean_proximity" in the imported dataset. Hence, I make use of pd.get_dummies(df) which divides that variable or feature into five variables(All are binary features).
ii) In EDA, I simply print the first five rows of imported dataset.
iii) Also, I renamed the features of variables which are complicated or large.
4) Data Preprocessing
