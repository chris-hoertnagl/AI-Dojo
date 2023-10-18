import pandas as pd
import sklearn
import pickle
import gzip
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor

#Loading data
#!wget https://raw.githubusercontent.com/Seb1703/AI-Dojo/main/Basics/sample_data/housing_new.csv
#!mv housing_new.csv /content/sample_data/
df = pd.read_csv("https://raw.githubusercontent.com/Seb1703/AI-Dojo/main/Basics/sample_data/housing_new.csv")

#Cleaning data
x = df["total_rooms"].mean()/df["total_bedrooms"].mean()
df["total_bedrooms"].fillna(df["total_rooms"]*x, inplace=True)
bord1 = df.quantile(q=0.33333)["median_house_value"]
bord2 = df.quantile(q=0.66666)["median_house_value"]
def get_value(x):
  if x < bord1:
    return "low"
  if x > bord2:
    return "high"
  return "medium"

df_class = df.copy()
df_class["house_value"] = df_class.median_house_value.apply(get_value)

#Classifier
#Splitting the data
features = ["longitude", "latitude"]
#features = ["longitude",	"latitude",	"housing_median_age",	"total_rooms",	"total_bedrooms",	"population",	"households",	"median_income"]
label = ["house_value"]

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(df_class[features], df_class[label], test_size=0.30, random_state=42)


#Initializing the knn-model
class_knn = KNeighborsClassifier(n_neighbors=5)
#training the model on the training data
class_knn.fit(X_train_class, y_train_class["house_value"])
#generating the classifications on our test data
y_pred_knn = class_knn.predict(X_test_class)

def get_house_value_class(lat, long):

  input = pd.DataFrame([{"longitude": long, "latitude": lat}])
  output = class_knn.predict(input)
  return output[0]

#Regression
#Splitting the data
features = ["longitude",	"latitude", "median_income", "total_rooms"]
label = ["median_house_value"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(df[features], df[label], test_size=0.30, random_state=42)

#Initializing the random forest regression model
reg_rfr = RandomForestRegressor(max_depth=15)
#training the model on the training data
reg_rfr.fit(X_train_reg, y_train_reg.values.ravel())
#generating the classifications on our test data
y_pred_rfr = reg_rfr.predict(X_test_reg)

def get_house_value_reg(lat, long, inc, rooms):

  input = pd.DataFrame([{"longitude": long, "latitude": lat, "median_income": inc, "total_rooms": rooms}])
  output = reg_rfr.predict(input)
  return output[0]


# über pickle pipeline in git ignore, siehe video ml ops, 

with gzip.open('class_model.pkl.gz', 'wb') as class_model_gzip:

    pickle.dump(class_knn, class_model_gzip)
    