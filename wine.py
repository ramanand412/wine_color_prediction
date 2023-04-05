import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.write("""
# sample wine prediction app
This app predicts color of the wine!
""")
st.sidebar.header("User input parameters")

def user_input_parameters():

    fixed_acidity = st.sidebar.slider('fixed_acidity', 3.8, 15.9, 5.4)
    volatile_acidity = st.sidebar.slider('volatile_acidity', 0.08, 1.58, 1.4)
    citric_acid = st.sidebar.slider('citric_acid', 0.0, 1.66, 1.3)
    residual_sugar = st.sidebar.slider('residual_sugar', 0.6, 65.8, 1.2)
    chlorides = st.sidebar.slider('chlorides', 0.009, 0.611, 0.4)
    free_sulfur_dioxide = st.sidebar.slider('free_sulphur_dioxide', 1.0, 289.0, 38.0)
    total_sulfur_dioxide = st.sidebar.slider('total_sulphur_dioxide', 6.0, 440.0, 10.0)
    density = st.sidebar.slider('density', 0.99, 1.04, 1.0)
    pH = st.sidebar.slider('pH', 2.72, 4.01, 3.4)
    sulphates = st.sidebar.slider('sulphates', 0.22, 2.0, 1.3)
    alcohol = st.sidebar.slider('alcohol', 8.0, 14.9, 5.0)
    quality = st.sidebar.slider('quality', 3.0, 9.0, 5.0)

    data = {'fixed_acidity': fixed_acidity,
                'volatile_acidity': volatile_acidity,
                'citric_acid': citric_acid,
                'residual_sugar': residual_sugar,
                'chlorides': chlorides,
                'free_sulfur_dioxide': free_sulfur_dioxide,
                'total_sulfur_dioxide': total_sulfur_dioxide,
                'density': density,
                'pH': pH,
                'sulphates': sulphates,
                'alcohol': alcohol,
                'quality': quality
                }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_parameters()
st.subheader("user input parameters")
st.write(df)

wine_df=pd.read_csv("Wine_Quality_Data.csv")


X=wine_df.drop("color",axis=1)
y=wine_df["color"]
scaler=StandardScaler()
scaler.fit(X)
standardised_df=scaler.transform(X)
df_standard=pd.DataFrame(standardised_df,columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality'])

log_reg=LogisticRegression()
log_reg.fit(df_standard,y)

prediction = log_reg.predict(df)
prediction_proba=log_reg.predict_proba(df)
st.subheader("class label and their corresponding")
st.write(y.unique())
st.subheader("prediction")
st.write(prediction)
st.subheader("prediction probability")
st.write(prediction_proba)