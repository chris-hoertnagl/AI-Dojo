import streamlit as st
import pandas as pd
import pydeck as pdk

from streamlit_option_menu import option_menu

selected = option_menu(None, ["Home", "Data Set", "Classification", 'About'], 
    icons=['house', 'table', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected

#Inhalt der Menü Punkte
if selected == "Home":
    st.title("🌉 Golden Gate Estate")
    st.subheader("Your California Real Estate Partner.🏡")
    st.markdown("""
    ##### Property Valuation using Data Science
    
    At Golden Gate Estate, we recognize that your property is more than just a building; it's a home, an investment, and a part of your story. Our passion for real estate and our commitment to innovative technology have made us a leading estate agency in California.
    This app uses machine learning to predict the price of the house. It loads a pre-trained linear regression model, which takes as input various features of the house, such as the number of rooms, the number of bedrooms, the population of the house's neighborhood, 
    and the distance to the nearest city. The app preprocesses the input data by combining some of the features and adding new features, such as the distance to the nearest city.
    """)
    
elif selected == "Data Set":
    data = pd.read_csv('Basics/sample_data/california_housing_test.csv')

    st.write("Hier ist der Datensatz als Tabelle:")
    st.table(data.head(21))
    
elif selected == "Classification":
    st.title("Einstellungen")
    st.write("Passen Sie Ihre Einstellungen an.")
    
        
    
elif selected == "About":
    st.title("About California House Price Prediction")
    st.subheader("Check out the colab workbook 🔗.")
    


