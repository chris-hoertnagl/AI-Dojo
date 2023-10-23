# Golden Gate Estate - Data Science Project

Welcome to the Golden Gate Estate Data Science project repository! In this repository, you will find a comprehensive guide to our data analysis and modeling process, following the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology.

## Project Overview
We have created a Jupyter Notebook, named `Ai_Workshop.ipynb`, that takes you through our journey of understanding, analyzing, and modeling data for a fictional company, Golden Gate Estate. The primary objective of this project is to predict house prices within the estate.

## Opening the Jupyter Notebook

To open and work with the Jupyter Notebook, follow these steps:

**Step 1:** Make sure Visual Studio Code (VSC) is installed on your system.

**Step 2:** Open VSC.

**Step 3:** If you don't already have Jupyter Notebook installed, you can install it using pip by running this command in VSC's integrated terminal:

```bash
pip install notebook
```

**Step 4:** Install all the necessary packages with:

```bash
pip install pandas numpy matplotlib scikit-learn seaborn streamlit geopandas pydeck folium requests
```

These commands will install the following packages:

- `pandas`: Data manipulation and analysis library.
- `numpy`: Numerical computing library for handling arrays and matrices.
- `matplotlib`: Data visualization library for creating charts and plots.
- `scikit-learn`: A machine learning library for building and evaluating models.
- `seaborn`: A data visualization library based on Matplotlib.
- `streamlit`: A tool for creating web apps for machine learning and data science.
- `geopandas`: For working with geospatial data.
- `pydeck`: A high-level library for creating 3D maps and data visualizations.
- `folium`: For creating interactive maps.
- `requests`: For making HTTP requests.
- `gzip`: For working with compressed files.

Make sure to execute this `pip install` command before running the code to ensure you have all the necessary dependencies.

**Step 5** Open Jupyter Notebook:
```bash
jupyter notebook
```
A browser tab will open where you can choose the notebook file.

## Models Built
Our analysis and modeling process involved the development of both a classifier and a regression model. These models were constructed using state-of-the-art data science techniques to ensure accurate predictions and insights.

## Streamlit App Deployment

To make our models accessible via the Streamlit app, follow these steps:

**Step 1 - Run the Jupyter Notebook**: First, run the complete Jupyter Notebook (`Ai_Workshop.ipynb`) locally. This notebook contains the data analysis and modeling steps. Running the notebook will generate two pickle files in the local folder "models."

**Step 2 - Run the Streamlit App**: After successfully running the Jupyter Notebook, you can proceed to run the Streamlit app. Open your terminal and navigate to the project directory containing `streamlit_app.py`. Run the following command:

```bash
streamlit run streamlit_app.py
```

This command will start the Streamlit app and make the models accessible via the web interface. It's essential to run the Jupyter Notebook first to ensure the required model files are available for the Streamlit app.

This repository is designed to provide a transparent and replicable view of our data science project for the Golden Gate Estate. We hope it serves as a valuable resource for anyone interested in data analysis, modeling, or the CRISP-DM process.

Happy exploring!
