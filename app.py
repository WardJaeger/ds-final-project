import pandas as pd 
import joblib
import streamlit as st


# Load model pipeline 
model = joblib.load(open("model-tpot.joblib","rb"))

