import pandas as pd 
import joblib
import streamlit as st


# Load model pipeline 
model = joblib.load(open("model-tpot.joblib","rb"))

def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data, columns = ['Percentage'], index = ['Dairy', 'Fruit', 'Grain', 'Protein', 'Sweets', 'Vegetable'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel("Percentage(%) Confidence Level", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Food Group", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    return

st.write("""
# Wine Quality Prediction ML Web-App 
This app predicts the ** Quality of Wine **  using **wine features** input via the **side panel** 
""")

st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar
st.sidebar.write('Please provide all the following nutritional values per serving.')

def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe
    """
    factor = st.sidebar.number_input('Serving size (g)', 0.0, 1000.0, 100.0)/100.0

    water = st.sidebar.slider('Water (g / serving)', 0.0, 100.0*factor, 61.7*factor)
    energy = st.sidebar.slider('Energy (Kcal / serving)', 0.0, 1000.0*factor, 203.0*factor)
    lipids  = st.sidebar.slider('Lipids (g / serving)', 0.0, 100.0*factor, 5.9*factor)
    saturated = st.sidebar.slider('Saturated Fat (g / serving)', 0.0, 100.0*factor, 1.7*factor)
    cholesterol = st.sidebar.slider('Cholesterol (mg / serving)', 0.0, 3500.0*factor, 6.0*factor)
    sodium = st.sidebar.slider('Sodium (mg / serving)', 0.0, 30000.0*factor, 80.0*factor)
    carbohydrates  = st.sidebar.slider('Carbohydrates (g / serving)', 0.0, 100.0*factor, 7.7*factor)
    protein = st.sidebar.slider('Protein (g / serving)', 0.0, 100.0*factor, 10.3*factor)
    calcium  = st.sidebar.slider('Calcium (g)', 0.0, 8000.0*factor, 21.0*factor)
    iron = st.sidebar.slider('Iron (mg / serving)', 0.0, 75.0*factor, 1.6*factor)
    potassium = st.sidebar.slider('Potassium (mg / serving)', 0.0, 17500.0*factor, 258.0*factor)
    
    features = {
        'Water (g)': water/factor,
        'Energy (Kcal)': energy/factor,
        'Protein (g)': protein/factor,
        'Lipids (g)': lipids/factor,
        'Carbohydrates (g)': carbohydrates/factor,
        'Calcium (mg)': calcium/factor,
        'Iron (mg)': iron/factor,
        'Potassium (mg)': potassium/factor,
        'Sodium (mg)': sodium/factor,
        'Saturated Fat (g)': saturated/factor,
        'Cholesterol (mg)': cholesterol/factor,
    }
    data = pd.DataFrame(features, index=[0])

    return data

user_input = get_user_input()

st.subheader('User Input parameters')
st.write(user_input)

prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

visualize_confidence_level(prediction_proba)
