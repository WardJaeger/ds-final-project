import joblib
import matplotlib.pyplot as plt
import pandas as pd 
import streamlit as st


# Load model pipeline 
model = joblib.load(open("model-tpot.joblib","rb"))

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


def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    ax = prediction_proba.plot(kind='barh', figsize=(7, 4), color='#722f37', zorder=10, width=0.5)
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

    st.pyplot(ax.get_figure())


st.header('Food Group Classification')
st.write('This application predicts the food groups of various food items, based on nutritional data provided by the user.')

st.sidebar.header('User Input Parameters')
st.sidebar.write('Please provide all the following nutritional values per serving.')
user_input = get_user_input()

st.subheader('User Input parameters')
st.write('The predictive model accepts features that have been normalized for a 100 gram sample. The data shown below has been adjusted accordingly, and may differ from what the user inputs into the sidebar.')
st.write(user_input)

st.subheader('Prediction')
prediction = model.predict(user_input)[0]
prediction_proba = pd.DataFrame(
    data = (model.predict_proba(user_input)[0]*100).round(2), 
    index = ['Dairy', 'Fruits', 'Grains', 'Protein', 'Sweets', 'Vegetables']
)
st.write(f"The model classifies this food into the food group of {prediction}, with a confidence level of {prediction_proba.loc[prediction][0]}%.")
visualize_confidence_level(prediction_proba)
