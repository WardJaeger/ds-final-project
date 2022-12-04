import joblib
import matplotlib.pyplot as plt
import pandas as pd 
import streamlit as st


# Load model pipeline 
nearest_neighbors = joblib.load(open("model-nearest-neighbors.joblib", "rb"))
decision_tree = joblib.load(open("model-decision-tree.joblib", "rb"))
neural_network = joblib.load(open("model-neural-network.joblib", "rb"))
tpot = joblib.load(open("model-tpot.joblib", "rb"))


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe
    """
    factor = st.sidebar.number_input('Serving size (g)', 0.0, 1000.0, 100.0)/100.0

    water = st.sidebar.number_input('Water (g / serving)', 0.0, 100.0*factor, 61.7*factor)
    energy = st.sidebar.number_input('Energy (Kcal / serving)', 0.0, 1000.0*factor, 203.0*factor)
    lipids  = st.sidebar.number_input('Lipids (g / serving)', 0.0, 100.0*factor, 5.9*factor)
    saturated = st.sidebar.number_input('Saturated Fat (g / serving)', 0.0, lipids, 0.0)
    cholesterol = st.sidebar.number_input('Cholesterol (mg / serving)', 0.0, 3500.0*factor, 6.0*factor)
    sodium = st.sidebar.number_input('Sodium (mg / serving)', 0.0, 30000.0*factor, 80.0*factor)
    potassium = st.sidebar.number_input('Potassium (mg / serving)', 0.0, 17500.0*factor, 258.0*factor)
    carbohydrates  = st.sidebar.number_input('Carbohydrates (g / serving)', 0.0, 100.0*factor, 7.7*factor)
    protein = st.sidebar.number_input('Protein (g / serving)', 0.0, 100.0*factor, 10.3*factor)
    calcium  = st.sidebar.number_input('Calcium (mg / serving)', 0.0, 8000.0*factor, 21.0*factor)
    iron = st.sidebar.number_input('Iron (mg / serving)', 0.0, 75.0*factor, 1.6*factor)
    
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


def get_predictions(user_input, model):
    prediction = model.predict(user_input)[0]
    prediction_proba = pd.DataFrame(
        data = (model.predict_proba(user_input)[0]*100).round(2), 
        index = ['Dairy', 'Fruits', 'Grains', 'Protein', 'Sweets', 'Vegetables']
    )
    prediction_proba.plot()
    return (prediction, prediction_proba)


def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.barh(prediction_proba.index, prediction_proba[0], color=('#08e', '#d00', '#f90', '#527', '#f4a', '#5d5'))
    
    ax.set_xlim(xmin=0, xmax=100)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    for tick in ax.get_xticks():
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel("Confidence Level (%)", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Food Group", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot(fig)


st.header('Food Group Classification')
st.write('This application predicts the food groups of various food items, based on nutritional data provided by the user. It shows the predictions of four different models.')

st.sidebar.header('User Input Parameters')
st.sidebar.write('Please provide all the following nutritional values per serving.')
user_input = get_user_input()

st.subheader('User Input Parameters')
st.write('The predictive models accept features that have been normalized for a 100 gram sample. The data shown below has been adjusted accordingly and may differ from what the user inputs into the sidebar.')
st.write(user_input)

st.subheader('Nearest Neighbor Prediction')
prediction, prediction_proba = get_predictions(user_input, nearest_neighbors)
st.write(f"This model classifies the food into the food group of {prediction}, with a confidence level of {prediction_proba.loc[prediction][0]}%.")
visualize_confidence_level(prediction_proba)

st.subheader('Decision Tree Prediction')
prediction, prediction_proba = get_predictions(user_input, decision_tree)
st.write(f"This model classifies the food into the food group of {prediction}, with a confidence level of {prediction_proba.loc[prediction][0]}%.")
visualize_confidence_level(prediction_proba)

st.subheader('Neural Network Prediction')
prediction, prediction_proba = get_predictions(user_input, neural_network)
st.write(f"This model classifies the food into the food group of {prediction}, with a confidence level of {prediction_proba.loc[prediction][0]}%.")
visualize_confidence_level(prediction_proba)

st.subheader('TPOT Prediction')
prediction, prediction_proba = get_predictions(user_input, tpot)
st.write(f"This model classifies the food into the food group of {prediction}, with a confidence level of {prediction_proba.loc[prediction][0]}%.")
visualize_confidence_level(prediction_proba)
