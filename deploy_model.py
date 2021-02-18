

# from pyngrok import ngrok
# public_url = ngrok.connect(port='80')
# print(public_url)

import pickle
import streamlit as st
import numpy as np


# loading the trained model
model = pickle.load(open('income_predictor.pkl', 'rb'))


@st.cache()
def prediction(workclass, education, marital_status, occupation, relationship, race, sex, native_country,
               age, hours_per_week):

    # Pre-processing user input
    X_cat = np.array([workclass, education, marital_status, occupation, relationship, race, sex, native_country]).reshape(1, -1)
    X_num = np.array([age, hours_per_week]).reshape(1, -1)

    with open('encoder', 'rb') as file:
        encoder = pickle.load(file)

    X_cat_ohe = encoder.fit_transform(X_cat).toarray()
    X = np.concatenate([X_cat_ohe, X_num], axis=1)

    # Making predictions
    prediction = model.predict(X)

    if prediction == 1:
        pred = 'Estimated Annual Income Higher than $50,000'
    else:
        pred = 'Estimated Annual Income Lower than $50,000'
    return pred


# print(prediction('Private', 'Masters', 'Divorced', 'Exec-managerial', 'Unmarried', 'White', 'Male', 'United-States',
#                50, 60))
#
# print(prediction('Local-gov', '7th-8th', 'Never-married', 'Handlers-cleaners', 'Unmarried', 'Black', 'Female', 'Cuba',
#                25, 40))


# this is the main function in which we define our webpage
def main():

    # front end elements of the web page
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Income Prediction</h1>
    </div>
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction
    workclass = st.selectbox('Workclass', ('Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov', 'Local-gov', 'Not Telling',
                                           'Self-emp-inc', 'Without-pay', 'Never-worked'))
    education = st.selectbox('Education', ('Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm',
                                           'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th',
                                           '1st-4th', 'Preschool', '12th'))
    marital_status = st.selectbox('Marital Status', ('Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
                                                        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'))
    occupation = st.selectbox('Occupation', ('Exec-managerial', 'Handlers-cleaners', 'Prof-specialty',
                                                'Other-service', 'Adm-clerical', 'Sales', 'Craft-repair',
                                                'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
                                                'Tech-support', 'Not Telling', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'))
    relationship = st.selectbox('Relationship', ('Husband', 'Not-in-family', 'Wife', 'Own-child',
                                                 'Unmarried', 'Other-relative'))
    race = st.selectbox('Race', ('White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'))
    sex = st.selectbox('Gender', ('Male', 'Female'))
    native_country = st.selectbox('Gender', ('United-States', 'Cuba', 'Jamaica', 'India', 'Not Telling', 'Mexico', 'South',
                                             'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
                                             'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
                                             'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic',
                                             'El-Salvador', 'France', 'Guatemala', 'China', 'Japan', 'Yugoslavia',
                                             'Peru', 'Outlying-US(Guam-USVI-etc),' 'Scotland', 'Trinadad&Tobago',
                                             'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
                                             'Holand-Netherlands'))

    age = st.selectbox('Age', range(16, 80), 1)
    hours_per_week = st.selectbox('Hours Per Week', range(10, 80), 1)

    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict Annual Income"):
        result = prediction(workclass, education, marital_status, occupation, relationship, race, sex, native_country,
               age, hours_per_week)
        st.success(result)


if __name__ == "__main__":
    main()
