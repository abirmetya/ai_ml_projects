import joblib
import re
import streamlit as st
import cleaning
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Title of the application 
st.title('News Classification\n', )
st.subheader("by Chandrima Chakrabarty")

@st.cache_resource

def load_vect_and_model():
    text_vectorizer = joblib.load('vectorizer.joblib')
    model           = joblib.load('svc_model.joblib')
    return text_vectorizer,model

text_vectorizer,model = load_vect_and_model()

def cleaning_(texts):
    texts = pd.DataFrame(texts)
    texts.columns = ['content']
    txt = cleaning.clean_data(texts)
    stop = stopwords.words('english')
    txt['content'] = txt['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    porter_stemmer = PorterStemmer()
    txt['content'] = txt['content'].apply(lambda x: ' '.join([porter_stemmer.stem(word) for word in x.split() ]))
    return txt


def vectorize_text(texts):
    # text = cleaning_(texts)
    text_transformed = text_vectorizer.transform(texts)
    return text_transformed

def pred_class(texts):
    return model.predict(vectorize_text(texts))

def pred_proba(texts):
    return model.predict_proba(vectorize_text(texts))

input_message = st.text_area(label='Enter your news',value='Please specify your input',height=20)
submit        = st.button('Check')

if submit and input_message:
    prediction,proba = pred_class([input_message]) , pred_proba([input_message])

    st.markdown('### Prediction: {}'.format(prediction[0]))
    # st.metric