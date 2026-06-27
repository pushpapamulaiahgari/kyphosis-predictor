import streamlit as st
import joblib
#load trined model
model=joblib.load('model_trained.pkl')

#title
st.title('💊🩹👩‍🔬kyphosis prediction app🩺💉')

#input fields
Age=st.number_input("Age in months",1,250,70)
Number=st.number_input("Number of vetebrae",1,20,5)
Start=st.number_input("Start level",1,18,10)

#prediction button
if st.button("Prediction"):
    pred=model.predict([[Age,Number,Start]])
    proba=model.predict_proba([[Age,Number,Start]])[0][1]
    if pred[0]==1: st.error(f'PRESENT | risk:{proba*100:.1f}%')
    else:
        st.success(f'ABSENT | risk:{proba*100:.1f}%')