import streamlit as st
from main import chatbot
import pickle

with open('.\\resources\\clf.pkl', 'rb') as f:
    clf = pickle.loads(f.read())

with open('.\\resources\\vec.pkl', 'rb') as f:
    vectorizer = pickle.loads(f.read())

counter = 0
st.title("Chatbot")
st.write("welcome to the chatbot. Please type the message and press enter to start the coversation.")
counter += 1
user_input = st.text_input("You:", key=f"user_input_{counter}")

if user_input:
    response = chatbot(user_input)
    st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

    if response.lower() in ['goodbye', 'bye']:
        st.write("Thank you for chatting with me. Have a great day!")
        st.stop()