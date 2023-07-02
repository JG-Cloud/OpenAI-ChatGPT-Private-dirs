import sys
import os

# suppress python warning messages
import warnings
warnings.filterwarnings("ignore")

# Langchain depedencies
from langchain import OpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# web framework
import streamlit as st

# Importing langchain config/backend code
import query_data


### streamlit webapp config/framework

# main body
st.title("🦜️🔗 Jatin's data sourcing AI app")
query_prompt = st.text_input('What are you looking for?', placeholder='Search inside my docs...')

if query_prompt:
    # Return query answers to the screen
    st.write("ANSWER:")
    st.write(query_data.get_qa_chain().run(query_prompt))