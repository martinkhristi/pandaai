import streamlit as st
import pandas as pd
import os
import time
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe
import seaborn as sns

# Load environment variables
load_dotenv()

# Define functions to load language models
@st.cache_resource
def load_groq_llm():
    return ChatGroq(model_name="Llama-3.1-8b-Instant", api_key=os.getenv('GROQ_API_KEY'))

@st.cache_resource
def load_openai_llm():
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4")

# Set up Streamlit page
st.set_page_config(page_title="Data Chat App", layout="wide")

# Sidebar for user inputs
with st.sidebar:
    st.title("Settings")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    llm_choice = st.selectbox("Select Language Model", ("Groq", "OpenAI"))

# Function to read uploaded file
def read_file(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    if file_extension == '.csv':
        return pd.read_csv(file)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

# Main application content
def main():
    st.title("Interactive Data Analysis with Groq API and Llama 3.1 & PandasAI")
    
    if uploaded_file is not None:
        try:
            data = read_file(uploaded_file)
            
            # Display first five rows
            st.header("Data Preview")
            st.write(data.head())
            
            # Chat window
            st.header("Chat with your data")
            query = st.text_area("Enter your query about the data:")
            
            if st.button("Ask"):
                if query:
                    try:
                        with st.spinner('Analyzing data...'):
                            # Start timer
                            start_time = time.time()
                            
                            # Load selected LLM
                            if llm_choice == "Groq":
                                llm = load_groq_llm()
                            else:
                                llm = load_openai_llm()
                            
                            # Create SmartDataframe
                            df_smart = SmartDataframe(data, config={'llm': llm})
                            
                            # Get response
                            response = df_smart.chat(query)
                            
                            # End timer
                            end_time = time.time()
                            
                            # Calculate elapsed time
                            elapsed_time = end_time - start_time
                            
                        # Display response
                        st.write("Response:")
                        st.write(response)
                        
                        # Display elapsed time
                        st.info(f"Time taken to answer: {elapsed_time:.2f} seconds")
                    
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.warning("Please enter a query.")
        except ValueError as e:
            st.error(str(e))
    else:
        st.warning("Please upload a CSV or Excel file to begin.")

if __name__ == "__main__":
    main()
