import streamlit as st
import os
from dotenv import load_dotenv
from tp import SimpleRAGSystem

# Load environment variables
load_dotenv()
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

st.set_page_config(page_title="RAG Q&A Bot")
st.title("RAG Q&A Bot Demo")
st.write("Ask questions about Python, Machine Learning, or Data Science. The bot retrieves relevant info from your documents and answers using an LLM.")

# Initialize the RAG system only once
if 'rag' not in st.session_state:
    st.session_state['rag'] = SimpleRAGSystem(openrouter_api_key=openrouter_api_key)

question = st.text_input("Your question:")

if question:
    with st.spinner('Searching and generating answer...'):
        result = st.session_state['rag'].ask_question(question)
    st.markdown(f"**Answer:**\n{result['answer']}")
    st.markdown(f"**Sources:** {', '.join(set([os.path.basename(s) for s in result['metadata']['sources']]))}") 