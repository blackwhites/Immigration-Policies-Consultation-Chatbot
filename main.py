import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.vectorstores import Vectara

load_dotenv()

# Initialize st.session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF upload and API keys
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    # customer_id = st.text_input("Vectara Customer ID", value=os.getenv("CUSTOMER_ID", ""))
    # api_key = st.text_input("Vectara API Key", value=os.getenv("API_KEY", ""))
    # corpus_id = st.text_input("Vectara Corpus ID", value=str(os.getenv("CORPUS_ID", "")))
    # openai_api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""))
    submit_button = st.button("Submit")

customer_id = os.getenv("CUSTOMER_ID", "")
api_key = os.getenv("API_KEY", "")
corpus_id = int(os.getenv("CORPUS_ID", ""))  # Assuming CORPUS_ID should be an integer
openai_api_key = os.getenv("OPENAI_API_KEY", "")
# Constants
CUSTOMER_ID = customer_id if customer_id else os.getenv("")
API_KEY = api_key if api_key else os.getenv("")
CORPUS_ID = int(corpus_id) if corpus_id else int(os.getenv("CORPUS_ID", ))  # Assuming CORPUS_ID should be an integer
OPENAI_API_KEY = openai_api_key if openai_api_key else os.getenv("")
       
       
def initialize_vectara():
    vectara = Vectara(
        vectara_customer_id=CUSTOMER_ID,
        vectara_corpus_id=CORPUS_ID,
        vectara_api_key=API_KEY
    )
    return vectara

vectara_client = initialize_vectara()

def get_knowledge_content(vectara, query, threshold=0.5):
    found_docs = vectara.similarity_search_with_score(
        query,
        score_threshold=threshold,
    )
    knowledge_content = ""
    for number, (score, doc) in enumerate(found_docs):
        knowledge_content += f"Document {number}: {found_docs[number][0].page_content}\n"
    return knowledge_content


prompt = PromptTemplate.from_template(
    """You are a professional and friendly immigration policies Consultant and you are helping a client with a immigration policies. The client is asking you for advice on a immigration policies. Just explain him in detail the answer and nothing else. This is the issue: {issue} 
    To assist him with his issue, you need to know the following information: {knowledge} 
    """
)


runnable = prompt | ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], openai_api_key=OPENAI_API_KEY) | StrOutputParser()

# Main Streamlit App
st.title("Immigration Policies Consultation Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if user_input := st.chat_input("Enter your issue:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    knowledge_content = get_knowledge_content(vectara_client, user_input)
    print("__________________ Start of knowledge content __________________")
    print(knowledge_content)
    response = runnable.invoke({"knowledge": knowledge_content, "issue": user_input})

    response_words = response.split()
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for word in response_words:
            full_response += word + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Run when the submit button is pressed
if submit_button and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        tmp_filename = tmpfile.name

    try:
        vectara_client.add_files([tmp_filename])
        st.sidebar.success("PDF file successfully uploaded to Vectara!")
    except Exception as e:
        st.sidebar.error(f"An error occurred: {str(e)}")
    finally:
        os.remove(tmp_filename)