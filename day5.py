import os 
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stChatMessage {
        background-color: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 10px;
        backdrop-filter: blur(10px);
    }
    .stChatInput input {
        background-color: rgba(255,255,255,0.2);
        color: white;
        border-radius: 25px;
    }
    h1 {
        color: white !important;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title = "AI Customer Support", page_icon = "🤖")
st.title("🤖 AI Customer Support Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite",
    google_api_key = api_key
)

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful customer support agent. Be polite and helpful."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{input}")
])

chain = prompt | llm

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

user_input = st.chat_input("Type your message here...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    response = chain.invoke({
        "input" : user_input,
        "chat_history" : st.session_state.chat_history
    })

    with st.chat_message("assistant"):
        st.write(response.content)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response.content))