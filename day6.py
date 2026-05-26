import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

loader = TextLoader("business_info.txt")
documents = loader.load()
print("Document loaded!")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks!")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Vector store created!")

retriever = vectorstore.as_retriever()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite",
    google_api_key = api_key
)

questions = [
    "What is the return policy?",
    "How much does express delivery cost?",
    "What are the working hours?",
]

for question in questions:
    relevant_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in relevant_docs]) 

    response = llm.invoke(
        f"Answer this question using only the context below.\n\nContext: {context}\n\nQuestion: {question}"
    )
    print(f"\nQ: {question}")
    print(f"\nA: {response.content}")