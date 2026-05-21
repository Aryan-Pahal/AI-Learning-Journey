import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts  import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite",
    google_api_key = api_key
)

chat_history = []

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpfulcustomer support agent."),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

def chat(user_input):
    response = chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))

    return response.content

print("🤖 AI Customer Support (type 'quit' to exit)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    response = chat(user_input)
    print(f"AI: {response}\n")
