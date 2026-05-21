import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite",
    google_api_key = api_key
)

classifier_prompt = PromptTemplate(
    input_variables = ["email"],
    template = "Classify this email as 'Purchase' or 'Complaint'. One word only \n\nEmail: {email}"
)
classifier_chain = classifier_prompt | llm | StrOutputParser()

reply_prompt = PromptTemplate(
    input_variables = ["category","email"],
    template = "You are a professional customer support agent. Write a polite reply under 100 wordsto this {category} email.\n\nEmail: {email}"
    )
reply_chain = reply_prompt | llm | StrOutputParser()

emails = [
    "I just bought your shoes and they are amazing!",
    "My order is delayed by 2 weeks and nobody is responding!",
]

for i, email in enumerate(emails):
    print(f"\n--- Email{i+1} ---")

    category = classifier_chain.invoke({"email": email}).strip()
    print(f"Category: {category}")

    reply = reply_chain.invoke({"category": category, "email": email})
    print(f"Reply: {reply}")

    if category.lower() == "complaint":
        print("🚨 Alert: Complaint detected!")