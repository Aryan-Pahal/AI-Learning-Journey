from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

classifier_prompt = "You are an email classifier. You must reply with one word only: eithe 'Purchase' or 'Complaint'. No other words , no punctuation, no explaination."

#system promt for classifier
reply_prompt = """
You are a professional customer support agent for an online store .
Always be polite, enpathetic and helpful.
keep replies under 100 words.
Always end with: 'Team Support' """

emails = [
    "I just bought your shoes and they are amazing! Best purchase ever!",
    "My order has been delayed by 2 weeks. Nobody is responding. Very frustated!",
    "I love your brand but the product i received was completely wrong ."
]

for i, email in enumerate(emails):
    print (f"\n--- Email {i+1} ---")
    print( f"Email:{email}")

    # step1 : classify
    category = client.models.generate_content(
    model = "gemini-2.5-flash",
    config = types.GenerateContentConfig(
        system_instruction = classifier_prompt
    ),
    contents = f"classify: {email}"
    ).text.strip()

    print(f"Category: {category}")

    # step2 - generate reply
    reply = client.models.generate_content(
    model = "gemini-2.5-flash",
    config = types.GenerateContentConfig(
        system_instruction = reply_prompt
    ),
    contents = f"Write a reply to this {category} email: {email}"
    ).text.strip()

    print(f"Reply: {reply}")

    # step3 - Alert if complaint
    if category.lower() == "complaint":
        print("🚨 Alert: Complaint detected! Notify company!")


