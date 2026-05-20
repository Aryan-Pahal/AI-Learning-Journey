
from google import genai
from dotenv import load_dotenv 
import os 

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) 

#a sample customer email
# email = """
# Hi, I recently bought your blue wireless headphones
# and I am very happy with the sound quality.
# Looking forward to buying more products!
# """

#A complain email
email = """
Hi, I recently bought your blue wireless headphones
and I am very unhappy with the sound quality.
I would like to return them.
"""

category_response = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents = f"Categorize this customer emailas either 'Purchase' or 'Complaint'.Reply in word only. \n\nEmail:{email}"
)
category = category_response.text.strip()

reply_response = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents = f"You are a professional customer support agent. Write a polite and helpful reply to this {category} email.\n\nEmail:{email}")
    

print("Category:", category)
print("\nAuto Reply:")
print(reply_response.text)

