from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = "chroma_db"

model = "openai/gpt-oss-20b"
run_name = "CustomerSupportLangGraph"

LLM = ChatGroq(
    model=model,
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.2
)