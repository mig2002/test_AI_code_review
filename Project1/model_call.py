from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import os
load_dotenv()


def model(essay_text)-> str:
    Evaluator = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=os.environ["NVIDIA_API_KEY"], 
    temperature=0.2,


    
    top_p=0.7,
    max_tokens=1024,
    )
    prompt = f"You are an UPSC Evaluator, Evaluate this essay: {essay_text} and give the feedback and score only. The feedback should be around 50 words and the score should be from 0 to 10."
    response = Evaluator.invoke(prompt)
    return response.content

