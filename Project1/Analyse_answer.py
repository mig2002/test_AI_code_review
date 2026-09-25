# For using NVIDIA NIM (LLM) models

# For building the LangGraph state machine/graph
from langgraph.graph import StateGraph, START, END, MessagesState
import os
from dotenv import load_dotenv
from model_call import model
import json
load_dotenv()

with open("Essay.json", "r") as f:
    Essay = json.load(f)
    for essay_id, essay_text in Essay.items():
        print(f"\n--- Evaluating {essay_id} ---")
        response = model(essay_text)
        print(f"Feedback and Score:\n{response}\n") 

