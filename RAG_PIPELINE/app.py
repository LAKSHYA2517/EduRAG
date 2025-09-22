# --- FastAPI Server for the RAG Pipeline ---
# This file provides the API endpoints to interact with the RAG agent.

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

# Import the core logic from our refactored script
import ragpipeline as rag_core

# --- Global Variables ---
# This will hold the single, active agent instance.
# A more advanced implementation could manage multiple agents for multiple users.
AGENT_EXECUTOR = None
CHAT_HISTORY = []
DOCS_PATH = "./documents" # Ensure this matches the core logic

# --- FastAPI App and Pydantic Models ---
app = FastAPI()

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str

class UploadResponse(BaseModel):
    message: str
    filename: str

# --- API Endpoints ---

@app.post("/upload", response_model=UploadResponse)
async def upload_and_process_file(file: UploadFile = File(...)):
    """
    Receives a file, saves it, and initializes the RAG agent for that file.
    This replaces the existing knowledge base with the new file's content.
    """
    global AGENT_EXECUTOR, CHAT_HISTORY
    
    # Ensure the documents directory exists
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        
    file_path = os.path.join(DOCS_PATH, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        # Initialize the agent with the new file. This is a blocking call.
        AGENT_EXECUTOR = rag_core.initialize_agent_for_file(file.filename)
        # Reset chat history for the new document
        CHAT_HISTORY = []
        return {"message": "File processed successfully. The assistant is ready.", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask_agent(request: AskRequest):
    """
    Receives a user query and returns the agent's response.
    This endpoint simulates the interaction from your Cell 7.
    """
    global AGENT_EXECUTOR, CHAT_HISTORY

    if AGENT_EXECUTOR is None:
        raise HTTPException(status_code=400, detail="No document has been processed yet. Please upload a file first via the /upload endpoint.")

    try:
        response = AGENT_EXECUTOR.invoke({
            "input": request.query,
            "chat_history": CHAT_HISTORY
        })
        
        answer = response["output"]

        # Update the server's chat history
        CHAT_HISTORY.extend([
            {"role": "user", "content": request.query},
            {"role": "assistant", "content": answer},
        ])
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while getting the answer: {str(e)}")

# --- Main Runner ---
if __name__ == "__main__":
    import uvicorn
    print("--- 🚀 Starting FastAPI RAG Server ---")
    print("--- Visit http://127.0.0.1:8000/docs for API documentation ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)
