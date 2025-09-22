# Your file: main.py

import os
import uuid  # Import the uuid library for generating session IDs
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
# Import your refactored module
import ragpipeline as rag_core 

# --- FastAPI App ---
app = FastAPI(
    title="AI Curriculum Assistant API",
    description="API with session management for the RAG agent."
)

# --- Pydantic Models ---
class AskRequest(BaseModel):
    query: str
    session_id: str  # The frontend MUST provide this

class AskResponse(BaseModel):
    answer: str
    session_id: str

# --- API Endpoints ---
@app.post("/upload")
async def upload_and_process_file(file: UploadFile = File(...)):
    """
    Receives a file, saves it, and initializes/re-initializes the RAG agent.
    This will clear all existing chat sessions.
    """
    file_path = os.path.join(rag_core.DOCS_PATH, file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        rag_core.initialize_agent_for_file(file.filename)
        return {"message": "File processed. The assistant is ready and all old sessions are cleared.", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.get("/new_session")
async def get_new_session():
    """
    Starts a new chat session and returns a unique session ID.
    The frontend should call this once when a new chat starts.
    """
    return {"session_id": str(uuid.uuid4())}

@app.post("/ask", response_model=AskResponse)
async def ask_agent(request: AskRequest):
    """
    Receives a user query for a specific session and returns the agent's response.
    """
    if not rag_core.assistant_state.get("agent_executor"):
        raise HTTPException(status_code=400, detail="No document processed. Please upload a file first.")

    try:
        answer = rag_core.get_agent_response(request.query, request.session_id)
        return AskResponse(answer=answer, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# --- Main Runner ---
if __name__ == "__main__":
    print("--- 🚀 Starting FastAPI RAG Server ---")
    print("--- Visit http://127.0.0.1:8000/docs for API documentation ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)