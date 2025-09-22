# --- Cell 1: Imports and Environment Setup ---
import os
import re
import base64
import io
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from typing import List, Dict
import fitz  # PyMuPDF
from PIL import Image
from langchain.retrievers import ContextualCompressionRetriever
# ⬇️ MODIFIED: The reranker class now needs its specific import
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain import hub
from langchain_experimental.tools import PythonREPLTool
from langchain_community.cross_encoders import HuggingFaceCrossEncoder




os.makedirs('./documents', exist_ok=True)


# Load environment variables from .env file
load_dotenv()


# --- Cell 2: Configuration ---
DOCS_PATH = "./documents"
CHROMA_PERSIST_PATH = "./chroma_db"
# ⬇️ MODIFIED: This variable now MANDATES a single file to be processed.
# The script will raise an error if this is empty or the file is not found.
PROCESS_SPECIFIC_FILE = "DataStructures.pdf" # Example: "DataStructures.pdf"

EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
LLM_MODEL = "gemini-1.5-flash-latest" 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# --- Cell 3: Helper Functions (Document Processors & Vector Store Setup) ---

def get_image_summary(image_bytes: bytes, llm: ChatGoogleGenerativeAI) -> str:
    """Generates a summary for an image using a multi-modal LLM."""
    print("Generating image summary...")
    prompt_messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "You are an expert at analyzing academic images, diagrams, and charts. Describe this image in detail. What is its main purpose? What key information does it convey? If it's a chart or graph, describe the data, axes, and trend. This summary will be used for a Retrieval-Augmented Generation (RAG) system, so be comprehensive."},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"}
            ]
        )
    ]
    try:
        response = llm.invoke(prompt_messages)
        return response.content
    except Exception as e:
        print(f"❌ Error generating image summary: {e}")
        return "Could not generate summary for this image."

class SmartPDFProcessor:
    def __init__(self, embeddings, llm=None):
        self.text_splitter = SemanticChunker(embeddings)
        self.llm = llm
    def process_pdf(self, pdf_path: str) -> List[Document]:
        print(f"Processing PDF with PyPDFLoader: {pdf_path}")
        all_docs = []
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            full_text = "\n\n".join([self._clean_text(page.page_content) for page in pages])
            chunks = self.text_splitter.create_documents([full_text])
            for chunk in chunks: chunk.metadata['source'] = pdf_path
            all_docs.extend(chunks)

            if self.llm:
                pdf_document = fitz.open(pdf_path)
                for page_num in range(len(pdf_document)):
                    for img_index, img in enumerate(pdf_document.get_page_images(page_num)):
                        xref, base_image = img[0], pdf_document.extract_image(img[0])
                        summary = get_image_summary(base_image["image"], self.llm)
                        all_docs.append(Document(page_content=summary, metadata={ "source": pdf_path, "page": page_num + 1, "chunk_method": "pdf_image_summary", "image_index": img_index }))
            print(f"✅ Successfully processed {len(all_docs)} chunks and summaries from {pdf_path}")
            return all_docs
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}"); return []
    def _clean_text(self, text: str) -> str: return re.sub(r'\s+', ' ', text).strip().replace("ﬁ", "fi").replace("ﬂ", "fl")

class SmartDocProcessor:
    def __init__(self, embeddings): self.text_splitter = SemanticChunker(embeddings)
    def process_document(self, doc_path: str) -> List[Document]:
        print(f"Processing document: {doc_path}")
        try:
            if doc_path.lower().endswith(".docx"): loader = Docx2txtLoader(doc_path)
            elif doc_path.lower().endswith(".txt"): loader = TextLoader(doc_path, encoding='utf-8')
            else: return []
            documents = loader.load()
            full_text = "\n\n".join([self._clean_text(doc.page_content) for doc in documents if len(self._clean_text(doc.page_content).strip()) >= 50])
            if not full_text: return []
            splits = self.text_splitter.create_documents([full_text])
            for split in splits: split.metadata.update({ "source": doc_path, "chunk_method": "semantic_chunker_text", "char_count": len(split.page_content)})
            print(f"✅ Successfully processed {len(splits)} chunks from {doc_path}")
            return splits
        except Exception as e: print(f"❌ Error processing {doc_path}: {e}"); return []
    def _clean_text(self, text: str) -> str: return re.sub(r'\s+', ' ', text).strip()

class SmartLatexProcessor:
    def __init__(self, embeddings): self.text_splitter = SemanticChunker(embeddings)
    def process_latex(self, tex_path: str) -> List[Document]:
        print(f"Processing LaTeX file: {tex_path}")
        try:
            loader = TextLoader(tex_path, encoding='utf-8')
            documents = loader.load()
            full_text, cleaned_text = "\n".join([doc.page_content for doc in documents]), self._clean_latex(full_text)
            if len(cleaned_text.strip()) < 100: return []
            splits = self.text_splitter.create_documents([cleaned_text])
            for split in splits: split.metadata.update({ "source": tex_path, "chunk_method": "semantic_chunker_latex", "char_count": len(split.page_content)})
            print(f"✅ Successfully processed {len(splits)} chunks from {tex_path}"); return splits
        except Exception as e: print(f"❌ Error processing {tex_path}: {e}"); return []
    def _clean_latex(self, text: str) -> str:
        if "\\begin{document}" in text: text = text.split("\\begin{document}")[1]
        text = re.sub(r"%.*?\n", "\n", text)
        text = re.sub(r"\\begin\{(?:figure|table|tabular|verbatim|lstlisting)\*?\}[\s\S]*?\\end\{(?:figure|table|tabular|verbatim|lstlisting)\*?\}", "", text, flags=re.MULTILINE)
        text = re.sub(r"\\documentclass(?:\[.*?\])?\{.*?\}|\\usepackage(?:\[.*?\])?\{.*?\}|\\(title|author|date|thanks)\{.*?\}", "", text, flags=re.DOTALL)
        text = re.sub(r"\\(maketitle|tableofcontents|listoffigures|listoftables|centering|newpage|section\*|subsection\*|subsubsection\*)\b|\\(begin|end)\{.*?\}", "", text)
        return re.sub(r'\s+', ' ', text).strip()

class SmartSheetProcessor:
    def process_sheet(self, sheet_path: str) -> List[Document]:
        try:
            if sheet_path.lower().endswith(".csv"): loader = CSVLoader(file_path=sheet_path, encoding='utf-8')
            elif sheet_path.lower().endswith(".xlsx"): loader = UnstructuredExcelLoader(sheet_path, mode="elements")
            else: return []
            return loader.load()
        except Exception as e: print(f"❌ Error processing {sheet_path}: {e}"); return []

class SmartPPTProcessor:
    def process_ppt(self, ppt_path: str) -> List[Document]:
        try: loader = UnstructuredPowerPointLoader(ppt_path, mode="elements"); return loader.load()
        except Exception as e: print(f"❌ Error processing {ppt_path}: {e}"); return []

# ⬇️ MODIFIED: This function is now simplified to only process one specified file.
def process_single_file(embedding_function, llm_for_summaries) -> List[Document]:
    """
    Processes a single file specified by the PROCESS_SPECIFIC_FILE variable.
    """
    if not PROCESS_SPECIFIC_FILE:
        raise ValueError("The 'PROCESS_SPECIFIC_FILE' variable is not set. Please specify a filename in Cell 2.")

    specific_file_path = os.path.join(DOCS_PATH, PROCESS_SPECIFIC_FILE)
    
    if not os.path.exists(specific_file_path):
        raise FileNotFoundError(f"The specified file '{PROCESS_SPECIFIC_FILE}' was not found in the '{DOCS_PATH}' directory.")

    print(f"--- 🎯 Processing specific file: {PROCESS_SPECIFIC_FILE} ---")
    
    all_splits, processors = [], {".pdf": SmartPDFProcessor(embeddings=embedding_function, llm=llm_for_summaries), ".txt": SmartDocProcessor(embeddings=embedding_function), ".docx": SmartDocProcessor(embeddings=embedding_function), ".tex": SmartLatexProcessor(embeddings=embedding_function), ".csv": SmartSheetProcessor(), ".xlsx": SmartSheetProcessor(), ".pptx": SmartPPTProcessor(), ".ppt": SmartPPTProcessor()}
    
    filename = PROCESS_SPECIFIC_FILE
    file_path, file_ext = os.path.join(DOCS_PATH, filename), os.path.splitext(filename)[1].lower()
    
    if file_ext in processors:
        processor = processors[file_ext]
        if hasattr(processor, 'process_pdf'): all_splits.extend(processor.process_pdf(file_path))
        elif hasattr(processor, 'process_document'): all_splits.extend(processor.process_document(file_path))
        elif hasattr(processor, 'process_latex'): all_splits.extend(processor.process_latex(file_path))
        elif hasattr(processor, 'process_sheet'): all_splits.extend(processor.process_sheet(file_path))
        elif hasattr(processor, 'process_ppt'): all_splits.extend(processor.process_ppt(file_path))
    else:
        print(f"❌ Warning: File type '{file_ext}' for file '{filename}' is not supported.")

    return all_splits


# --- Cell 4: RAG Chain and Tool Creation ---
def create_rag_chain(retriever, llm):
    contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_system_prompt = ("You are an expert AI Curriculum Assistant. Your task is to answer user questions accurately and concisely based ONLY on the provided context. This context contains text excerpts and detailed summaries of images, charts, or diagrams. When referencing visual content, explicitly mention it (e.g., 'As seen in the diagram...'). If the context does not contain the answer, state that you cannot find the information in the provided materials. Do not use any external knowledge.\n\nContext:\n{context}")
    qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# --- Cell 5: Assistant Tools and Features ---
progress_tracker: Dict[str, str] = {}
llm = None
retriever = None
all_documents_for_tools = []
python_repl_tool = None
rag_chain_for_tools = None

@tool
def curriculum_qa_tool(input: str, chat_history: List = []):
    """
    Use this tool to answer any question about the content of the uploaded curriculum documents.
    It is the primary tool for factual questions and information retrieval from the knowledge base.
    For example: 'What is photosynthesis?', 'Summarize the section on data structures.'
    """
    if rag_chain_for_tools is None: return "RAG chain not initialized."
    response = rag_chain_for_tools.invoke({"input": input, "chat_history": chat_history})
    return response['answer']

@tool
def python_math_solver(problem: str):
    """
    Use this specialized tool to solve mathematical problems, perform calculations, or answer word problems that require logic and computation.
    It works by writing and executing Python code. For example: 'What is 24 * 5?', 'If a car travels at 80 km/h for 2.5 hours, how far does it go?'.
    """
    if not llm or not python_repl_tool: return "Math solver components not initialized."
    print(f"--- 🐍 Solving math problem with Python: '{problem}' ---")
    code_gen_prompt_text = (
        "You are an expert Python programmer tasked with solving a mathematical problem. "
        "Based on the user's problem, write a single, executable block of Python code to find the solution. "
        "IMPORTANT:\n"
        "- Only output the Python code. Do not include any explanation, comments, or markdown formatting like ```python. "
        "- The code MUST print the final answer to the console using the print() function. "
        "- You can use standard libraries like 'math'.\n\n"
        "Problem: {problem}"
    )
    code_gen_prompt = ChatPromptTemplate.from_template(code_gen_prompt_text)
    code_gen_chain = code_gen_prompt | llm
    code_to_execute = code_gen_chain.invoke({"problem": problem}).content
    try:
        result = python_repl_tool.run(code_to_execute)
        return f"Solution:\n{result}"
    except Exception as e:
        return f"Error executing the Python code: {e}"

@tool
def enhanced_quiz_generator(topic: str, num_questions: int = 3, question_type: str = "multiple choice"):
    """
    Generates a quiz on a specific topic based on the curriculum. 
    Use this when the user explicitly asks for a quiz, test, or to check their knowledge.
    Arguments: topic (str), num_questions (int, default 3), question_type (str, default 'multiple choice').
    """
    if not retriever or not llm: return "Retriever or LLM not initialized."
    print(f"--- ❓ Generating a {num_questions}-question quiz on '{topic}'... ---")
    context_docs = retriever.invoke(topic)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    if not context_text.strip(): return f"Sorry, I couldn't find enough information on '{topic}' to create a quiz."
    quiz_prompt_text = (
        "You are an expert quiz creator. Based ONLY on the provided context, create a quiz with {num_questions} "
        "{question_type} questions about '{topic}'. For each multiple-choice question, provide 4 options (A, B, C, D) "
        "and clearly mark the correct answer. For other question types, provide the question and the correct answer based on the text.\n\n"
        "Context:\n---\n{context}\n---"
    )
    quiz_prompt = ChatPromptTemplate.from_template(quiz_prompt_text)
    quiz_chain = quiz_prompt | llm
    response = quiz_chain.invoke({"num_questions": num_questions, "topic": topic, "question_type": question_type, "context": context_text})
    return response.content

@tool
def learning_path_suggester(topic: str):
    """
    Suggests a step-by-step learning path for a given topic based on the provided documents.
    Use this when the user asks 'How should I study for X?', 'What's the learning path for Y?', or 'Suggest a study plan.'
    """
    if not llm: return "LLM not initialized."
    print(f"--- 🗺️ Generating a learning path for '{topic}'... ---")
    full_context = "\n\n".join([doc.page_content for doc in all_documents_for_tools])
    path_prompt_text = (
        "You are an expert academic advisor. Based on the entire curriculum provided, analyze the content related to '{topic}'. "
        "Create a logical, step-by-step learning path for a student to master this topic. Break it down into key concepts, "
        "suggesting the order in which they should be studied. Your output should be a clear, actionable list.\n\n"
        "Full Curriculum Context:\n---\n{context}\n---"
    )
    path_prompt = ChatPromptTemplate.from_template(path_prompt_text)
    path_chain = path_prompt | llm
    response = path_chain.invoke({"topic": topic, "context": full_context})
    return response.content

@tool
def mark_topic_as_studied(topic: str):
    """
    Marks a topic as 'studied' in the progress tracker. Use this when a user says they have finished studying a topic.
    """
    print(f"--- ✅ Marking '{topic}' as studied. ---")
    progress_tracker[topic.lower()] = "Studied"
    return f"Great! I've marked '{topic}' as studied. Keep up the great work!"

@tool
def view_study_progress(input: str = ""):
    """
    Shows the user's study progress. Use this when the user asks 'What have I studied?' or 'Show my progress.'
    This tool takes no arguments.
    """
    print("--- 📊 Displaying study progress... ---")
    if not progress_tracker:
        return "You haven't marked any topics as studied yet."
    progress_report = "Here's your study progress so far:\n"
    for topic, status in progress_tracker.items():
        progress_report += f"- {topic.capitalize()}: {status}\n"
    return progress_report


# --- Cell 6: Initialization and Main Execution ---
print("--- 🚀 AI Curriculum Assistant Initializing (Agent Mode) 🚀 ---")
if not GOOGLE_API_KEY:
    print("❌ Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
else:
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.5)
        
        # ⬇️ MODIFIED: Function call updated to the new single-file processor.
        all_documents_for_tools = process_single_file(embeddings, llm)
        if not all_documents_for_tools: raise ValueError("No documents were processed. Halting initialization.")

        if os.path.exists(CHROMA_PERSIST_PATH) and os.listdir(CHROMA_PERSIST_PATH):
            print(f"✅ Loading existing vector store from '{CHROMA_PERSIST_PATH}'...")
            vectorstore = Chroma(persist_directory=CHROMA_PERSIST_PATH, embedding_function=embeddings)
        else:
            print(f"⚠️ No existing vector store found. Creating and persisting a new one at '{CHROMA_PERSIST_PATH}'...")
            vectorstore = Chroma.from_documents(documents=all_documents_for_tools, embedding=embeddings, persist_directory=CHROMA_PERSIST_PATH)
        
        vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        bm25_retriever = BM25Retriever.from_documents(all_documents_for_tools, k=10)
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
        
        print(f"Initializing cross-encoder for reranking: '{CROSS_ENCODER_MODEL}'")
        model = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL)
        compressor = CrossEncoderReranker(model=model, top_n=3)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
        
        rag_chain_for_tools = create_rag_chain(retriever, llm)
        python_repl_tool = PythonREPLTool()
        
        tools = [curriculum_qa_tool, python_math_solver, enhanced_quiz_generator, learning_path_suggester, mark_topic_as_studied, view_study_progress]
        
        agent_prompt = hub.pull("hwchase17/react-chat")
        agent = create_react_agent(llm, tools, agent_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        
        chat_history = []
        print("\n✅ Multi-tool Assistant is ready! Ask me a curriculum question or give me a math problem to solve.")
        
    except (ValueError, FileNotFoundError) as e: print(f"❌ Error: {e}")
    except Exception as e: print(f"❌ An unexpected error occurred during initialization: {e}")


    #Cell 7: Interactive Chat Loop ---
if 'agent_executor' in locals():
    user_input = "Give C Code for Bubble Sort"
    
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    answer = response["output"]
    print(f"🤖 Assistant: {answer}")
    
    chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=answer),
    ])
else:
    print("The Agent Executor is not initialized. Please run the setup cells successfully.")