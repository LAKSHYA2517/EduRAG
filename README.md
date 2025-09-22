📘 AI Curriculum Assistant

An AI-powered tool to help students and advisors easily navigate through scattered academic materials like syllabi, lecture notes, and course catalogs. By uploading curriculum documents, the system builds a personalized Retrieval-Augmented Generation (RAG) pipeline that allows users to ask complex academic queries and get accurate, context-grounded answers.

🚀 Problem Statement

Navigating a college curriculum is often overwhelming due to information being scattered across disconnected documents. Students struggle to answer simple but important questions such as:

Which courses satisfy the quantitative reasoning requirement and also cover topics in AI?

What were the key concepts from the first half of my Networks course?

This decentralized nature of information makes academic planning and guidance time-consuming.

💡 Solution Overview

The AI Curriculum Assistant provides a centralized, intelligent knowledge base for all academic materials. Students and advisors can upload documents in PDF or .txt format, which are processed into a searchable vector database. Queries are answered with context-aware AI responses, strictly grounded in the uploaded curriculum.

✨ Key Features

📂 Multi-Document Upload – Upload syllabi, notes, and course catalogs.

🔍 Unified Knowledge Base – All materials indexed in a single vector store.

🤖 Contextual Q&A – Ask complex academic questions that span multiple documents.

⚡ In-Memory Vector Store – Fast performance (FAISS/Chroma) for hackathon prototyping.

📑 Source Referencing (Stretch Goal) – Display snippets and document references for transparency.

🛠 Tech Stack
Backend & AI Pipeline

Python / Node.js

LangChain or LlamaIndex

FAISS or ChromaDB

Gemini API (LLM & Embeddings)

PyMuPDF / pdfplumber (PDF parsing)

Frontend & UI

React.js, HTML, CSS

Streamlit / Gradio (for quick prototype UI)

Figma (for design prototyping)

⚙️ High-Level Workflow

Upload → User uploads .pdf or .txt curriculum documents.

Load & Chunk → Documents split into meaningful chunks.

Embed & Store → Chunks converted into vectors & stored in FAISS.

Query → User asks a question.

Retrieve → Similarity search finds most relevant document chunks.

Generate → AI synthesizes a context-grounded response.

Output → User sees the final answer (with references in stretch goals).

⚡ Hackathon Goals

Build a working prototype that can handle at least 2 different documents.

Accurately answer 8/10 test questions about uploaded content.

🔮 Future Improvements

Better handling of diverse document formats.

Scalable vector databases for large user bases.

Advanced analytics and recommendation features for academic planning.

👨‍💻 Team Members

Aryan Choudhary

Lakshay Asani

Pranavi Gupta

Abhinav Sharma
