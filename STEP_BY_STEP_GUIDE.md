# Medical Chatbot with Pinecone & Gemini AI - Complete Step-by-Step Guide

## 🎯 Project Overview

This guide walks you through building an end-to-end medical chatbot using:

- **Flask** for web framework
- **Pinecone** for vector database
- **Google Gemini AI** for language model
- **LangChain** for RAG (Retrieval Augmented Generation)
- **HuggingFace** for embeddings

---

## 📋 Prerequisites

### 1. System Requirements

- Python 3.8 or higher
- Git installed
- Internet connection for API access

### 2. API Keys Required

- **Pinecone API Key**: Sign up at [pinecone.io](https://pinecone.io)
- **Google API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

---

## 🛠️ Step 1: Project Setup

### Create Project Directory

```bash
mkdir Ayucore-AI
cd Ayucore-AI
```

### Initialize Git Repository

```bash
git init
git remote add origin https://github.com/your-username/Ayucore-AI.git
```

---

## 📁 Step 2: Project Structure Creation

### Create `template.py` (Automated Setup Script)

```python
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_directory_structure():
    """Create the complete project directory structure"""

    directories = [
        "src",
        "templates",
        "static",
        "data",
        "research"
    ]

    files = {
        # Source files
        "src/__init__.py": "",
        "src/helper.py": """# Helper functions for PDF processing and embeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf_file(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
""",

        "src/prompt.py": """system_prompt = '''
You are a helpful medical assistant. Answer questions based on the provided medical context.
If you don't know the answer, say so clearly. Always recommend consulting healthcare professionals for serious concerns.

Context: {context}

Question: {input}
'''
""",

        # Flask application
        "app.py": """from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

# Load existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Use Gemini instead of OpenAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
    max_output_tokens=500
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        input = msg
        print(f"User input: {input}")

        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        print(f"Bot response: {answer}")
        return str(answer)

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try again or rephrase your question."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
""",

        # HTML Template
        "templates/chat.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Chatbot</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Medical Assistant Chatbot</h1>
            <p>Ask your medical questions and get informed responses</p>
        </div>

        <div class="chat-container" id="chatContainer">
            <div class="message bot-message">
                <div class="message-content">
                    Hello! I'm your medical assistant. How can I help you today?
                </div>
            </div>
        </div>

        <form class="input-form" id="messageForm">
            <div class="input-group">
                <input type="text" id="messageInput" name="msg" placeholder="Ask a medical question..." required>
                <button type="submit" id="sendButton">Send</button>
            </div>
        </form>
    </div>

    <script>
    $(document).ready(function(){
        $("#messageForm").submit(function(e){
            e.preventDefault();

            var userMessage = $("#messageInput").val();
            if(userMessage.trim() === "") return;

            // Add user message to chat
            $("#chatContainer").append(`
                <div class="message user-message">
                    <div class="message-content">${userMessage}</div>
                </div>
            `);

            // Clear input and show typing indicator
            $("#messageInput").val("");
            $("#chatContainer").append(`
                <div class="message bot-message typing" id="typingIndicator">
                    <div class="message-content">Typing...</div>
                </div>
            `);

            // Scroll to bottom
            $("#chatContainer").scrollTop($("#chatContainer")[0].scrollHeight);

            $.ajax({
                type: "POST",
                url: "/get",
                data: {msg: userMessage},
                success: function(response){
                    $("#typingIndicator").remove();
                    $("#chatContainer").append(`
                        <div class="message bot-message">
                            <div class="message-content">${response}</div>
                        </div>
                    `);
                    $("#chatContainer").scrollTop($("#chatContainer")[0].scrollHeight);
                },
                error: function(){
                    $("#typingIndicator").remove();
                    $("#chatContainer").append(`
                        <div class="message bot-message error">
                            <div class="message-content">Sorry, something went wrong. Please try again.</div>
                        </div>
                    `);
                    $("#chatContainer").scrollTop($("#chatContainer")[0].scrollHeight);
                }
            });
        });
    });
    </script>
</body>
</html>
""",

        # CSS Styles
        "static/style.css": """* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

.container {
    width: 90%;
    max-width: 800px;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    backdrop-filter: blur(10px);
}

.header {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 2rem;
    text-align: center;
}

.header h1 {
    margin-bottom: 0.5rem;
    font-size: 2rem;
}

.header p {
    opacity: 0.9;
    font-size: 1rem;
}

.chat-container {
    height: 400px;
    overflow-y: auto;
    padding: 1rem;
    background: #f8f9fa;
}

.message {
    margin-bottom: 1rem;
    display: flex;
}

.user-message {
    justify-content: flex-end;
}

.bot-message {
    justify-content: flex-start;
}

.message-content {
    max-width: 70%;
    padding: 0.75rem 1rem;
    border-radius: 18px;
    font-size: 0.9rem;
    line-height: 1.4;
}

.user-message .message-content {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.bot-message .message-content {
    background: white;
    color: #333;
    border: 1px solid #e0e0e0;
}

.typing .message-content {
    background: #e9ecef;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.input-form {
    padding: 1rem;
    background: white;
    border-top: 1px solid #e0e0e0;
}

.input-group {
    display: flex;
    gap: 0.5rem;
}

#messageInput {
    flex: 1;
    padding: 0.75rem 1rem;
    border: 2px solid #e0e0e0;
    border-radius: 25px;
    font-size: 1rem;
    outline: none;
    transition: border-color 0.3s;
}

#messageInput:focus {
    border-color: #667eea;
}

#sendButton {
    padding: 0.75rem 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 25px;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 600;
    transition: transform 0.2s;
}

#sendButton:hover {
    transform: translateY(-2px);
}

.chat-container::-webkit-scrollbar {
    width: 6px;
}

.chat-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

.chat-container::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

.chat-container::-webkit-scrollbar-thumb:hover {
    background: #a1a1a1;
}
""",

        # Vector Database Setup
        "store_index.py": """from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

print("Loading PDF documents...")
extracted_data = load_pdf_file(data='data/')

print("Splitting text into chunks...")
text_chunks = text_split(extracted_data)
print(f"Created {len(text_chunks)} text chunks")

print("Downloading embeddings model...")
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# Check if index exists, if not create it
if index_name not in [idx.name for idx in pc.list_indexes()]:
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
    print("Index created successfully!")
else:
    print(f"Index '{index_name}' already exists")

print("Creating vector store and uploading documents...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("✅ Vector database setup complete!")
print(f"Uploaded {len(text_chunks)} document chunks to Pinecone index '{index_name}'")
""",

        # Environment file template
        ".env": """# API Keys - Replace with your actual keys
PINECONE_API_KEY=your_pinecone_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
""",

        # Requirements
        "requirements.txt": """flask==3.0.0
python-dotenv==1.0.0
langchain==0.1.0
langchain-community==0.0.10
langchain-pinecone==0.0.1
langchain-google-genai==0.0.6
pinecone-client==3.0.0
sentence-transformers==2.2.2
pypdf==3.17.4
numpy<2
huggingface_hub==0.20.0
""",

        # Test script
        "test_setup.py": """#!/usr/bin/env python3
\"\"\"
Test script to verify all components are working
\"\"\"

import os
from dotenv import load_dotenv

print("🧪 Testing Medical Chatbot Setup...")
print("=" * 50)

# Test 1: Environment Variables
print("\\n1️⃣ Testing Environment Variables...")
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

if PINECONE_API_KEY and PINECONE_API_KEY != 'your_pinecone_api_key_here':
    print("✅ Pinecone API key found")
else:
    print("❌ Pinecone API key missing or default")

if GOOGLE_API_KEY and GOOGLE_API_KEY != 'your_google_api_key_here':
    print("✅ Google API key found")
else:
    print("❌ Google API key missing or default")

# Test 2: Dependencies
print("\\n2️⃣ Testing Dependencies...")
try:
    import flask
    print("✅ Flask imported successfully")
except ImportError:
    print("❌ Flask not installed")

try:
    from langchain_pinecone import PineconeVectorStore
    print("✅ LangChain Pinecone imported successfully")
except ImportError:
    print("❌ LangChain Pinecone not installed")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("✅ LangChain Google GenAI imported successfully")
except ImportError:
    print("❌ LangChain Google GenAI not installed")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ Sentence Transformers imported successfully")
except ImportError:
    print("❌ Sentence Transformers not installed")

# Test 3: File Structure
print("\\n3️⃣ Testing File Structure...")
required_files = [
    'src/helper.py',
    'src/prompt.py',
    'templates/chat.html',
    'static/style.css',
    'app.py',
    'store_index.py'
]

for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✅ {file_path} exists")
    else:
        print(f"❌ {file_path} missing")

# Test 4: Pinecone Connection
if PINECONE_API_KEY and PINECONE_API_KEY != 'your_pinecone_api_key_here':
    print("\\n4️⃣ Testing Pinecone Connection...")
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        indexes = pc.list_indexes()
        print(f"✅ Connected to Pinecone. Available indexes: {[idx.name for idx in indexes]}")
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")

# Test 5: Gemini AI
if GOOGLE_API_KEY and GOOGLE_API_KEY != 'your_google_api_key_here':
    print("\\n5️⃣ Testing Gemini AI...")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.4,
            max_output_tokens=500,
            google_api_key=GOOGLE_API_KEY
        )
        response = llm.invoke("Hello, test message")
        print("✅ Gemini AI connection successful")
    except Exception as e:
        print(f"❌ Gemini AI connection failed: {e}")

print("\\n" + "=" * 50)
print("🏁 Test Complete!")
""",

        # Setup checker
        "check_pinecone.py": """#!/usr/bin/env python3
\"\"\"
Simple script to check and fix Pinecone index
\"\"\"

print("🔍 Checking Pinecone Index...")

try:
    from dotenv import load_dotenv
    from pinecone import Pinecone
    import os

    load_dotenv()
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

    if not PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY not found in .env file")
        exit(1)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes]

    print(f"📋 Available indexes: {index_names}")

    if "medicalbot" in index_names:
        print("✅ 'medicalbot' index exists!")
        print("🎉 Your Pinecone setup is ready!")
    else:
        print("⚠️  'medicalbot' index not found.")
        print("📝 Run: python store_index.py to create it")

except Exception as e:
    print(f"❌ Error: {e}")
""",

        # Data directory placeholder
        "data/.gitkeep": "# This file ensures the data directory is tracked by git",

        # README
        "README.md": """# Medical Chatbot with Pinecone & Gemini AI

A sophisticated medical assistant chatbot built with Flask, Pinecone vector database, and Google Gemini AI.

## Features
- 🏥 Medical knowledge base powered by PDF documents
- 🧠 AI responses using Google Gemini
- 🔍 Vector similarity search with Pinecone
- 💬 Real-time chat interface
- 🎨 Modern, responsive UI

## Quick Start
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up API keys in `.env` file
4. Add medical PDF to `data/` folder
5. Run `python store_index.py` to create vector database
6. Start the app: `python app.py`
7. Visit `http://localhost:8080`

## API Keys Required
- Pinecone API Key from [pinecone.io](https://pinecone.io)
- Google API Key from [Google AI Studio](https://makersuite.google.com/app/apikey)

See `STEP_BY_STEP_GUIDE.md` for detailed setup instructions.
""",

        # License
        "LICENSE": """MIT License

Copyright (c) 2025 Ayucore AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""",

        # Setup script
        "setup.py": """from setuptools import setup, find_packages

setup(
    name="ayucore-ai",
    version="1.0.0",
    description="Medical Chatbot with Pinecone & Gemini AI",
    author="Ayucore AI",
    packages=find_packages(),
    install_requires=[
        "flask>=3.0.0",
        "python-dotenv>=1.0.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-pinecone>=0.0.1",
        "langchain-google-genai>=0.0.6",
        "pinecone-client>=3.0.0",
        "sentence-transformers>=2.2.2",
        "pypdf>=3.17.4",
        "numpy<2",
        "huggingface_hub>=0.20.0",
    ],
    python_requires=">=3.8",
)
"""
    }

    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

    # Create files
    for file_path, content in files.items():
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logging.info(f"Created file: {file_path}")

    print("🎉 Project structure created successfully!")
    print("\\n📝 Next steps:")
    print("1. Update .env file with your API keys")
    print("2. Add medical PDF files to data/ directory")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Run: python store_index.py (to create vector database)")
    print("5. Run: python app.py (to start the application)")

if __name__ == "__main__":
    create_directory_structure()
```

### Run the Template Script

```bash
python template.py
```

---

## 🔧 Step 3: Environment Setup

### Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Step 4: API Configuration

### Update `.env` File

Replace the placeholder values in `.env`:

```env
# API Keys - Replace with your actual keys
PINECONE_API_KEY=pcsk_your_actual_pinecone_key_here
GOOGLE_API_KEY=AIzaSy_your_actual_google_key_here
```

### Get API Keys:

#### Pinecone API Key:

1. Visit [pinecone.io](https://pinecone.io)
2. Sign up/Login
3. Go to API Keys section
4. Copy your API key

#### Google API Key:

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create new API key
3. Copy the key

---

## 📚 Step 5: Add Medical Data

### Add PDF Documents

1. Place your medical PDF files in the `data/` directory
2. Example: `data/Medical_book.pdf`

### Supported Formats

- PDF files (.pdf)
- Multiple PDFs can be added to the same directory

---

## 🗄️ Step 6: Vector Database Setup

### Create Pinecone Index and Upload Data

```bash
python store_index.py
```

This script will:

- Load PDF documents from `data/` directory
- Split text into chunks
- Create embeddings using HuggingFace model
- Create Pinecone index named "medicalbot"
- Upload all document chunks to the vector database

### Verify Setup

```bash
python check_pinecone.py
```

---

## 🧪 Step 7: Testing

### Run Comprehensive Tests

```bash
python test_setup.py
```

This will test:

- Environment variables
- Dependencies
- File structure
- Pinecone connection
- Gemini AI connection

---

## 🚀 Step 8: Launch Application

### Start the Flask Server

```bash
python app.py
```

### Access the Application

- Open browser and go to: `http://localhost:8080`
- You should see the medical chatbot interface

---

## 📱 Step 9: Using the Chatbot

### Sample Questions to Test:

1. "What are the symptoms of diabetes?"
2. "How to treat high blood pressure?"
3. "What causes heart disease?"
4. "Explain different types of antibiotics"

### Expected Behavior:

- Bot responds based on your uploaded medical documents
- Responses include relevant context from the PDFs
- Professional medical advice disclaimer

---

## 🔧 Step 10: Troubleshooting

### Common Issues and Solutions:

#### 1. NumPy Compatibility Error

```bash
# If you get NumPy errors, downgrade:
pip install "numpy<2"
```

#### 2. Gemini Model Not Found

- Ensure you're using `gemini-1.5-flash` model name
- Check your Google API key is valid

#### 3. Pinecone Connection Issues

```bash
# Test Pinecone connection:
python check_pinecone.py
```

#### 4. Dependency Conflicts

```bash
# Create fresh virtual environment:
deactivate
rm -rf venv
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

## 📊 Step 11: File Structure Overview

```
Ayucore-AI/
├── src/
│   ├── __init__.py
│   ├── helper.py          # PDF processing & embeddings
│   └── prompt.py          # System prompts
├── templates/
│   └── chat.html          # Web interface
├── static/
│   └── style.css          # Styling
├── data/
│   └── Medical_book.pdf   # Your medical documents
├── research/
│   └── trials.ipynb       # Jupyter notebooks
├── app.py                 # Main Flask application
├── store_index.py         # Vector database setup
├── test_setup.py          # Testing script
├── check_pinecone.py      # Pinecone verification
├── template.py            # Project setup script
├── requirements.txt       # Dependencies
├── .env                   # API keys (keep private)
├── README.md              # Project documentation
└── LICENSE               # License file
```

---

## 🔄 Step 12: Version Control

### Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial commit: Medical chatbot setup"
```

### Create GitHub Repository

1. Go to GitHub and create new repository named "Ayucore-AI"
2. Add remote origin:

```bash
git remote add origin https://github.com/your-username/Ayucore-AI.git
git branch -M main
git push -u origin main
```

### Save Changes to GitHub

```bash
# Add all changes
git add .

# Commit with message
git commit -m "Complete medical chatbot with Pinecone and Gemini AI integration"

# Push to GitHub
git push origin main
```

---

## 🔒 Step 13: Security Best Practices

### Protect API Keys

1. Never commit `.env` file to GitHub
2. Add `.env` to `.gitignore`:

```bash
echo ".env" >> .gitignore
```

### Environment Variables in Production

For deployment, set environment variables directly on the server:

```bash
export PINECONE_API_KEY=your_key_here
export GOOGLE_API_KEY=your_key_here
```

---

## 🚀 Step 14: Deployment (Optional)

### Local Development

```bash
python app.py
```

### Production Deployment Options:

1. **Heroku**: Use Procfile
2. **AWS EC2**: Deploy on virtual server
3. **DigitalOcean**: App platform deployment
4. **Vercel**: Serverless deployment

---

## 📈 Step 15: Customization

### Modify System Prompt

Edit `src/prompt.py` to change bot behavior:

```python
system_prompt = '''
Your custom medical assistant instructions here...
'''
```

### Adjust Model Parameters

In `app.py`, modify:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,        # Creativity (0-1)
    max_output_tokens=500   # Response length
)
```

### Change UI Theme

Edit `static/style.css` for different colors and layout.

---

## ✅ Final Checklist

- [ ] Project structure created
- [ ] Dependencies installed
- [ ] API keys configured
- [ ] Medical PDFs added to data/
- [ ] Vector database created
- [ ] Application tested locally
- [ ] Code committed to GitHub
- [ ] Documentation updated

---

## 🎉 Congratulations!

You've successfully built a complete medical chatbot with:

- ✅ AI-powered responses using Gemini
- ✅ Vector database for document search
- ✅ Professional web interface
- ✅ Secure API key management
- ✅ Version control with Git

Your medical chatbot is now ready to help users with medical questions based on your uploaded documents!

---

## 📞 Support

If you encounter any issues:

1. Check the troubleshooting section
2. Run `python test_setup.py` for diagnostics
3. Verify API keys are correctly set
4. Ensure all dependencies are installed

Happy coding! 🚀
