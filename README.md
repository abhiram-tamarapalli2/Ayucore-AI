# Medical Chatbot - End-to-End RAG Application

A complete medical chatbot built with **Pinecone Vector Database** and **Google Gemini AI** using RAG (Retrieval Augmented Generation).

## 🏗️ Project Architecture

```
User Query → Vector Search (Pinecone) → Context Retrieval → Gemini AI → Response
```

## ✅ Setup Complete

Your project is ready to use! Here's what has been set up:

### 📁 Project Structure

```
d:/Ayucore-AI/
├── src/
│   ├── __init__.py
│   ├── helper.py          # PDF processing & embeddings
│   └── prompt.py          # System prompts
├── templates/
│   └── chat.html          # Web interface
├── static/
│   └── style.css          # Styling
├── Data/                  # 📂 ADD YOUR PDF FILES HERE
├── research/
│   ├── trials.ipynb       # Experiments
│   └── setup_test.ipynb   # Setup testing
├── .env                   # API keys (configured ✅)
├── app.py                 # Flask web app
├── store_index.py         # Vector database setup
└── requirements.txt       # Dependencies

```

### 🔑 API Keys Configured

- ✅ Pinecone API Key: Set up
- ✅ Google Gemini API Key: Set up

### 📦 Required Packages

```bash
pip install sentence-transformers langchain flask pypdf python-dotenv pinecone-client langchain-pinecone langchain_community langchain_google_genai google-generativeai langchain_experimental
```

## 🚀 How to Run

### Step 1: Add Medical PDF Files

1. Download medical reference PDFs (textbooks, medical guides, etc.)
2. Place them in the `Data/` folder
3. Supported formats: PDF files only

**Recommended Sources:**

- Medical textbooks (PDF format)
- Medical reference guides
- Healthcare documentation
- Clinical guidelines

### Step 2: Create Vector Database

```bash
python store_index.py
```

This will:

- Extract text from your PDFs
- Create embeddings using HuggingFace
- Store vectors in Pinecone cloud database

### Step 3: Run the Web Application

```bash
python app.py
```

- Opens at: http://localhost:8080
- Beautiful chat interface
- Powered by Gemini AI

## 🔧 Key Features

- **📚 Custom Medical Knowledge**: Uses your PDF documents
- **🧠 Vector Search**: Semantic search with Pinecone
- **🤖 AI Responses**: Google Gemini for natural answers
- **💬 Web Interface**: Beautiful Flask chat UI
- **⚡ Fast Retrieval**: Cloud-based vector database

## 📋 Usage Examples

Ask questions like:

- "What is diabetes?"
- "What are the symptoms of hypertension?"
- "How to treat fever?"
- "What medications are used for asthma?"

## 🛠️ Customization

### Change AI Model Settings

Edit `app.py`:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.4,        # Creativity (0-1)
    max_output_tokens=500   # Response length
)
```

### Modify System Prompt

Edit `src/prompt.py` to change how the AI responds.

### Adjust Chunk Settings

Edit `src/helper.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # Text chunk size
    chunk_overlap=20   # Overlap between chunks
)
```

## 📊 Technology Stack

- **Frontend**: HTML, CSS, JavaScript, Flask
- **Backend**: Python, LangChain
- **Vector DB**: Pinecone (Cloud)
- **AI Model**: Google Gemini Pro
- **Embeddings**: HuggingFace sentence-transformers
- **PDF Processing**: PyPDF

## 🎯 Next Steps

1. **Add PDF Files**: Place medical PDFs in `Data/` folder
2. **Run Setup**: Execute `python store_index.py`
3. **Start Chatbot**: Run `python app.py`
4. **Test**: Visit http://localhost:8080

## 🔒 Security Notes

- API keys are stored in `.env` file
- Don't commit `.env` to version control
- Keep your Pinecone and Gemini keys secure

## 📞 Support

If you encounter any issues:

1. Check that all packages are installed
2. Verify API keys are correctly set
3. Ensure PDF files are in the Data folder
4. Check terminal output for error messages

---

🎉 **Your medical chatbot is ready to use!** Add your medical PDFs and start chatting with your AI medical assistant.
