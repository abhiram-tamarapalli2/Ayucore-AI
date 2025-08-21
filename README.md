# Ayucore-AI Medical Chatbot - Advanced RAG Application

🩺 **Professional Medical Consultation Chatbot** built with **Pinecone Vector Database** and **Google Gemini AI** using advanced RAG (Retrieval Augmented Generation) architecture.

## 🏗️ Project Architecture

```
Patient Input → Educational Detection → Vector Search (Pinecone) → Medical Knowledge → Gemini AI → Personalized Medical Response
```

## 🌟 Key Features

- 🩺 **Doctor-like Consultation**: Progressive questioning and symptom analysis
- 📚 **Educational System**: Comprehensive "what is" medical guides with spelling tolerance
- 👤 **Patient Management**: Name, age, gender integration throughout consultation
- 🧠 **Intelligent Responses**: Detailed medical explanations, treatments, and precautions
- ⚡ **Session Management**: Conversation history and question tracking
- 🛡️ **Error Handling**: Timeout protection and medical fallback responses
- 🎨 **Modern UI**: Professional medical interface with patient information panel

## ✅ Project Ready for Deployment

Your advanced medical chatbot is complete and ready to use! Here's the optimized project structure:

### 📁 Current Project Structure

```
Ayucore-AI/
├── 📱 CORE APPLICATION
│   ├── app.py                 # Flask backend with RAG chain
│   ├── requirements.txt       # Python dependencies
│   ├── setup.py              # Package configuration
│   └── .env                  # Environment variables
├── 💻 SOURCE CODE
│   └── src/
│       ├── __init__.py       # Package initialization
│       ├── helper.py         # PDF processing & embeddings
│       └── prompt.py         # System prompts & templates
├── 🎨 FRONTEND
│   ├── templates/
│   │   └── chat.html         # Medical chatbot interface
│   └── static/
│       └── style.css         # Modern UI styling
├── 🗃️ DATA & SETUP
│   ├── data/
│   │   └── Medical_book.pdf  # Medical knowledge base
│   ├── store_index.py        # Pinecone vector store setup
│   └── template.py           # Project structure template
├── 📚 DOCUMENTATION
│   ├── README.md             # This file
│   ├── PROJECT_DOCUMENTATION.txt    # Technical documentation
│   ├── PROJECT_DEMONSTRATION.txt    # Sample conversations
│   ├── PROJECT_STRUCTURE.txt        # Final structure guide
│   ├── ARCHITECTURE_DESIGN.md       # System architecture
│   ├── STEP_BY_STEP_GUIDE.md       # Setup guide
│   ├── DIAGRAMS_GUIDE.md           # System diagrams
│   └── LICENSE               # MIT License
└── 📊 SYSTEM DIAGRAMS
    └── diagrams/
        ├── component_diagram.puml     # Component architecture
        ├── data_flow_diagram.puml     # Data flow
        ├── deployment_diagram.puml    # Deployment
        ├── sequence_diagram.puml      # Interactions
        └── system_architecture.puml  # Overall design
```

### 🔑 API Keys Configuration

Set up your environment variables in `.env` file:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

**Getting API Keys:**

- **Pinecone**: Sign up at [pinecone.io](https://pinecone.io) (Free tier available)
- **Google Gemini**: Get API key from [Google AI Studio](https://aistudio.google.com)

### 📦 Installation & Setup

**1. Clone and Install Dependencies:**

```bash
git clone https://github.com/abhiram-tamarapalli2/Ayucore-AI
cd Ayucore-AI
pip install -r requirements.txt
```

**2. Required Packages:**

```bash
pip install sentence-transformers langchain flask pypdf python-dotenv pinecone-client langchain-pinecone langchain_community langchain_google_genai google-generativeai langchain_experimental
```

## 🚀 Quick Start Guide

### Step 1: Set Up Vector Database (One-time setup)

```bash
python store_index.py
```

This will:

- Extract text from the medical PDF in `data/` folder
- Create embeddings using HuggingFace Sentence Transformers
- Store vectors in Pinecone cloud database (index: "medicalbot")

### Step 2: Run the Medical Chatbot

```bash
python app.py
```

- **Access at**: http://localhost:8080
- **Features**: Professional medical consultation interface
- **AI**: Powered by Google Gemini 1.5 Flash

### Step 3: Start Medical Consultation

1. **Enter Patient Information**: Name, age, gender in the side panel
2. **Begin Consultation**: Describe symptoms or ask educational questions
3. **Interactive Dialogue**: AI asks progressive diagnostic questions
4. **Get Comprehensive Advice**: Detailed medical guidance and treatment plans

## 🩺 Medical Consultation Features

### 📋 **Doctor-like Consultation Process**

- Progressive symptom questioning (up to 5 targeted questions)
- Severity assessment and duration analysis
- Comprehensive medical evaluation
- Personalized treatment recommendations

### 📚 **Educational Medical Guides**

- Ask "What is [condition]?" for detailed medical information
- Spelling tolerance (e.g., "ferver" → "fever")
- Comprehensive disease explanations
- Treatment protocols and prevention strategies

### 👤 **Patient Information Management**

- Personal details integration throughout consultation
- Session-based conversation history
- Customized medical advice based on patient profile

### 🎯 **Professional Medical Responses**

- Detailed medication recommendations with dosages
- Comprehensive nutritional guidance
- Recovery timelines and monitoring instructions
- Emergency care criteria and red flag symptoms

## � Usage Examples

### Medical Consultation Examples:

```
User: "Hi, I have fever and headache for 2 days"
AI: "Hello! I need to understand your condition better. What specific symptoms are you experiencing? (fever, pain, cough, etc.)"

User: "I have fever around 101F, headache, body ache and little cough"
AI: "How long have you been experiencing these symptoms? This helps me determine if it's acute or chronic."
```

### Educational Queries:

```
User: "What is diabetes?"
AI: [Comprehensive medical guide with definition, types, symptoms, treatments, monitoring, and prevention]

User: "what is ferver" (spelling tolerance)
AI: [Complete fever guide with mechanisms, types, treatments, and care instructions]
```

### Patient Information Integration:

```
Patient: Sarah Johnson, Age: 28, Female
AI: "Hello Sarah! Based on your symptoms, this appears to be a viral upper respiratory infection..."
```

## � Advanced Configuration

### AI Model Settings (app.py)

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,          # Medical precision (0-1)
    max_output_tokens=1000,   # Detailed responses
    timeout=20               # Extended for comprehensive answers
)
```

### Vector Search Settings

```python
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}    # Number of relevant documents
)
```

### Consultation Behavior (app.py)

```python
session_data = {
    'questions_asked': 0,
    'max_questions': 5,       # Flexible question limit
    'consultation_active': True
}
```

## 📊 Technology Stack

- **🎨 Frontend**: HTML5, CSS3, JavaScript, Flask Templates
- **⚙️ Backend**: Python 3.12, Flask Web Framework
- **🧠 AI**: Google Gemini 1.5 Flash with LangChain
- **🗄️ Vector Database**: Pinecone (Cloud-based semantic search)
- **📄 Embeddings**: HuggingFace Sentence Transformers (384 dimensions)
- **📚 RAG Framework**: LangChain for document retrieval and chain management
- **💾 Session Management**: Flask sessions with conversation history
- **🔒 Security**: Environment variable API key management

## 🎯 Project Highlights

- **🩺 Professional Medical Consultation**: Simulates doctor-patient interaction
- **📚 Comprehensive Education System**: Detailed medical guides with spelling tolerance
- **👤 Patient-Centered Design**: Personal information integration throughout consultation
- **⚡ Advanced RAG Architecture**: Semantic search with medical knowledge base
- **🛡️ Robust Error Handling**: Timeout protection and medical fallback responses
- **📱 Modern UI/UX**: Professional medical interface with responsive design
- **📖 Complete Documentation**: Technical docs, demos, and architecture guides

## 📚 Documentation Files

- **`PROJECT_DOCUMENTATION.txt`**: Complete technical documentation
- **`PROJECT_DEMONSTRATION.txt`**: Sample conversations and feature demos
- **`PROJECT_STRUCTURE.txt`**: Final project structure and submission guide
- **`ARCHITECTURE_DESIGN.md`**: System architecture and design patterns
- **`STEP_BY_STEP_GUIDE.md`**: Detailed setup and usage instructions
- **`DIAGRAMS_GUIDE.md`**: System diagrams and PlantUML guides

## 🏥 Medical Knowledge Base

The system includes comprehensive medical knowledge covering:

- **Common Conditions**: Fever, cold, flu, headaches, migraines
- **Chronic Diseases**: Diabetes, hypertension, respiratory conditions
- **Treatment Protocols**: Medications, dosages, natural remedies
- **Emergency Care**: Red flag symptoms and when to seek help
- **Prevention**: Lifestyle recommendations and health maintenance

## 🔒 Security & Best Practices

- **🔑 API Key Security**: Environment variables in `.env` file (never commit to git)
- **🛡️ Input Validation**: Secure handling of user inputs and patient information
- **⏱️ Timeout Protection**: Prevents system hang with comprehensive error handling
- **🚨 Medical Disclaimers**: Appropriate medical advice warnings and limitations
- **📝 Session Security**: Secure session management with unique session IDs

## � Deployment Options

### Local Development

```bash
python app.py  # Runs on localhost:8080
```

### Production Deployment

- **Heroku**: Ready for Heroku deployment with Procfile
- **Docker**: Containerization support
- **Cloud Platforms**: AWS, GCP, Azure compatible
- **Environment Variables**: Production-ready configuration management

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/medical-enhancement`)
3. Commit changes (`git commit -am 'Add comprehensive diabetes guide'`)
4. Push to branch (`git push origin feature/medical-enhancement`)
5. Create Pull Request

## 📞 Support & Troubleshooting

### Common Issues:

1. **API Key Errors**: Verify `.env` file configuration
2. **Pinecone Connection**: Check API key and internet connection
3. **Package Issues**: Run `pip install -r requirements.txt`
4. **PDF Processing**: Ensure medical PDF is in `data/` folder

### Getting Help:

- **📖 Documentation**: Check `PROJECT_DOCUMENTATION.txt`
- **💬 Demo**: Review `PROJECT_DEMONSTRATION.txt` for examples
- **🏗️ Architecture**: See `ARCHITECTURE_DESIGN.md` for system details

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **Google Gemini AI** for advanced medical reasoning capabilities
- **Pinecone** for high-performance vector search
- **LangChain** for RAG framework and document processing
- **HuggingFace** for sentence transformer embeddings
- **Flask** for web application framework

---

🩺 **Your Ayucore-AI Medical Chatbot is ready to provide professional medical consultations!**

**Start the application and experience advanced AI-powered healthcare guidance with comprehensive medical knowledge and doctor-like consultation capabilities.**
