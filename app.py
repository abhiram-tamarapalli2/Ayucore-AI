from flask import Flask, render_template, request, jsonify, session, send_from_directory
import os
from datetime import datetime
from src.helper import download_hugging_face_embeddings
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize components with error handling
try:
    embeddings = download_hugging_face_embeddings()
    
    # Pinecone initialization
    index_name = "medicalbot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Initialize Google Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # Create retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    AI_AVAILABLE = True
    print("[SUCCESS] AI components initialized successfully!")
    
except Exception as e:
    print(f"[WARNING] AI initialization failed: {str(e)}")
    print("Running in fallback mode with basic responses...")
    AI_AVAILABLE = False
    llm = None
    retrieval_chain = None

def get_fallback_response(user_message):
    """Provide comprehensive medical guidance with detailed format"""
    user_message_lower = user_message.lower()
    
    if any(word in user_message_lower for word in ['fever', 'temperature', 'hot', 'chills']):
        return """**🩺 FEVER - Complete Medical Guide**

**📋 What's Happening:**
Fever is your body's natural immune response to infections. When pathogens invade, your hypothalamus raises body temperature to create an inhospitable environment for viruses and bacteria while boosting immune cell function.

**💊 Treatment & Medications:**
**Immediate Relief:**
• **Acetaminophen (Tylenol):** 500-1000mg every 6-8 hours (max 3000mg/day)
• **Ibuprofen (Advil):** 400-600mg every 6-8 hours (max 2400mg/day)
• **Aspirin:** 325-650mg every 4 hours (adults only, not for children)

**Alternative Treatments:**
• **Cool compress** on forehead and wrists for 10-15 minutes
• **Lukewarm bath** (not cold - can cause shivering and raise temperature)
• **Willow bark tea** (natural salicin, aspirin-like effects)

**🥗 Nutritional Therapy:**
**Foods to Include:**
• **Chicken soup** (proven to reduce inflammation and provide electrolytes)
• **Citrus fruits** (vitamin C: oranges, lemons, grapefruits)
• **Ginger tea** (anti-inflammatory, reduces nausea)
• **Coconut water** (natural electrolytes, easy to digest)
• **Bananas** (potassium, easy on stomach)
• **Bone broth** (minerals, amino acids for recovery)

**Foods to Avoid:**
• **Alcohol** (dehydrates and interferes with immune function)
• **Caffeine** (can worsen dehydration)
• **Heavy, fatty foods** (hard to digest when sick)
• **Sugary foods** (can suppress immune function)

**Supplements & Vitamins:**
• **Vitamin C:** 1000mg daily (immune support)
• **Zinc:** 15-30mg daily (antiviral properties)
• **Elderberry:** 10ml syrup 2x daily (reduces duration)

**⚠️ Important Precautions:**
**Warning Signs to Watch:**
• Temperature above 103°F (39.4°C)
• Severe headache with neck stiffness
• Difficulty breathing or chest pain
• Persistent vomiting preventing fluid intake
• Signs of dehydration (dizziness, dark urine)

**When to Seek Emergency Care:**
• Temperature above 104°F (40°C)
• Severe difficulty breathing
• Chest pain or rapid heartbeat
• Confusion or altered mental state
• Severe dehydration symptoms

**🔄 Recovery Timeline:**
• **Day 1-2:** Fever peaks, focus on rest and hydration
• **Day 3-4:** Temperature should start declining with treatment
• **Day 5-7:** Most fevers resolve completely
• **Week 2:** Full energy and strength return

**🏠 Self-Care Protocol:**
• **Rest:** Sleep 8-10 hours, avoid strenuous activity
• **Hydration:** 8-10 glasses of water daily, herbal teas
• **Cool environment:** Keep room temperature around 65-70°F
• **Light clothing:** Dress in breathable, lightweight fabrics
• **Monitor temperature:** Check every 4 hours

**🛡️ Prevention for Future:**
• **Hand washing:** 20 seconds with soap frequently
• **Immune support:** Regular vitamin D, C, zinc
• **Sleep hygiene:** 7-8 hours nightly
• **Stress management:** Chronic stress weakens immunity
• **Vaccination:** Stay current with flu and other vaccines

**📞 Follow-up Care:**
• **Monitor closely** if fever lasts more than 3 days
• **See doctor** if accompanied by severe symptoms
• **Blood tests** may be needed if fever persists over 7 days

*This comprehensive guide provides thorough medical information. Most fevers resolve with proper care in 2-3 days! 🌡️➡️😊*"""
    
    elif any(word in user_message_lower for word in ['headache', 'head pain', 'migraine']):
        return """**🩺 HEADACHE - Complete Medical Guide**

**📋 What's Happening:**
Headaches result from various triggers affecting blood vessels, muscles, and nerves in your head and neck. Most are tension-type (muscle contraction) or vascular (blood vessel changes). Understanding your type helps target treatment effectively.

**💊 Treatment & Medications:**
**Immediate Relief:**
• **Ibuprofen:** 400-600mg every 8 hours (anti-inflammatory, ideal for tension headaches)
• **Acetaminophen:** 1000mg every 6 hours (effective for mild-moderate pain)
• **Aspirin:** 325-650mg every 4 hours (good for vascular headaches)
• **Naproxen:** 220mg every 12 hours (longer-lasting relief)

**Alternative Treatments:**
• **Cold compress:** 15-20 minutes on forehead/temples for migraines
• **Warm compress:** On neck/shoulders for tension headaches
• **Peppermint oil:** Diluted, applied to temples (natural muscle relaxant)
• **Magnesium:** 400mg daily (prevents migraine recurrence)

**🥗 Nutritional Therapy:**
**Foods to Include:**
• **Magnesium-rich foods:** Almonds, spinach, avocados, dark chocolate
• **Ginger tea:** 2-3 cups daily (anti-inflammatory, nausea relief)
• **Peppermint tea:** Natural muscle relaxant and pain reliever
• **Quinoa:** Complex carbs stabilize blood sugar
• **Fatty fish:** Omega-3s reduce inflammation (salmon, mackerel)
• **Cherries:** Natural melatonin for sleep-related headaches

**Foods to Avoid:**
• **Tyramine foods:** Aged cheese, wine, processed meats (migraine triggers)
• **MSG:** Chinese food, processed snacks
• **Artificial sweeteners:** Aspartame can trigger headaches
• **Caffeine excess:** More than 2 cups coffee daily
• **Alcohol:** Especially red wine (histamine trigger)

**Supplements & Vitamins:**
• **Magnesium glycinate:** 400-600mg daily (most absorbable form)
• **Riboflavin (B2):** 400mg daily (reduces migraine frequency)
• **CoQ10:** 100mg 3x daily (cellular energy, migraine prevention)
• **Feverfew:** 100mg daily (traditional migraine herb)

**⚠️ Important Precautions:**
**Warning Signs to Watch:**
• Sudden, severe "thunderclap" headache
• Headache with fever and neck stiffness
• Vision changes or speech problems
• Weakness on one side of body
• Headache after head injury

**When to Seek Emergency Care:**
• Worst headache of your life (sudden onset)
• Headache with high fever (104°F+)
• Changes in vision, speech, or coordination
• Severe headache with vomiting
• Headache with confusion or memory loss

**🔄 Recovery Timeline:**
• **0-2 hours:** Medication takes effect, symptoms ease
• **2-6 hours:** Most tension headaches resolve completely
• **6-24 hours:** Migraines may take longer but improve significantly
• **Ongoing:** Preventive measures reduce future episodes

**🏠 Self-Care Protocol:**
• **Dark, quiet room:** Light and sound sensitivity relief
• **Sleep:** 7-8 hours nightly, consistent schedule
• **Hydration:** Dehydration is a common trigger
• **Neck massage:** Gentle circular motions, 5-10 minutes
• **Deep breathing:** 4-7-8 technique for stress relief
• **Regular meals:** Low blood sugar triggers headaches

**🛡️ Prevention for Future:**
• **Stress management:** Yoga, meditation, regular exercise
• **Sleep schedule:** Same bedtime/wake time daily
• **Trigger identification:** Keep headache diary
• **Regular exercise:** 30 minutes, 3x weekly
• **Ergonomic workspace:** Proper monitor height, lighting

**📞 Follow-up Care:**
• **See doctor** if headaches become more frequent
• **Neurologist referral** for chronic migraines (15+ days/month)
• **Blood pressure check** if headaches worsen
• **Eye exam** if visual symptoms accompany headaches

*Most headaches respond excellently to proper treatment! Relief typically comes within 1-2 hours. 🤕➡️😊*"""
    
    elif any(word in user_message_lower for word in ['cold', 'cough', 'runny nose', 'congestion', 'flu', 'sore throat']):
        return """**🩺 COLD/FLU - Complete Medical Guide**

**📋 What's Happening:**
Common cold is caused by rhinoviruses, while flu is from influenza viruses. Both are upper respiratory infections that trigger inflammatory responses. Your immune system creates symptoms while fighting off the virus - this is actually a sign your body is working properly!

**💊 Treatment & Medications:**
**Immediate Relief:**
• **Decongestants:** Pseudoephedrine 30mg every 6 hours (clears nasal passages)
• **Expectorants:** Guaifenesin 400mg every 12 hours (thins mucus)
• **Cough suppressants:** Dextromethorphan 15mg every 4 hours (dry cough)
• **Pain relief:** Acetaminophen 1000mg every 6 hours (body aches)
• **Throat lozenges:** Menthol/benzocaine for sore throat

**Alternative Treatments:**
• **Saline nasal rinse:** Neti pot 2x daily (flushes irritants)
• **Steam inhalation:** Hot shower or bowl of hot water with towel
• **Honey:** 1-2 tsp before bed (natural cough suppressant)
• **Zinc lozenges:** Within 24 hours of symptoms (reduces duration)

**🥗 Nutritional Therapy:**
**Foods to Include:**
• **Chicken soup:** Proven anti-inflammatory effects, hydration
• **Garlic:** Allicin has antimicrobial properties (crush before eating)
• **Ginger:** Fresh ginger tea reduces inflammation and nausea
• **Citrus fruits:** Vitamin C supports immune function
• **Mushrooms:** Shiitake, reishi boost immune system
• **Hot herbal teas:** Echinacea, elderberry, green tea
• **Spicy foods:** Cayenne, horseradish clear sinuses naturally

**Foods to Avoid:**
• **Dairy products:** May increase mucus production in some people
• **Refined sugars:** Can suppress immune function temporarily
• **Alcohol:** Dehydrates and impairs immune response
• **Fried foods:** Inflammatory, hard to digest when sick
• **Cold foods/drinks:** May worsen throat irritation

**Supplements & Vitamins:**
• **Vitamin C:** 1000mg every 4 hours (maximum 4000mg/day)
• **Zinc:** 13-23mg daily (take with food to avoid nausea)
• **Elderberry:** 15ml syrup 3x daily (antiviral properties)
• **Echinacea:** 300mg 3x daily (immune system support)
• **Vitamin D3:** 4000 IU daily (immune regulation)

**⚠️ Important Precautions:**
**Warning Signs to Watch:**
• Fever above 103°F (39.4°C)
• Difficulty breathing or wheezing
• Chest pain or pressure
• Severe sore throat with white patches
• Persistent vomiting preventing fluid intake

**When to Seek Emergency Care:**
• Severe difficulty breathing or shortness of breath
• Chest pain or pressure
• High fever with severe headache and neck stiffness
• Signs of dehydration (dizziness, dark urine)
• Symptoms worsen significantly after initial improvement

**🔄 Recovery Timeline:**
• **Day 1-3:** Symptoms peak, focus on rest and symptom relief
• **Day 4-7:** Gradual improvement, energy slowly returns
• **Day 8-10:** Most symptoms resolve, may have residual cough
• **Week 2-3:** Complete recovery, normal energy levels

**🏠 Self-Care Protocol:**
• **Rest:** 8-10 hours sleep, avoid strenuous activities
• **Hydration:** 10-12 glasses warm fluids daily (water, tea, broth)
• **Humidifier:** 40-60% humidity prevents nasal drying
• **Throat care:** Warm salt water gargle (1/2 tsp salt in warm water)
• **Hand hygiene:** Wash frequently to prevent spreading
• **Isolation:** Stay home until fever-free 24 hours

**🛡️ Prevention for Future:**
• **Hand washing:** 20 seconds with soap, especially after public spaces
• **Immune support:** Regular vitamin D, C, zinc supplementation
• **Sleep hygiene:** 7-8 hours nightly boosts immune function
• **Exercise regularly:** Moderate exercise strengthens immunity
• **Stress management:** Chronic stress weakens immune system
• **Annual flu shot:** Reduces risk by 40-60% when well-matched

**📞 Follow-up Care:**
• **See doctor** if no improvement after 7-10 days
• **Possible bacterial infection** if symptoms worsen after initial improvement
• **Antibiotic consideration** only if bacterial complications develop
• **Chest X-ray** if persistent cough with breathing difficulties

*Most cold/flu resolves naturally in 7-10 days! Your immune system has this covered. 🤧➡️😊*"""
    
    elif any(word in user_message_lower for word in ['hello', 'hi', 'hey', 'good', 'morning', 'afternoon', 'evening']):
        return "Hello! I'm Dr. Ayucore, your AI medical assistant. How can I help you today?"
    
    else:
        return f"""**🩺 General Health Guidance**

Thank you for your question about '{user_message}'. While I'd love to provide specific guidance, I want to ensure you get the most accurate information for your particular situation.

**💊 General Health Recommendations:**
• **Stay hydrated:** 8-10 glasses of water daily
• **Rest well:** 7-8 hours of quality sleep
• **Balanced nutrition:** Focus on whole foods, fruits, vegetables
• **Regular exercise:** 30 minutes daily, as tolerated
• **Stress management:** Deep breathing, meditation, or relaxation techniques

**⚠️ When to Seek Medical Care:**
• **Persistent symptoms:** Lasting more than a few days
• **Severe pain:** Interfering with daily activities
• **Fever:** Above 103°F (39.4°C)
• **Breathing difficulties:** Shortness of breath or wheezing
• **Concerning changes:** In vision, speech, or mental clarity

**📞 Follow-up Recommendations:**
For specific medical concerns, I recommend consulting with a healthcare provider who can:
• Perform physical examination
• Review your medical history
• Order appropriate tests if needed
• Provide personalized treatment plans

Feel free to ask me about specific symptoms, conditions, or general health topics for more detailed guidance!"""

@app.route("/")
def index():
    """Homepage route"""
    return render_template('index.html')

@app.route("/features")
def features():
    """Features page route"""
    return render_template('features.html')

@app.route("/about")
def about():
    """About page route"""
    return render_template('about.html')

@app.route("/chat")
def chat():
    """Chatbot interface route"""
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat_get():
    """Legacy endpoint for backward compatibility with form data"""
    try:
        # Handle form data from the chat interface
        user_message = request.form.get('msg', '').strip()
        patient_info_str = request.form.get('patient_info', '')
        
        if not user_message:
            return "Please provide a message."
        
        # Parse patient info if provided
        patient_info = {}
        if patient_info_str:
            try:
                # Simple parsing of patient info string
                lines = patient_info_str.strip().split('\n')
                for line in lines[1:]:  # Skip first line "Patient Details:"
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        if key == 'name':
                            patient_info['name'] = value
                        elif key == 'age':
                            patient_info['age'] = value.replace(' years', '')
                        elif key == 'gender':
                            patient_info['gender'] = value
                        elif key == 'weight':
                            patient_info['weight'] = value.replace(' kg', '')
                        elif key == 'height':
                            patient_info['height'] = value.replace(' cm', '')
            except:
                pass
        
        # Store patient info in session
        if patient_info:
            session['patient_info'] = patient_info
        
        # Initialize chat history if not exists
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        # Create context with patient information
        context = ""
        if session.get('patient_info'):
            patient_data = session['patient_info']
            context = f"""
Patient Information:
- Name: {patient_data.get('name', 'Not provided')}
- Age: {patient_data.get('age', 'Not provided')}
- Gender: {patient_data.get('gender', 'Not provided')}
- Weight: {patient_data.get('weight', 'Not provided')} kg
- Height: {patient_data.get('height', 'Not provided')} cm
- Medical History: {patient_data.get('medical_history', 'None provided')}
- Current Medications: {patient_data.get('medications', 'None provided')}

"""
        
        # Add chat history context
        if session['chat_history']:
            context += "Previous conversation:\n"
            for msg in session['chat_history'][-5:]:  # Last 5 messages for context
                context += f"Patient: {msg['user']}\nDoctor: {msg['bot']}\n"
        
        # Combine context with current question
        full_query = context + f"Current question: {user_message}"
        
        # Get AI response using retrieval chain
        print(f"[DEBUG] AI_AVAILABLE: {AI_AVAILABLE}, retrieval_chain exists: {retrieval_chain is not None}")
        if AI_AVAILABLE and retrieval_chain:
            try:
                print(f"[DEBUG] Attempting AI call with query: {full_query[:100]}...")
                result = retrieval_chain.invoke({"input": full_query})
                ai_response = result["answer"]
                print(f"[DEBUG] AI Response: {ai_response[:100]}...")
            except Exception as llm_error:
                print(f"[ERROR] LLM Error: {str(llm_error)}")
                ai_response = get_fallback_response(user_message)
                print(f"[DEBUG] Using Fallback after error: {ai_response[:100]}...")
        else:
            # Use fallback responses when AI is not available
            ai_response = get_fallback_response(user_message)
            print(f"[DEBUG] AI Not Available, Using Fallback: {ai_response[:100]}...")
        
        # Store in chat history
        chat_entry = {
            'user': user_message,
            'bot': ai_response,
            'timestamp': datetime.now().isoformat()
        }
        session['chat_history'].append(chat_entry)
        
        # Keep only last 20 exchanges to manage session size
        if len(session['chat_history']) > 20:
            session['chat_history'] = session['chat_history'][-20:]
        
        return ai_response
        
    except Exception as e:
        error_message = str(e)
        print(f"Error in chat_get: {error_message}")
        return f"I apologize, but I encountered an error: {error_message}"

@app.route("/reset_conversation", methods=["POST"])
def reset_conversation():
    """Reset the chat conversation and patient info"""
    try:
        session.pop('chat_history', None)
        return jsonify({"status": "success", "message": "Conversation reset successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat_response():
    """Handle chat messages and return AI responses"""
    user_message = ""
    patient_info = {}
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        patient_info = data.get('patient_info', {})
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Initialize chat history if not exists
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        # Store patient info in session
        if patient_info:
            session['patient_info'] = patient_info
        
        # Create context with patient information
        context = ""
        if session.get('patient_info'):
            patient_data = session['patient_info']
            context = f"""
Patient Information:
- Name: {patient_data.get('name', 'Not provided')}
- Age: {patient_data.get('age', 'Not provided')}
- Gender: {patient_data.get('gender', 'Not provided')}
- Weight: {patient_data.get('weight', 'Not provided')} kg
- Height: {patient_data.get('height', 'Not provided')} cm
- Medical History: {patient_data.get('medical_history', 'None provided')}
- Current Medications: {patient_data.get('medications', 'None provided')}

"""
        
        # Add chat history context
        if session['chat_history']:
            context += "Previous conversation:\n"
            for msg in session['chat_history'][-5:]:  # Last 5 messages for context
                context += f"Patient: {msg['user']}\nDoctor: {msg['bot']}\n"
        
        # Combine context with current question
        full_query = context + f"Current question: {user_message}"
        
        # Get AI response using retrieval chain
        print(f"[DEBUG] AI_AVAILABLE: {AI_AVAILABLE}, retrieval_chain exists: {retrieval_chain is not None}")
        if AI_AVAILABLE and retrieval_chain:
            try:
                print(f"[DEBUG] Attempting AI call with query: {full_query[:100]}...")
                result = retrieval_chain.invoke({"input": full_query})
                ai_response = result["answer"]
                print(f"[DEBUG] AI Response: {ai_response[:100]}...")
            except Exception as llm_error:
                print(f"[ERROR] LLM Error: {str(llm_error)}")
                ai_response = get_fallback_response(user_message)
                print(f"[DEBUG] Using Fallback after error: {ai_response[:100]}...")
        else:
            # Use fallback responses when AI is not available
            ai_response = get_fallback_response(user_message)
            print(f"[DEBUG] AI Not Available, Using Fallback: {ai_response[:100]}...")
        
        # Store in chat history
        chat_entry = {
            'user': user_message,
            'bot': ai_response,
            'timestamp': datetime.now().isoformat()
        }
        session['chat_history'].append(chat_entry)
        
        # Keep only last 20 exchanges to manage session size
        if len(session['chat_history']) > 20:
            session['chat_history'] = session['chat_history'][-20:]
        
        return jsonify({
            "response": ai_response,
            "timestamp": chat_entry['timestamp']
        })
        
    except Exception as e:
        error_message = str(e)
        print(f"Error in chat_response: {error_message}")
        
        # Log more details for debugging
        print(f"User message: {user_message}")
        print(f"Patient info: {patient_info}")
        print(f"Full error: {e}")
        
        # Return a more helpful error message
        if "API key" in error_message.lower() or "unauthorized" in error_message.lower():
            error_response = "I'm having trouble connecting to the AI service. Please check the API configuration."
        elif "pinecone" in error_message.lower():
            error_response = "I'm having trouble accessing the medical knowledge base. Please try again."
        elif "timeout" in error_message.lower():
            error_response = "The request timed out. Please try with a shorter question."
        else:
            error_response = f"I'm experiencing technical difficulties. Error details: {error_message[:100]}..."
        
        return jsonify({
            "error": "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
            "response": error_response,
            "debug_info": error_message if app.debug else None
        }), 500

@app.route("/reset", methods=["POST"])
def reset_chat():
    """Reset chat history and patient information"""
    try:
        session.pop('chat_history', None)
        session.pop('patient_info', None)
        return jsonify({"status": "success", "message": "Chat reset successfully"})
    except Exception as e:
        print(f"Error in reset_chat: {str(e)}")
        return jsonify({"error": "Failed to reset chat"}), 500

@app.route("/history")
def chat_history():
    """Get chat history"""
    history = session.get('chat_history', [])
    return jsonify({"history": history})

@app.route("/patient-info")
def patient_info():
    """Get current patient information"""
    info = session.get('patient_info', {})
    return jsonify({"patient_info": info})

@app.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.svg', mimetype='image/svg+xml')

@app.route('/sw.js')
def service_worker():
    """Serve service worker - return empty response for now"""
    return "", 204

@app.route("/test-chat", methods=["POST"])
def test_chat():
    """Test endpoint for debugging chat functionality"""
    try:
        data = request.get_json()
        return jsonify({
            "status": "success",
            "received_data": data,
            "message": "Test endpoint working!"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health-check")
def health_check():
    """Health check endpoint to verify AI components"""
    try:
        # Test Pinecone connection
        test_results = docsearch.similarity_search("test query", k=1)
        
        # Test LLM
        test_prompt = "Say 'AI is working' if you can understand this."
        result = retrieval_chain.invoke({"input": test_prompt})
        
        return jsonify({
            "status": "healthy",
            "pinecone_docs_found": len(test_results),
            "llm_response": result.get("answer", "No response")[:100]
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route("/test-llm")
def test_llm():
    """Test just the LLM without Pinecone"""
    try:
        # Test Google Gemini directly
        simple_response = llm.invoke("Hello, are you working?")
        
        return jsonify({
            "status": "success",
            "llm_direct_response": simple_response.content if hasattr(simple_response, 'content') else str(simple_response)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
