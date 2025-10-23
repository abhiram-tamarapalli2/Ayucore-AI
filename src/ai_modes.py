"""
AI Mode Handlers for AyuCore AI Medical Chatbot
Implements three specialized modes using different AI models and techniques
"""

import os
import logging
from typing import Dict, Any, Optional, List
import requests
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalAIModes:
    """
    Handles three specialized AI modes for medical consultation
    """
    
    def __init__(self):
        """Initialize all AI models and configurations"""
        self.initialize_models()
        self.setup_gemini()
        
    def initialize_models(self):
        """Initialize Hugging Face models for different modes"""
        try:
            # Medical conversation model for doctor mode
            logger.info("Loading medical conversation model...")
            self.conversation_model = None  # Will load on first use for memory efficiency
            
            # Medical QA model for knowledge mode
            logger.info("Loading medical QA model...")
            self.qa_pipeline = None  # Will load on first use
            
            # Medical text analysis for report mode
            logger.info("Loading medical text analysis model...")
            self.report_analyzer = None  # Will load on first use
            
            # Sentence transformer for RAG
            logger.info("Loading sentence transformer for RAG...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("AI models initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            
    def setup_gemini(self):
        """Setup Gemini API as fallback"""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                # Use a simpler, more reliable model
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-8b')
                logger.info("✅ Gemini API configured successfully with gemini-1.5-flash-8b")
                
                # Test the API connection
                test_response = self.gemini_model.generate_content("Hello, respond with 'API working'")
                if test_response and test_response.text:
                    logger.info("✅ Gemini API test successful")
                else:
                    logger.warning("⚠️ Gemini API test failed - empty response")
            else:
                logger.warning("❌ No Gemini API key found")
                self.gemini_model = None
        except Exception as e:
            logger.error(f"❌ Error setting up Gemini: {e}")
            self.gemini_model = None
    
    def load_conversation_model(self):
        """Load conversation model on demand"""
        if self.conversation_model is None:
            try:
                # Use a medical conversation model from Hugging Face
                model_name = "microsoft/DialoGPT-medium"  # Can be upgraded to medical-specific model
                self.conversation_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.conversation_model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Add special tokens for medical context
                special_tokens = {"pad_token": "<pad>", "additional_special_tokens": ["<medical>", "<patient>", "<doctor>"]}
                self.conversation_tokenizer.add_special_tokens(special_tokens)
                self.conversation_model.resize_token_embeddings(len(self.conversation_tokenizer))
                
                logger.info("Conversation model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading conversation model: {e}")
                self.conversation_model = "error"
    
    def load_qa_model(self):
        """Load QA model on demand"""
        if self.qa_pipeline is None:
            try:
                # Use medical QA model
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model="deepset/roberta-base-squad2",  # Can upgrade to medical-specific model
                    tokenizer="deepset/roberta-base-squad2"
                )
                logger.info("QA model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading QA model: {e}")
                self.qa_pipeline = "error"
    
    def load_report_analyzer(self):
        """Load report analysis model on demand"""
        if self.report_analyzer is None:
            try:
                # Use medical text classification model
                self.report_analyzer = pipeline(
                    "text-classification",
                    model="emilyalsentzer/Bio_ClinicalBERT",  # Medical text understanding
                    return_all_scores=True
                )
                
                # Also load named entity recognition for medical entities
                self.medical_ner = pipeline(
                    "ner",
                    model="d4data/biomedical-ner-all",  # Medical NER
                    aggregation_strategy="simple"
                )
                
                logger.info("Report analyzer loaded successfully")
            except Exception as e:
                logger.error(f"Error loading report analyzer: {e}")
                self.report_analyzer = "error"
    
    def is_medical_symptom(self, message: str) -> bool:
        """Check if message describes a medical symptom or concern"""
        symptom_keywords = [
            'fever', 'headache', 'pain', 'ache', 'hurt', 'sick', 'nausea', 'vomit', 'dizzy',
            'cough', 'cold', 'flu', 'sore', 'throat', 'stomach', 'chest', 'back', 'joint',
            'rash', 'itch', 'swelling', 'bleeding', 'bruise', 'tired', 'fatigue', 'weak',
            'breathe', 'breathing', 'short of breath', 'anxiety', 'stress', 'depression',
            'sleep', 'insomnia', 'appetite', 'weight', 'blood pressure', 'diabetes',
            'infection', 'allergy', 'redness', 'bump', 'lump', 'symptom', 'problem',
            'feel', 'feeling', 'unwell', 'bad', 'worse', 'better', 'treatment', 'medicine'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in symptom_keywords)

    async def handle_symptom_consultation(self, message: str, context: str, patient_info: str) -> str:
        """Handle medical symptom consultation with structured approach"""
        try:
            # Analyze the symptom mentioned
            message_lower = message.lower()
            
            # Fever consultation
            if 'fever' in message_lower:
                return await self.fever_consultation(message, context, patient_info)
            
            # Headache consultation
            elif any(word in message_lower for word in ['headache', 'head pain', 'migraine']):
                return await self.headache_consultation(message, context, patient_info)
            
            # Cough/Cold consultation
            elif any(word in message_lower for word in ['cough', 'cold', 'flu', 'congestion', 'runny nose']):
                return await self.cold_consultation(message, context, patient_info)
            
            # Stomach/Digestive issues
            elif any(word in message_lower for word in ['stomach', 'nausea', 'vomit', 'diarrhea', 'constipation']):
                return await self.stomach_consultation(message, context, patient_info)
            
            # Pain consultation
            elif any(word in message_lower for word in ['pain', 'ache', 'hurt']):
                return await self.pain_consultation(message, context, patient_info)
            
            # General symptom handling
            else:
                return await self.general_symptom_consultation(message, context, patient_info)
                
        except Exception as e:
            logger.error(f"Error in symptom consultation: {e}")
            return await self.gemini_fallback(message, "doctor", context, patient_info)

    async def fever_consultation(self, message: str, context: str, patient_info: str) -> str:
        """Structured fever consultation"""
        if "temperature" not in context.lower() and "how long" not in context.lower():
            return """I understand you have a fever. Let me help you with this step by step.

**First, let's check your temperature:**
Have you measured your temperature with a thermometer? 

**Examples of what to tell me:**
• "My temperature is 101.5°F" 
• "I haven't checked yet, but I feel very hot"
• "It was 38.5°C this morning"
• "I don't have a thermometer"

Please share your temperature reading or let me know if you haven't measured it yet."""
        
        elif "duration" not in context.lower() and "other symptoms" not in context.lower():
            return """**Next, let's understand the timeline:**
How long have you had this fever?

**Examples of what to tell me:**
• "Started this morning"
• "For about 2 days now"
• "Since yesterday evening"
• "Just noticed it an hour ago"

Also, are you experiencing any other symptoms along with the fever?

**Common symptoms to mention:**
• Chills or shivering
• Body aches or muscle pain
• Headache
• Sore throat
• Cough
• Nausea or vomiting
• Fatigue or weakness"""
        
        else:
            return """**Thank you for the information. Here's my immediate advice:**

**🌡️ Temperature Management:**
• Take acetaminophen (Tylenol) 500-1000mg every 6-8 hours OR
• Take ibuprofen (Advil) 400-600mg every 6-8 hours
• Use cool compresses on forehead for 10-15 minutes

**💧 Stay Hydrated:**
• Drink 8-10 glasses of water, herbal tea, or clear broths
• Avoid alcohol and excessive caffeine

**🛏️ Rest:**
• Get plenty of sleep (8-10 hours)
• Stay in a cool, comfortable room

**⚠️ See a doctor immediately if:**
• Temperature rises above 103°F (39.4°C)
• Severe headache with neck stiffness
• Difficulty breathing
• Persistent vomiting

Is there anything specific about your fever that concerns you the most?"""

    async def headache_consultation(self, message: str, context: str, patient_info: str) -> str:
        """Structured headache consultation"""
        if "type" not in context.lower() and "location" not in context.lower():
            return """I understand you're experiencing a headache. Let me help you identify the best treatment.

**First, describe your headache type:**
What does your headache feel like?

**Examples of what to tell me:**
• "Sharp, stabbing pain on one side"
• "Dull, constant pressure around my forehead"
• "Throbbing pain behind my eyes"
• "Tight band feeling around my head"
• "Severe pain with nausea and light sensitivity"

**Also, where exactly is the pain located?**
• Front of head (forehead)
• Back of head (neck area)
• Sides (temples)
• Top of head
• Behind eyes
• One side only"""
        
        else:
            return """**Here's immediate relief for your headache:**

**💊 Quick Relief:**
• Ibuprofen 400-600mg (best for tension headaches)
• Acetaminophen 1000mg (for general pain)
• Apply cold compress to forehead for 15-20 minutes

**🏠 Self-Care:**
• Rest in a dark, quiet room
• Gentle neck and shoulder massage
• Stay hydrated - drink 2-3 glasses of water
• Practice deep breathing (4 counts in, 7 hold, 8 out)

**⚠️ Seek immediate medical care if:**
• Sudden, severe "worst headache of your life"
• Headache with fever and neck stiffness
• Vision changes or speech problems
• Headache after head injury

**Prevention tips:**
• Regular sleep schedule (7-8 hours)
• Stay hydrated throughout the day
• Limit screen time if sensitive to light

How long have you had this headache, and have you tried any treatments yet?"""

    async def cold_consultation(self, message: str, context: str, patient_info: str) -> str:
        """Structured cold/flu consultation"""
        return """I see you're dealing with cold/flu symptoms. Let me help you feel better.

**🤧 Immediate Symptom Relief:**
• **For congestion:** Saline nasal spray or rinse (neti pot)
• **For cough:** Honey (1-2 teaspoons) or throat lozenges
• **For body aches:** Acetaminophen 1000mg every 6 hours

**💧 Hydration & Rest:**
• Drink warm liquids: herbal tea, chicken soup, warm water with honey
• Get 8-10 hours of sleep
• Use a humidifier or breathe steam from hot shower

**🥗 Helpful Foods:**
• Ginger tea (reduces inflammation)
• Garlic (natural antimicrobial)
• Citrus fruits (vitamin C)
• Chicken soup (proven to help recovery)

**⚠️ See a doctor if:**
• Fever above 103°F (39.4°C)
• Difficulty breathing or chest pain
• Symptoms worsen after 7-10 days
• Severe sore throat with white patches

**What specific symptoms are bothering you most right now?**
• Stuffy nose
• Sore throat  
• Cough
• Body aches
• Fatigue

This will help me give you more targeted advice."""

    async def stomach_consultation(self, message: str, context: str, patient_info: str) -> str:
        """Structured stomach/digestive consultation"""
        return """I understand you're having stomach issues. Let me help you feel better.

**🤢 Immediate Relief:**
• **For nausea:** Ginger tea or ginger candies
• **For upset stomach:** BRAT diet (Bananas, Rice, Applesauce, Toast)
• **For acid reflux:** Avoid spicy, fatty foods; eat smaller meals

**💧 Stay Hydrated:**
• Sip clear fluids slowly: water, herbal tea, clear broth
• Avoid dairy, caffeine, and alcohol temporarily
• Try electrolyte solutions if vomiting

**🍽️ Gentle Foods:**
• Plain crackers or toast
• Bananas (easy to digest, replace potassium)
• Rice or pasta (plain)
• Chamomile tea (soothes stomach)

**⚠️ Seek medical care if:**
• Severe abdominal pain
• Blood in vomit or stool
• Signs of dehydration (dizziness, dark urine)
• Unable to keep fluids down for 24 hours

**Tell me more about your symptoms:**
• Is it nausea, pain, or both?
• When did it start?
• Any recent food changes or stress?

**Example responses:**
• "Started after eating spicy food yesterday"
• "Nauseous for 2 days, can't keep food down"
• "Sharp pain in upper stomach area"

This helps me give you the best advice for your specific situation."""

    async def pain_consultation(self, message: str, context: str, patient_info: str) -> str:
        """Structured pain consultation"""
        return """I understand you're experiencing pain. Let me help you manage it effectively.

**First, help me understand your pain:**

**📍 Location:** Where exactly does it hurt?
**Examples:** "Lower back," "Right shoulder," "Left knee," "Chest area"

**📊 Intensity:** On a scale of 1-10, how severe is the pain?
**Examples:** "About a 6/10," "Mild (2-3/10)," "Severe (8-9/10)"

**⏰ Duration:** When did it start?
**Examples:** "This morning," "3 days ago," "After exercising yesterday"

**🔄 Type:** How would you describe the pain?
**Examples:** 
• Sharp and stabbing
• Dull and constant
• Throbbing
• Burning sensation
• Cramping

**💊 Quick Relief Options:**
• **For inflammation:** Ibuprofen 400-600mg every 8 hours
• **For general pain:** Acetaminophen 1000mg every 6 hours
• **For muscle pain:** Apply heat or ice for 15-20 minutes

**⚠️ Seek immediate medical care if:**
• Chest pain with breathing difficulty
• Severe abdominal pain
• Pain after injury/accident
• Pain with numbness or weakness

Please describe your pain using the examples above, and I'll give you specific treatment advice."""

    async def general_symptom_consultation(self, message: str, context: str, patient_info: str) -> str:
        """Handle general medical symptoms"""
        return f"""I understand you're experiencing: "{message}"

Let me help you with this step by step.

**📋 To give you the best advice, please tell me:**

**1. When did this start?**
**Examples:** "This morning," "2 days ago," "Been ongoing for a week"

**2. How severe is it?**
**Examples:** "Mild but annoying," "Moderate - affecting daily activities," "Severe - can't function normally"

**3. Any other symptoms?**
**Examples:** "Also have a headache," "Feeling tired too," "No other symptoms"

**4. What makes it better or worse?**
**Examples:** "Worse when I move," "Better when I rest," "Gets worse at night"

**💊 General Self-Care:**
• Rest and avoid strenuous activities
• Stay hydrated (8-10 glasses of water)
• Consider over-the-counter pain relief if needed
• Monitor symptoms for changes

**⚠️ See a doctor if:**
• Symptoms worsen rapidly
• New concerning symptoms develop
• No improvement after 2-3 days
• You're worried about the symptoms

Please answer the questions above, and I'll provide specific guidance for your situation."""

    def is_medical_symptom(self, message: str) -> bool:
        """Check if message contains medical symptoms"""
        medical_keywords = [
            'fever', 'headache', 'pain', 'hurt', 'ache', 'sick', 'nausea', 
            'tired', 'fatigue', 'cough', 'cold', 'flu', 'dizzy', 'weak',
            'sore', 'throat', 'stomach', 'chest', 'back', 'breathing',
            'temperature', 'chills', 'vomit', 'diarrhea', 'constipation'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in medical_keywords)
    
    def extract_symptom_info(self, message: str) -> Dict[str, str]:
        """Extract symptom information from user message"""
        info = {}
        message_lower = message.lower()
        
        # Time indicators
        time_patterns = {
            'today': ['today', 'this morning', 'this afternoon', 'this evening'],
            'yesterday': ['yesterday'],
            '2_days': ['2 days', 'two days', 'couple days', 'couple of days'],
            '3_days': ['3 days', 'three days'],
            'week': ['week', '7 days', 'seven days'],
            'ongoing': ['ongoing', 'always', 'chronic', 'long time']
        }
        
        for timeframe, patterns in time_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                info['timing'] = timeframe
                break
        
        # Severity indicators
        severity_patterns = {
            'mild': ['mild', 'little', 'slight', 'not bad', 'annoying'],
            'moderate': ['moderate', 'affecting', 'daily activities', 'interfering'],
            'severe': ['severe', 'terrible', 'unbearable', 'can\'t function', 'worst', 'extreme']
        }
        
        for severity, patterns in severity_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                info['severity'] = severity
                break
        
        # Additional symptoms
        additional_symptoms = []
        symptom_keywords = {
            'tired': ['tired', 'fatigue', 'exhausted', 'weak'],
            'headache': ['headache', 'head pain'],
            'nausea': ['nausea', 'sick', 'queasy'],
            'chills': ['chills', 'cold', 'shivering'],
            'cough': ['cough', 'coughing'],
            'sore_throat': ['sore throat', 'throat pain'],
            'aches': ['aches', 'body pain', 'muscle pain']
        }
        
        for symptom, keywords in symptom_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                additional_symptoms.append(symptom)
        
        if additional_symptoms:
            info['additional_symptoms'] = additional_symptoms
        
        return info

    def has_sufficient_info(self, symptom_info: Dict[str, str]) -> bool:
        """Check if we have enough information to provide specific advice"""
        required_fields = ['timing', 'severity']
        return all(field in symptom_info for field in required_fields)

    async def handle_symptom_consultation(self, message: str, context: str = "", patient_info: str = "") -> str:
        """Handle symptom consultation with intelligent parsing"""
        try:
            # Extract information from the message
            symptom_info = self.extract_symptom_info(message)
            
            # Check if user has provided comprehensive information
            if self.has_sufficient_info(symptom_info):
                return await self.provide_specific_advice(message, symptom_info, patient_info)
            else:
                # Ask for missing information step by step
                return await self.ask_follow_up_questions(message, symptom_info, patient_info)
                
        except Exception as e:
            logger.error(f"Error in symptom consultation: {e}")
            return await self.gemini_fallback(message, "doctor", context, patient_info)

    async def provide_specific_advice(self, message: str, symptom_info: Dict[str, str], patient_info: str = "") -> str:
        """Provide specific medical advice based on gathered information"""
        
        timing = symptom_info.get('timing', 'unknown')
        severity = symptom_info.get('severity', 'unknown')
        additional_symptoms = symptom_info.get('additional_symptoms', [])
        
        # Create detailed advice based on the information
        advice_prompt = f"""
        You are Dr. Ayucore providing specific medical advice. A patient has reported:
        
        Original Message: {message}
        Duration: {timing}
        Severity: {severity}
        Additional Symptoms: {', '.join(additional_symptoms) if additional_symptoms else 'None reported'}
        Patient Info: {patient_info}
        
        Based on this information, provide a comprehensive response including:
        1. **Assessment**: What this likely indicates
        2. **Immediate Care**: Specific steps to take now
        3. **Treatment**: Medications, remedies, or therapies
        4. **Timeline**: Expected recovery time
        5. **Warning Signs**: When to seek immediate care
        
        Be specific, practical, and reassuring. Format professionally with clear sections.
        """
        
        return await self.gemini_fallback(advice_prompt, "doctor", "", patient_info)

    async def ask_follow_up_questions(self, message: str, symptom_info: Dict[str, str], patient_info: str = "") -> str:
        """Ask targeted follow-up questions based on missing information"""
        
        missing_info = []
        if 'timing' not in symptom_info:
            missing_info.append('timing')
        if 'severity' not in symptom_info:
            missing_info.append('severity')
        
        # Ask for the most important missing information first
        if 'timing' in missing_info:
            question_prompt = f"""
            The patient said: "{message}"
            
            I need to know when this started to provide the best advice. Ask ONE specific question about timing.
            
            Examples of good timing questions:
            - "When did this start exactly?"
            - "How long have you been experiencing this?"
            
            Be concise and ask only about timing.
            """
        elif 'severity' in missing_info:
            question_prompt = f"""
            The patient said: "{message}"
            
            I need to understand the severity to provide appropriate advice. Ask ONE specific question about how severe it is.
            
            Examples of good severity questions:
            - "How would you rate the severity on a scale of 1-10?"
            - "Is this interfering with your daily activities?"
            
            Be concise and ask only about severity.
            """
        else:
            # Ask for additional context
            question_prompt = f"""
            The patient said: "{message}"
            
            Ask ONE specific follow-up question to better understand their condition.
            Focus on the most important missing detail.
            Be concise and medical.
            """
        
        return await self.gemini_fallback(question_prompt, "doctor", "", patient_info)

    def is_greeting(self, message: str) -> bool:
        """Check if message is a greeting"""
        greeting_words = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings', 'hola', 'welcome']
        message_lower = message.lower().strip()
        
        # Check for explicit greeting words
        if any(greeting in message_lower for greeting in greeting_words):
            return True
            
        # Check for very short messages that are likely greetings (but exclude medical terms)
        medical_keywords = ['fever', 'pain', 'hurt', 'sick', 'ache', 'feel', 'symptom', 'have', 'got', 'headache', 'cough', 'cold', 'flu']
        if len(message_lower) <= 10 and not any(keyword in message_lower for keyword in medical_keywords):
            return True
            
        return False

    async def doctor_conversation_mode(self, message: str, context: str = "", patient_info: str = "") -> str:
        """
        Doctor Conversation Mode: Interactive consultation like a real doctor
        Uses conversational AI with medical context
        """
        try:
            # Handle greetings only if there's no existing conversation context
            if self.is_greeting(message) and not context.strip():
                return await self.get_doctor_greeting(patient_info, context)
            
            # For doctor mode, always respond conversationally like a real doctor
            return await self.get_conversational_doctor_response(message, context, patient_info)
            
        except Exception as e:
            logger.error(f"Error in doctor conversation mode: {e}")
            return "I'm sorry, I'm having a technical issue. Can you tell me more about how you're feeling so I can help you?"
    
    async def get_conversational_doctor_response(self, message: str, context: str, patient_info: str) -> str:
        """Generate human-like conversational doctor responses"""
        try:
            # Create a conversational doctor prompt with better context handling
            doctor_prompt = f"""
You are Dr. Ayucore, a warm and caring family doctor having an ongoing conversation with your patient. You must remember and refer to the entire conversation history.

{context}

Current patient message: "{message}"

Patient information: {patient_info}

Important Instructions:
- Continue the conversation naturally - DO NOT restart or forget previous messages
- Reference symptoms and information already discussed in the conversation
- Build on what the patient has already told you
- Ask specific follow-up questions based on what you already know
- Show that you remember what they've said before
- Be empathetic and caring like a real doctor
- Keep responses conversational, not like a medical textbook
- Ask ONE relevant follow-up question to gather more information

Respond as Dr. Ayucore continuing this medical consultation.
            """

            # Use Gemini for conversational response or fallback to pre-defined responses
            if self.gemini_model:
                try:
                    logger.info(f"📤 Sending conversation to Gemini API (context: {len(context)} chars)")
                    
                    # Configure safety settings for medical conversation
                    safety_settings = [
                        {"category": "HARM_CATEGORY_MEDICAL", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HEALTH", "threshold": "BLOCK_NONE"}
                    ]
                    
                    generation_config = {
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "top_k": 40,
                        "max_output_tokens": 200,
                    }
                    
                    response = self.gemini_model.generate_content(
                        doctor_prompt,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    if response and response.text and response.text.strip():
                        logger.info("✅ Gemini API response received successfully")
                        return response.text.strip()
                    else:
                        logger.warning("⚠️ Gemini API returned empty or blocked response")
                        if hasattr(response, 'prompt_feedback'):
                            logger.warning(f"Prompt feedback: {response.prompt_feedback}")
                        
                except Exception as e:
                    logger.error(f"❌ Gemini conversation error: {type(e).__name__}: {e}")
            else:
                logger.error("❌ Gemini model not initialized")
            
            # Fallback to conversational responses for common symptoms
            logger.info("🔄 Using fallback response due to Gemini API issue")
            return self.get_conversational_fallback(message, patient_info, context)
            
        except Exception as e:
            logger.error(f"Error in conversational doctor response: {e}")
            return self.get_conversational_fallback(message, patient_info, context)

    def get_conversational_fallback(self, message: str, patient_info: str = "", context: str = "") -> str:
        """Provide intelligent fallback responses when Gemini API fails"""
        # Extract patient name if available
        patient_name = "there"
        if patient_info and "name:" in patient_info.lower():
            lines = patient_info.split('\n')
            for line in lines:
                if 'name:' in line.lower():
                    name_part = line.split(':', 1)[1].strip()
                    if name_part and name_part != 'Not provided':
                        patient_name = name_part.split()[0]
                    break
        
        message_lower = message.lower()
        context_lower = context.lower() if context else ""
        
        # Determine conversation stage based on context
        conversation_count = context_lower.count('patient:') if context else 0
        
        # Stage-based responses that follow conversation flow
        if conversation_count == 0:
            # First message - greeting
            return f"Hello {patient_name}, I'm Dr. Ayucore. I'm here to help you with your health concerns. Please tell me more about what's been bothering you - I'm listening carefully."
            
        elif conversation_count == 1:
            # Second message - initial symptom description
            return f"I can see you're dealing with multiple symptoms - headaches, dizziness, and fatigue. That combination can really affect your daily life. Can you tell me more about when these headaches are most severe? Are they throbbing, sharp, or more like a dull pressure?"
            
        elif conversation_count == 2:
            # Third message - symptom timing and patterns
            return f"Thank you for those details, {patient_name}. Morning headaches that worsen during the day, combined with dizziness when standing - that's giving me some important clues. Tell me about your hydration and eating patterns lately. Are you drinking enough water throughout the day?"
            
        elif conversation_count == 3:
            # Fourth message - lifestyle factors
            return f"I see the bigger picture now, {patient_name}. Office work, irregular eating, stress, and that hydration pattern could definitely be contributing factors. Based on what you've told me, this sounds like it could be related to dehydration, stress, and possibly some blood pressure changes when you stand. Have you noticed any other symptoms like nausea or vision changes?"
            
        elif conversation_count == 4:
            # Fifth message - concern about urgency
            return f"I understand your concern, {patient_name}. Based on your symptoms - the headache pattern, dizziness when standing, fatigue, plus your lifestyle factors - I'd recommend starting with some immediate self-care: increase your water intake significantly, try to eat regular meals, and monitor your symptoms. However, if headaches worsen or you develop severe dizziness, see a doctor promptly."
            
        elif conversation_count >= 5:
            # Later messages - emergency signs or wrap-up
            if any(word in message_lower for word in ['warning signs', 'emergency', 'watch out']):
                return f"Great question, {patient_name}. Seek immediate medical care if you experience: severe sudden headache unlike any before, headache with fever or neck stiffness, persistent vomiting, confusion, vision changes, or if dizziness becomes severe or you feel faint. Otherwise, follow up with your doctor within a few days if symptoms don't improve with better hydration and stress management."
            elif any(word in message_lower for word in ['thank', 'helpful', 'appreciate']):
                return f"You're very welcome, {patient_name}. I'm glad I could help clarify things for you. Remember to take care of yourself - proper hydration, regular meals, and stress management can make a huge difference. Feel better, and don't hesitate to seek care if you need it!"
            else:
                return f"I'm following what you're telling me, {patient_name}. Each detail you share helps me understand your situation better. Can you tell me a bit more about how this is affecting your daily routine?"
        
        else:
            # Default conversational response that acknowledges we're listening
            return f"I'm following what you're telling me, {patient_name}. Each detail you share helps me understand your situation better. Can you tell me a bit more about how this is affecting your daily routine?"

    async def medical_knowledge_mode(self, message: str, context: str = "", patient_info: str = "") -> str:
        """
        Medical Knowledge Mode: Comprehensive medical guides and Q&A
        Uses specialized medical knowledge models with RAG
        """
        try:
            # Handle greetings specifically
            if self.is_greeting(message):
                return await self.get_knowledge_greeting(patient_info, context)
            
            self.load_qa_model()
            
            # Use RAG to find relevant medical context
            relevant_context = self.retrieve_medical_knowledge(message, context)
            
            if self.qa_pipeline and self.qa_pipeline != "error":
                try:
                    # Use QA model if we have relevant context
                    if relevant_context:
                        result = self.qa_pipeline(question=message, context=relevant_context)
                        
                        if result['score'] > 0.3:  # Confidence threshold
                            return self.format_knowledge_response(message, result['answer'], relevant_context)
                    
                except Exception as e:
                    logger.error(f"QA pipeline error: {e}")
            
            # Enhanced Gemini fallback for comprehensive medical knowledge
            knowledge_prompt = f"""
            You are Dr. AyuCore, a highly experienced medical professional and educator providing comprehensive medical guidance.
            
            Patient Question: {message}
            Relevant Medical Context: {relevant_context}
            Patient Information: {patient_info}
            
            Provide a complete, comprehensive medical guide following this exact structure with clean formatting (NO markdown symbols like **, ***, ---, etc.):
            
            🏥 WHAT IS [CONDITION/TOPIC]:
            - Clear definition and explanation
            - Medical terminology with patient-friendly explanations
            - Who it affects and prevalence
            
            🧬 MEDICAL BACKGROUND & CAUSES:
            - Pathophysiology (how it develops in the body)
            - Root causes and risk factors
            - Contributing factors and triggers
            
            🔍 SYMPTOMS & SIGNS:
            - Early symptoms to watch for
            - Progressive symptoms
            - Severe symptoms requiring immediate care
            - How symptoms may vary between individuals
            
            🩺 DIAGNOSIS:
            - How healthcare providers diagnose this condition
            - Tests and examinations typically performed
            - Differential diagnosis (ruling out similar conditions)
            
            💊 TREATMENT OPTIONS:
            
            Over-the-Counter Medications:
            - Specific medications with names and dosages
            - How they work and expected timeline
            - Important precautions and contraindications
            
            Prescription Medications:
            - Common prescription treatments
            - When they're typically prescribed
            - Expected outcomes and side effects
            
            Non-Medication Treatments:
            - Lifestyle modifications
            - Home remedies and natural approaches
            - Physical therapies or procedures
            
            🛡️ PREVENTION STRATEGIES:
            - Primary prevention (preventing development)
            - Secondary prevention (preventing worsening)
            - Lifestyle recommendations
            - Environmental factors to consider
            
            ⚠️ WHEN TO SEEK MEDICAL CARE:
            
            Routine Care:
            - When to schedule regular check-ups
            - Monitoring recommendations
            
            Urgent Care:
            - Symptoms requiring prompt medical attention
            - Warning signs of complications
            
            Emergency Care:
            - Life-threatening symptoms requiring immediate ER visit
            - Red flags that cannot wait
            
            📋 LONG-TERM MANAGEMENT:
            - Ongoing care requirements
            - Follow-up schedules
            - Monitoring for complications
            - Quality of life considerations
            
            🏠 SELF-CARE & HOME MANAGEMENT:
            - Daily care routines
            - Dietary recommendations
            - Activity and exercise guidelines
            - Stress management techniques
            
            IMPORTANT: Use clean, readable formatting without any markdown symbols (**, ***, ---, etc.). Use emojis and clear section headers for organization. Make your response detailed, educational, and professionally comprehensive while remaining accessible to patients. Include specific medication names, dosages when appropriate, and clear action steps.
            """
            
            return await self.gemini_fallback(knowledge_prompt, "knowledge", context, patient_info)
            
        except Exception as e:
            logger.error(f"Error in medical knowledge mode: {e}")
            return "I apologize for the technical difficulty. Let me provide you with general medical information about your query using my knowledge base."
    
    async def report_analysis_mode(self, message: str, context: str = "", patient_info: str = "") -> str:
        """
        Report Analysis Mode: Analyze medical reports and lab results
        Uses medical NLP models for document analysis
        """
        try:
            # Handle greetings specifically
            if self.is_greeting(message):
                return await self.get_report_greeting(patient_info, context)
            
            self.load_report_analyzer()
            
            analysis_results = {}
            
            # Analyze with medical text classifier
            if self.report_analyzer and self.report_analyzer != "error":
                try:
                    # Extract medical entities
                    if hasattr(self, 'medical_ner') and self.medical_ner:
                        entities = self.medical_ner(message)
                        analysis_results['entities'] = entities
                    
                    # Classify text type
                    classification = self.report_analyzer(message)
                    analysis_results['classification'] = classification
                    
                except Exception as e:
                    logger.error(f"Report analysis error: {e}")
            
            # Format analysis using Gemini with structure
            analysis_prompt = f"""
            You are a medical report analysis specialist. Analyze the following medical report/results:
            
            Report Content: {message}
            Patient Information: {patient_info}
            Analysis Results: {analysis_results}
            
            Provide a comprehensive analysis including:
            
            🔬 **REPORT ANALYSIS:**
            - Type of report/test
            - Key findings and values
            - Normal vs abnormal ranges
            
            📊 **CLINICAL INTERPRETATION:**
            - What the results mean
            - Significance of abnormal values
            - Potential health implications
            
            ⚠️ **RECOMMENDATIONS:**
            - Follow-up actions needed
            - When to consult a doctor
            - Lifestyle recommendations
            
            📋 **SUMMARY:**
            - Overall health assessment
            - Key points to discuss with your doctor
            
            Be thorough, accurate, and explain medical terms clearly.
            """
            
            return await self.gemini_fallback(analysis_prompt, "report", context, patient_info)
            
        except Exception as e:
            logger.error(f"Error in report analysis mode: {e}")
            return "I apologize for the technical difficulty. Please share your report details, and I'll do my best to analyze and explain the results."
    
    async def get_doctor_greeting(self, patient_info: str = "", context: str = "") -> str:
        """Generate personalized doctor greeting"""
        try:
            # Extract patient name if available
            patient_name = "there"
            if patient_info and "name:" in patient_info.lower():
                lines = patient_info.split('\n')
                for line in lines:
                    if 'name:' in line.lower():
                        name_part = line.split(':', 1)[1].strip()
                        if name_part and name_part != 'Not provided':
                            patient_name = name_part.split()[0]  # First name only
                        break
            
            greetings = [
                f"Hi {patient_name}. It's nice to hear from you. \"{patient_name}\" is a little brief, though. Is there something specific you'd like to talk about today? I'm here to listen and help in any way I can. Are you feeling unwell, or did you have a question about your health or a recent appointment? Please tell me more so I can understand how to best assist you.",
                
                f"Hello {patient_name}. I'm Dr. Ayucore, and I'm here to help with any health concerns you might have. Whether you're feeling unwell, have questions about symptoms, or need guidance about a medical condition, I'm ready to listen and provide you with comprehensive care. What's on your mind today?",
                
                f"Good to see you, {patient_name}. I'm Dr. Ayucore, your medical assistant. I understand that reaching out about health concerns can sometimes feel overwhelming, but you're in the right place. Whether you have specific symptoms to discuss, questions about your health, or just need some medical guidance, I'm here to help. What would you like to talk about today?"
            ]
            
            # Use first greeting as primary
            return greetings[0]
            
        except Exception as e:
            logger.error(f"Error generating doctor greeting: {e}")
            return "Hello! I'm Dr. Ayucore, your AI medical assistant. I'm here to help with any health concerns or questions you might have. Please tell me what's on your mind today."

    async def get_knowledge_greeting(self, patient_info: str = "", context: str = "") -> str:
        """Generate simple knowledge mode greeting"""
        try:
            patient_name = "there"
            if patient_info and "name:" in patient_info.lower():
                lines = patient_info.split('\n')
                for line in lines:
                    if 'name:' in line.lower():
                        name_part = line.split(':', 1)[1].strip()
                        if name_part and name_part != 'Not provided':
                            patient_name = name_part.split()[0]
                        break
            
            return f"Hello {patient_name}! I'm ready to provide comprehensive medical information. What would you like to know about?"
            
        except Exception as e:
            logger.error(f"Error generating knowledge greeting: {e}")
            return "Hello! I'm ready to provide comprehensive medical information. What would you like to know about?"

    async def get_report_greeting(self, patient_info: str = "", context: str = "") -> str:
        """Generate report analysis mode greeting"""
        try:
            patient_name = "there"
            if patient_info and "name:" in patient_info.lower():
                lines = patient_info.split('\n')
                for line in lines:
                    if 'name:' in line.lower():
                        name_part = line.split(':', 1)[1].strip()
                        if name_part and name_part != 'Not provided':
                            patient_name = name_part.split()[0]
                        break
            
            return f"Hello {patient_name}! Welcome to the Medical Report Analysis Center. I'm here to help you understand your medical reports, lab results, test findings, and diagnostic documents. I can break down complex medical terminology, explain what your numbers mean, identify important findings, and help you understand the clinical significance of your results. Please share your report or describe what you'd like me to analyze."
            
        except Exception as e:
            logger.error(f"Error generating report greeting: {e}")
            return "Hello! Welcome to the Medical Report Analysis Center. I can help you understand medical reports, lab results, and diagnostic findings. Please share your report or tell me what you'd like me to analyze."

    def retrieve_medical_knowledge(self, query: str, context: str) -> str:
        """
        RAG implementation for medical knowledge retrieval
        """
        try:
            # Simple implementation - can be enhanced with vector database
            if context:
                # Use sentence transformer to find relevant parts
                query_embedding = self.sentence_model.encode([query])
                context_sentences = context.split('.')
                
                if len(context_sentences) > 1:
                    context_embeddings = self.sentence_model.encode(context_sentences)
                    
                    # Calculate similarities
                    similarities = np.dot(query_embedding, context_embeddings.T)[0]
                    
                    # Get top relevant sentences
                    top_indices = np.argsort(similarities)[-3:][::-1]
                    relevant_sentences = [context_sentences[i] for i in top_indices if similarities[i] > 0.3]
                    
                    return '. '.join(relevant_sentences)
            
            return context
            
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {e}")
            return context
    
    def format_knowledge_response(self, question: str, answer: str, context: str) -> str:
        """Format comprehensive knowledge mode response with clean UI formatting"""
        # Check if the answer already has comprehensive formatting
        if "🏥" in answer or "🧬" in answer or "🔍" in answer:
            return answer  # Already properly formatted
        
        # Otherwise, enhance the basic answer with comprehensive structure
        return f"""🩺 Medical Guidance

Thank you for reaching out about: "{question}"

📋 Comprehensive Medical Information:

{answer}

💊 General Health Recommendations:
• Stay hydrated: 8-10 glasses of water daily
• Rest: Adequate sleep (7-8 hours)
• Nutrition: Balanced diet with fruits and vegetables
• Exercise: As tolerated, gentle movement
• Monitor symptoms: Keep track of changes

⚠️ When to Seek Medical Care:
• Symptoms persist or worsen
• New concerning symptoms develop
• Fever above 103°F (39.4°C)
• Difficulty breathing
• Severe pain

📞 Emergency Care If:
• Life-threatening symptoms
• Severe difficulty breathing
• Chest pain
• Signs of stroke or heart attack

For specific medical concerns, it's always best to consult with a healthcare provider who can perform a physical examination and review your medical history.

Feel free to provide more details about your symptoms for more specific guidance.
        """
    
    async def gemini_fallback(self, message: str, mode: str, context: str = "", patient_info: str = "") -> str:
        """
        Fallback to Gemini API when Hugging Face models fail
        """
        try:
            if not self.gemini_model:
                # If no Gemini, provide structured medical responses
                return self.get_structured_medical_response(message, mode, context, patient_info)
            
            mode_prompts = {
                "doctor": f"""
                You are Dr. Ayucore, a compassionate medical doctor in consultation mode.
                
                Patient Information: {patient_info}
                Medical Context: {context}
                Patient Message: {message}
                
                Respond as a caring doctor would - ask follow-up questions, provide medical guidance, and show empathy.
                """,
                
                "knowledge": f"""
                You are a medical knowledge expert. Provide comprehensive, detailed medical information.
                
                Question: {message}
                Context: {context}
                
                Provide thorough medical information covering causes, symptoms, treatments, and prevention.
                """,
                
                "report": f"""
                You are a medical report analysis specialist.
                
                Report/Results: {message}
                Patient Info: {patient_info}
                
                Analyze this medical report thoroughly, explaining findings, normal ranges, and recommendations.
                """
            }
            
            prompt = mode_prompts.get(mode, message)
            response = self.gemini_model.generate_content(prompt)
            
            return response.text if response else "I'm having trouble processing your request. Please try again."
            
        except Exception as e:
            logger.error(f"Gemini fallback error: {e}")
            return self.get_structured_medical_response(message, mode, context, patient_info)

    def get_structured_medical_response(self, message: str, mode: str, context: str = "", patient_info: str = "") -> str:
        """
        Provide structured medical responses when all AI services are unavailable
        """
        message_lower = message.lower()
        
        # Specific medical conditions - high priority matches
        if any(word in message_lower for word in ['acne', 'pimple', 'blackhead', 'whitehead', 'zit']):
            return self.get_comprehensive_acne_response()
        elif any(word in message_lower for word in ['diabetes', 'blood sugar', 'insulin']):
            return self.get_comprehensive_diabetes_response()
        elif any(word in message_lower for word in ['hypertension', 'high blood pressure', 'blood pressure']):
            return self.get_comprehensive_hypertension_response()
        elif any(word in message_lower for word in ['migraine', 'headache', 'head pain']):
            return self.get_comprehensive_migraine_response()
        
        # Common symptoms - lower priority
        elif any(word in message_lower for word in ['fever', 'temperature', 'hot', 'chills']):
            return self.get_fever_response()
        elif any(word in message_lower for word in ['cold', 'cough', 'runny nose', 'congestion', 'flu', 'sore throat']):
            return self.get_cold_flu_response()
        elif any(word in message_lower for word in ['pain', 'ache', 'hurt', 'sore']) and not any(word in message_lower for word in ['acne', 'skin', 'face']):
            return self.get_pain_response()
        elif any(word in message_lower for word in ['stomach', 'nausea', 'vomit', 'digest']):
            return self.get_digestive_response()
        else:
            return self.get_general_medical_response(message, mode)

    def get_comprehensive_acne_response(self) -> str:
        """Comprehensive acne guide with clean UI-friendly formatting"""
        return """🩺 Medical Guidance

Thank you for reaching out about: "what is acne"

🏥 WHAT IS ACNE:
• Acne is a common skin condition that occurs when hair follicles become clogged with oil (sebum) and dead skin cells
• Affects primarily the face, chest, back, and shoulders where oil glands are most active  
• Most common during teenage years but can persist into adulthood
• Affects approximately 85% of people aged 12-24

🧬 MEDICAL BACKGROUND & PATHOPHYSIOLOGY:
• Root Cause: Combination of four main factors working together
• Excess Sebum Production: Hormones (especially androgens) increase oil gland activity
• Follicular Hyperkeratinization: Abnormal shedding of skin cells blocks pores
• Bacterial Overgrowth: Propionibacterium acnes (P. acnes) thrives in clogged follicles
• Inflammation: Body's immune response to bacteria and blocked follicles
• Hormonal Triggers: Puberty, menstrual cycles, pregnancy, PCOS
• Genetic Factors: Family history increases likelihood by 70-80%

🔍 SYMPTOMS & DIAGNOSIS:
• Mild Acne: Blackheads (open comedones) and whiteheads (closed comedones)
• Moderate Acne: Red, tender bumps (papules) and pus-filled lesions (pustules)
• Severe Acne: Large, painful lumps (nodules) and deep cysts
• Location: Face, neck, chest, back, shoulders
• Diagnosis: Visual examination by healthcare provider, severity assessment
• Associated Signs: Post-inflammatory hyperpigmentation, potential scarring

💊 TREATMENT OPTIONS:

Over-the-Counter Medications:
• Benzoyl Peroxide (2.5-10%): Kills bacteria, reduces inflammation
  - Start with 2.5% to minimize irritation
  - Apply once daily, increase gradually
• Salicylic Acid (0.5-2%): Unclogs pores, gentle exfoliation
• Sulfur Products: Anti-inflammatory and antimicrobial properties
• Retinol Creams: Milder vitamin A derivatives for maintenance

Prescription Medications:
• Topical Retinoids: Tretinoin, adapalene, tazarotene
  - Prevent new comedones, reduce inflammation
  - Apply at night, start 2-3 times per week
• Topical Antibiotics: Clindamycin 1%, erythromycin 2%
  - Usually combined with benzoyl peroxide
• Oral Antibiotics: Doxycycline 50-100mg daily, minocycline
  - For moderate to severe inflammatory acne
• Hormonal Therapy: Birth control pills, spironolactone (for women)
• Isotretinoin: For severe, cystic, or treatment-resistant acne
  - Requires close monitoring, highly effective

🛡️ PREVENTION STRATEGIES:
• Gentle Skincare Routine: Wash face twice daily with mild, non-comedogenic cleanser
• Avoid Over-cleansing: Harsh scrubbing can worsen acne
• Non-comedogenic Products: Choose makeup and moisturizers labeled "won't clog pores"
• Don't Pick or Squeeze: Can cause scarring and worsen inflammation
• Hair Care: Keep hair clean, avoid oil-based products near face
• Diet Considerations: Limit high glycemic foods (white bread, sugary foods) and dairy
• Stress Management: High stress can worsen acne through hormonal changes
• Clean Pillowcases: Change weekly to reduce bacteria transfer

⚠️ WHEN TO SEEK MEDICAL CARE:

Routine Dermatology Care:
• Over-the-counter treatments haven't worked after 6-8 weeks
• Acne is affecting self-esteem or quality of life
• Mild acne that's persistent or worsening

Urgent Dermatological Care:
• Moderate to severe inflammatory acne (multiple papules/pustules)
• Deep, painful nodules or cysts
• Signs of scarring or post-inflammatory changes
• Acne accompanied by excessive hair growth or irregular periods (may indicate hormonal issues)

Emergency Care:
• Signs of severe infection (fever, spreading redness, severe pain)
• Severe allergic reaction to acne medications (difficulty breathing, swelling)

📋 LONG-TERM MANAGEMENT:
• Maintenance Treatment: Continue successful treatments to prevent recurrence
• Patience Required: Most treatments take 8-12 weeks to show full results
• Combination Therapy: Often more effective than single treatments
• Regular Follow-ups: Monitor progress and adjust treatments
• Scar Prevention: Early, effective treatment prevents permanent scarring
• Adult Acne: May require different approach, often hormonal factors involved

🏠 SELF-CARE & HOME MANAGEMENT:
• Daily Routine: Gentle cleanser morning and night
• Moisturize: Even oily skin needs non-comedogenic moisturizer
• Sun Protection: SPF 30+ daily (some acne treatments increase sun sensitivity)
• Makeup Tips: Remove completely before bed, use non-comedogenic products
• Exercise Hygiene: Shower immediately after sweating
• Sleep Hygiene: Clean pillowcases, avoid sleeping face-down

Important Note: This comprehensive guide provides evidence-based medical information. However, individual cases vary, and what works for one person may not work for another. For personalized treatment plans and prescription medications, consult with a dermatologist or healthcare provider who can assess your specific condition and medical history."""

    def get_comprehensive_diabetes_response(self) -> str:
        """Comprehensive diabetes guide with clean formatting"""
        return """🩺 Medical Guidance

Thank you for reaching out about: "diabetes"

🏥 WHAT IS DIABETES:
• Diabetes is a group of metabolic disorders characterized by high blood glucose (sugar) levels
• Type 1: Autoimmune destruction of insulin-producing cells (5-10% of cases)
• Type 2: Insulin resistance and relative insulin deficiency (90-95% of cases)
• Affects over 422 million people worldwide
• Leading cause of blindness, kidney failure, heart attacks, and lower limb amputation

🧬 MEDICAL BACKGROUND & PATHOPHYSIOLOGY:
• Normal Function: Insulin helps glucose enter cells for energy
• Type 1: Immune system destroys pancreatic beta cells, no insulin production
• Type 2: Cells become resistant to insulin, pancreas can't produce enough
• Risk Factors: Genetics, obesity, physical inactivity, age, ethnicity
• Metabolic Effects: High glucose damages blood vessels and organs over time

💊 TREATMENT OPTIONS:
• Type 1: Insulin therapy (multiple daily injections or pump)
• Type 2: Lifestyle changes, oral medications (metformin), insulin if needed
• Blood glucose monitoring: Regular testing to guide treatment
• A1C testing: Measures average blood sugar over 2-3 months (goal <7%)

🛡️ PREVENTION STRATEGIES:
• Healthy diet: Limit refined sugars and processed foods
• Regular exercise: 150 minutes moderate activity per week
• Weight management: Maintain healthy BMI
• Regular screening: Especially if family history or risk factors present

⚠️ When to seek immediate medical care: Symptoms of diabetic ketoacidosis, severe hypoglycemia, or blood sugar >400 mg/dL"""

    def get_comprehensive_hypertension_response(self) -> str:
        """Comprehensive hypertension guide with clean formatting"""
        return """🩺 Medical Guidance

Thank you for reaching out about: "high blood pressure"

🏥 WHAT IS HYPERTENSION:
• High blood pressure occurs when blood pushes against artery walls with consistently high force
• Normal: <120/80 mmHg, Elevated: 120-129/<80, Stage 1: 130-139/80-89, Stage 2: ≥140/90
• Often called "silent killer" because it usually has no symptoms
• Affects nearly half of American adults
• Major risk factor for heart disease, stroke, and kidney disease

💊 TREATMENT OPTIONS:
• Lifestyle first: Diet, exercise, weight loss, sodium reduction
• ACE inhibitors: Lisinopril, enalapril (relax blood vessels)
• Diuretics: Hydrochlorothiazide (remove excess sodium and fluid)
• Beta-blockers: Metoprolol (slow heart rate, reduce force)
• Calcium channel blockers: Amlodipine (relax blood vessel muscles)

🛡️ PREVENTION & MANAGEMENT:
• DASH diet: Rich in fruits, vegetables, whole grains, lean proteins
• Limit sodium: <2300mg daily (ideal <1500mg)
• Regular exercise: 150 minutes moderate activity weekly
• Maintain healthy weight: BMI 18.5-24.9
• Limit alcohol: ≤1 drink/day women, ≤2 drinks/day men
• Don't smoke: Smoking damages blood vessels
• Manage stress: Meditation, deep breathing, regular sleep

⚠️ When to seek emergency care: Severe headache, chest pain, difficulty breathing, vision changes with very high BP readings"""

    def get_comprehensive_migraine_response(self) -> str:
        """Comprehensive migraine guide with clean formatting"""
        return """🩺 Medical Guidance

Thank you for reaching out about: "migraines"

🏥 WHAT ARE MIGRAINES:
• Neurological disorder causing severe, recurring headaches
• Often one-sided, throbbing or pulsating pain
• Can include nausea, vomiting, sensitivity to light and sound
• Affects about 12% of population, more common in women
• Can significantly impact quality of life and daily functioning

💊 TREATMENT OPTIONS:

Acute Treatment:
• Triptans: Sumatriptan, rizatriptan (first-line for moderate-severe)
• NSAIDs: Ibuprofen 400-600mg, naproxen
• Anti-nausea: Metoclopramide, ondansetron
• Ergotamines: For patients who can't use triptans

Preventive Treatment:
• Beta-blockers: Propranolol, metoprolol
• Antidepressants: Amitriptyline, venlafaxine
• Anti-seizure: Topiramate, valproate
• CGRP inhibitors: Newer preventive medications

🛡️ PREVENTION STRATEGIES:
• Identify triggers: Keep headache diary
• Regular sleep: 7-9 hours nightly, consistent schedule
• Manage stress: Relaxation techniques, regular exercise
• Dietary factors: Avoid known trigger foods
• Stay hydrated: Adequate water intake
• Limit caffeine: Sudden changes can trigger migraines

⚠️ When to seek emergency care: Sudden severe headache, headache with fever/stiff neck, headache after head injury, or changes in headache pattern"""

    def get_fever_response(self) -> str:
        """Comprehensive fever guidance with clean formatting"""
        return """🌡️ FEVER - Medical Guidance

📋 Understanding Fever:
Fever is your body's natural defense against infections. It helps your immune system fight off viruses and bacteria.

💊 Immediate Treatment:
• Acetaminophen: 500-1000mg every 6 hours
• Ibuprofen: 400-600mg every 8 hours  
• Stay hydrated: 8-10 glasses of fluids daily
• Rest: Get plenty of sleep

🥗 Supportive Care:
• Cool compresses on forehead
• Light, breathable clothing
• Room temperature: Keep cool (65-70°F)
• Nutritious foods: Soup, fruits, easy-to-digest meals

⚠️ When to Seek Medical Care:
• Temperature above 103°F (39.4°C)
• Fever lasting more than 3 days
• Severe headache with neck stiffness
• Difficulty breathing
• Signs of dehydration

📞 Emergency Signs:
• Temperature above 104°F (40°C)
• Confusion or altered mental state
• Severe difficulty breathing
• Chest pain

Most fevers resolve in 2-3 days with proper care. Monitor your symptoms and don't hesitate to contact a healthcare provider if concerned."""

    def get_headache_response(self) -> str:
        """Comprehensive headache guidance"""
        return """**🤕 HEADACHE - Medical Guidance**

**📋 Understanding Headaches:**
Most headaches are tension-type or related to stress, dehydration, or minor illness.

**💊 Treatment Options:**
• **Ibuprofen:** 400-600mg every 8 hours
• **Acetaminophen:** 1000mg every 6 hours
• **Apply cold/warm compress:** 15-20 minutes
• **Massage:** Gentle temple and neck massage

**🛠️ Self-Care Measures:**
• **Hydration:** Drink plenty of water
• **Rest:** Dark, quiet room
• **Sleep:** Regular sleep schedule
• **Stress management:** Deep breathing, relaxation

**⚠️ See a Doctor If:**
• Sudden, severe "thunderclap" headache
• Headache with fever and neck stiffness
• Changes in vision or speech
• Headache after head injury
• Headaches becoming more frequent/severe

**🔄 Prevention:**
• Stay hydrated
• Regular sleep schedule
• Manage stress
• Regular exercise
• Limit screen time

Most headaches improve within 2-4 hours with treatment."""

    def get_cold_flu_response(self) -> str:
        """Comprehensive cold/flu guidance"""
        return """**🤧 COLD/FLU - Medical Guidance**

**📋 Understanding Cold/Flu:**
Viral infections affecting your upper respiratory system. Your body is fighting off the virus.

**💊 Symptom Relief:**
• **Decongestants:** For stuffy nose
• **Cough suppressants:** For dry cough
• **Pain relievers:** Acetaminophen or ibuprofen
• **Throat lozenges:** For sore throat

**🏠 Home Care:**
• **Rest:** 8-10 hours of sleep
• **Fluids:** Water, warm tea, soup
• **Humidity:** Use humidifier or steam
• **Salt water gargle:** For sore throat
• **Honey:** Natural cough suppressant

**⚠️ See a Doctor If:**
• Fever above 103°F (39.4°C)
• Difficulty breathing
• Chest pain or pressure
• Symptoms worsen after improving
• No improvement after 10 days

**🛡️ Prevention:**
• Wash hands frequently
• Avoid touching face
• Stay away from sick people
• Get adequate sleep
• Eat nutritious foods

Most colds resolve in 7-10 days."""

    def get_pain_response(self) -> str:
        """General pain management guidance"""
        return """**😣 PAIN MANAGEMENT - Medical Guidance**

**💊 Pain Relief Options:**
• **Ibuprofen:** 400-600mg every 8 hours (anti-inflammatory)
• **Acetaminophen:** 1000mg every 6 hours
• **Topical creams:** For localized pain
• **Ice/Heat:** 15-20 minutes at a time

**🛠️ Non-Medication Approaches:**
• **Rest:** Avoid aggravating activities
• **Gentle movement:** As tolerated
• **Relaxation:** Deep breathing, meditation
• **Positioning:** Elevate if appropriate

**⚠️ Seek Medical Care For:**
• Severe pain (8/10 or higher)
• Pain after injury
• Signs of infection (redness, warmth, swelling)
• Pain that interferes with daily activities
• No improvement after 2-3 days

**📞 Emergency Care If:**
• Severe chest pain
• Signs of heart attack or stroke
• Severe abdominal pain
• Pain with fever and confusion

Pain management is individualized. What works for one person may not work for another."""

    def get_digestive_response(self) -> str:
        """Digestive issues guidance"""
        return """**🤢 DIGESTIVE ISSUES - Medical Guidance**

**💊 Immediate Relief:**
• **Clear fluids:** Water, clear broth, electrolyte drinks
• **BRAT diet:** Bananas, Rice, Applesauce, Toast
• **Anti-nausea:** Ginger tea or ginger supplements
• **Rest:** Avoid solid foods initially

**🥗 Gradual Recovery:**
• **Small, frequent meals**
• **Bland foods:** Crackers, plain pasta
• **Avoid:** Dairy, fatty, spicy, or high-fiber foods
• **Probiotics:** After acute phase

**⚠️ See a Doctor If:**
• Severe dehydration
• Blood in vomit or stool
• High fever with digestive symptoms
• Severe abdominal pain
• Unable to keep fluids down for 24 hours

**📞 Emergency Care If:**
• Signs of severe dehydration
• Severe abdominal pain
• Bloody or black stools
• Persistent vomiting

Most digestive upsets resolve in 24-48 hours with proper care."""

    def get_general_medical_response(self, message: str, mode: str) -> str:
        """General medical response for unspecified symptoms"""
        return f"""**🩺 Medical Guidance**

Thank you for reaching out about: "{message}"

**💊 General Health Recommendations:**
• **Stay hydrated:** 8-10 glasses of water daily
• **Rest:** Adequate sleep (7-8 hours)
• **Nutrition:** Balanced diet with fruits and vegetables
• **Exercise:** As tolerated, gentle movement
• **Monitor symptoms:** Keep track of changes

**⚠️ When to Seek Medical Care:**
• Symptoms persist or worsen
• New concerning symptoms develop
• Fever above 103°F (39.4°C)
• Difficulty breathing
• Severe pain

**📞 Emergency Care If:**
• Life-threatening symptoms
• Severe difficulty breathing
• Chest pain
• Signs of stroke or heart attack

For specific medical concerns, it's always best to consult with a healthcare provider who can perform a physical examination and review your medical history.

Feel free to provide more details about your symptoms for more specific guidance."""

# Global instance
ai_modes = MedicalAIModes()