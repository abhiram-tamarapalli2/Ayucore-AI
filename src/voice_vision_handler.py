"""
Voice and Vision Handler for AyuCore AI Medical Chatbot
Integrates voice input, voice output, and image analysis capabilities using HuggingFace models (primary) and Google Gemini API (fallback)
"""

import os
import logging
import base64
import subprocess
import platform
import tempfile
from io import BytesIO
from typing import Optional, Dict, Any

# Audio processing
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS

# Google Gemini AI
import google.generativeai as genai
from PIL import Image

# HuggingFace imports for image analysis
try:
    from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("HuggingFace transformers not available. Install with: pip install transformers torch torchvision")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceVisionHandler:
    """
    Handles voice input, voice output, and image analysis for medical consultation
    Uses HuggingFace models primarily with Google Gemini API as fallback
    """
    
    def __init__(self):
        """Initialize voice and vision handlers with HuggingFace models and Gemini API"""
        # Initialize HuggingFace models
        self.hf_models = {}
        self.hf_available = HF_AVAILABLE
        
        # Setup APIs and models
        self.setup_apis()
        if self.hf_available:
            self.setup_huggingface_models()
        
    def setup_apis(self):
        """Setup Google Gemini API"""
        # Ensure .env is loaded
        from dotenv import load_dotenv
        load_dotenv()
        
        try:
            # Google Gemini API for all processing
            google_api_key = os.environ.get('GOOGLE_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if google_api_key:
                genai.configure(api_key=google_api_key)
                
                # Try different models in order of preference (lightweight first to avoid quota issues)
                models_to_try = [
                    ('gemini-1.5-flash-8b', 'gemini-1.5-flash'),  # Lightweight text, standard vision
                    ('gemini-1.5-flash', 'gemini-1.5-flash'),     # Standard for both
                    ('gemini-2.0-flash', 'gemini-2.0-flash'),     # Latest version
                    ('gemini-flash-latest', 'gemini-flash-latest'), # Latest alias
                ]
                
                self.gemini_model = None
                self.gemini_vision_model = None
                
                for text_model, vision_model in models_to_try:
                    try:
                        # Test text model
                        test_model = genai.GenerativeModel(text_model)
                        test_response = test_model.generate_content("Hi")
                        
                        if test_response and test_response.text:
                            self.gemini_model = test_model
                            self.gemini_vision_model = genai.GenerativeModel(vision_model)
                            logger.info(f"Successfully configured Gemini models - Text: {text_model}, Vision: {vision_model}")
                            break
                            
                    except Exception as model_e:
                        logger.warning(f"Failed to use model {text_model}: {model_e}")
                        continue
                
                if not self.gemini_model:
                    logger.error("No working Gemini models found - all models failed or quota exceeded")
                    self.gemini_model = None
                    self.gemini_vision_model = None
                    
            else:
                logger.error("Google API key not found! Please add GOOGLE_API_KEY to your .env file")
                self.gemini_model = None
                self.gemini_vision_model = None
                
        except Exception as e:
            logger.error(f"Error setting up Gemini API: {e}")
            self.gemini_model = None
            self.gemini_vision_model = None
    
    def setup_huggingface_models(self):
        """Set up HuggingFace models for local image analysis"""
        if not self.hf_available:
            logger.warning("HuggingFace transformers not available. Install with: pip install transformers torch torchvision")
            return
            
        try:
            logger.info("🔧 Setting up HuggingFace image analysis models...")
            
            # BLIP model for medical image captioning and analysis
            logger.info("Loading BLIP model for image captioning...")
            self.hf_models['blip_processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.hf_models['blip_model'] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # Vision-to-Text model for detailed analysis
            logger.info("Loading ViT-GPT2 model for detailed image analysis...")
            self.hf_models['vit_model'] = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.hf_models['vit_processor'] = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.hf_models['vit_tokenizer'] = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            
            # Check if CUDA is available for faster processing
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Move models to appropriate device
            if device == "cuda":
                self.hf_models['blip_model'] = self.hf_models['blip_model'].to(device)
                self.hf_models['vit_model'] = self.hf_models['vit_model'].to(device)
                logger.info("✅ HuggingFace models loaded on GPU")
            else:
                logger.info("✅ HuggingFace models loaded on CPU")
                
            self.hf_models['device'] = device
            logger.info("🎯 HuggingFace image analysis models ready!")
            
        except Exception as e:
            logger.error(f"Failed to setup HuggingFace models: {e}")
            self.hf_available = False
    
    def analyze_image_with_huggingface(self, image_path):
        """Analyze image using HuggingFace models with comprehensive medical diagnosis"""
        if not self.hf_available or not self.hf_models:
            return None
            
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Get basic caption using BLIP
            inputs = self.hf_models['blip_processor'](image, return_tensors="pt")
            if self.hf_models['device'] == "cuda":
                inputs = {k: v.to(self.hf_models['device']) for k, v in inputs.items()}
                
            with torch.no_grad():
                out = self.hf_models['blip_model'].generate(**inputs, max_length=50, num_beams=3)
            basic_caption = self.hf_models['blip_processor'].decode(out[0], skip_special_tokens=True)
            
            # Get detailed analysis using ViT-GPT2
            pixel_values = self.hf_models['vit_processor'](images=image, return_tensors="pt").pixel_values
            if self.hf_models['device'] == "cuda":
                pixel_values = pixel_values.to(self.hf_models['device'])
                
            with torch.no_grad():
                output_ids = self.hf_models['vit_model'].generate(pixel_values, max_length=100, num_beams=4)
            detailed_caption = self.hf_models['vit_tokenizer'].decode(output_ids[0], skip_special_tokens=True)
            
            # Analyze visual elements for medical conditions
            medical_analysis = self._analyze_medical_condition(basic_caption, detailed_caption)
            
            # Combine comprehensive medical analysis
            analysis = f"""
🔍 Comprehensive Medical Image Analysis

Visual Description: {basic_caption}

Detailed Observation: {detailed_caption}

{medical_analysis}

Processing Details:
• Technology: Local HuggingFace models (BLIP + ViT-GPT2)
• Performance: {'GPU-accelerated' if self.hf_models.get('device') == 'cuda' else 'CPU-based'} analysis
• Privacy: Analysis performed locally without sending data to external servers

⚠️ Medical Disclaimer: This AI analysis is for informational purposes only and should not replace professional medical consultation, diagnosis, or treatment. Always consult qualified healthcare professionals for medical advice.
"""
            
            logger.info("✅ Comprehensive HuggingFace medical analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"HuggingFace comprehensive medical analysis failed: {e}")
            return None

    def _analyze_medical_condition(self, basic_description, detailed_description):
        """Analyze potential medical conditions based on visual description"""
        analysis_text = f"{basic_description} {detailed_description}".lower()
        
        # Medical condition analysis based on common visual indicators
        conditions = []
        prevention_tips = []
        general_recommendations = []
        
        # Skin conditions analysis
        if any(keyword in analysis_text for keyword in ['red', 'spots', 'rash', 'skin', 'face', 'acne']):
            conditions.append({
                'condition': 'Possible Skin Condition (Acne, Dermatitis, or Rash)',
                'description': 'Visible red spots or skin irregularities detected',
                'prevention': [
                    'Maintain good facial hygiene with gentle cleansing',
                    'Avoid touching face with unwashed hands',
                    'Use non-comedogenic skincare products',
                    'Stay hydrated and maintain healthy diet'
                ],
                'care_recommendations': [
                    'Keep affected area clean and dry',
                    'Apply topical treatments as recommended by dermatologist',
                    'Avoid picking or squeezing affected areas',
                    'Consider consulting dermatologist for persistent issues'
                ],
                'when_to_see_doctor': 'If condition persists >2 weeks, shows signs of infection, or causes significant discomfort'
            })
        
        # Eye conditions
        if any(keyword in analysis_text for keyword in ['eye', 'eyes', 'pupil', 'iris']):
            conditions.append({
                'condition': 'Eye-related Observation',
                'description': 'Visual elements related to eyes detected',
                'prevention': [
                    'Regular eye examinations (annually or as recommended)',
                    'Protect eyes from UV radiation with sunglasses',
                    'Take breaks from screen time (20-20-20 rule)',
                    'Maintain good lighting when reading or working'
                ],
                'care_recommendations': [
                    'Avoid rubbing eyes excessively',
                    'Use artificial tears if eyes feel dry',
                    'Remove contact lenses before sleeping',
                    'Follow proper contact lens hygiene'
                ],
                'when_to_see_doctor': 'For sudden vision changes, persistent irritation, or pain'
            })
        
        # General health indicators
        if any(keyword in analysis_text for keyword in ['pale', 'discoloration', 'swelling']):
            conditions.append({
                'condition': 'General Health Indicator',
                'description': 'Visual changes that may indicate health status',
                'prevention': [
                    'Maintain balanced nutrition and regular exercise',
                    'Ensure adequate sleep (7-9 hours nightly)',
                    'Stay properly hydrated',
                    'Manage stress through healthy coping mechanisms'
                ],
                'care_recommendations': [
                    'Monitor symptoms and note any changes',
                    'Take photos to track progression over time',
                    'Maintain healthy lifestyle habits',
                    'Consider keeping a symptom diary'
                ],
                'when_to_see_doctor': 'If symptoms worsen, persist, or are accompanied by other concerning signs'
            })
        
        # Build comprehensive analysis
        if not conditions:
            # Default general medical analysis
            analysis = """
🏥 Medical Assessment:
Based on the visual analysis, this appears to be a medical image. Without specific visual indicators, here's general health guidance:

🛡️ General Prevention & Wellness:
• Maintain regular health check-ups
• Follow balanced diet rich in fruits and vegetables  
• Exercise regularly (150 minutes moderate activity weekly)
• Practice good hygiene habits
• Get adequate sleep and manage stress
• Stay hydrated and limit alcohol/tobacco

👩‍⚕️ General Care Recommendations:
• Monitor any changes in your health condition
• Keep record of symptoms or concerns
• Follow prescribed treatments consistently
• Maintain open communication with healthcare providers

⚠️ When to Seek Medical Care:
• New or worsening symptoms
• Persistent discomfort or pain
• Changes that concern you
• Any signs of infection or complications
"""
        else:
            analysis = "🏥 Medical Assessment:\n"
            for i, condition in enumerate(conditions, 1):
                analysis += f"""
{i}. {condition['condition']}
Description: {condition['description']}

🛡️ Prevention Strategies:
{chr(10).join('• ' + tip for tip in condition['prevention'])}

💊 Care Recommendations:
{chr(10).join('• ' + rec for rec in condition['care_recommendations'])}

⚠️ When to See a Doctor:
{condition['when_to_see_doctor']}

"""
        
        return analysis

    # ========== VOICE INPUT FUNCTIONS ==========
    
    def record_audio(self, file_path: str, timeout: int = 20, phrase_time_limit: int = None) -> bool:
        """
        Record audio from microphone and save as WAV (better for speech recognition)
        
        Args:
            file_path: Path to save the recorded audio
            timeout: Maximum time to wait for speech
            phrase_time_limit: Maximum time for recording
        
        Returns:
            bool: True if recording successful
        """
        recognizer = sr.Recognizer()
        
        try:
            with sr.Microphone() as source:
                logger.info("Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("🎤 Start speaking now...")
                
                # Record the audio
                audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                logger.info("Recording complete.")
                
                # Save as WAV (better for speech recognition)
                wav_data = audio_data.get_wav_data()
                with open(file_path, "wb") as f:
                    f.write(wav_data)
                
                logger.info(f"Audio saved to {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return False
    
    def transcribe_audio(self, audio_filepath: str) -> Optional[str]:
        """
        Transcribe audio to text using Python SpeechRecognition with Google Speech Recognition
        
        Args:
            audio_filepath: Path to audio file
        
        Returns:
            str: Transcribed text or None if failed
        """
        try:
            recognizer = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(audio_filepath) as audio_file:
                audio_data = recognizer.record(audio_file)
            
            # Use Google Speech Recognition (free)
            text = recognizer.recognize_google(audio_data)
            logger.info("Audio transcription successful")
            return text
            
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return "Sorry, I couldn't understand the audio clearly. Please try speaking again."
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition; {e}")
            return None
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    def voice_to_text_with_fallback(self, audio_file_path: str = None, timeout: int = 20) -> Optional[str]:
        """
        Complete voice input pipeline with multiple recognition methods
        
        Args:
            audio_file_path: Path to existing audio file, if None will record live
            timeout: Recording timeout in seconds
        
        Returns:
            str: Transcribed text or None if failed
        """
        temp_audio_path = "temp_voice_input.wav"
        
        try:
            # If no audio file provided, record new audio
            if not audio_file_path:
                if not self.record_audio(temp_audio_path, timeout=timeout):
                    return None
                audio_file_path = temp_audio_path
            
            # Try transcription with multiple methods
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(audio_file_path) as audio_file:
                # Adjust for noise and record
                recognizer.adjust_for_ambient_noise(audio_file)
                audio_data = recognizer.record(audio_file)
            
            # Try Google Speech Recognition first (most accurate for English)
            try:
                text = recognizer.recognize_google(audio_data, language='en-US')
                logger.info("Google Speech Recognition successful")
                return text
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                logger.warning(f"Google Speech Recognition error: {e}")
            
            # Fallback to offline recognition if Google fails
            try:
                text = recognizer.recognize_sphinx(audio_data)
                logger.info("Sphinx (offline) recognition successful")
                return text
            except sr.UnknownValueError:
                logger.warning("Sphinx could not understand audio")
            except sr.RequestError:
                logger.warning("Sphinx recognition error")
            
            # If all methods fail
            return "Sorry, I couldn't understand the audio clearly. Please try speaking again."
            
        except Exception as e:
            logger.error(f"Voice to text error: {e}")
            return None
        finally:
            # Cleanup temp file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except:
                    pass
    
    # ========== VOICE OUTPUT FUNCTIONS ==========
    
    def text_to_speech_gtts(self, text: str, output_filepath: str, auto_play: bool = False) -> bool:
        """
        Convert text to speech using Google Text-to-Speech
        
        Args:
            text: Text to convert
            output_filepath: Output audio file path
            auto_play: Whether to play audio automatically
        
        Returns:
            bool: True if successful
        """
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_filepath)
            
            if auto_play:
                self.play_audio(output_filepath)
            
            logger.info(f"Text-to-speech saved to {output_filepath}")
            return True
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False
    
    def play_audio(self, audio_filepath: str) -> bool:
        """
        Play audio file on different operating systems
        
        Args:
            audio_filepath: Path to audio file
        
        Returns:
            bool: True if playback successful
        """
        try:
            os_name = platform.system()
            
            if os_name == "Darwin":  # macOS
                subprocess.run(['afplay', audio_filepath])
            elif os_name == "Windows":  # Windows
                subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{audio_filepath}").PlaySync();'])
            elif os_name == "Linux":  # Linux
                subprocess.run(['aplay', audio_filepath])
            else:
                logger.warning("Unsupported operating system for audio playback")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            return False
    
    # ========== IMAGE ANALYSIS FUNCTIONS ==========
    
    def encode_image(self, image_path: str) -> Optional[str]:
        """
        Encode image to base64 for API processing
        
        Args:
            image_path: Path to image file
        
        Returns:
            str: Base64 encoded image or None if failed
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Image encoding error: {e}")
            return None
    
    def analyze_medical_image_gemini(self, image_path: str, query: str = "") -> Optional[str]:
        """
        Analyze medical image using Google Gemini Vision
        
        Args:
            image_path: Path to medical image
            query: Optional specific query about the image
        
        Returns:
            str: Medical analysis or None if failed
        """
        try:
            if not self.gemini_vision_model:
                logger.error("Gemini Vision model not available for image analysis")
                return "Image analysis is currently unavailable. Please ensure your Google API key is properly configured."
            
            # Verify image file exists and is readable
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
                
            # Load and prepare image
            try:
                image = Image.open(image_path)
                logger.info(f"Image loaded successfully: {image_path}, Size: {image.size}, Mode: {image.mode}")
                
                # Convert to RGB if necessary (for better compatibility)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    logger.info(f"Image converted to RGB mode")
                    
            except Exception as img_e:
                logger.error(f"Error loading image: {img_e}")
                return "Could not load the image. Please ensure it's a valid image file."
            
            # Comprehensive medical analysis prompt
            medical_prompt = f"""
            You are Dr. Ayucore, a professional medical doctor with extensive experience in medical image analysis. 
            
            {query if query else "Please provide a comprehensive medical analysis of this image."}
            
            Please provide a detailed medical assessment including:
            
            🔍 **VISUAL OBSERVATIONS:**
            - Describe what you see in the image in medical terms
            - Note any abnormalities, lesions, discoloration, or unusual features
            - Assess the general appearance and any concerning elements
            
            🏥 **POTENTIAL MEDICAL CONDITIONS:**
            - List possible diagnoses based on visual findings
            - Explain the reasoning behind each potential condition
            - Indicate severity level (mild, moderate, severe) if applicable
            
            🛡️ **PREVENTION STRATEGIES:**
            - Provide specific prevention tips for the identified/suspected condition
            - Include lifestyle modifications and risk reduction measures
            - Recommend hygiene practices or environmental considerations
            
            💊 **TREATMENT RECOMMENDATIONS:**
            - Suggest appropriate care measures and treatments
            - Include both immediate care and long-term management
            - Mention over-the-counter options if applicable (with cautions)
            - Specify when prescription medication might be needed
            
            ⚠️ **WHEN TO SEEK MEDICAL CARE:**
            - Clearly outline warning signs that require immediate medical attention
            - Specify timeframes for follow-up care
            - Indicate when specialist referral might be necessary
            
            📝 **ADDITIONAL RECOMMENDATIONS:**
            - Suggest monitoring strategies or documentation
            - Provide general wellness advice related to the condition
            - Include any relevant dietary or activity modifications
            
            Please format your response clearly with headers and bullet points. Always emphasize that this AI analysis should complement, not replace, professional medical consultation with a qualified healthcare provider.
            """
            
            # Analyze with Gemini Vision
            logger.info("Sending image to Gemini Vision for analysis...")
            response = self.gemini_vision_model.generate_content([medical_prompt, image])
            
            if response and response.text:
                logger.info("Medical image analysis completed successfully with Gemini Vision")
                return response.text.strip()
            else:
                logger.error("Empty response from Gemini Vision")
                return "The image analysis did not return any results. Please try again with a clearer medical image."
            
        except Exception as e:
            logger.error(f"Gemini image analysis error: {e}")
            
            # Provide specific error messages based on the error type
            error_msg = str(e).lower()
            if "quota" in error_msg or "exceeded" in error_msg:
                return "⚠️ API quota exceeded: You've reached the free tier limits for today. Please try again tomorrow or upgrade to a paid plan. Meanwhile, I can still help with text-based medical questions!"
            elif "api key" in error_msg or "invalid" in error_msg:
                return "❌ API key error: Please check that your Google API key is valid and has Gemini API access enabled."
            elif "permission" in error_msg or "access" in error_msg:
                return "🔒 Permission error: Please ensure your Google API key has permission to use Gemini API. You may need to enable the Generative Language API in Google Cloud Console."
            elif "not found" in error_msg or "404" in error_msg:
                return "🔍 Model not available: The AI vision model is temporarily unavailable. Please try again later or use text-based consultation."
            elif "billing" in error_msg:
                return "💳 Billing issue: Please check your Google Cloud billing account settings."
            else:
                return f"❌ Image analysis encountered an error. Please try again later or consult a healthcare professional. (Error: {str(e)[:100]}...)"
    
    def analyze_medical_image(self, image_path: str, query: str = "") -> Optional[str]:
        """
        Main method to analyze medical images using HuggingFace models first, then Gemini Vision as fallback
        
        Args:
            image_path: Path to medical image
            query: Optional specific query about the image
        
        Returns:
            str: Medical analysis or None if failed
        """
        logger.info("🏥 Starting medical image analysis...")
        
        # First try HuggingFace models (local processing)
        if self.hf_available and self.hf_models:
            logger.info("🔬 Attempting HuggingFace image analysis...")
            hf_result = self.analyze_image_with_huggingface(image_path)
            
            if hf_result:
                logger.info("✅ HuggingFace analysis successful - returning result")
                return hf_result
            else:
                logger.warning("⚠️ HuggingFace analysis failed, trying Gemini API fallback...")
        else:
            logger.info("📡 HuggingFace not available, using Gemini API directly...")
        
        # Fallback to Gemini API
        gemini_result = self.analyze_medical_image_gemini(image_path, query)
        
        if gemini_result:
            logger.info("✅ Gemini API analysis successful")
            # Add a note that this was processed via Gemini API
            return f"""🌟 **Google Gemini AI Analysis**

{gemini_result}

---
*Analysis provided by Google Gemini AI (cloud-based processing)*"""
        
        # If both methods fail
        logger.error("❌ Both HuggingFace and Gemini analysis failed")
        return "I'm sorry, I'm unable to analyze the medical image at this time. Both local and cloud-based analysis methods are currently unavailable. Please consult with a healthcare professional for proper medical evaluation."
    
    # ========== INTEGRATED FUNCTIONS ==========
    
    def process_voice_and_image(self, audio_filepath: str = None, image_filepath: str = None, 
                               voice_timeout: int = 20) -> Dict[str, Any]:
        """
        Complete pipeline: process voice input and/or image analysis
        
        Args:
            audio_filepath: Path to audio file (if None, will record)
            image_filepath: Path to image file
            voice_timeout: Timeout for voice recording
        
        Returns:
            dict: Results containing transcription, analysis, and audio response
        """
        results = {
            'transcription': None,
            'image_analysis': None,
            'response_text': None,
            'response_audio': None,
            'success': False
        }
        
        try:
            # Process voice input
            if audio_filepath:
                results['transcription'] = self.transcribe_audio(audio_filepath)
            else:
                results['transcription'] = self.voice_to_text_with_fallback(timeout=voice_timeout)
            
            # Process image
            if image_filepath and os.path.exists(image_filepath):
                query = results['transcription'] if results['transcription'] else ""
                results['image_analysis'] = self.analyze_medical_image(image_filepath, query)
            
            # Generate response
            if results['image_analysis']:
                response_text = results['image_analysis']
            elif results['transcription']:
                response_text = f"I heard you say: '{results['transcription']}'. Could you please provide more details or share an image for medical analysis?"
            else:
                response_text = "I didn't receive any voice input or image. Please try again."
            
            results['response_text'] = response_text
            
            # Generate voice response
            audio_output_path = "temp_response.mp3"
            if self.text_to_speech_gtts(response_text, audio_output_path):
                results['response_audio'] = audio_output_path
            
            results['success'] = True
            logger.info("Voice and image processing completed successfully")
            
        except Exception as e:
            logger.error(f"Voice and image processing error: {e}")
            results['error'] = str(e)
        
        return results

# Global instance
voice_vision_handler = VoiceVisionHandler()