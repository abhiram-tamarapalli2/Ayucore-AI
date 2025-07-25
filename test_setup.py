#!/usr/bin/env python3
"""
Test script to verify the medical chatbot setup
"""

print("🔍 Testing Medical Chatbot Setup...")
print("=" * 50)

# Test 1: Import all required modules
try:
    print("1. Testing imports...")
    from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
    from pinecone import Pinecone
    from langchain_google_genai import ChatGoogleGenerativeAI
    from dotenv import load_dotenv
    import os
    print("   ✅ All imports successful!")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    exit(1)

# Test 2: Load environment variables
try:
    print("2. Testing environment variables...")
    load_dotenv()
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    
    if PINECONE_API_KEY and GOOGLE_API_KEY:
        print("   ✅ API keys loaded successfully!")
    else:
        print("   ❌ API keys not found in .env file")
        exit(1)
except Exception as e:
    print(f"   ❌ Environment error: {e}")
    exit(1)

# Test 3: Test PDF loading
try:
    print("3. Testing PDF loading...")
    extracted_data = load_pdf_file(data='Data/')
    print(f"   ✅ Loaded {len(extracted_data)} PDF pages!")
except Exception as e:
    print(f"   ❌ PDF loading error: {e}")
    exit(1)

# Test 4: Test text chunking
try:
    print("4. Testing text chunking...")
    text_chunks = text_split(extracted_data)
    print(f"   ✅ Created {len(text_chunks)} text chunks!")
except Exception as e:
    print(f"   ❌ Text chunking error: {e}")
    exit(1)

# Test 5: Test embeddings
try:
    print("5. Testing embeddings...")
    embeddings = download_hugging_face_embeddings()
    print("   ✅ Embeddings model loaded successfully!")
except Exception as e:
    print(f"   ❌ Embeddings error: {e}")
    exit(1)

# Test 6: Test Pinecone connection
try:
    print("6. Testing Pinecone connection...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    indexes = pc.list_indexes()
    print(f"   ✅ Pinecone connected! Available indexes: {[idx.name for idx in indexes]}")
except Exception as e:
    print(f"   ❌ Pinecone connection error: {e}")
    exit(1)

# Test 7: Test Gemini connection
try:
    print("7. Testing Gemini AI connection...")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)
    
    # Test a simple generation
    test_response = llm.invoke("Hello, this is a test.")
    print("   ✅ Gemini AI model initialized and tested successfully!")
    print(f"   Test response: {test_response.content[:50]}...")
except Exception as e:
    print(f"   ❌ Gemini AI error: {e}")
    print("   💡 Tip: Check if your Google API key is valid and has Gemini access enabled")
    exit(1)

print("=" * 50)
print("🎉 ALL TESTS PASSED! Your medical chatbot is ready!")
print("📋 Next steps:")
print("   1. Run: python store_index.py (to create vector database)")
print("   2. Run: python app.py (to start the web application)")
print("   3. Open: http://localhost:8080 (to use the chatbot)")
