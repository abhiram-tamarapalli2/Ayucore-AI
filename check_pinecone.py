#!/usr/bin/env python3
"""
Simple script to check and fix Pinecone index
"""

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
