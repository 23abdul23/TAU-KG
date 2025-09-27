#!/usr/bin/env python3
"""
Setup and test script for the Gene/Protein Knowledge Chat application
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def test_vector_db():
    """Test the vector database functionality"""
    print("\nTesting vector database...")
    try:
        from vector_db_manager import VectorDBManager
        
        # Initialize database
        print("Initializing vector database...")
        db_manager = VectorDBManager()
        
        # Load CSV data
        print("Loading CSV data...")
        db_manager.load_csv_to_vectordb("nodes_main.csv")
        
        # Test search functionality
        print("Testing search functionality...")
        results = db_manager.search_similar("protein kinase", n_results=3)
        
        print(f"✅ Search test successful! Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['metadata']['node_name']} (Score: {(1-result['distance'])*100:.1f}%)")
        
        # Test enhanced response generation
        print("\nTesting enhanced response generation...")
        try:
            enhanced_result = db_manager.generate_enhanced_response("protein kinase", results, max_tokens=512)
            print(f"✅ Enhanced response test successful!")
            print(f"  GPT-4 available: {enhanced_result['has_openai']}")
            print(f"  Citations available: {enhanced_result['has_citations']}")
            if enhanced_result['citations']:
                print(f"  Found {len(enhanced_result['citations'])} citations")
            print(f"  Response preview: {enhanced_result['gpt_response'][:100]}...")
        except Exception as e:
            print(f"⚠️  Enhanced response test failed: {e}")
            print("  This is expected if API keys are not configured")
        
        # Get database stats
        stats = db_manager.get_database_stats()
        print(f"\n📊 Database Statistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Sources: {list(stats['sources'].keys())}")
        print(f"  Node types: {list(stats['node_types'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing vector database: {e}")
        return False

def main():
    """Main setup and test function"""
    print("🧬 Gene/Protein Knowledge Chat - Setup and Test")
    print("=" * 50)
    
    # Check if nodes_main.csv exists
    if not os.path.exists("nodes_main.csv"):
        print("❌ Error: nodes_main.csv not found in current directory!")
        print("Please make sure the CSV file is in the same directory as this script.")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Test vector database
    if not test_vector_db():
        return False
    
    print("\n🎉 Setup and testing completed successfully!")
    print("\nTo start the chat application, run:")
    print("  streamlit run chat_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
