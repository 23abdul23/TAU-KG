# 🧬 Gene/Protein Knowledge Chat Application

A local vector database-powered chat application that allows you to query and interact with gene/protein data from your CSV files using natural language, enhanced with GPT-4 and PubMed citations.

## Features

- **Local Vector Database**: Uses ChromaDB to store and search gene/protein information
- **GPT-4 Enhanced Responses**: Intelligent, grounded responses using OpenAI's GPT-4
- **PubMed Citations**: Automatic literature search with 5 recent citations per query
- **Semantic Search**: Powered by sentence transformers for intelligent similarity matching
- **Interactive Chat Interface**: Streamlit-based web application with real-time responses
- **Rich Information Display**: Shows detailed gene/protein metadata with relevance scores
- **Paper Ingestion**: Upload PDFs or paste PMC/NLM links to ingest papers into the same extraction pipeline
- **Database Statistics**: View comprehensive stats about your dataset
- **Quick Actions**: Random gene lookup, database stats, and gene name browsing
- **Fallback Mode**: Works without API keys in basic mode

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Install dependencies and test the system:**
   ```bash
   python setup_and_test.py
   ```

2. **Start the chat application:**
   ```bash
   streamlit run chat_app.py
   ```

### Docker Deployment

1. Copy `.env.example` to `.env` and set your API keys.
2. Build and start the containerized app:
   ```bash
   docker compose up -d --build
   ```
3. Open the app on port `8778`:
   ```text
   http://YOUR_SERVER_IP:8778
   ```

The Docker setup runs `chat_app.py`, persists `chroma_db`, `uploaded_papers`, `data`, and the model cache in Docker volumes, and restarts automatically unless you stop it.

### Manual Setup

If you prefer to install manually:

```bash
# Install required packages
pip install -r requirements.txt

# Test the vector database
python vector_db_manager.py

# Start the chat application
streamlit run chat_app.py
```

## Usage

### Starting the Application

1. Run the setup script to install dependencies and test the system
2. Launch the Streamlit app using `streamlit run chat_app.py`
3. Open your browser to the provided URL (usually `http://localhost:8501`)
4. Click "Initialize Database" in the sidebar to load your CSV data

### Asking Questions

The chat application supports various types of queries:

**Direct Gene/Protein Queries:**
- "Tell me about MYC protein"
- "What is BRCA2?"
- "Show me information about EGFR"

**Functional Queries:**
- "What genes are related to kinase?"
- "Find proteins involved in transcription"
- "Show me genes related to cancer"

**General Searches:**
- "protein kinase"
- "transcription factor"
- "tumor suppressor"

### Features Overview

#### Chat Interface
- **Real-time responses** with gene/protein information
- **Relevance scoring** showing how well results match your query
- **Chat history** that persists during your session
- **Detailed search results** in expandable sections

#### Sidebar Information
- **Database statistics** showing total genes, sources, and types
- **Sample queries** to help you get started
- **Clear chat history** option

#### Paper Upload Workflow
- **PDF upload** for local papers
- **PMC/NLM links** for direct ingestion from National Library of Medicine article pages
- **Same downstream pipeline** for metadata extraction, entity extraction, review, and graph merge
- **Direct article access** links stored with the paper record when available

#### Quick Actions
- **🎲 Random Gene Info**: Get information about a random gene
- **📊 Database Stats**: View detailed database statistics
- **📋 All Gene Names**: Browse all available gene names

## File Structure

```
├── nodes_main.csv          # Your gene/protein data (CSV format)
├── requirements.txt        # Python dependencies
├── vector_db_manager.py    # Vector database management class
├── chat_app.py            # Main Streamlit chat application
├── setup_and_test.py      # Setup and testing script
├── README_CHAT_APP.md     # This documentation
└── chroma_db/             # ChromaDB storage (created automatically)
```

## How It Works

### Vector Database
1. **Data Loading**: CSV data is converted into text documents with metadata
2. **Embedding Generation**: Each document is converted to vector embeddings using sentence transformers
3. **Storage**: Embeddings and metadata are stored in ChromaDB for fast retrieval
4. **Search**: User queries are converted to embeddings and matched against stored documents

### Chat Application
1. **Query Processing**: User input is processed and sent to the vector database
2. **Similarity Search**: The system finds the most relevant gene/protein information
3. **Response Generation**: Results are formatted into a comprehensive response
4. **Display**: Information is presented with relevance scores and detailed metadata

## Data Format

The application expects a CSV file with the following columns:
- `node_index`: Unique index for each entry
- `node_id`: Gene/protein identifier
- `node_type`: Type of biological entity (e.g., "gene/protein")
- `node_name`: Name of the gene/protein
- `node_source`: Data source (e.g., "NCBI")

Example:
```csv
node_index,node_id,node_type,node_name,node_source
0,9796,gene/protein,PHYHIP,NCBI
1,7918,gene/protein,GPANK1,NCBI
```

## Customization

### Adding More Data
To add more CSV files or different data sources:
1. Modify the `load_csv_to_vectordb()` method in `vector_db_manager.py`
2. Update the document text format to include additional fields
3. Adjust the metadata structure as needed

### Changing the Embedding Model
To use a different sentence transformer model:
1. Update the model name in `VectorDBManager.__init__()`
2. Available models: https://huggingface.co/sentence-transformers

### Customizing the Chat Interface
The Streamlit interface can be customized by modifying `chat_app.py`:
- Change the page configuration
- Add new sidebar features
- Modify the response generation logic
- Add new quick action buttons

## Troubleshooting

### Common Issues

**"ModuleNotFoundError" when running the application:**
- Run `python setup_and_test.py` to install all dependencies

**"Collection already exists" warning:**
- This is normal - the database is reusing existing data
- Delete the `chroma_db/` folder to start fresh

**Slow initial startup:**
- The first run downloads the sentence transformer model
- Subsequent runs will be faster

**No search results:**
- Check that your CSV file is properly formatted
- Verify the database was initialized correctly
- Try more general search terms

### Performance Tips

- **First-time setup**: Initial model download may take a few minutes
- **Database size**: Larger datasets will take longer to load initially
- **Search speed**: Once loaded, searches are very fast (< 1 second)

## Technical Details

### Dependencies
- **ChromaDB**: Vector database for storing and searching embeddings
- **Sentence Transformers**: For generating text embeddings
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and CSV handling

### Architecture
```
User Query → Streamlit Interface → Vector DB Manager → ChromaDB → Search Results → Response Generation → Display
```

## License

This project uses the same license as your main project. See the LICENSE file for details.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are properly installed
4. Verify your CSV file format matches the expected structure
