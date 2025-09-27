# Gene/Protein Knowledge Chat Application - Sequence Flow Diagram

## Overview
This document presents the sequence flow diagrams for the Gene/Protein Knowledge Chat Application, showing the interactions between users, components, and external services.

## Main Chat Query Flow

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#2E86AB',
    'primaryTextColor': '#000000',
    'primaryBorderColor': '#1B5E7F',
    'lineColor': '#4A90A4',
    'secondaryColor': '#A23B72',
    'tertiaryColor': '#F18F01',
    'background': '#F8F9FA',
    'mainBkg': '#FFFFFF',
    'secondBkg': '#E8F4F8',
    'tertiaryBkg': '#FFF3E0',
    'actorTextColor': '#000000',
    'noteTextColor': '#000000'
  }
}}%%
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant CA as Chat App
    participant VDB as Vector DB Manager
    participant CDB as ChromaDB
    participant ST as Sentence Transformer
    participant AI as OpenAI GPT-4
    participant CS as Citation System
    participant PM as PubMed API

    rect rgb(240, 248, 255)
        Note over U, PM: User Query Processing Phase
        U->>+UI: Enter gene/protein query
        UI->>+CA: Process chat input
        CA->>CA: Add to chat history
        CA->>+VDB: search_similar(query, n_results=5)
    end
    
    rect rgb(245, 255, 245)
        Note over VDB, CDB: Vector Search Phase
        VDB->>+ST: Encode query to vector
        ST-->>-VDB: Query embedding
        VDB->>+CDB: query(query_texts, n_results)
        CDB-->>-VDB: Search results with distances
        VDB-->>-CA: Formatted search results
    end
    
    rect rgb(255, 248, 240)
        Note over CA, AI: AI Enhancement Phase
        CA->>+VDB: generate_enhanced_response(query, results)
        VDB->>VDB: _format_context_for_gpt(results)
        VDB->>+AI: chat.completions.create(GPT-4)
        AI-->>-VDB: Enhanced response
    end
    
    rect rgb(255, 240, 245)
        Note over VDB, PM: Literature Search Phase
        par Citation Fetching
            VDB->>+CS: _get_pubmed_citations(query)
            CS->>CS: extract_entities_with_llm(query)
            CS->>+AI: Extract biomedical entities
            AI-->>-CS: Extracted entities
            CS->>CS: optimize_query(query, entities)
            CS->>+PM: esearch.fcgi (search PMIDs)
            PM-->>-CS: PMID list
            CS->>+PM: efetch.fcgi (fetch details)
            PM-->>-CS: Article XML data
            CS->>CS: parse_pubmed_xml(xml_data)
            CS-->>-VDB: Citation objects
        end
    end
    
    rect rgb(248, 255, 248)
        Note over VDB, U: Response Display Phase
        VDB-->>-CA: Enhanced response with citations
        CA->>UI: Display response
        CA->>UI: Display citations
        CA->>UI: Show search results dropdown
        UI-->>-U: Complete response with literature
    end
```

## Database Initialization Flow

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#6C5CE7',
    'primaryTextColor': '#FFFFFF',
    'primaryBorderColor': '#5A4FCF',
    'lineColor': '#74B9FF',
    'secondaryColor': '#00B894',
    'tertiaryColor': '#FDCB6E',
    'background': '#F8F9FA',
    'mainBkg': '#FFFFFF',
    'secondBkg': '#F1F2F6',
    'tertiaryBkg': '#DDD6FE'
  }
}}%%
sequenceDiagram
    participant U as 👤 User
    participant UI as 🖥️ Streamlit UI
    participant CA as 🎯 Chat App
    participant VDB as 🧠 Vector DB Manager
    participant CDB as 📊 ChromaDB
    participant ST as 🔤 Sentence Transformer
    participant CSV as 📄 CSV File

    rect rgb(245, 245, 255)
        Note over U, CSV: 🚀 Database Initialization Phase
        U->>+UI: 🔧 Click "Initialize Database"
        UI->>+CA: ⚡ initialize_database()
        CA->>+VDB: 🏗️ VectorDBManager()
        VDB->>+CDB: 🔗 PersistentClient(path)
        VDB->>+ST: 🤖 SentenceTransformer('all-MiniLM-L6-v2')
        VDB->>CDB: 📦 get_collection() or create_collection()
        CDB-->>-VDB: ✅ Collection reference
    end
    
    rect rgb(240, 255, 240)
        Note over CA, CSV: 📊 Data Loading Phase
        CA->>VDB: 📥 load_csv_to_vectordb("nodes_main.csv")
        VDB->>+CDB: 🔢 collection.count()
        CDB-->>-VDB: 📈 Document count
        
        alt 📋 Collection is empty
            rect rgb(255, 250, 240)
                VDB->>+CSV: 📖 pd.read_csv(csv_path)
                CSV-->>-VDB: 📊 DataFrame
                
                loop 🔄 For each row in DataFrame
                    VDB->>VDB: 📝 Create document text
                    VDB->>VDB: 🏷️ Create metadata dict
                end
                
                VDB->>+CDB: 💾 collection.add(documents, metadatas, ids)
                CDB-->>-VDB: ✅ Success confirmation
            end
        else 📚 Collection exists
            VDB->>VDB: ⏭️ Skip loading (already populated)
        end
    end
    
    rect rgb(248, 255, 248)
        Note over VDB, U: 🎉 Completion Phase
        VDB-->>-CA: ✅ Load complete
        CA->>UI: 🔄 Update session state
        UI-->>-U: 🎉 "Database initialized successfully!"
    end
```

## Error Handling and Fallback Flow

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#E74C3C',
    'primaryTextColor': '#FFFFFF',
    'primaryBorderColor': '#C0392B',
    'lineColor': '#E67E22',
    'secondaryColor': '#F39C12',
    'tertiaryColor': '#27AE60',
    'background': '#F8F9FA',
    'mainBkg': '#FFFFFF',
    'secondBkg': '#FDEDEC',
    'tertiaryBkg': '#FEF9E7'
  }
}}%%
sequenceDiagram
    participant CA as 🎯 Chat App
    participant VDB as 🧠 Vector DB Manager
    participant AI as 🤖 OpenAI API
    participant CS as 📚 Citation System

    rect rgb(255, 245, 245)
        Note over CA, CS: ⚠️ Error Handling & Fallback Phase
        CA->>+VDB: 🚀 generate_enhanced_response()
        
        alt 🤖 OpenAI Available
            rect rgb(240, 255, 240)
                VDB->>+AI: 🧠 GPT-4 API call
                alt ✅ API Success
                    AI-->>-VDB: ✨ Enhanced response
                else ❌ API Failure
                    AI-->>VDB: 🚨 Error
                    VDB->>VDB: 📝 Log error
                    VDB->>CA: 🔄 generate_basic_response()
                end
            end
        else 🚫 OpenAI Not Available
            rect rgb(255, 250, 240)
                VDB->>CA: 🔄 generate_basic_response()
            end
        end
        
        par 📚 Citation Handling
            rect rgb(245, 250, 255)
                VDB->>+CS: 📖 fetch_pubmed_citations()
                alt ✅ Citations Available
                    CS-->>-VDB: 📋 Citation list
                else ❌ Citations Failed
                    CS-->>VDB: 📭 Empty list
                    VDB->>VDB: ⚠️ Log warning
                end
            end
        end
        
        VDB-->>-CA: 📊 Display response with status indicators
    end
```

## Quick Actions Flow

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#00B894',
    'primaryTextColor': '#FFFFFF',
    'primaryBorderColor': '#00A085',
    'lineColor': '#00CEC9',
    'secondaryColor': '#6C5CE7',
    'tertiaryColor': '#FDCB6E',
    'background': '#F8F9FA',
    'mainBkg': '#FFFFFF',
    'secondBkg': '#E8F8F5',
    'tertiaryBkg': '#F0F8FF'
  }
}}%%
sequenceDiagram
    participant U as 👤 User
    participant UI as 🖥️ Streamlit UI
    participant VDB as 🧠 Vector DB Manager
    participant CDB as 📊 ChromaDB

    alt 🎲 Random Gene Info
        rect rgb(240, 255, 240)
            Note over U, CDB: 🎯 Random Gene Discovery
            U->>+UI: 🎲 Click "Random Gene Info"
            UI->>+VDB: 📋 get_all_gene_names()
            VDB->>+CDB: 📊 collection.get()
            CDB-->>-VDB: 📄 All documents with metadata
            VDB->>VDB: 🔄 Extract and deduplicate gene names
            VDB-->>UI: 📝 Gene names list
            UI->>UI: 🎰 random.choice(gene_names)
            UI->>+VDB: 🔍 get_gene_info(random_gene)
            VDB->>+CDB: 🎯 query(gene_name, where clause)
            CDB-->>-VDB: 🧬 Gene information
            VDB-->>-UI: 📊 Gene details
            UI-->>-U: 🎉 Display random gene info
        end
    
    else 📊 Database Stats
        rect rgb(245, 250, 255)
            Note over U, CDB: 📈 Statistics Overview
            U->>+UI: 📊 Click "Database Stats"
            UI->>+VDB: 📈 get_database_stats()
            VDB->>+CDB: 📊 collection.get()
            CDB-->>-VDB: 📄 All metadata
            VDB->>VDB: 🔢 Count by source and type
            VDB-->>UI: 📊 Statistics dict
            UI-->>-U: 📋 Display stats as JSON
        end
    
    else 📋 All Gene Names
        rect rgb(255, 250, 240)
            Note over U, CDB: 📚 Gene Catalog Browse
            U->>+UI: 📋 Click "All Gene Names"
            UI->>+VDB: 📝 get_all_gene_names()
            VDB->>+CDB: 📊 collection.get()
            CDB-->>-VDB: 📄 All metadata
            VDB->>VDB: 🔤 Extract, sort, deduplicate names
            VDB-->>UI: 📋 Sorted gene names
            UI-->>-U: 📜 Display gene names (first 20)
        end
    end
```

## Entity Extraction and Query Optimization Flow

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#A23B72',
    'primaryTextColor': '#FFFFFF',
    'primaryBorderColor': '#8E2A5B',
    'lineColor': '#D63384',
    'secondaryColor': '#F18F01',
    'tertiaryColor': '#2E86AB',
    'background': '#F8F9FA',
    'mainBkg': '#FFFFFF',
    'secondBkg': '#FDF2F8',
    'tertiaryBkg': '#FFF8E1'
  }
}}%%
sequenceDiagram
    participant CS as 📚 Citation System
    participant AI as 🤖 OpenAI API
    participant RE as 🔍 Regex Engine

    rect rgb(253, 242, 248)
        Note over CS, RE: 🧬 Entity Extraction Phase
        CS->>CS: 🔍 extract_entities_with_llm(query)
        
        alt 🤖 OpenAI Available
            rect rgb(240, 248, 255)
                CS->>+AI: 🧬 Extract biomedical entities
                Note over AI: 🎯 System prompt for entity extraction<br/>📋 Categories: genes, proteins, diseases, pathways, keywords
                AI-->>-CS: 📊 JSON with categorized entities
                CS->>CS: ✅ Validate and format entities
            end
        else 🚫 OpenAI Not Available
            rect rgb(255, 248, 240)
                CS->>+RE: 🔍 extract_entities_from_text(query)
                RE->>RE: 🎯 Apply regex patterns
                RE-->>-CS: 🏷️ Extracted entities
            end
        end
    end
    
    rect rgb(255, 248, 225)
        Note over CS, AI: ⚡ Query Optimization Phase
        CS->>CS: ⚡ optimize_query(query, entities)
        
        alt 🧠 LLM Query Generation Enabled
            rect rgb(245, 255, 245)
                CS->>+AI: 🔧 generate_improved_query_with_llm()
                Note over AI: 🎯 System prompt for PubMed query optimization<br/>🏷️ Uses MeSH terms, field tags, boolean logic
                AI-->>-CS: ✨ Optimized PubMed query
            end
        else 📋 Rule-based Optimization
            rect rgb(255, 250, 240)
                CS->>CS: 🏗️ Build structured query
                Note over CS: 🏷️ Add MeSH terms, field tags<br/>🔗 Combine with boolean operators
            end
        end
        
        CS-->>CS: 🎯 Final optimized query
    end
```

## Configuration and Environment Setup Flow

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#F18F01',
    'primaryTextColor': '#FFFFFF',
    'primaryBorderColor': '#D67E00',
    'lineColor': '#FF7675',
    'secondaryColor': '#6C5CE7',
    'tertiaryColor': '#00B894',
    'background': '#F8F9FA',
    'mainBkg': '#FFFFFF',
    'secondBkg': '#FFF8E1',
    'tertiaryBkg': '#F0F8FF'
  }
}}%%
sequenceDiagram
    participant APP as 🚀 Application
    participant ENV as 📄 .env File
    participant CFG as ⚙️ config.json
    participant AI as 🤖 OpenAI Client
    participant NCBI as 🔬 NCBI API

    rect rgb(255, 248, 225)
        Note over APP, NCBI: 🔧 Environment Configuration Phase
        APP->>+ENV: 📥 load_dotenv()
        ENV-->>-APP: 🔑 Environment variables loaded
        
        APP->>+CFG: ⚙️ load_config("config.json")
        CFG-->>-APP: 📊 Configuration settings
    end
    
    rect rgb(240, 248, 255)
        Note over APP, AI: 🤖 OpenAI Setup Phase
        alt 🔑 OpenAI Setup
            APP->>APP: 🔍 Check OPENAI_API_KEY
            alt ✅ API Key Available
                APP->>+AI: 🚀 Initialize OpenAI client
                alt 🌐 Proxy Configured
                    rect rgb(245, 255, 245)
                        APP->>APP: 🔧 Setup httpx client with proxy
                        APP->>AI: 🌐 OpenAI(http_client=proxy_client)
                    end
                else 🚫 No Proxy
                    rect rgb(255, 250, 240)
                        APP->>AI: 🔗 OpenAI(api_key)
                    end
                end
                AI-->>-APP: ✅ Client initialized
            else ❌ No API Key
                APP->>APP: 🚫 Set OPENAI_CLIENT = None
            end
        end
    end
    
    rect rgb(245, 255, 245)
        Note over APP, NCBI: 🔬 NCBI Setup Phase
        alt 🔬 NCBI Setup
            APP->>APP: 🔍 Check NCBI_API_KEY
            alt ✅ API Key Available
                APP->>APP: ⚡ Set rate limit to 10/sec
            else ❌ No API Key
                APP->>APP: 🐌 Set rate limit to 3/sec
            end
        end
    end
    
    rect rgb(248, 255, 248)
        Note over APP, APP: 🎯 Final Setup Phase
        APP->>APP: 📝 Initialize logging configuration
        APP->>APP: 🌐 Setup proxy configuration from config
    end
```

## Key Flow Characteristics

### Performance Considerations
- **Caching**: ChromaDB provides persistent storage, avoiding re-computation
- **Rate Limiting**: Different rates for NCBI API based on key availability
- **Parallel Processing**: Citations fetched in parallel with response generation
- **Lazy Loading**: Database initialization only when requested

### Error Resilience
- **Graceful Degradation**: Falls back to basic responses if GPT-4 fails
- **Retry Logic**: Multiple attempts for network requests with exponential backoff
- **Validation**: Input validation and sanitization at multiple levels
- **Logging**: Comprehensive logging for debugging and monitoring

### Security Features
- **API Key Management**: Secure handling of OpenAI and NCBI keys
- **Proxy Support**: Configurable proxy settings for enterprise environments
- **Input Sanitization**: Safe handling of user queries and responses
- **Rate Limiting**: Respects API rate limits to avoid blocking
