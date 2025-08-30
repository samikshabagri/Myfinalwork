import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Configure Streamlit page
st.set_page_config(
    page_title="🌊 OceanFlow Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Modern gradient background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Card styling */
    .stCard {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
    }
    
    /* Chat message styling */
    .chat-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        margin-left: 2rem;
    }
    
    .chat-message.assistant {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin-right: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed #667eea;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Success/Error messages */
    .success-message {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# API base URL
API_BASE_URL = "http://localhost:8000"

def make_api_request(endpoint, method="GET", data=None):
    """Make API request to the backend."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = None
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        elif method == "PUT":
            response = requests.put(url, json=data)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None

        if response is not None:
            response.raise_for_status()
            return response.json()
        else:
            st.error("No response received from server")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def main():
    # Sidebar
    st.sidebar.title("🚢 Enhanced Maritime AI Agent")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["🚀 Enhanced RAG AI Chat", "🗺️ Voyage Planning", "📦 Cargo Matching", "📊 Market Analysis", "💰 PDA Calculator", "🔍 Data Explorer"]
    )
    
    # Main content
    if page == "🚀 Enhanced RAG AI Chat":
        chat_interface()
    elif page == "🗺️ Voyage Planning":
        voyage_planning()
    elif page == "📦 Cargo Matching":
        cargo_matching()
    elif page == "📊 Market Analysis":
        market_analysis()
    elif page == "💰 PDA Calculator":
        pda_calculator()
    elif page == "🔍 Data Explorer":
        data_explorer()

def chat_interface():
    # Header with animated gradient
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    ">
        <h1 style="color: white; font-size: 3rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🌊 OceanFlow Intelligence
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.95);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        ">
            <h3 style="color: #667eea; margin: 0;">🧠 GPT-4 Powered</h3>
            <p style="color: #666; margin: 0.5rem 0;">Advanced language understanding and generation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.95);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        ">
            <h3 style="color: #667eea; margin: 0;">🔍 LlamaIndex RAG</h3>
            <p style="color: #666; margin: 0.5rem 0;">Advanced retrieval with document context</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.95);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        ">
            <h3 style="color: #667eea; margin: 0;">📄 Document Upload</h3>
            <p style="color: #666; margin: 0.5rem 0;">Upload PDF, DOCX, TXT for enhanced knowledge</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Initialize session state for advanced RAG
    if "advanced_messages" not in st.session_state:
        st.session_state.advanced_messages = []
    
    # Create tabs for different chat modes
    tab1, tab2 = st.tabs(["💬 Basic Chat", "🧠 Advanced RAG Chat"])
    
    with tab1:
        basic_chat_interface()
    
    with tab2:
        advanced_rag_chat_interface()

def basic_chat_interface():
    """Basic chat interface without document context"""
    st.subheader("💬 Basic Maritime AI Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show additional data if available
            if message.get("data"):
                with st.expander("🔍 View Detailed Data"):
                    st.json(message["data"])
            
            # Show suggestions if available
            if message.get("suggestions"):
                st.markdown("**💡 Suggestions:**")
                for suggestion in message["suggestions"]:
                    st.markdown(f"• {suggestion}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about maritime operations..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Processing..."):
                response = make_api_request("/chat", method="POST", data={"query": prompt})
                
                if response and response.get("success"):
                    # Display main response
                    st.markdown(response["response"])
                    
                    # Show action results if available
                    if response.get("action_result", {}).get("status") == "success":
                        action_data = response["action_result"]["data"]
                        st.success(f"✅ **Action Completed**: {response['action_result']['action'].replace('_', ' ').title()}")
                        
                        # Display action-specific data
                        if response["action_result"]["action"] == "voyage_planning":
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Distance", f"{action_data.get('distance_nm', 0):.0f} NM")
                                st.metric("Duration", f"{action_data.get('total_voyage_days', 0):.1f} days")
                            with col2:
                                st.metric("Total Cost", f"${action_data.get('total_voyage_cost_usd', 0):,.0f}")
                                st.metric("Fuel Cost", f"${action_data.get('fuel_cost_usd', 0):,.0f}")
                            with col3:
                                st.metric("ETA", action_data.get('eta', 'N/A')[:10] if action_data.get('eta') else 'N/A')
                                st.metric("Risk", action_data.get('piracy_risk', 'N/A').title())
                        
                        elif response["action_result"]["action"] == "cargo_matching":
                            matches = action_data.get("matches", [])
                            if matches:
                                st.metric("Matches Found", len(matches))
                                # Show top matches
                                if len(matches) > 0:
                                    top_match = matches[0]
                                    st.info(f"**Top Match**: {top_match.get('cargo_id', 'N/A')} - {top_match.get('commodity', 'N/A')} ({top_match.get('quantity_mt', 0):,.0f} MT)")
                        
                        elif response["action_result"]["action"] == "market_analysis":
                            st.metric("BDI", action_data.get('current_bdi', 0))
                            st.metric("VLSFO Price", f"${action_data.get('current_vlsfo_usd_per_mt', 0)}/MT")
                    
                    # Show suggestions
                    if response.get("suggestions"):
                        st.markdown("**💡 Try These Queries:**")
                        for suggestion in response["suggestions"]:
                            st.markdown(f"• {suggestion}")
                    
                    # Store response with additional data
                    assistant_message = {
                        "role": "assistant",
                        "content": response["response"],
                        "data": response.get("action_result", {}).get("data"),
                        "suggestions": response.get("suggestions")
                    }
                    
                else:
                    error_msg = response.get("response", "I couldn't process that request. Please try again.")
                    st.markdown(error_msg)
                    assistant_message = {"role": "assistant", "content": error_msg}
        
        # Add assistant response to chat history
        st.session_state.messages.append(assistant_message)
    
    # Sidebar for basic chat
    with st.sidebar:
        st.markdown("### 💬 Basic Chat")
        
        if st.button("🗑️ Clear Chat", key="clear_advanced_rag_chat"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("### 💡 Example Queries")
        example_queries = [
            "Plan voyage for vessel 9700001 from BRSSZ to CNSHA",
            "Find cargo matches for Panamax vessels",
            "Show market trends for Capesize",
            "Calculate PDA for vessel 9700001"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"basic_{hash(query)}", use_container_width=True):
                st.session_state.basic_example_query = query
                st.rerun()

def advanced_rag_chat_interface():
    """Advanced RAG chat interface with document upload and context"""
    
    # Modern header for Advanced RAG
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    ">
        <h2 style="color: white; margin: 0; font-size: 2rem;">🧠 Advanced RAG Chat</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;">Intelligent Document Processing & Context-Aware Responses</p>
    </div>
    """, unsafe_allow_html=True)
    
        # Document upload section with modern styling
    with st.container():
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.95);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin-bottom: 2rem;
        ">
            <h3 style="color: #667eea; margin: 0 0 1rem 0;">📄 Upload Documents for Enhanced Knowledge</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose documents to upload",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files for enhanced RAG capabilities"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Prepare file for upload
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    try:
                        # Upload to advanced RAG system
                        response = requests.post(f"{API_BASE_URL}/advanced-chat/upload", files=files)
                        response.raise_for_status()
                        result = response.json()
                        
                        if result.get("success"):
                            st.success(f"✅ {uploaded_file.name} processed successfully!")
                            st.info(f"📄 Added to knowledge base for enhanced responses")
                        else:
                            st.error(f"❌ Error processing {uploaded_file.name}: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Error uploading {uploaded_file.name}: {str(e)}")
    
    # Display uploaded documents
    with st.expander("📚 Knowledge Base Documents"):
        documents_result = make_api_request("/advanced-chat/documents")
        
        if documents_result and documents_result.get("success"):
            documents = documents_result.get("documents", [])
            if documents:
                for doc in documents:
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(f"📄 **{doc['filename']}** ({doc['file_type'].upper()})")
                    with col2:
                        st.write(f"📅 {doc['upload_date'][:10]}")
                    with col3:
                        if st.button("🗑️", key=f"delete_rag_{doc['id']}", help="Delete document"):
                            delete_result = make_api_request(f"/advanced-chat/documents/{doc['id']}", method="DELETE")
                            if delete_result and delete_result.get("success"):
                                st.success("Document deleted!")
                                st.rerun()
                            else:
                                st.error("Error deleting document")
            else:
                st.info("No documents uploaded yet. Upload some documents to enhance the AI's knowledge!")
        else:
            st.warning("Unable to load documents list")
    
    # Display advanced chat history with modern styling
    for message in st.session_state.advanced_messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                color: white;
                padding: 1rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                margin-left: 2rem;
            ">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                margin-right: 2rem;
            ">
                <strong>🤖 AI Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show source documents if available
            if message.get("source_documents"):
                st.markdown("""
                <div style="
                    background: rgba(255, 255, 255, 0.95);
                    padding: 1rem;
                    border-radius: 10px;
                    margin: 0.5rem 0;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                ">
                    <h4 style="color: #667eea; margin: 0 0 0.5rem 0;">📄 Sources:</h4>
                """, unsafe_allow_html=True)
                
                for source in message["source_documents"]:
                    if isinstance(source, str):
                        st.markdown(f"<p style='margin: 0.25rem 0;'><strong>📎</strong> {source}</p>", unsafe_allow_html=True)
                    elif isinstance(source, dict):
                        filename = source.get('filename', 'Unknown')
                        score = source.get('score', 'N/A')
                        content = source.get('content', '')
                        st.markdown(f"<p style='margin: 0.25rem 0;'><strong>📎 {filename}</strong> (Score: {score})</p>", unsafe_allow_html=True)
                        if content:
                            st.markdown(f"<p style='margin: 0.25rem 0; color: #666; font-style: italic;'>{content}</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='margin: 0.25rem 0;'><strong>📎</strong> {str(source)}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Show suggestions if available
            if message.get("suggestions"):
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                    padding: 1rem;
                    border-radius: 10px;
                    margin: 0.5rem 0;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                ">
                    <h4 style="color: #d63031; margin: 0 0 0.5rem 0;">💡 Suggestions:</h4>
                """, unsafe_allow_html=True)
                
                for suggestion in message["suggestions"]:
                    st.markdown(f"<p style='margin: 0.25rem 0;'>• {suggestion}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Advanced chat input
    if prompt := st.chat_input("Ask me anything with document context..."):
        # Add user message
        st.session_state.advanced_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response with context
        with st.chat_message("assistant"):
            with st.spinner("🧠 Processing with Advanced RAG..."):
                # Convert chat history to the format expected by the API
                chat_history = []
                for i in range(0, len(st.session_state.advanced_messages) - 1, 2):
                    if i + 1 < len(st.session_state.advanced_messages):
                        chat_history.append({
                            "role": st.session_state.advanced_messages[i]["role"],
                            "content": st.session_state.advanced_messages[i]["content"]
                        })
                        chat_history.append({
                            "role": st.session_state.advanced_messages[i + 1]["role"],
                            "content": st.session_state.advanced_messages[i + 1]["content"]
                        })
                
                response = make_api_request("/advanced-chat", method="POST", data={
                    "query": prompt,
                    "chat_history": chat_history
                })
                
                if response and response.get("success"):
                    st.markdown(response["response"])
                    
                    # Store response with source documents
                    assistant_message = {
                        "role": "assistant",
                        "content": response["response"],
                        "source_documents": response.get("source_documents", []),
                        "suggestions": response.get("suggestions", [])
                    }
                else:
                    if response is None:
                        error_msg = "I couldn't connect to the server. Please try again."
                    else:
                        error_msg = response.get("error", "I couldn't process that question. Please try again.")
                    st.markdown(error_msg)
                    assistant_message = {"role": "assistant", "content": error_msg}
        
        # Add assistant response to chat history
        st.session_state.advanced_messages.append(assistant_message)
    
    # Sidebar for advanced RAG chat with modern styling
    with st.sidebar:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        ">
            <h3 style="color: white; margin: 0;">🧠 Advanced RAG Chat</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Advanced Chat", key="clear_advanced_chat"):
            st.session_state.advanced_messages = []
            st.rerun()
        
        # System status with modern cards
        status_result = make_api_request("/advanced-chat/status")
        if status_result and status_result.get("success"):
            status = status_result["status"]
            st.markdown("""
            <div style="
                background: rgba(255, 255, 255, 0.95);
                padding: 1rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            ">
                <h4 style="color: #667eea; margin: 0 0 1rem 0;">📊 System Status</h4>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Documents", status["documents_loaded"])
                st.metric("🧠 LlamaIndex", "✅" if status["llamaindex_ready"] else "❌")
                st.metric("🤖 LLM Model", status["llm_model"])
            with col2:
                st.metric("📊 Chunks", status["chunks_available"])
                st.metric("🔑 OpenAI", "✅" if status["openai_configured"] else "❌")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Advanced examples with modern styling
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.95);
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        ">
            <h4 style="color: #667eea; margin: 0 0 1rem 0;">💡 Quick Examples</h4>
        """, unsafe_allow_html=True)
        
        advanced_examples = [
            "What are the key points in the uploaded documents?",
            "Summarize the main findings from all documents",
            "Find information about costs and pricing in the documents",
            "What are the recommendations mentioned in the documents?",
            "Extract all dates and deadlines from the documents"
        ]
        
        for example in advanced_examples:
            if st.button(example, key=f"advanced_{hash(example)}", use_container_width=True):
                st.session_state.advanced_example_query = example
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    


def voyage_planning():
    st.title("🗺️ Voyage Planning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Plan Voyage")
        
        vessel_imo = st.text_input("Vessel IMO", value="9700001")
        load_port = st.text_input("Load Port", value="BRSSZ")
        disch_port = st.text_input("Discharge Port", value="CNSHA")
        speed_knots = st.slider("Speed (knots)", 10.0, 18.0, 14.0, 0.5)
        route_variant = st.selectbox("Route Variant", ["DIRECT", "SUEZ", "PANAMA", "CAPE"])
        
        if st.button("Plan Voyage", key="plan_voyage"):
            with st.spinner("Planning voyage..."):
                data = {
                    "vessel_imo": vessel_imo,
                    "load_port": load_port,
                    "disch_port": disch_port,
                    "speed_knots": speed_knots,
                    "route_variant": route_variant
                }
                
                result = make_api_request("/voyage/plan", method="POST", data=data)
                
                if result:
                    st.success("Voyage planned successfully!")
                    
                    # Display voyage details
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Distance", f"{result['distance_nm']:.0f} NM")
                        st.metric("Duration", f"{result['total_voyage_days']:.1f} days")
                    with col2:
                        st.metric("Total Cost", f"${result['total_voyage_cost_usd']:,.0f}")
                        st.metric("Fuel Cost", f"${result['fuel_cost_usd']:,.0f}")
                    with col3:
                        st.metric("ETA", result['eta'][:10])
                        st.metric("Risk", result['piracy_risk'].title())
                    
                    # Show detailed breakdown
                    with st.expander("Voyage Details"):
                        st.json(result)
    
    with col2:
        st.subheader("Compare Routes")
        
        comp_vessel_imo = st.text_input("Vessel IMO (Compare)", value="9700001")
        comp_load_port = st.text_input("Load Port (Compare)", value="BRSSZ")
        comp_disch_port = st.text_input("Discharge Port (Compare)", value="CNSHA")
        
        if st.button("Compare Routes", key="compare_routes"):
            with st.spinner("Comparing routes..."):
                data = {
                    "vessel_imo": comp_vessel_imo,
                    "load_port": comp_load_port,
                    "disch_port": comp_disch_port
                }
                
                result = make_api_request("/voyage/compare-routes", method="POST", data=data)
                
                if result and result.get("comparisons"):
                    comparisons = result["comparisons"]
                    
                    # Create comparison table
                    df = pd.DataFrame(comparisons[:5])  # Top 5
                    df = df[['route_variant', 'speed_knots', 'total_voyage_cost_usd', 'total_voyage_days', 'distance_nm']]
                    df.columns = ['Route', 'Speed (kts)', 'Cost (USD)', 'Days', 'Distance (NM)']
                    df['Cost (USD)'] = df['Cost (USD)'].apply(lambda x: f"${x:,.0f}")
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Cost comparison chart
                    fig = px.bar(
                        comparisons[:5],
                        x='route_variant',
                        y='total_voyage_cost_usd',
                        title="Route Cost Comparison",
                        labels={'route_variant': 'Route', 'total_voyage_cost_usd': 'Cost (USD)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

def cargo_matching():
    st.title("📦 Cargo Matching")
    
    tab1, tab2, tab3 = st.tabs(["🚢 Find Cargo for Vessel", "📦 Find Vessel for Cargo", "🎯 Optimal Matches"])
    
    with tab1:
        st.subheader("Find Cargo Matches for Vessel")
        
        vessel_imo = st.text_input("Vessel IMO", value="9700001")
        min_tce = st.number_input("Minimum TCE (USD/day)", value=5000.0)
        max_ballast = st.number_input("Max Ballast Distance (NM)", value=2000.0)
        
        if st.button("Find Cargo Matches", key="find_cargo_matches"):
            with st.spinner("Finding cargo matches..."):
                data = {
                    "vessel_imo": vessel_imo,
                    "min_tce_usd_per_day": min_tce,
                    "max_ballast_distance_nm": max_ballast
                }
                
                result = make_api_request("/cargo/find-matches", method="POST", data=data)
                
                if result and result.get("matches"):
                    matches = result["matches"]
                    
                    if matches:
                        # Create matches table
                        df = pd.DataFrame(matches[:10])  # Top 10
                        df = df[['cargo_id', 'commodity', 'quantity_mt', 'load_port', 'disch_port', 'tce_analysis']]
                        df['TCE (USD/day)'] = df['tce_analysis'].apply(lambda x: x['tce_usd_per_day'])
                        df = df[['cargo_id', 'commodity', 'quantity_mt', 'load_port', 'disch_port', 'TCE (USD/day)']]
                        df.columns = ['Cargo ID', 'Commodity', 'Quantity (MT)', 'Load Port', 'Disch Port', 'TCE (USD/day)']
                        
                        st.dataframe(df, use_container_width=True)
                        
                        # TCE distribution chart
                        tce_values = [m['tce_analysis']['tce_usd_per_day'] for m in matches[:10]]
                        fig = px.histogram(x=tce_values, title="TCE Distribution", labels={'x': 'TCE (USD/day)', 'y': 'Count'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No suitable cargoes found.")
    
    with tab2:
        st.subheader("Find Vessel Matches for Cargo")
        
        cargo_id = st.text_input("Cargo ID", value="CARG-001")
        min_tce = st.number_input("Minimum TCE (USD/day)", value=5000.0, key="vessel_tce")
        
        if st.button("Find Vessel Matches", key="find_vessel_matches"):
            with st.spinner("Finding vessel matches..."):
                data = {
                    "cargo_id": cargo_id,
                    "min_tce_usd_per_day": min_tce
                }
                
                result = make_api_request("/cargo/find-vessel-matches", method="POST", data=data)
                
                if result and result.get("matches"):
                    matches = result["matches"]
                    
                    if matches:
                        # Create matches table
                        df = pd.DataFrame(matches[:10])
                        df = df[['vessel_name', 'vessel_type', 'dwt', 'tce_analysis']]
                        df['TCE (USD/day)'] = df['tce_analysis'].apply(lambda x: x['tce_usd_per_day'])
                        df = df[['vessel_name', 'vessel_type', 'dwt', 'TCE (USD/day)']]
                        df.columns = ['Vessel Name', 'Type', 'DWT', 'TCE (USD/day)']
                        
                        st.dataframe(df, use_container_width=True)
                        st.success(f"✅ Found {len(matches)} vessel matches")
                    else:
                        st.warning("No suitable vessels found.")
                else:
                    st.error("❌ Error: Unable to fetch vessel matches")
                    if result:
                        st.json(result)
    
    with tab3:
        st.subheader("Optimal Vessel-Cargo Combinations")
        
        min_tce = st.number_input("Minimum TCE (USD/day)", value=5000.0, key="optimal_tce")
        max_matches = st.number_input("Max Matches", value=20, min_value=1, max_value=50)
        
        if st.button("Find Optimal Matches", key="find_optimal_matches"):
            with st.spinner("Finding optimal matches..."):
                result = make_api_request(f"/cargo/optimal-matches?min_tce_usd_per_day={min_tce}&max_matches={max_matches}")
                
                if result and result.get("matches"):
                    matches = result["matches"]
                    
                    if matches:
                        # Create matches table
                        df = pd.DataFrame(matches[:10])
                        df = df[['vessel_name', 'cargo_id', 'commodity', 'tce_analysis']]
                        df['TCE (USD/day)'] = df['tce_analysis'].apply(lambda x: x['tce_usd_per_day'])
                        df = df[['vessel_name', 'cargo_id', 'commodity', 'TCE (USD/day)']]
                        df.columns = ['Vessel', 'Cargo', 'Commodity', 'TCE (USD/day)']
                        
                        st.dataframe(df, use_container_width=True)
                        st.success(f"✅ Found {len(matches)} optimal matches")
                    else:
                        st.warning("No optimal matches found.")
                else:
                    st.error("❌ Error: Unable to fetch optimal matches")
                    if result:
                        st.json(result)

def market_analysis():
    st.title("📊 Market Analysis")
    
    # Market summary
    st.subheader("Market Summary")
    
    if st.button("Refresh Market Data", key="refresh_market_data"):
        with st.spinner("Loading market data..."):
            summary = make_api_request("/market/summary")
            
            if summary:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("BDI", summary['current_bdi'])
                with col2:
                    st.metric("VLSFO Price", f"${summary['current_vlsfo_usd_per_mt']}/MT")
                with col3:
                    st.metric("Market Sentiment", summary['market_sentiment'].title())
                with col4:
                    st.metric("BDI Trend", f"{summary['bdi_trend_percent']}%")
                
                # Vessel analysis
                st.subheader("Vessel Type Analysis")
                vessel_data = summary.get('vessel_analysis', {})
                
                if vessel_data:
                    vessel_df = pd.DataFrame([
                        {
                            'Vessel Type': k,
                            'Estimated Rate': v['estimated_rate_usd_per_mt'],
                            'Outlook': v['market_outlook']
                        }
                        for k, v in vessel_data.items()
                    ])
                    
                    st.dataframe(vessel_df, use_container_width=True)
    
    # Freight rate trends
    st.subheader("Freight Rate Trends")
    
    vessel_type = st.selectbox("Vessel Type", ["Capesize", "Panamax", "Supramax", "Handysize", "Kamsarmax"])
    route = st.text_input("Route", value="BRSSZ-CNSHA")
    
    if st.button("Get Trends", key="get_trends"):
        with st.spinner("Loading trends..."):
            trends = make_api_request(f"/market/trends/{vessel_type}?route={route}")
            
            if trends:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current BDI", trends['current_bdi'])
                with col2:
                    st.metric("Estimated Rate", f"${trends['estimated_freight_rate_usd_per_mt']}/MT")
                with col3:
                    st.metric("Trend", trends['trend_direction'].title())
                
                st.info(trends['recommendation'])
                
                # Historical data chart
                if trends.get('historical_data'):
                    hist_df = pd.DataFrame(trends['historical_data'])
                    hist_df['date'] = pd.to_datetime(hist_df['date'])
                    
                    fig = px.line(hist_df, x='date', y='BDI', title=f"{vessel_type} BDI Trend")
                    st.plotly_chart(fig, use_container_width=True)

def pda_calculator():
    st.title("💰 PDA Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Calculate PDA")
        
        vessel_imo = st.text_input("Vessel IMO", value="9700001")
        load_port = st.text_input("Load Port", value="BRSSZ")
        disch_port = st.text_input("Discharge Port", value="CNSHA")
        bunker_port = st.text_input("Bunker Port (optional)", value="")
        fuel_type = st.selectbox("Fuel Type", ["VLSFO", "HSFO", "LSMGO"])
        
        if st.button("Calculate PDA", key="calculate_pda"):
            with st.spinner("Calculating PDA..."):
                data = {
                    "vessel_imo": vessel_imo,
                    "load_port": load_port,
                    "disch_port": disch_port,
                    "fuel_type": fuel_type
                }
                
                if bunker_port:
                    data["bunker_port"] = bunker_port
                
                result = make_api_request("/pda/calculate", method="POST", data=data)
                
                if result:
                    st.success("PDA calculated successfully!")
                    
                    # Display PDA breakdown
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total PDA", f"${result['total_pda_usd']:,.0f}")
                        st.metric("Load Port Fees", f"${result['load_port_fees']['total']:,.0f}")
                        st.metric("Discharge Port Fees", f"${result['disch_port_fees']['total']:,.0f}")
                    with col2:
                        st.metric("Bunker Costs", f"${result['bunker_costs']['total_cost']:,.0f}")
                        st.metric("Canal Costs", f"${result['canal_costs']['canal_toll_usd']:,.0f}")
                        st.metric("Budget Status", result['budget_analysis']['status'])
                    
                    # Cost breakdown pie chart
                    cost_data = {
                        'Port Fees': result['load_port_fees']['total'] + result['disch_port_fees']['total'],
                        'Bunker': result['bunker_costs']['total_cost'],
                        'Canal': result['canal_costs']['canal_toll_usd'],
                        'Additional': result['additional_costs']['total']
                    }
                    
                    fig = px.pie(
                        values=list(cost_data.values()),
                        names=list(cost_data.keys()),
                        title="PDA Cost Breakdown"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Bunker Port Comparison")
        
        comp_vessel_imo = st.text_input("Vessel IMO (Compare)", value="9700001")
        comp_load_port = st.text_input("Load Port (Compare)", value="BRSSZ")
        comp_disch_port = st.text_input("Discharge Port (Compare)", value="CNSHA")
        candidate_ports = st.text_input("Candidate Ports (comma-separated)", value="BRSSZ,SGSIN,AEJEA")
        
        if st.button("Compare Bunker Ports", key="compare_bunker_ports"):
            with st.spinner("Comparing bunker ports..."):
                ports_list = [p.strip() for p in candidate_ports.split(",")]
                
                data = {
                    "vessel_imo": comp_vessel_imo,
                    "load_port": comp_load_port,
                    "disch_port": comp_disch_port,
                    "fuel_type": "VLSFO",
                    "candidate_ports": ports_list
                }
                
                result = make_api_request("/pda/compare-bunker-ports", method="POST", data=data)
                
                if result and result.get("comparisons"):
                    comparisons = result["comparisons"]
                    
                    # Create comparison table
                    df = pd.DataFrame(comparisons)
                    df = df[['port', 'price_usd_per_mt', 'total_cost_usd', 'savings_vs_load_port']]
                    df.columns = ['Port', 'Price (USD/MT)', 'Total Cost (USD)', 'Savings vs Load Port (USD)']
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Cost comparison chart
                    fig = px.bar(
                        df,
                        x='Port',
                        y='Total Cost (USD)',
                        title="Bunker Port Cost Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True)



def data_explorer():
    st.title("🔍 Data Explorer")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚢 Vessels", "📦 Cargos", "🏢 Ports", "Market Data"])
    
    with tab1:
        st.subheader("Vessel Data")
        
        vessel_type = st.selectbox("Vessel Type", ["", "Capesize", "Panamax", "Supramax", "Handysize", "Kamsarmax"])
        min_dwt = st.number_input("Min DWT", value=0.0)
        max_dwt = st.number_input("Max DWT", value=200000.0)
        
        if st.button("Load Vessels", key="load_vessels"):
            with st.spinner("Loading vessel data..."):
                params = f"?min_dwt={min_dwt}&max_dwt={max_dwt}"
                if vessel_type:
                    params += f"&vessel_type={vessel_type}"
                
                result = make_api_request(f"/data/vessels{params}")
                
                if result and result.get("vessels"):
                    vessels = result["vessels"]
                    df = pd.DataFrame(vessels)
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Vessel type distribution
                    if 'type' in df.columns:
                        fig = px.pie(df, names='type', title="Vessel Type Distribution")
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Cargo Data")
        
        cargo_type = st.selectbox("Cargo Type", ["", "Iron Ore", "Coal", "Grain", "Soybeans", "Sugar", "Fertilizer", "Cement"])
        load_port = st.text_input("Load Port Filter", value="")
        min_qty = st.number_input("Min Quantity (MT)", value=0.0)
        max_qty = st.number_input("Max Quantity (MT)", value=200000.0)
        
        if st.button("Load Cargos", key="load_cargos"):
            with st.spinner("Loading cargo data..."):
                params = f"?min_quantity={min_qty}&max_quantity={max_qty}"
                if cargo_type:
                    params += f"&cargo_type={cargo_type}"
                if load_port:
                    params += f"&load_port={load_port}"
                
                result = make_api_request(f"/data/cargos{params}")
                
                if result and result.get("cargos"):
                    cargos = result["cargos"]
                    df = pd.DataFrame(cargos)
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Cargo type distribution
                    if 'commodity' in df.columns:
                        fig = px.pie(df, names='commodity', title="Cargo Type Distribution")
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Port Information")
        
        port_code = st.text_input("Port Code", value="BRSSZ")
        
        if st.button("Get Port Details", key="get_port_details"):
            with st.spinner("Loading port details..."):
                result = make_api_request(f"/data/ports/{port_code}")
                
                if result:
                    st.json(result)
    
    with tab4:
        st.subheader("Market Data")
        
        if st.button("Load Market Summary", key="load_market_summary"):
            with st.spinner("Loading market data..."):
                summary = make_api_request("/utils/summary")
                
                if summary:
                    st.json(summary)

def add_footer():
    """Add beautiful footer with developer credit"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    ">
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        ">
            <h3 style="color: white; margin: 0; font-size: 1.5rem;">🌊 OceanFlow Intelligence</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0; font-size: 1rem;">
                Advanced Maritime AI Assistant
            </p>
        </div>
                 <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;">
             <strong>🚀 Developed with ❤️ by:</strong>
         </p>
         <h3 style="color: white; margin: 0.5rem 0; font-size: 2rem; font-weight: bold;">
             👩‍💻 Samiksha Bagri
         </h3>
         <h4 style="color: rgba(255,255,255,0.9); margin: 0.5rem 0; font-size: 1.5rem; font-weight: bold;">
             🌟 NEXT GEN
         </h4>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    add_footer()
