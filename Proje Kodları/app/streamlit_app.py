import streamlit as st
import sys
import os

# Set page config as FIRST Streamlit command!
st.set_page_config(
    page_title="RAG + Intent Classifier Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    
    .intent-message {
        background-color: #fff3e0;
        border-left-color: #ff9800;
        font-size: 0.9rem;
    }
    
    .sidebar-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .model-selector {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .input-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .confidence-badge {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Path ayarları - doğru yolları ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'models'))

from models.llama_model import LlamaModel
from models.gemini_model import GeminiModel

# --- Streamlit UI ---
st.markdown("""
<div class="main-header">
    <h1>🤖 RAG + Intent Classifier Chatbot</h1>
    <p>Advanced AI-Supported Question-Answer System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with model selection
with st.sidebar:
    st.markdown("""
    <div class="model-selector">
        <h3>🎯 Model Selection</h3>
    </div>
    """, unsafe_allow_html=True)
    
    model_option = st.selectbox(
        "Select the AI model:",
        ["Gemini", "Llama"],
        help="Select the AI model you want to use"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div class="sidebar-info">
        <h4>📖 Usage Guide</h4>
        <ol>
            <li>Select the model (Gemini or Llama)</li>
            <li>Type your question here...</li>
            <li>Optional: Use only classifier</li>
            <li>Send button</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-info">
        <h4>✨ Features</h4>
        <div class="feature-card">
            <strong>🎯 Intent Classification</strong><br>
            Predict the intent of the user's question
        </div>
        <div class="feature-card">
            <strong>🔍 RAG (Retrieval Augmented Generation)</strong><br>
            Find similar questions and answer
        </div>
        <div class="feature-card">
            <strong>🛡️ Fallback</strong><br>
            Low confidence score fallback answer
        </div>
    </div>
    """, unsafe_allow_html=True)

# State init for model and messages
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = model_option
    st.session_state.messages = []

# Clear chat if model changes
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

# API Key dosya yolları
api_key_path = os.path.join(parent_dir, "API_KEY.txt")

# Model yükleme
@st.cache_resource
def load_model(model_name):
    try:
        if model_name == "Gemini":
            return GeminiModel(
                api_key_path=api_key_path,
                data_path=os.path.join(parent_dir, "data/data_set.xlsx"),
                vector_store_path=os.path.join(parent_dir, "models/vector_store.pkl"),
                intent_model_path=os.path.join(parent_dir, "models/intent_classifier.pkl")
            )
        else:  # Llama
            return LlamaModel(
                api_key_path=api_key_path,
                data_path=os.path.join(parent_dir, "data/data_set.xlsx"),
                vector_store_path=os.path.join(parent_dir, "models/vector_store.pkl"),
                intent_model_path=os.path.join(parent_dir, "models/intent_classifier.pkl")
            )
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        return None

# Model yükle
with st.spinner(f"🔄 {model_option} model is loading..."):
    model = load_model(model_option)

if model is None:
    st.error("❌ Model could not be loaded. Please check the API key files.")
    st.stop()

# Chat container
chat_container = st.container()

with chat_container:
    # Display chat messages
    if st.session_state.messages:
        st.markdown("### 💬 Chat History")
        
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 Siz:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            elif msg["role"] == "assistant":
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Bot:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            elif msg["role"] == "intent":
                confidence_color = "🟢" if msg['confidence'] > 70 else "🟡" if msg['confidence'] > 50 else "🔴"
                st.markdown(f"""
                <div class="chat-message intent-message">
                    <strong>🎯 Predicted Intent:</strong> <code>{msg['intent']}</code> 
                    {confidence_color} Confidence: <span class="confidence-badge">{msg['confidence']:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

# Input area
st.markdown("---")
st.markdown("""
<div class="input-container">
    <h3>💭 Type your question here...</h3>
</div>
""", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_question = st.text_input(
            "Type your question here...",
            placeholder="Example: Hello, how are you?",
            help="Type your question here"
        )
    
    with col2:
        use_classifier_only = st.checkbox(
            "Only Classifier", 
            value=False,
            help="Answer only with intent classifier, without RAG"
        )
    
    submitted = st.form_submit_button("🚀 Send", use_container_width=True)

# On user submit
if submitted and user_question:
    try:
        with st.spinner("🤔 Thinking..."):
            # Model response
            final_reply, intent_pred, confidence = model.get_response(
                user_question, 
                use_classifier_only=use_classifier_only
            )
        
        # Store messages
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "assistant", "content": final_reply})
        st.session_state.messages.append({"role": "intent", "intent": intent_pred, "confidence": confidence})
        
        # Force rerun to scroll
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 RAG + Intent Classifier Chatbot | Advanced AI-Supported Question-Answer System</p>
</div>
""", unsafe_allow_html=True) 