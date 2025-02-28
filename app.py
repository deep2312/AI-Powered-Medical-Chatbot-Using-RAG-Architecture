import streamlit as st
import sqlite3
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
from llama_cpp import Llama
import torch
import bcrypt

# Paths
DB_FAISS_PATH = 'vectorstore/db_faiss'
BIOMISTRAL_MODEL_PATH = "path/to/your/BioMistral-7B.Q2_K.gguf"  # Update with the local path to your GGUF model

# Load PubMedBERT (for embeddings)
embed_tokenizer = AutoTokenizer.from_pretrained("NeuML/pubmedbert-base-embeddings")
embed_model = AutoModel.from_pretrained("NeuML/pubmedbert-base-embeddings")

# Load BioMistral GGUF model
llama = Llama(model_path=BIOMISTRAL_MODEL_PATH)

# Generate embeddings using PubMedBERT
def generate_embeddings(text):
    inputs = embed_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Load FAISS vector database
@st.cache_resource
def load_vector_db():
    db = FAISS.load_local(DB_FAISS_PATH, allow_dangerous_deserialization=True)
    return db

# Semantic search
def query_vector_db(query, db, k=3):
    query_embedding = generate_embeddings(query)
    results = db.similarity_search_by_vector(query_embedding, k=k)
    return results

# Generate response using the locally stored BioMistral model
def generate_response(prompt):
    response = llama(prompt, max_tokens=200, temperature=0.7, top_p=0.9)
    return response["choices"][0]["text"]

# SQLite database setup
conn = sqlite3.connect('chatbot.db', check_same_thread=False)
c = conn.cursor()

# Create tables for user authentication and chat history
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    topic TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER,
    sender TEXT,
    message TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
)
''')
conn.commit()

# Password hashing functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def check_password(hashed_password: str, password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

# User management functions
def create_user(username: str, password: str):
    hashed_password = hash_password(password)
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
    conn.commit()

def authenticate_user(username: str, password: str):
    c.execute('SELECT id, password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    if result and check_password(result[1], password):
        return result[0]  # User ID
    return None

# Streamlit UI setup
st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.title("Medical Chatbot")

# Authentication
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

if st.session_state["user_id"] is None:
    option = st.selectbox("Choose option", ["Login", "Signup"])

    if option == "Signup":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Create Account"):
            if password != confirm_password:
                st.error("Passwords do not match!")
            else:
                create_user(username, password)
                st.success("Account created successfully. Please log in.")

    elif option == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user_id = authenticate_user(username, password)
            if user_id:
                st.session_state["user_id"] = user_id
                st.session_state["username"] = username
                st.success(f"Welcome back, {username}!")
            else:
                st.error("Invalid username or password")

# Chatbot functionality
if st.session_state["user_id"] is not None:
    db = load_vector_db()

    # Sidebar for conversation management
    st.sidebar.header("Conversation History")

    # Start new conversation
    new_conversation = st.sidebar.text_input("New Conversation Topic")
    if st.sidebar.button("Start New Conversation"):
        if new_conversation:
            c.execute("INSERT INTO conversations (user_id, topic) VALUES (?, ?)", (st.session_state["user_id"], new_conversation))
            conn.commit()
            conversation_id = c.lastrowid
            st.session_state["conversation_id"] = conversation_id
            st.session_state["topic"] = new_conversation
            st.session_state["messages"] = []

    # Load previous conversations
    conversations = c.execute("SELECT id, topic FROM conversations WHERE user_id = ?", (st.session_state["user_id"],)).fetchall()
    conversation_to_load = st.sidebar.selectbox("Select Conversation to Load", [None] + [row[1] for row in conversations])
    if st.sidebar.button("Load Selected Conversation"):
        if conversation_to_load:
            conversation_id = c.execute("SELECT id FROM conversations WHERE topic = ? AND user_id = ?", (conversation_to_load, st.session_state["user_id"])).fetchone()[0]
            st.session_state["conversation_id"] = conversation_id
            st.session_state["topic"] = conversation_to_load
            st.session_state["messages"] = c.execute("SELECT sender, message FROM messages WHERE conversation_id = ?", (conversation_id,)).fetchall()

    # Main chat area
    if "conversation_id" in st.session_state and st.session_state["conversation_id"]:
        st.header(f"Topic: {st.session_state['topic']}")
        user_input = st.text_input("You: ")

        if st.button("Send"):
            if user_input:
                # Query FAISS database
                search_results = query_vector_db(user_input, db)

                # Combine results into a prompt for BioMistral
                context = "\n\n".join([result.page_content for result in search_results])
                prompt = f"Context:\n{context}\n\nUser: {user_input}\n\nAnswer concisely and accurately:"

                # Generate response using BioMistral
                bot_response = generate_response(prompt)

                # Save messages to database
                c.execute("INSERT INTO messages (conversation_id, sender, message) VALUES (?, ?, ?)", (st.session_state["conversation_id"], "User", user_input))
                c.execute("INSERT INTO messages (conversation_id, sender, message) VALUES (?, ?, ?)", (st.session_state["conversation_id"], "Bot", bot_response))
                conn.commit()

                # Update session history
                st.session_state["messages"].append(("User", user_input))
                st.session_state["messages"].append(("Bot", bot_response))

        # Display conversation history
        st.header("Conversation")
        for sender, message in st.session_state["messages"]:
            st.write(f"**{sender}:** {message}")
            st.write("---")
