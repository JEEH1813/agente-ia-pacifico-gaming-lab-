import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Configurar ruta absoluta para el archivo .env
ruta_env = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ruta_env)

# 2. Configuración estética de la interfaz de Streamlit (Ancho completo para mejor visualización)
st.set_page_config(
    page_title="Pacífico Gaming Lab - IA", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para presencia corporativa
st.markdown("""
    <style>
    /* Estilos de tarjetas para el chat */
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
    }
    
    /* Encabezado principal estilizado */
    .header-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px;
        border-radius: 16px;
        border-left: 6px solid #2563eb;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Insignia de estado */
    .badge-online {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #059669;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# BARRA LATERAL (SIDEBAR) CORPORATIVA
with st.sidebar:
    st.markdown("## 🤖 Pacífico Gaming Lab")
    st.markdown("<span class='badge-online'>🟢 RAG Activo & En Línea</span>", unsafe_allow_html=True)
    st.caption("Uramba Centro de Innovación")
    
    st.divider()
    st.markdown("### 📌 Arquitectura del Sistema")
    st.markdown("""
    * **LLM Engine:** Groq (Llama-3.1-8B)
    * **Vector Store:** FAISS Vector Index
    * **Embeddings:** HuggingFace MiniLM
    * **Seguridad:** Aislamiento por Contexto
    """)
    
    st.divider()
    st.markdown("### 💡 Preguntas Rápidas (Demo)")
    st.caption("Selecciona una pregunta frecuente para interactuar de inmediato:")
    
    # Botones interactivos para demostración rápida ante la Junta Directiva
    p1 = st.button("❓ ¿Qué es Pacífico Gaming Lab?", use_container_width=True)
    p2 = st.button("📋 ¿Qué servicios o reglas aplican?", use_container_width=True)
    p3 = st.button("👥 ¿Cómo contactar con soporte?", use_container_width=True)

# ENCABEZADO PRINCIPAL DE LA APLICACIÓN
st.markdown("""
    <div class="header-box">
        <h1 style="margin: 0; font-size: 2rem; color: #ffffff;">🤖 Agente Virtual Corporativo</h1>
        <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
            Asistente Inteligente basado en RAG para la consulta de manuales y documentos internos.
        </p>
    </div>
""", unsafe_allow_html=True)

# 3. Inicialización de la base de datos de vectores (Cacheada)
@st.cache_resource
def inicializar_base_conocimiento():
    ruta_documentos = "documentos"
    if not os.path.exists(ruta_documentos):
        return None
    
    loader = PyPDFDirectoryLoader(ruta_documentos)
    documentos = loader.load()
    if not documentos:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = text_splitter.split_documents(documentos)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    base_vectores = FAISS.from_documents(fragmentos, embeddings)
    return base_vectores

# Intentar cargar los documentos e indexar los vectores
resultado = inicializar_base_conocimiento()

if resultado:
    base_vectores = resultado
    retriever = base_vectores.as_retriever(search_kwargs={"k": 3})
    
    # 4. Configurar el modelo de Groq
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.2, 
        groq_api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    )
    
    system_prompt = (
        "Eres el agente de IA corporativo oficial de Pacífico Gaming Lab. "
        "Tu objetivo es asistir a los usuarios y clientes usando únicamente la información proporcionada en el contexto.\n"
        "Si la respuesta no se encuentra en los documentos de contexto, responde amablemente que no posees esa "
        "información detallada en tus manuales actuales y ofrece escalarlo con soporte humano.\n"
        "Mantén un tono profesional, tecnológico y servicial.\n\n"
        "Contexto de la empresa:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Algoritmo RAG
    def ejecutar_flujo_rag(pregunta_usuario):
        documentos_relevantes = retriever.invoke(pregunta_usuario)
        texto_contexto = "\n\n".join([doc.page_content for doc in documentos_relevantes])
        
        prompt_formateado = prompt.format_messages(
            context=texto_contexto,
            input=pregunta_usuario
        )
        
        respuesta_modelo = llm.invoke(prompt_formateado)
        return respuesta_modelo.content

    # 5. Historial de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial previo
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Captura de input: Ya sea por los botones del Sidebar o por la barra de chat tradicional
    user_query = None
    
    if p1:
        user_query = "¿Qué es Pacífico Gaming Lab y cuál es su objetivo principal?"
    elif p2:
        user_query = "¿Cuáles son los servicios principales o reglamentos descritos en los manuales?"
    elif p3:
        user_query = "¿Cuál es el canal para contactar con soporte o atención?"
    
    # Si el usuario escribió en el chat box, sobreescribe
    if chat_input_query := st.chat_input("¿En qué puedo ayudarte hoy sobre Pacífico Gaming Lab?"):
        user_query = chat_input_query

    # Si hay una consulta (de botón o de teclado), procesar
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Consultando base de conocimiento RAG..."):
                answer = ejecutar_flujo_rag(user_query)
                st.markdown(answer)
                
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.error("⚠️ No se pudo inicializar la base de datos. Por favor, verifica que la carpeta 'documentos' contenga los PDFs corporativos.")