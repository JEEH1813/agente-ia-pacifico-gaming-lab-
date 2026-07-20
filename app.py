import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# === IMPORTACIONES 100% SEGURAS (EVITA CUALQUIER FUNCIÓN MÓVIL DE LANGCHAIN) ===
from langchain_core.prompts import ChatPromptTemplate

# 1. Configurar ruta absoluta para el archivo .env
ruta_env = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ruta_env)

# 2. Configuración estética de la interfaz de Streamlit
st.set_page_config(page_title="Pacífico Gaming Lab - IA", page_icon="🤖", layout="centered")
st.title("🤖 Pacífico Gaming Lab")
st.write("Agente de IA Corporativo - Base de conocimiento RAG activa.")

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
    
# El spinner le avisa a Render que la app sigue viva y trabajando en segundo plano
    with st.spinner("Descargando e inicializando modelos de lenguaje... Esto puede tardar un momento la primera vez."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        base_vectores = FAISS.from_documents(fragmentos, embeddings)
    return base_vectores

# Intentar cargar los documentos e indexar los vectores
resultado = inicializar_base_conocimiento()

if resultado:
    base_vectores = resultado
    retriever = base_vectores.as_retriever(search_kwargs={"k": 3})
    
    # 4. Configurar el modelo de Groq y el Prompt Corporativo
    llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.2, 
    groq_api_key=os.getenv("GROQ_API_KEY")
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

    # === ALGORITMO RAG PURO (Solución definitiva sin chains importadas) ===
    def ejecutar_flujo_rag(pregunta_usuario):
        # A. Recuperar los documentos relevantes
        documentos_relevantes = retriever.invoke(pregunta_usuario)
        
        # B. Juntar los textos de los fragmentos en una sola cadena de texto clara
        texto_contexto = "\n\n".join([doc.page_content for doc in documentos_relevantes])
        
        # C. Darle formato al prompt inyectando las variables reales directamente
        prompt_formateado = prompt.format_messages(
            context=texto_contexto,
            input=pregunta_usuario
        )
        
        # D. Invocar el modelo y retornar solo la respuesta en texto
        respuesta_modelo = llm.invoke(prompt_formateado)
        return respuesta_modelo.content

    # 5. Historial de Chat en Streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de texto del usuario
    if user_query := st.chat_input("¿En qué puedo ayudarte hoy?"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Consultando manuales internos..."):
                answer = ejecutar_flujo_rag(user_query)
                st.markdown(answer)
                
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.error("Por favor, verifica que la carpeta 'documentos' contenga los PDFs corporativos.")