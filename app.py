import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Configuración de la interfaz web en Streamlit
st.set_page_config(page_title="Pacífico Gaming Lab - IA", page_icon="🤖", layout="centered")
st.title("🤖 Pacífico Gaming Lab - Agente de IA Corporativo")
st.write("Sistema centralizado de consultas basado en documentación interna.")

# 2. Función encargada de cargar, procesar y vectorizar los documentos PDF
@st.cache_resource
def inicializar_base_conocimiento():
    ruta_documentos = "documentos"
    
    if not os.path.exists(ruta_documentos):
        st.error(f"La carpeta '{ruta_documentos}' no fue encontrada.")
        return None
        
    # A. Cargar los PDFs
    loader = PyPDFDirectoryLoader(ruta_documentos)
    documentos = loader.load()
    
    if not documentos:
        return None
        
    # B. Dividir el texto en fragmentos pequeños (Chunks)
    # Usamos fragmentos de 1000 caracteres con una superposición de 200 para no perder contexto entre cortes
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = text_splitter.split_documents(documentos)
    
    # C. Crear los Embeddings usando un modelo liviano y gratuito que corre local
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # D. Crear la base de datos de vectores local (FAISS)
    base_vectores = FAISS.from_documents(fragmentos, embeddings)
    
    return base_vectores, len(documentos), len(fragmentos)

# 3. Ejecutar el procesamiento con un indicador visual de carga
with st.spinner("Procesando documentación y generando base de datos de vectores local..."):
    resultado = inicializar_base_conocimiento()

if resultado:
    base_vectores, total_paginas, total_fragmentos = resultado
    st.success(f"¡Base de conocimiento indexada con éxito!")
    
    # Mostramos métricas informativas en la interfaz
    col1, col2 = st.columns(2)
    col1.metric("Documentos leídos (Páginas)", total_paginas)
    col2.metric("Fragmentos indexados (Chunks)", total_fragmentos)
else:
    st.warning("No se pudieron procesar los documentos corporativos.")