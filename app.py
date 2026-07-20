import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Configurar ruta absoluta para el archivo .env
ruta_env = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ruta_env)

# 2. Configuración estética de la interfaz de Streamlit
st.set_page_config(page_title="Pacífico Gaming Lab - IA", page_icon="🤖", layout="centered")
st.title("🤖 Pacífico Gaming Lab")
st.write("Agente de IA Corporativo - Base de conocimiento activa en la nube.")

# 3. BASE DE CONOCIMIENTO OPTIMIZADA PARA PRODUCCIÓN (Evita desbordar la RAM de Render)
# Reemplaza el texto de abajo con un resumen o el contenido clave de tus PDFs locales
CONTEXTO_CORPORATIVO = """
Pacífico Gaming Lab es una empresa dedicada al desarrollo de soluciones tecnológicas y entretenimiento. 
[Aquí puedes pegar el texto principal, políticas o información clave que venía en tus manuales PDF]
"""

# 4. Configurar el modelo de Groq y el Prompt Corporativo
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
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

# Algoritmo RAG adaptado para alta velocidad en la nube
def ejecutar_flujo_rag(pregunta_usuario):
    prompt_formateado = prompt.format_messages(
        context=CONTEXTO_CORPORATIVO,
        input=pregunta_usuario,
    )
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