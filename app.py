import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pytesseract
from pdf2image import convert_from_bytes
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Function to extract text from PDFs (including OCR for scanned text)
def extract_text_from_pdf(pdfs):
    """Extract text from PDFs, converting non-selectable text to selectable using OCR."""
    text = ""
    
    for pdf in pdfs:
        pdf_bytes = pdf.read()
        reader = PdfReader(BytesIO(pdf_bytes))
        
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
            else:
                images = convert_from_bytes(pdf_bytes)
                for img in images:
                    ocr_text = pytesseract.image_to_string(img)
                    text += ocr_text

    return text.strip()

# Function to split text into chunks
def get_text_chunks(text):
    """Split text into chunks for better processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    return text_splitter.split_text(text)

# Function to store embeddings in FAISS
def get_vector_store(text_chunks):
    """Generate embeddings and store them in FAISS."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Function to set up conversational AI chain
def get_conversational_chain():
    """Set up the conversational chain with a prompt template."""
    prompt_template = """
    Answer the question in detail based on the given context. If the answer is not available, reply:
    "Answer is not available in the provided context."

    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    return load_qa_chain(model, chain_type='stuff', prompt=prompt)

# Function to process user input
def user_input(user_question, chat_history):
    """Process user queries and generate responses."""
    if st.session_state.vector_store is None:
        st.warning("Please upload and process PDFs first.")
        return chat_history

    docs = st.session_state.vector_store.similarity_search(user_question)
    chain = get_conversational_chain()

    with st.spinner("Thinking..."):
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
    
    chat_history.append({"user": user_question, "bot": response['output_text']})
    return chat_history

# Function to display chat UI
def display_chat(chat_history):
    """Display chat history in a modern UI."""
    for chat in chat_history:
        st.markdown(f"""
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <div style='background-color: #007bff; color: white; border-radius: 20px; padding: 10px; margin-right: 10px;'>\U0001F464 User</div>
                <div style='background-color: #f1f1f1; padding: 10px; border-radius: 10px; flex: 1; color: black;'>{chat['user']}</div>
            </div>
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <div style='background-color: #28a745; color: white; border-radius: 20px; padding: 10px; margin-right: 10px;'>\U0001F916 Bot</div>
                <div style='background-color: #e8f5e9; padding: 10px; border-radius: 10px; flex: 1; color: black;'>{chat['bot']}</div>
            </div>
        """, unsafe_allow_html=True)

# Main function for Streamlit app
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title='QueryBot', layout='wide')
    st.markdown("""
        <h1 style='text-align: center;'>📚 QueryBot</h1>
        <p style='text-align: center;'>Upload Document and ask questions interactively.</p>
    """, unsafe_allow_html=True)
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    user_question = st.text_input('Ask a Question From the Files', placeholder='Type your question here...')
    if user_question:
        st.session_state.chat_history = user_input(user_question=user_question, chat_history=st.session_state.chat_history)
    
    display_chat(st.session_state.chat_history)
    
    with st.sidebar:
        st.title('📂 Document Upload & Processing')
        pdf_docs = st.file_uploader('Upload your Files and Click on Submit & Process', accept_multiple_files=True, type=["pdf"])
        
        if st.button('Submit & Process', key="process"):
            with st.spinner("Processing PDF..."):
                raw_text = extract_text_from_pdf(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks=text_chunks)
                st.success('Processing Complete! You can now ask questions.')

if __name__ == "__main__":
    main()
