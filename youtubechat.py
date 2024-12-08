import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import docx
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Dict
import json

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def initialize_session_state():
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "current_score" not in st.session_state:
        st.session_state.current_score = 0

def get_youtube_transcript(url):
    try:
        video_id = url.split('v=')[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        raise ValueError(f"Error fetching YouTube transcript: {str(e)}")

def get_pdf_text(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def get_document_text(file):
    if file.name.endswith('.pdf'):
        return get_pdf_text(file)
    elif file.name.endswith('.docx'):
        return get_docx_text(file)
    else:
        raise ValueError("Unsupported file format")

def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def generate_quiz(content: str, num_questions: int = 5) -> List[Dict]:
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        model_name="llama-3.2-3b-instruct",
        api_key="lm-studio"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a quiz generator. You must return ONLY a JSON array of questions.
        Do not include any explanatory text or markdown code blocks."""),
        ("human", """Generate exactly {num_questions} multiple choice questions based on this content:
{content}

Return ONLY a JSON array where each question has:
- question: the question text
- options: array of exactly 4 possible answers
- correct_answer: index of the correct answer (0-3)""")
    ])

    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "content": content,
            "num_questions": num_questions
        })
        
        # Clean up the response
        response = response.strip()
        if response.startswith('```'):
            response = response[response.find('['):response.rfind(']')+1]
        
        # Parse JSON
        questions = json.loads(response)
        
        # Validate structure
        if not isinstance(questions, list):
            raise ValueError("Response must be a JSON array")
        
        for q in questions:
            if not all(k in q for k in ["question", "options", "correct_answer"]):
                raise ValueError("Each question must have required fields")
            if len(q["options"]) != 4:
                raise ValueError("Each question must have exactly 4 options")
                
        return questions
        
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return []

def get_response(query):
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        model_name="llama-3.2-3b-instruct", 
        api_key="lm-studio"
    )

    context = ""
    if st.session_state.vector_store:
        docs = st.session_state.vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the provided context and chat history to answer questions."),
        ("human", "Context:\n{context}\n\nChat History:\n{chat_history}\nHuman: {question}")
    ])

    chain = prompt | llm | StrOutputParser()

    chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
    response = chain.invoke({
        "question": query,
        "chat_history": chat_history,
        "context": context
    })

    st.session_state.memory.save_context(
        {"input": query},
        {"output": response}
    )
    
    return response

def main():
    st.title("Educational AI Assistant")
    initialize_session_state()

    with st.sidebar:
        st.title("Content Upload")
        
        tab1, tab2 = st.tabs(["Document Upload", "YouTube URL"])
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Upload a document", 
                type=["pdf", "docx"],
                help="Supported formats: PDF, DOCX"
            )
            
            if st.button("Process Document"):
                if uploaded_file is not None:
                    try:
                        with st.spinner("Processing document..."):
                            raw_text = get_document_text(uploaded_file)
                            st.session_state.vector_store = create_vector_store(raw_text)
                            st.success(f"Successfully processed {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                else:
                    st.error("Please upload a document first")
        
        with tab2:
            youtube_url = st.text_input("Enter YouTube URL")
            if st.button("Process YouTube Video"):
                if youtube_url:
                    try:
                        with st.spinner("Processing YouTube transcript..."):
                            transcript_text = get_youtube_transcript(youtube_url)
                            st.session_state.vector_store = create_vector_store(transcript_text)
                            st.success("Successfully processed YouTube video")
                    except Exception as e:
                        st.error(f"Error processing YouTube video: {str(e)}")
                else:
                    st.error("Please enter a YouTube URL")

    # Main content area with tabs
    tab1, tab2 = st.tabs(["Chat", "Quiz"])
    
    with tab1:
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        if prompt := st.chat_input("Type your message here..."):
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                response = get_response(prompt)
                st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with tab2:
        if st.session_state.vector_store is None:
            st.warning("Please load a document or YouTube video first to generate a quiz.")
        else:
            if not st.session_state.quiz_questions:
                if st.button("Generate New Quiz"):
                    with st.spinner("Generating quiz questions..."):
                        content = "\n".join([doc.page_content for doc in 
                            st.session_state.vector_store.similarity_search("", k=10)])
                        st.session_state.quiz_questions = generate_quiz(content)
                        st.session_state.current_score = 0
                        st.rerun()
            
            else:
                st.subheader("Quiz Time!")
                st.write(f"Current Score: {st.session_state.current_score}/{len(st.session_state.quiz_questions)}")
                
                for i, q in enumerate(st.session_state.quiz_questions):
                    st.write(f"\n**Question {i+1}:** {q['question']}")
                    answer = st.radio(
                        f"Select your answer for question {i+1}:",
                        q['options'],
                        key=f"q_{i}"
                    )
                    
                    if answer:
                        selected_index = q['options'].index(answer)
                        if selected_index == q['correct_answer']:
                            st.success("Correct!")
                            st.session_state.current_score += 1
                        else:
                            st.error(f"Incorrect. The correct answer was: {q['options'][q['correct_answer']]}")
                
                if st.button("New Quiz"):
                    st.session_state.quiz_questions = []
                    st.session_state.current_score = 0
                    st.rerun()

if __name__ == "__main__":
    main()