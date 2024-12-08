import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import docx
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Dict
import json
from streamlit_player import st_player
import requests
import pyttsx3

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

engine = pyttsx3.init()

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def get_available_models():
    try:
        response = requests.get("http://localhost:1234/api/v0/models")
        if response.status_code == 200:
            models_data = response.json()
            return [model['id'] for model in models_data['data'] if model['state'] == 'loaded']
        else:
            st.error("Failed to fetch models from LM Studio")
            return ['llama-3.2-3b-qnn:2','llama-3.2-3b-instruct']
    except Exception as e:
        print(f"Error connecting to LM Studio: {str(e)}")
        return ['llama-3.2-3b-qnn:2','llama-3.2-3b-instruct']

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
    # Add content cache
    if "content_cache" not in st.session_state:
        st.session_state.content_cache = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

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
    # Cache the content
    st.session_state.content_cache = text
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def ensure_vector_store():
    if st.session_state.vector_store is None and st.session_state.content_cache is not None:
        st.session_state.vector_store = create_vector_store(st.session_state.content_cache)

def generate_quiz(content: str, num_questions: int = 5, model=None) -> List[Dict]:
    model = model or st.session_state.selected_model
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

def get_response(query, model=None):
    model = model or st.session_state.selected_model
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

def analyze_transcript_segments(transcript, url):
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        model_name="llama-3.2-3b-instruct",
        api_key="lm-studio"
    )
 
    # Divide transcript into 5 equal parts by duration
    total_duration = transcript[-1]['start'] + transcript[-1].get('duration', 0)
    segment_duration = total_duration / 5
    segments = []
    current_segment = []
 
    # Group entries by time segments
    for entry in transcript:
        if entry['start'] <= (len(segments) + 1) * segment_duration:
            current_segment.append(entry)
        else:
            if current_segment:
                segments.append(current_segment)
                current_segment = [entry]
    if current_segment:
        segments.append(current_segment)
 
    # Enhanced prompt for concise, heading-style topics
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at creating clear, concise section headings for video content.
        For each segment, return a brief (2-5 words) topic heading that captures the main concept.
        Use proper title case and professional terminology.
        Examples:
        - "Introduction and Overview"
        - "Key Methodology Steps"
        - "Performance Evaluation"
        - "Implementation Results"
        - "Future Research Directions"
        """),
        ("human", "Create a concise section heading for this content:\n{text}")
    ])
 
    chain = prompt | llm | StrOutputParser()
 
    # Process each segment to identify main topics
    main_topics = []
    for segment in segments:
        if len(main_topics) < 5:
            segment_text = ' '.join(entry['text'] for entry in segment[:10])
            try:
                topic = chain.invoke({"text": segment_text}).strip()
                main_topics.append({
                    "topic": topic,
                    "start": segment[0]['start'],
                    "text": segment[0]['text']  # Keep original text for context
                })
            except Exception:
                continue
 
    return main_topics

def main():
    st.set_page_config(page_title="SnapLearnAI", page_icon="📚")
    
    st.title("📚 SnapLearn - AI Assistant")
    st.caption("💡Your offline learning companion powered by local LLMs")
    
    initialize_session_state()

    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Add model selector dropdown
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Select LLM Model",
            available_models,
            index=0 if st.session_state.selected_model is None else available_models.index(st.session_state.selected_model)
        )
        st.session_state.selected_model = selected_model
        
        st.title("Content Upload")
        
        tab1, tab2 = st.tabs(["📄 Document Upload", "🎥 YouTube"])
        
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
 
            if youtube_url:
                # Display video player
                player = st_player(youtube_url)
 
                if st.button("Process YouTube Video"):
                    try:
                        with st.spinner("Processing YouTube transcript..."):
                            video_id = youtube_url.split('v=')[1]
                            transcript = YouTubeTranscriptApi.get_transcript(video_id)
                            transcript_text = ' '.join([entry['text'] for entry in transcript])
                            st.session_state.vector_store = create_vector_store(transcript_text)
 
                            # Store important segments in session state
                            st.session_state.important_segments = analyze_transcript_segments(transcript, youtube_url)
                            st.success("Successfully processed YouTube video")
                    except Exception as e:
                        st.error(f"Error processing YouTube video: {str(e)}")
 
                # Display segments from session state
                # Display segments from session state
                if hasattr(st.session_state, 'important_segments'):
                    with st.sidebar:
                        st.subheader("Key Moments")
                        for segment in st.session_state.important_segments:
                            start_time = int(segment["start"])
 
                            # Clean up the topic text by removing all explanatory text
                            topic = segment['topic']
                            timestamped_url = f"{youtube_url}&t={start_time}s"
 
                            # Remove common explanatory phrases
                            cleanup_phrases = [
                                "Here is a concise section heading:",
                                "Here is a possible section heading:",
                                "This heading captures",
                                "Both of these headings capture",
                                "or alternatively,",
                                "(or alternatively,"
                            ]
 
                            for phrase in cleanup_phrases:
                                if phrase.lower() in topic.lower():
                                    topic = topic.split(phrase)[-1]
 
                            # Remove quotes and clean up extra whitespace
                            topic = topic.strip('"').strip()
 
                            # Remove any text after a period (explanatory text often follows)
                            if "." in topic:
                                topic = topic.split(".")[0]
 
                            # Display only timestamp and cleaned topic
                            with st.expander(f"{int(start_time//60)}:{int(start_time%60):02d} - {topic}"):
                                if st.button("Jump to timestamp", key=f"ts_{start_time}"):
                                    st_player(timestamped_url)
        
        

    # Main content area with tabs
    tab1, tab2 = st.tabs(["Chat", "Quiz"])
    
    with tab1:
        ensure_vector_store()
        
        # Create a container with fixed height for chat history
        chat_container = st.container()
        
        # Display chat history in scrollable container
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Input field fixed at bottom
        if prompt := st.chat_input("Type your message here..."):
            with chat_container:
                with st.chat_message("user"):
                    st.write(prompt)
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("🤔 Thinking...")
                    
                    # Add a spinner while generating response
                    with st.spinner("Generating response..."):
                        response = get_response(prompt)
                    
                    # Replace placeholder with actual response
                    message_placeholder.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        if st.button("🔊"):
        # Ensure there is a response to speak
            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                last_response = st.session_state.chat_history[-1]["content"]
                speak_text(last_response)
            else:
                st.warning("No assistant response to speak!")

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
                        st.session_state.question_feedback = {}  # Initialize feedback storage
                        st.rerun()
            
            else:
                st.subheader("Quiz Time!")
                
                # Initialize feedback storage if not exists
                if "question_feedback" not in st.session_state:
                    st.session_state.question_feedback = {}
                
                for i, q in enumerate(st.session_state.quiz_questions):
                    question_container = st.container()
                    
                    with question_container:
                        st.write(f"\n**Question {i+1}:** {q['question']}")
                        answer = st.radio(
                            f"Select your answer for question {i+1}:",
                            q['options'],
                            key=f"q_{i}",
                            index=None
                        )
                        
                        # Show feedback if answer exists
                        if answer:
                            selected_index = q['options'].index(answer)
                            
                            # Only update score and show feedback if this is a new answer
                            if i not in st.session_state.question_feedback:
                                if selected_index == q['correct_answer']:
                                    feedback = {"correct": True, "message": "Correct!"}
                                    st.session_state.current_score += 1
                                else:
                                    feedback = {"correct": False, 
                                            "message": f"Incorrect. The correct answer was: {q['options'][q['correct_answer']]}"}
                                st.session_state.question_feedback[i] = feedback
                            
                            # Display feedback from storage
                            if i in st.session_state.question_feedback:
                                if st.session_state.question_feedback[i]["correct"]:
                                    st.success(st.session_state.question_feedback[i]["message"])
                                else:
                                    st.error(st.session_state.question_feedback[i]["message"])
                
                st.write(f"Current Score: {st.session_state.current_score}/{len(st.session_state.quiz_questions)}")
                
                if st.button("New Quiz"):
                    st.session_state.quiz_questions = []
                    st.session_state.current_score = 0
                    st.session_state.question_feedback = {}
                    st.rerun()
    # custom_css = """
    # <style>
    #     #main-container {
    #         display: flex;
    #         flex-direction: column;
    #         min-height: 50vh;
    #     }
    #     footer {
    #         margin-top: auto;
    #         text-align: center;
    #         padding: 10px;
    #     }
    # </style>
    # """
    # st.markdown(custom_css, unsafe_allow_html=True)
    # st.markdown("<div id='main-container'>", unsafe_allow_html=True)
    # # Footer with copyright and social links
    # footer = """
    # <div style='text-align: center; margin-top: 50px;'>
    #     <p>&copy; 2024 SnapLearnAI</p>
    #     <p>
    #         Follow us: 
    #         <a href='https://twitter.com/SnapLearnAI' target='_blank'>Twitter</a> | 
    #         <a href='https://facebook.com/SnapLearnAI' target='_blank'>Facebook</a> | 
    #         <a href='https://linkedin.com/company/SnapLearnAI' target='_blank'>LinkedIn</a>
    #     </p>
    # </div>
    # """
    # st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()