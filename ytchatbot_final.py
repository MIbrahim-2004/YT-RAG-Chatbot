import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables (Make sure your .env file has OPENAI_API_KEY)
load_dotenv()

st.set_page_config(page_title="YouTube Chatbot", page_icon="🎥", layout="centered")
st.title("YouTube Video Chatbot")
st.markdown("Enter a YouTube Video ID below to index its transcript, then chat with the video in real-time!")

# Streamlit re-runs the script on every interaction. We use session_state to remember things.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = ""

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def process_video(video_id):
    """Fetches transcript, creates chunks, and builds the FAISS vector store."""
    try:
        yt_api = YouTubeTranscriptApi()
        transcript_list = yt_api.fetch(video_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        
        # Splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        
        # Embeddings & Vector Store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Save to session state
        st.session_state.vector_store = vector_store
        st.session_state.current_video_id = video_id
        st.session_state.chat_history = [] # Clear history for new video
        
        return True
    except TranscriptsDisabled:
        st.error("No Captions found for this video. Please try another one.")
        return False
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False

with st.sidebar:
    st.header("Configuration")
    vid_id = st.text_input("YouTube Video ID", placeholder="e.g., Gfr50f6ZBvo")
    
    if st.button("Load Video", use_container_width=True):
        if vid_id:
            with st.spinner("Fetching transcript and indexing..."):
                success = process_video(vid_id)
                if success:
                    st.success("Video processed successfully!")
        else:
            st.warning("Please enter a Video ID.")

if st.session_state.vector_store is not None:
    
    # Render previous chat history
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
            
    # Chat Input Box
    if user_question := st.chat_input("Ask a question about the video..."):
        
        # Display the user's message immediately
        with st.chat_message("user"):
            st.markdown(user_question)
            
        # Build the LangChain pipeline dynamically using the stored vector_store
        retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        llm = ChatOpenAI(model="gpt-5-nano", temperature=0.2)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.
            
            Context:
            {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        current_history = st.session_state.chat_history

        # Notice how we pull chat_history directly from st.session_state in the Lambda
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda x: current_history)
        })
        
        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser
        
        # Generate the response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = main_chain.invoke(user_question)
                st.markdown(response)
                
        # Append the new interaction to the session state memory
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response))
        
else:
    # Instructions to show when no video is loaded
    st.info("Please enter a YouTube Video ID in the sidebar to start.")