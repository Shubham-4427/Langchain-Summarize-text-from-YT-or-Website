import re
import validators
import streamlit as st

# -------- LangChain Core (Modern & Stable) --------
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------- LLM --------
from langchain_groq import ChatGroq

# -------- Loaders --------
from langchain_community.document_loaders import UnstructuredURLLoader

# -------- YouTube Transcript API --------
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(
    page_title="LangChain: Summarize Text From YouTube or Website",
    page_icon="🦜"
)

st.title("🦜 YouTube & Website Summarizer")
st.subheader("Summarize any public YouTube video or website")


# -------------------- SIDEBAR --------------------
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")

url = st.text_input("Enter YouTube or Website URL", label_visibility="collapsed")


# -------------------- LLM CONFIG --------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # ✅ supported Groq text model
    groq_api_key=groq_api_key,
    temperature=0
)


# -------------------- PROMPT --------------------
prompt = PromptTemplate.from_template(
    """
Provide a clear, well-structured summary of the following content
in about 300 words.

{text}
"""
)


# -------------------- YOUTUBE TRANSCRIPT LOADER --------------------
def load_youtube_transcript(video_url: str):
    match = re.search(r"(?:v=|youtu\.be/)([^&?/]+)", video_url)
    if not match:
        raise ValueError("Invalid YouTube URL")

    video_id = match.group(1)
    api = YouTubeTranscriptApi()  # instance REQUIRED

    try:
        transcript_list = api.list(video_id)

        try:
            transcript = transcript_list.find_transcript(["en"]).fetch()
        except NoTranscriptFound:
            # fallback to any available transcript
            keys = (
                list(transcript_list._manually_created_transcripts.keys())
                or list(transcript_list._generated_transcripts.keys())
            )
            if not keys:
                raise NoTranscriptFound(video_id)
            transcript = transcript_list.find_transcript(keys).fetch()

    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video")
    except NoTranscriptFound:
        raise RuntimeError("No transcript found for this video")

    text = " ".join(item.text for item in transcript)
    return [Document(page_content=text)]


# -------------------- LCEL SUMMARIZATION --------------------
def summarize_documents(docs):
    combined_text = "\n\n".join(doc.page_content for doc in docs)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": combined_text})


# -------------------- BUTTON ACTION --------------------
if st.button("Summarize Content"):
    if not groq_api_key.strip() or not url.strip():
        st.error("Please provide both Groq API key and a URL")

    elif not validators.url(url):
        st.error("Please enter a valid URL")

    else:
        try:
            with st.spinner("Loading and summarizing content..."):
                # -------- LOAD CONTENT --------
                if "youtube.com" in url or "youtu.be" in url:
                    docs = load_youtube_transcript(url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                if not docs:
                    st.error("No content could be extracted from the URL")
                    st.stop()

                # -------- SUMMARIZE --------
                summary = summarize_documents(docs)

                st.success("Summary generated successfully!")
                st.write(summary)

        except Exception as e:
            st.exception(e)
