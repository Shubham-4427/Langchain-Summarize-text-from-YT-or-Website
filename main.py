import re
import validators
import streamlit as st

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(
    page_title="LangChain: Summarize Text From YouTube or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YouTube or Website")
st.subheader("Summarize any public URL")


# -------------------- SIDEBAR --------------------
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")

url = st.text_input("Enter YouTube or Website URL", label_visibility="collapsed")


# -------------------- LLM (SUPPORTED GROQ MODEL) --------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # ✅ text-generation model
    groq_api_key=groq_api_key,
    temperature=0
)


# -------------------- PROMPT --------------------
prompt_template = """
Provide a clear and concise summary of the following content in about 300 words.

{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


# -------------------- YOUTUBE TRANSCRIPT LOADER (FOR YOUR API VERSION) --------------------
def load_youtube_transcript(video_url: str):
    """
    Works with youtube-transcript-api versions where
    list() and fetch() are INSTANCE methods.
    """
    match = re.search(r"(?:v=|youtu\.be/)([^&?/]+)", video_url)
    if not match:
        raise ValueError("Invalid YouTube URL")

    video_id = match.group(1)

    api = YouTubeTranscriptApi()  # IMPORTANT: instance required

    try:
        transcript_list = api.list(video_id)

        # Prefer English if available, else fall back to first available
        try:
            transcript = transcript_list.find_transcript(["en"]).fetch()
        except NoTranscriptFound:
            # Fallback to any available transcript
            # (generated or manually created)
            keys = list(transcript_list._manually_created_transcripts.keys()) \
                   or list(transcript_list._generated_transcripts.keys())
            if not keys:
                raise NoTranscriptFound(video_id)
            transcript = transcript_list.find_transcript(keys).fetch()

    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video")
    except NoTranscriptFound:
        raise RuntimeError("No transcript found for this video")

    text = " ".join(item.text for item in transcript)
    return [Document(page_content=text)]


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

                # -------- SUMMARIZATION --------
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                result = chain.invoke({"input_documents": docs})

                st.success("Summary generated successfully!")
                st.write(result["output_text"])

        except Exception as e:
            st.exception(e)
