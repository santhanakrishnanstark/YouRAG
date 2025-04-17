import streamlit as st
import langchain_helper as lch
import textwrap
import os  # ✅ Required for accessing environment variables
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.title("🎬 YouTube Video Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.text_area(
            label="🔗 YouTube Video URL",
            max_chars=100,
            value="https://www.youtube.com/watch?v=ejoOZ-swkOk"
        )
        query = st.text_area(
            label="❓ Ask a question about the video",
            max_chars=100,
            key="query",
            value="what is this video about ?"
        )
        submit_button = st.form_submit_button(label='Submit')

if query and youtube_url:
    if not huggingface_api_key:
        st.warning("🚫 Hugging Face API Key is missing. Please add it to your .env file.")
        st.stop()

    with st.spinner("⏳ Processing..."):
        db = lch.create_db_from_youtube_video_url(youtube_url)
        response, docs = lch.get_response_from_query(db, query, huggingface_api_key)
        st.subheader("📜 Answer:")
        st.text(textwrap.fill(response, width=85))
