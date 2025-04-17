import ssl
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  # Corrected import
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Explicitly specify the model name for HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Replace with your desired model

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    # Workaround for SSL certificate error on macOS without sudo
    ssl._create_default_https_context = ssl._create_unverified_context

    # Loading the YouTube video
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    # Splitting the transcript into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    # Creating a vector store (FAISS) from the documents
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, hf_api_key, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # ✅ Using T5-small for text2text-generation task
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",  # or "t5-small"
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=hf_api_key
    )

    # ✅ Better prompt template for T5
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant. Use the provided transcript to answer the question clearly.

        Context: {docs}
        Question: {question}
        Answer:
        """

        # template="""
        # You are a helpful assistant that answers questions based on the provided transcript.

        # Context: {docs}

        # Question: {question}

        # Provide only the direct answer to the question. Do not include any additional explanation or context.
        # """
    )

    # Run the prompt through the LLM
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)

    return response.split("Answer:")[-1].strip(), docs

