# 🎬 YouRAG - YouTube RAG Assistant

**YouRAG** is an AI-powered assistant that transcribes YouTube videos and allows you to ask questions about the content using a Retrieval-Augmented Generation (RAG) pipeline backed by FAISS and Hugging Face models.

Ask it anything about a video — summaries, key points, definitions, or anything context-related — and it will respond with an answer based on the actual transcript!

---

## 🚀 Features

- 🔗 Input a YouTube video URL
- 📜 Automatic transcript generation via Hugging Face models
- 🧠 Embedding & vector storage using `sentence-transformers` and FAISS
- ❓ Query the transcript using natural language
- 🗣️ Generates answers using large language models (e.g., Mistral-7B)

---

## 📦 Dependencies

- Python 3.8+
- Streamlit
- Langchain
- Hugging Face Hub
- FAISS
- python-dotenv

Install dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## 🛠️ Setup

1. **Clone the repository**

\`\`\`bash
git clone https://github.com/yourusername/yourag.git
cd yourag
\`\`\`

2. **Create a `.env` file** with your Hugging Face API key:

\`\`\`env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key_here
\`\`\`

3. **Run the Streamlit app**

\`\`\`bash
streamlit run main.py
\`\`\`

---

## 🧠 How It Works

1. **Transcription**: The app extracts and transcribes a YouTube video using `YoutubeLoader`.
2. **Embedding**: The transcript is chunked and embedded using `sentence-transformers/all-MiniLM-L6-v2`.
3. **Vector Search**: Chunks are stored and searched in FAISS based on user queries.
4. **LLM Response**: Relevant chunks are passed to a Hugging Face model (like `Mistral-7B-Instruct`) to generate an answer.

---

## 📂 Project Structure

\`\`\`text
.
├── main.py                  # Streamlit app entrypoint
├── langchain_helper.py     # RAG + FAISS helper functions
├── .env                    # Hugging Face API token (not tracked in Git)
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
\`\`\`

---

## 📹 Example Use Case

> **Input:**  
> 🔗 https://www.youtube.com/watch?v=ejoOZ-swkOk  
> ❓ What is this video about?

> **Output:**  
> 📜 *"The video discusses..."* (based on the transcript and model's understanding)

---

## 🧪 Models Used

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.3` (can be customized)

These can be changed in `langchain_helper.py` as needed.

---

## 📌 Roadmap

- [ ] Add multi-language support
- [ ] Support for longer videos with chunk caching
- [ ] Enhance UI with Streamlit components (e.g., transcript viewer)
- [ ] Export Q&A sessions
- [ ] Docker support for deployment

---

## 🙌 Acknowledgements

- [Langchain](https://github.com/langchain-ai/langchain)
- [Hugging Face](https://huggingface.co)
- [Streamlit](https://streamlit.io)
- [FAISS by Facebook Research](https://github.com/facebookresearch/faiss)

---

## 🧑‍💻 Author

Built with ❤️ by [Santhanakrishnan](https://github.com/santhanakrishnanstark)
Portfolio: [Santhanakrishnan Portfolio](https://sandykrish.netlify.app/)
Blog: [https://thedailyfrontend.com/]https://thedailyfrontend.com/

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
