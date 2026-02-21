# 🥷 Career Sensei

AI-powered resume gap analysis tool — upload your resume and a job URL to get an instant match score, strengths & gaps breakdown, actionable improvement plan, and a context-aware follow-up chatbot.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Gemini](https://img.shields.io/badge/Google-Gemini-yellow?logo=google)

---

## ✨ Features

- **Match Score** — 0–100 fit rating with color-coded indicator
- **Job Summary** — Quick overview of the role and ideal candidate
- **Key Strengths** — What you already bring to the table
- **Key Gaps** — Concrete missing skills and experience
- **Action Plan** — Specific courses, projects, or certs to close your gaps
- **Follow-up Chat** — Ask context-aware questions about the role and your fit

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Google Gemini (Flash) |
| **Embeddings** | Gemini Embedding API |
| **RAG Framework** | LangChain |
| **Vector Store** | ChromaDB (in-memory) |
| **Frontend** | Streamlit |
| **Scraping** | BeautifulSoup4 |

## 📁 Project Structure

```
Career Sensei/
├── app.py                  # Streamlit entry point
├── src/
│   ├── rag/
│   │   ├── utils.py        # Resume parsing & URL scraping
│   │   └── pipeline.py     # RAG chains, prompts, output parsing
│   └── ui/
│       ├── styles.py       # CSS injection
│       └── components.py   # Reusable UI components
├── requirements.txt
└── .env                    # Your API key (not committed)
```

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/saifrahman10/career-sensei.git
cd career-sensei
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
Get a free key at [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Run the app
```bash
streamlit run app.py
```

## 🔒 Privacy

Your resume and data stay on your device and are never stored. Everything is processed in-memory and cleared the moment you close the tab.

## 📝 License

MIT
