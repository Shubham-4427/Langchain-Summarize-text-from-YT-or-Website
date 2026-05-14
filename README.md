# LangChain Text Summarizer for YouTube & Websites

> An AI-powered summarization app that extracts content from YouTube videos and websites, then generates concise summaries using LangChain and Groq LLMs.

---

## Project Overview

**LangChain Text Summarizer for YouTube & Websites** is a Streamlit-based application that helps users quickly understand long videos, blogs, articles, and web pages.

The user only needs to provide a valid **YouTube URL** or **website URL**. The application extracts the available text content and passes it to a Large Language Model through LangChain to generate a clear and concise summary.

This project is useful for students, researchers, developers, and professionals who want to save time while consuming long-form online content.

---

## Key Features

* Summarizes YouTube videos using transcript extraction
* Summarizes website content from URLs
* Uses LangChain for LLM orchestration
* Uses Groq LLMs for fast response generation
* Built with a simple Streamlit interface
* Accepts user-provided Groq API key
* Validates URLs before processing
* Generates short and readable summaries
* Handles YouTube transcript and website loading errors
* Useful for blogs, tutorials, lectures, documentation, and articles

---

## System Architecture

```text
User Enters URL
      │
      ▼
Streamlit Interface
      │
      ▼
URL Validation
      │
      ├── YouTube URL
      │       └── Extract Transcript
      │
      └── Website URL
              └── Load Web Page Content
      │
      ▼
LangChain Prompt Template
      │
      ▼
Groq LLM
      │
      ▼
Generated Summary
```

---

## Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Frontend / UI | Streamlit |
| LLM Framework | LangChain |
| LLM Provider | Groq |
| YouTube Transcript Extraction | youtube-transcript-api |
| Website Content Loading | UnstructuredURLLoader |
| URL Validation | validators |
| Environment Variables | python-dotenv |

---

## Project Structure

```text
Langchain-Summarize-text-from-YT-or-Website/
│
├── main.py              # Main Streamlit application
├── requirements.txt     # Required Python packages
├── README.md            # Project documentation
└── LICENSE              # License file
```

---

## How It Works

1. The user enters a Groq API key.
2. The user provides a YouTube video URL or website URL.
3. The app validates the URL.
4. If the URL is from YouTube, the transcript is extracted.
5. If the URL is from a website, the webpage content is loaded.
6. The extracted text is passed into a LangChain prompt.
7. Groq LLM generates a concise summary.
8. The summary is displayed in the Streamlit app.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Shubham-4427/Langchain-Summarize-text-from-YT-or-Website.git
cd Langchain-Summarize-text-from-YT-or-Website
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
# macOS/Linux
source venv/bin/activate
```

```bash
# Windows
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the Streamlit application:

```bash
streamlit run main.py
```

After running the command:

1. Open the local Streamlit URL in your browser.
2. Enter your Groq API key.
3. Paste a YouTube video URL or website URL.
4. Click the summarize button.
5. View the generated summary.

---

## Example Inputs

### YouTube URL

```text
https://www.youtube.com/watch?v=VIDEO_ID
```

### Website URL

```text
https://example.com/article
```

---

## Example Use Cases

### YouTube Summarization

This app can summarize:

* Educational lectures
* Programming tutorials
* AI/ML videos
* Conference talks
* Podcasts with available transcripts
* Technical walkthroughs

### Website Summarization

This app can summarize:

* Blog posts
* News articles
* Documentation pages
* Research articles
* Online tutorials
* Long-form web content

---

## Core Components

### Streamlit

Streamlit is used to build the user interface where users enter their API key, provide URLs, and view generated summaries.

### LangChain

LangChain manages the prompt template and connects the extracted content with the Groq language model.

### Groq

Groq provides fast LLM inference for generating summaries from extracted text.

### YouTube Transcript API

The YouTube Transcript API extracts transcripts from YouTube videos when captions are available.

### UnstructuredURLLoader

UnstructuredURLLoader is used to load and process website content from URLs.

---

## Error Handling

The app handles common errors such as:

* Missing Groq API key
* Empty URL input
* Invalid URL format
* YouTube transcript not available
* Website content extraction failure
* LLM generation errors

---

## Future Enhancements

* Add PDF summarization
* Add support for multiple URLs
* Add summary length options
* Add bullet-point and detailed summary modes
* Add downloadable summaries
* Add multilingual summarization
* Add chat-with-video feature
* Add source citation support
* Add summary history
* Deploy on Streamlit Cloud or Hugging Face Spaces
* Add support for multiple LLM providers

---

## Author

**Shubham Kumar**  
AI Developer | Machine Learning Engineer | Creative Technologist

---

## License

This project is licensed under the GPL-3.0 License.

---

⭐ If you find this project useful, consider giving it a star on GitHub.
