# Chatbot using RAG (PDF)

## Description

This project demonstrates how to extend the capabilities of LLM models to answer questions related to a given document. The project consists of two main programs: one for preparation and the other for question-and-answering using LLM.

The preparation program reads a PDF file and generates vectors, which are then fed to the selected model. The model uses these special tokens to answer questions related to the document.

## Programs/Files

| #  | File Name                 | Description                                                                                       |
|----|---------------------------|---------------------------------------------------------------------------------------------------|
| 1  | `pdf_reader.py`           | LLM chatbot using OpenVINO that answers queries by referring to a vectorstore.                    |
| 2  | `llm-model-downloader.py` | Downloads LLM models from HuggingFace and converts them into OpenVINO IR models. Use hugging face to edit which models you want to download and use(edit them in the environment variables and this file)|
| 3  | `.env`                    | Contains configurations (model name, model precision, inference device, etc.)                     |

## How to run
0. Install prerequisites

```sh
python -m venv venv
venv\Scripts\activate
python -m pip install -U pip
pip install -U setuptools wheel
pip install -r requirements.txt
```

1. Download LLM models

This program downloads the LLM models and converts them into OpenVINO IR models.
If you don't want to download many LLM models, you can comment out the models in the code to save time.
```sh
phthon llm-model-downloader.py
```

2. Run streamlit app

```sh
streamlit run pdf_reader.py
```

Inspired from "https://github.com/yas-sim/openvino-chatbot-rag-pdf"
Tested various heavier models like Llama, Gemma and Mistral
Built a streamlit front end where you can upload the pdf file
Tested various other chunking strategies
Used a vector database(Pinecone) to help accommodate larger multiple files instead of a single file
