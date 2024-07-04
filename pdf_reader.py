import os
import logging
import argparse
from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from transformers.generation.streamers import TextStreamer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from optimum.intel.openvino import OVModelForCausalLM
import torch

# Load environment variables
load_dotenv(verbose=True)
env_model_embeddings = os.environ['MODEL_EMBEDDINGS']
env_cache_dir = os.environ['CACHE_DIR']
env_model_vendor = os.environ['MODEL_VENDOR']
env_model_name = os.environ['MODEL_NAME']
env_model_precision = os.environ['MODEL_PRECISION']
env_inference_device = os.environ['INFERENCE_DEVICE']
env_streaming = True if os.environ['STREAMING_OUTPUT'] == "True" else False
env_log_level = {'NOTSET': 0, 'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}.get(os.environ['LOG_LEVEL'], 20)

# Setup logging
logger = logging.getLogger('Logger')
logger.addHandler(logging.StreamHandler())
logger.setLevel(env_log_level)


# Helper functions
def generate_rag_prompt(question, vectorstore, bos_token='<s>', verbose=False):
    B_INST, E_INST = '[INST]', '[/INST]'
    B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'
    reference_documents = vectorstore.similarity_search(question, k=4)
    prompt = f'{bos_token}{B_INST} {B_SYS}You are responding to highly technical customers. '
    prompt += 'Answer the question based only on the following context:\n'
    for ref_doc in reference_documents:
        prompt += ref_doc.page_content.replace('\n', '') + '\n'
        prompt += '\n'
    prompt += f'{E_SYS}'
    prompt += f'Question: {question} {E_INST}'
    logger.debug(prompt)
    if verbose:
        print(prompt)
    return prompt

def run_llm_text_generation(pipeline, prompt, max_new_tokens=20, temperature=0.5, repetition_penalty=1.0, streaming=False, verbose=False):
    result = pipeline(prompt, max_new_tokens=max_new_tokens, temperature=temperature, repetition_penalty=repetition_penalty)
    answer = result[0]['generated_text']
    if '\nAnswer: ' in answer:
        answer = answer.split('\nAnswer: ')[1]
    return answer

# Streamlit app
def main():
    st.title("PDF Question Answering System")

    # Upload PDF
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        try:
            # Save the PDF file
            with open("uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract text from PDF
            with open("uploaded_file.pdf", "rb") as f:
                reader = PdfReader(f)
                text = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text()

            # Display the extracted text (optional)
            st.text_area("Extracted Text", text, height=200)

            # Create vectorstore
            embeddings = HuggingFaceEmbeddings(model_name=env_model_embeddings, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
            vectorstore = Chroma(embedding_function=embeddings, persist_directory="vectorstore")
            vectorstore.add_texts([text])

            # Load model and tokenizer using pipeline
            model_id = f'{env_model_vendor}/{env_model_name}'
            text_gen_pipeline = pipeline(
                "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
            )

            # Question input
            question = st.text_input("Enter your question:")
            if st.button("Get Answer"):
                prompt = generate_rag_prompt(question, vectorstore, bos_token='<s>')
                answer = run_llm_text_generation(text_gen_pipeline, prompt, max_new_tokens=20, streaming=env_streaming, temperature=0.2, repetition_penalty=1.0)
                st.write(f"Answer: {answer}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()