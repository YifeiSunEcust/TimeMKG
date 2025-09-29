import asyncio
import nest_asyncio

nest_asyncio.apply()
import os
import inspect
import logging
from lightrag.lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = 'TimeMKG/KG_generate/example'
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 32768},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)

def generate_prompts(rag, variables):
    prompts = {}
    for variable in variables:
        prompt = f"<|start_prompt|>You are a data scientist who is proficient in time series analysis.The dataset contains the following variables{variables}; 
        Current variable: {variable}, Task: Analyze the relationship between {variable} and other variables and provide feature engineering suggestions for time series modeling, covering the following aspects: 
        1. Causality/correlation logic 2. Impact direction and intensity 3. Modeling suggestions such as subsequent features and interaction terms.<|end_prompt|>"
        response = rag.query(prompt, param=QueryParam(mode="naive"))
        prompt = f"Variable: {variable}\nDescription: {response}"
        prompts[variable] = prompt
    return prompts

def MKG(txt_path, variables, output_file):
    rag = asyncio.run(initialize_rag())
    with open(txt_path, "r", encoding="utf-8") as f:
        rag.insert(f.read())
    prompts = generate_prompts(rag, variables)

    # Print prompts
    for variable, prompt in prompts.items():
        print(f"Prompt for {variable}:\n{prompt}\n")

    with open(output_file, "w", encoding="utf-8") as f:
        for variable, prompt in prompts.items():
            f.write(f"{variable}:\n{prompt}\n\n")

    print("MKG has been constructed and saved.")

def response_rag(prompt, txt_path):
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    # Insert example text
    with open(txt_path, "r", encoding="utf-8") as f:
        rag.insert(f.read())
    response = rag.query(prompt, param=QueryParam(mode="naive"))
    return response

if __name__ == "__main__":
    txt_path = "."
    variables = ["A","B","C"]  
    output_file = "." 
    MKG(txt_path, variables, output_file)

