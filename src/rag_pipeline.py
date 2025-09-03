import os
from dotenv import load_dotenv
import logging as log

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma

log.basicConfig(
            level=log.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

load_dotenv()

config = {
     "INPUT_FOLDER": os.getenv("INPUT_FOLDER"),
     "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
     "CHROMA_DIR": os.getenv("CHROMA_DIR")
}

input_path = Path(os.getenv("INPUT_FOLDER"))
text_files = list(input_path.glob("*.txt"))

embedding_model = GoogleGenerativeAIEmbeddings(
     google_api_key=os.getenv("GOOGLE_API_KEY"), 
     model="models/text-embedding-004"
)

vector_db = Chroma(
    persist_directory=os.getenv("CHROMA_DIR"),
     embedding_function=embedding_model
)

for file in text_files:
    loader = TextLoader(file.as_posix())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    data = loader.load_and_split(text_splitter=splitter)
    log.info(f"Inserindo: {file.stem}")

    vector_db.add_documents(documents=data)
    log.info(f"Documento inserido: {file.stem}")



