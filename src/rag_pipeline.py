import os
from dotenv import load_dotenv
import logging as log

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

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

input_path = Path(config["INPUT_FOLDER"])
text_files = list(input_path.glob("*.txt"))


def build_vector_db():
    """
    Inicializa o modelo de LLM para os embeddings, cria o banco vetorial e carrega os arquivos em chunks para esse banco
    """
    embedding_model = GoogleGenerativeAIEmbeddings(
        google_api_key=config["GOOGLE_API_KEY"], 
        model="models/text-embedding-004"
    )

    vector_db = Chroma(
        persist_directory=config["CHROMA_DIR"],
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


def query_rag(question: str):
    """
    Instancia o modelo de Emebeddings, banco vetorial e modelo de LLM para as respostas e faz a busca vetorial
    """
    embedding_model = GoogleGenerativeAIEmbeddings(
     google_api_key=config["GOOGLE_API_KEY"], 
     model="models/text-embedding-004"
    )

    vector_db = Chroma(
        persist_directory=config["CHROMA_DIR"],
        embedding_function=embedding_model
    )

    llm = ChatGoogleGenerativeAI(
        model = "gemini-1.5-flash",
        google_api_key = config["GOOGLE_API_KEY"],
        temperature = 0.3
    )

    retriever = vector_db.as_retriever(
        search_kwargs = {"k":3}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type = "stuff",
        retriever = retriever
    )

    result = qa_chain.invoke({"query":question})

    return result["result"]


if __name__ == "__main__":
    pergunta = "What is the core idea behind Reinforcement Learning?"
    resposta = query_rag(pergunta)

    print(f"\nPergunta: {pergunta}")
    print(f"\nResposta: {resposta}")