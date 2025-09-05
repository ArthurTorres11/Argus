import logging as log
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from src.rag_pipeline import query_rag

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)


class QuestionRequest(BaseModel):
    question:str

class AnswerResponse(BaseModel):
    answer: str


app = FastAPI(
    title="Argus",
    description="API para perguntas a um sistema RAG sobre artigos acadêmicos",
    version="1.0.0",
)

@app.get('/')
def read_root():
    """Verificar se a API está funcionando"""
    return {"status":"Argus API is running"}


@app.post('/ask', response_model=AnswerResponse)
async def as_question(request: QuestionRequest):
    """Recebe uma pergunta, processa com o motor RAG e retorna a resposta"""
    log.info(f"Pergunta recebida {request.question}")
    try:
        response_text = query_rag(request.question)
        return {"answer":response_text}
    
    except Exception as e:
        log.error(f"Erro ao processar a pergunta: {request.question} : {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno")


