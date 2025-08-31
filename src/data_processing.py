from pathlib import Path
import arxiv
import logging as log
from pypdf import PdfReader

log.basicConfig(
            level=log.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

def search_and_load(articles_themes: str, articles_numbers: int):
    """
    Faz a busca de artigos conforme o tema e o número de artigos
    """
    
    client = arxiv.Client()

    search = arxiv.Search(
        query=articles_themes,
        max_results=articles_numbers,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = client.results(search)

    for r in results:
        log.info(f"Baixando o artigo {r.title}")
        r.download_pdf(dirpath="./data/raw", filename=f"{r.title}.pdf")
        log.info(f"Artigo baixado com sucesso: {r.title}")


def pdf_to_text(input_dir: str, output_dir: str):
    """
    Lê todos os PDFs de um diretório de entrada, extrai o texto de cada um
    e salva como um arquivo .txt em um diretório de saída.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    log.info(f"Procurando arquivos  PDF em {input_path}")
    pdf_files = list(input_path.glob("*pdf"))

    if not pdf_files:
        log.warning(f"Nenhum arquivo encontrado em {input_path}")
        return
    
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            output_filename = pdf_file.stem + ".txt"
            output_filepath = output_path / output_filename

            with open(output_filepath, "w", encoding="utf_8") as f:
                f.write(full_text)
            log.info(f"{output_filename} salvo com sucesso em {output_filepath}")

        except Exception as e:
            log.error(f"Falha ao processor {pdf_file.name}: {e}")

if __name__ == "__main__":
    # 1. Baixa os artigos (se necessário)
    # search_and_load("Reinforcement Learning", 2)
    
    # 2. Extrai o texto dos PDFs baixados
    pdf_to_text(input_dir="./data/raw", output_dir="./data/processed")



