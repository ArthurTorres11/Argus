import arxiv
import logging as log

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
        

## Teste da função
## search_and_load("Reinforcement Learning", 2)