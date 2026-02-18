"""
retrieve_terms.py

Small-to-big retrieval tool for insurance policy terms.
Retrieves full articles from the vector database based on relevant chunks.

Usage in OnboardingState:
    docs = retrieve_terms(query, policy_date)
    relevant_terms = "\n\n".join(docs)  # → store in OnboardingState.relevant_terms
"""
import os
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain.tools import tool

load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "solar-embedding-1-large")
VECTOR_DIR = os.getenv("VECTOR_DIR", "../../data/chroma")

# ============================================================================
# Load Vector Database
# ============================================================================

def load_vectordb() -> Chroma:
    """Load the persistent vector database."""
    embeddings_model = UpstageEmbeddings(
        api_key=UPSTAGE_API_KEY,
        model=EMBEDDING_MODEL
    )
    return Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embeddings_model
    )

vectordb = load_vectordb()


# ============================================================================
# Retrieval Tool
# ============================================================================

class RetrieveTermsSchema(BaseModel):
    query: str = Field(
        ...,
        description="The question or denial statement text to search for in the policy."
    )
    policy_date: str = Field(
        ...,
        description="Policy version date in YYYYMMDD format (e.g., '20200101'). Must match the client's contract date."
    )
    k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of articles to retrieve. Default is 3."
    )

@tool(args_schema=RetrieveTermsSchema)
def retrieve_terms(query: str, policy_date: str, k: int = 3) -> str:
    """
    Retrieve full insurance policy articles relevant to a query.
    Use this tool when you need to look up what the policy says about a
    specific topic such as coverage exclusions, deductibles, or claim procedures.
    Returns the full text of the most relevant articles including section and article metadata.
    """
    # Step 1: Find top k most relevant chunks
    relevant_chunks = vectordb.similarity_search(
        query,
        k=k,
        filter={"policy_date": policy_date}
    )

    # Step 2: Extract unique (section, article) pairs
    articles_to_fetch = set()
    for chunk in relevant_chunks:
        article = chunk.metadata.get('Article')
        section = chunk.metadata.get('Document Section')
        if article:
            articles_to_fetch.add((section, article))

    # Step 3: Retrieve and combine all chunks per article
    full_terms_docs = []
    for section, article in articles_to_fetch:

        if section is not None:
            where_filter = {
                "$and": [
                    {"policy_date": policy_date},
                    {"Document Section": section},
                    {"Article": article}
                ]
            }
        else:
            where_filter = {
                "$and": [
                    {"policy_date": policy_date},
                    {"Article": article}
                ]
            }

        article_chunks = vectordb.get(where=where_filter)

        paired = list(zip(article_chunks['metadatas'], article_chunks['documents']))
        sorted_pairs = sorted(paired, key=lambda x: x[0].get('chunk_id', ''))

        combined_text = "\n".join([text for _, text in sorted_pairs])

        first_metadata = sorted_pairs[0][0]
        formatted = (
            f"[Section: {first_metadata.get('Document Section', 'N/A')} | "
            f"Article: {first_metadata.get('Article', 'N/A')} | "
            f"Policy: {first_metadata.get('policy_date', 'N/A')}]\n"
            f"{combined_text}"
        )
        full_terms_docs.append(formatted)

    return "\n\n".join(full_terms_docs)