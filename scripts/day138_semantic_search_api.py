from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
from time import perf_counter


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

documents = [
    "I love this product. The quality is amazing.",
    "The support team was very helpful and polite.",
    "Delivery was slow and the package arrived damaged.",
    "This laptop is good for programming and daily work.",
    "Я дуже задоволений якістю цього товару.",
    "Служба підтримки швидко відповіла на моє питання.",
]


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=5)


class SearchResult(BaseModel):
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    top_k: int
    latency_ms: float
    results: list[SearchResult]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer(MODEL_NAME)
document_embeddings = model.encode(documents)


def retrieve_documents(query: str, top_k: int) -> list[SearchResult]:
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]

    scored_documents = list(zip(documents, similarities))
    scored_documents_sorted = sorted(
        scored_documents,
        key=lambda x: x[1],
        reverse=True,
    )

    top_results = [
        SearchResult(text=doc, score=float(score))
        for doc, score in scored_documents_sorted[:top_k]
    ]

    return top_results


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    start_time = perf_counter()

    top_results = retrieve_documents(request.query, request.top_k)

    latency_ms = (perf_counter() - start_time) * 1000

    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        latency_ms=round(latency_ms, 2),
        results=top_results,
    )