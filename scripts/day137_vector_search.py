from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    documents = [
        "I love this product. The quality is amazing.",
        "The support team was very helpful and polite.",
        "Delivery was slow and the package arrived damaged.",
        "This laptop is good for programming and daily work.",
        "Я дуже задоволений якістю цього товару.",
        "Служба підтримки швидко відповіла на моє питання.",
    ]

    query = "I need help from customer support"
    
    document_embeddings = model.encode(documents)
    query_embedding = model.encode([query])

    similarities = cosine_similarity(query_embedding, document_embeddings)[0]

    scored_documents = list(zip(documents, similarities))
    scored_documents_sorted = sorted(
        scored_documents,
        key=lambda x: x[1],
        reverse=True,
    )

    # print()
    # print("Top-3 results:")
    # for doc, score in scored_documents_sorted[:3]:
    #     print(f"{score:.4f} -> {doc}")

    second_query = "I want a product with very good quality"
    second_query_embedding = model.encode([second_query])
    second_similarities = cosine_similarity(second_query_embedding, document_embeddings)[0]

    second_scored_documents = list(zip(documents, second_similarities))
    second_scored_documents_sorted = sorted(
        second_scored_documents,
        key=lambda x: x[1],
        reverse=True,
    )

    print()
    print("Second query:", second_query)
    print("Top-3 results for second query:")
    for doc, score in second_scored_documents_sorted[:3]:
        print(f"{score:.4f} -> {doc}")


if __name__ == "__main__":
    main()