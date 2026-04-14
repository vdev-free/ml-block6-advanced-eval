from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    sentences = [
        "I love this product",
        "Я дуже люблю цей продукт",
        "The quality is terrible",
        "Якість дуже погана",
    ]

    embeddings = model.encode(sentences)

    similarity_matrix = cosine_similarity(embeddings)

    print()
    print("Pairwise similarities:")

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            print(f"{sentences[i]!r} <-> {sentences[j]!r} = {similarity_matrix[i][j]:.4f}")


if __name__ == "__main__":
    main()