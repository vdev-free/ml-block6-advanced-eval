from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

words = ["I", "love", "machine", "learning"]

embeddings = model.encode(words)

similarity_matrix = cosine_similarity(embeddings)

# for i, word in enumerate(words):
#     print(f"\nWord: {word}")
#     for j, score in enumerate(similarity_matrix[i]):
#         print(f"  {word} ↔ {words[j]} = {score:.3f}")

# print("\nMost related words:")

for i, word in enumerate(words):
    scores = similarity_matrix[i]

    # ігноруємо схожість слова із самим собою
    scores[i] = -1

    best_index = np.argmax(scores)

    # print(f"{word} → {words[best_index]} ({scores[best_index]:.3f})")

x = torch.rand(4, 8)

W_q = torch.rand(8, 8)
W_k = torch.rand(8, 8)
W_v = torch.rand(8, 8)

Q = x @ W_q
K = x @ W_k
V = x @ W_v

scores = Q @ K.T

attention_weights = torch.softmax(scores, dim=-1)

output = attention_weights @ V
print("output shape:", output.shape)
print(output)
