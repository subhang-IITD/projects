import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticSearchEngine:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.documents = []
        self.embeddings = []

    def add_document(self, document):
        self.documents.append(document)
        embedding = self.get_embedding(document)
        self.embeddings.append(embedding)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        return embedding

    def search(self, query, top_k=5):
        query_embedding = self.get_embedding(query)
        similarities = cosine_similarity(query_embedding, np.vstack(self.embeddings))
        top_k_indices = similarities[0].argsort()[-top_k:][::-1]
        return [(self.documents[i], similarities[0][i]) for i in top_k_indices]

if __name__ == "__main__":
    # Initialize the search engine
    search_engine = SemanticSearchEngine()

    # Add documents to the engine
    search_engine.add_document("The cat sat on the mat.")
    search_engine.add_document("A quick brown fox jumps over the lazy dog.")
    search_engine.add_document("Artificial intelligence is the future of technology.")
    search_engine.add_document("BERT embeddings are useful for NLP tasks.")
    search_engine.add_document("The dog barked at the moon.")

    # Perform a search
    query = "What is the role of BERT in NLP?"
    results = search_engine.search(query, top_k=3)

    # Print the results
    print("Query:", query)
    print("Top results:")
    for doc, score in results:
        print(f"Document: {doc}, Similarity Score: {score:.4f}")
