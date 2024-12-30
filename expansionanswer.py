from helper_utils import project_embeddings, word_wrap, extract_text_from_pdf, load_chroma
from pypdf import PdfReader 
import os
from pypdf import PdfReader
#import umap
import umap.umap_ as umap
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

open_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=open_key)

reader = PdfReader("data/microsoft-annual-report.pdf")

pdf_texts = [p.extract_text().strip() for p in reader.pages]

pdf_texts = [text for text in pdf_texts if text]

# print(
#     word_wrap(
#         pdf_texts[0],
#         width=100,
#     )
# 
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)

character_split_texts = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)

character_split_texts = character_split_texts.split_text("\n\n".join(pdf_texts))

# print(word_wrap(character_split_texts[10]))
# print(f"\nTotal chunks: {len(character_split_texts)}")

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_size=256, chunk_overlap=0
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)
    
# print(word_wrap(token_split_texts[10]))
# print(f"\nTotal chunks: {len(token_split_texts)}")

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction()

chroma_client = chromadb.Client()

chroma_collection = chroma_client.create_collection("microsoft_collection", embedding_function=embedding_function)

ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
# print("\n\n-----------------------------------\n\n")
# print(chroma_collection.count())

query = "what was the total revenue for the year?"

result = chroma_collection.query(query_texts=[query], n_results=5)
# print(result["documents"][0])
retrieved_documents = result["documents"][0]

# for doc in retrieved_documents:
#     print(word_wrap(doc, width=100))
#     print("\n\n-----------------------------------\n\n")

def augmented_query_generated(query, mode= "gpt-3.5-turbo"):
    prompt =""" 
       You rae a helpful expert financial research assistant.
       provide an example answer to the given question, thta might be found in the document.
    """
    
    messages = [
                  {
                    "role": "system",
                    "content": prompt
                  },
                  {
                    "role": "user",
                    "content": query
                  },
                ]
    
    response = client.chat.completions.create(
        model=mode,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content


origional_query = "What was the total profit for the year, and how does it compare to the previous year?"
hypothetical_answer = augmented_query_generated(origional_query)

jointt_query = f"{origional_query}{hypothetical_answer}"

print(word_wrap(jointt_query))


results = chroma_collection.query(
    query_texts=jointt_query, 
    n_results=5,
    include=["documents", "embeddings"]
    )

retrieved_documents = results["documents"][0]


# for doc in retrieved_documents:
#     print(word_wrap(doc, width=100))
#     print("\n\n-----------------------------------\n\n")

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

retrieved_embeddings = results["embeddings"][0]
original_query_embedding = embedding_function([origional_query])
augmented_query_embedding = embedding_function([jointt_query])

projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

import matplotlib.pyplot as plt

# Plot the projected query and retrieved documents in the embedding space
plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{origional_query}")
plt.axis("off")
plt.show()  # display the plot
