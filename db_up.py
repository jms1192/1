#dao_all_comments_with_embeddings
import json
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pc = Pinecone(
    api_key="pcsk_3WQErp_AHcWoEVdvh1EuRkGzTDVndBnLVhSWURsYMerozMXADe3GSdRSe4DB7HD5v5msN7"
)

index_name = "morpho-governance"
if index_name not in pc.list_indexes().names():
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # Change this to match your embedding dimensions
        metric="cosine",  # Metric for similarity search
        spec=ServerlessSpec(cloud="aws", region="us-west-1"),
    )

index = pc.Index(index_name)

# Load the JSON data
with open("dao_all_comments_with_embeddings.json", "r") as f:
    comments_data = json.load(f)

# Prepare and upload embeddings
for comment in comments_data:
    dao_name = comment.get("dao_name", "unknown")
    comment_id = comment.get("comment_id")
    content = comment.get("content", "")
    embeddings = comment.get("embeddings", [])

    for idx, embedding in enumerate(embeddings):
        vector_id = f"{comment_id}_{idx}"  # Unique ID for each embedding part
        metadata = {
            "dao_name": dao_name,
            "comment_id": comment_id,
            "content": content,
            "part_index": idx,  # To track which part this is
        }

        try:
            index.upsert([(vector_id, embedding, metadata)])
            print(f"Uploaded vector ID: {vector_id}")
        except Exception as e:
            print(f"Error uploading vector ID {vector_id}: {e}")

print("Data upload complete.")