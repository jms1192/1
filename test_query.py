from openai import OpenAI
from pinecone import Pinecone

# Initialize OpenAI and Pinecone clients
client = OpenAI(api_key="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")
pc = Pinecone(api_key="pcsk_3WQErp_AHcWoEVdvh1EuRkGzTDVndBnLVhSWURsYMerozMXADe3GSdRSe4DB7HD5v5msN7")
index = pc.Index("morpho-governance")

def get_similar_context(input_text, top_k=4):
    """Query Pinecone for similar comments based on input text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=input_text
    )
    query_vector = response.data[0].embedding

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    return [
        {
            "content": match.metadata.get("content", ""),
            "source_url": match.metadata.get("source_url", "No link available")
        }
        for match in results.matches
    ]

def generate_concise_response(input_text):
    """Generate a concise response to the user's input using GPT-4o."""
    # Fetch context from Pinecone
    similar_comments = get_similar_context(input_text)
    print(similar_comments)
    # Combine context into a single string for the prompt
    context = "\n\n".join(
        f"Comment {i+1}: {comment['content']}"
        for i, comment in enumerate(similar_comments)
    )

    # Define the chat prompt
    messages = [
        {"role": "system", "content": "You are a DAO governance intern on Twitter, and all your context is from governance forums. Respond concisely in under 200 characters."},
        {"role": "user", "content": f"Using the context below, respond concisely:\n\nContext:\n{context}\n\nQuery: {input_text}"}
    ]

    # Query GPT-4o
    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=messages
    )
    
    # Extract the response content
    gpt_response = completion.choices[0].message.content

    # Add the source URL of the first comment
    link1 = similar_comments[0]["source_url"] if similar_comments else "No link available"
    link2 = similar_comments[1]["source_url"] if similar_comments else "No link available"
    link3 = similar_comments[2]["source_url"] if similar_comments else "No link available"
  
    print(link1)
    print(link2)
    print(link3)

    return f"{gpt_response}\n\nSource: {link1}"

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    response = generate_concise_response(user_query)
    print("\nResponse:\n")
    print(response)

