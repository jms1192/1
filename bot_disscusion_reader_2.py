#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
import os
import json
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Initialize OpenAI and SentenceTransformer clients
client = OpenAI(api_key="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")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# List of DAOs with their forum base URLs
DAO_FORUMS = [
    #{"name": "Jito", "base_url": "https://forum.jito.network"}
    #{"name": "Uniswap", "base_url": "https://gov.uniswap.org/"}
    {"name": "Aave", "base_url": "https://governance.aave.com/"},
    {"name": "Arbitrum", "base_url": "https://forum.arbitrum.foundation/"},
    {"name": "ENS", "base_url": "https://discuss.ens.domains/"},
    {"name": "Apecoin", "base_url": "https://forum.apecoin.com/"},
    {"name": "Balancer", "base_url": "https://forum.balancer.fi/"},
    {"name": "Lido", "base_url": "https://research.lido.fi/"},
    {"name": "Starknet", "base_url": "https://community.starknet.io/"},
    {"name": "Stargate", "base_url": "https://stargate.discourse.group/"},
    {"name": "Optimism", "base_url": "https://gov.optimism.io/"},
    {"name": "Gitcoin", "base_url": "https://gov.gitcoin.co/"},
    {"name": "GMX", "base_url": "https://gov.gmx.io/"},
    {"name": "Decentraland", "base_url": "https://forum.decentraland.org/"},
    {"name": "Radiant", "base_url": "https://community.radiant.capital/"},
    {"name": "Gnosis", "base_url": "https://forum.gnosis.io/"},
    {"name": "coW DAO", "base_url": "https://forum.cow.fi/"},
    {"name": "Hop", "base_url": "https://forum.hop.exchange/"},
    {"name": "Curve", "base_url": "https://gov.curve.fi/"},
    {"name": "Shapeshift", "base_url": "https://forum.shapeshift.com/"},
    {"name": "Gyroscope", "base_url": "https://forum.gyro.finance/"},
    {"name": "Frax", "base_url": "https://gov.frax.finance/"},
    {"name": "Compound", "base_url": "https://www.comp.xyz/"},
    {"name": "Across", "base_url": "https://forum.across.to/"},
    {"name": "Moonwell", "base_url": "https://forum.moonwell.fi/"},
    {"name": "Rocketpool", "base_url": "https://dao.rocketpool.net/"},
    {"name": "VitaDAO", "base_url": "https://gov.vitadao.com/"},
    {"name": "Goldfinch", "base_url": "https://gov.goldfinch.finance/"},
    {"name": "Cabin", "base_url": "https://forum.cabin.city/"}
]

MAX_TOKENS = 8192
CHUNK_SIZE = 3000


def get_embedding(text):
    """Get embedding for a given text, splitting if necessary."""
    if len(text) > CHUNK_SIZE:
        parts = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        embeddings = []
        for part in parts:
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=part
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Error embedding part: {e}")
                embeddings.append(None)
        return embeddings
    else:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return [response.data[0].embedding]
        except Exception as e:
            print(f"Error embedding text: {e}")
            return [None]


def fetch_all_categories(base_url):
    """Fetch all categories from the forum."""
    url = f"{base_url}/categories.json"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            return [cat.get("id") for cat in data.get("category_list", {}).get("categories", [])]
        except requests.exceptions.JSONDecodeError:
            print("Error decoding JSON for categories.")
    else:
        print(f"Failed to fetch categories. Status Code: {response.status_code}")
    return []


def fetch_category_topics(base_url, category_id):
    """Fetch all topics in a specific category with pagination."""
    url = f"{base_url}/c/{category_id}.json"
    topics = []
    while url:
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json()
                topics.extend(data.get("topic_list", {}).get("topics", []))
                # Handle pagination
                url = data.get("topic_list", {}).get("more_topics_url")
                if url:
                    url = f"{base_url}{url}"
            except requests.exceptions.JSONDecodeError:
                print("Error decoding JSON for category topics.")
                break
        else:
            print(f"Failed to fetch category topics. Status Code: {response.status_code}")
            break
    return topics


def fetch_topic_posts(base_url, topic_id):
    """Fetch all posts in a topic with error handling."""
    url = f"{base_url}/t/{topic_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"Error decoding JSON for topic {topic_id}.")
    else:
        print(f"Failed to fetch posts for topic {topic_id}. Status Code: {response.status_code}")
    return {}


def extract_metadata(post, topic, dao, category, base_url):
    """Format a single post into the desired metadata structure."""
    content = post.get("cooked", "").replace("\n", " ")
    embeddings = get_embedding(content)
    return {
        "dao_name": dao,
        "comment_id": post.get("id"),
        "content": content,
        "embeddings": embeddings,
        "author": post.get("username"),
        "author_metadata": {
            "trust_level": post.get("trust_level", None),
        },
        "topic_id": topic.get("id"),
        "topic_title": topic.get("title"),
        "category": category,
        "parent_comment_id": post.get("reply_to_post_number", None),
        "timestamp": post.get("created_at"),
        "source_url": f"{base_url}/t/{topic.get('id')}",
    }


def process_dao_forum(dao_name, base_url):
    """Process a DAO forum and extract all discussions."""
    print(f"Processing DAO: {dao_name}")
    categories = fetch_all_categories(base_url)
    if not categories:
        print(f"No categories found for DAO: {dao_name}")
        return []

    dao_comments = []
    for category_id in categories:
        topics = fetch_category_topics(base_url, category_id)
        for topic in topics:
            print(f"Processing topic: {topic.get('title', 'Unknown Title')} (ID: {topic['id']})")
            topic_data = fetch_topic_posts(base_url, topic['id'])
            posts = topic_data.get("post_stream", {}).get("posts", [])
            for post in posts:
                metadata = extract_metadata(post, topic, dao_name, category_id, base_url)
                dao_comments.append(metadata)
    return dao_comments


def main():
    all_comments = []
    for dao in DAO_FORUMS:
        comments = process_dao_forum(dao["name"], dao["base_url"])
        all_comments.extend(comments)

    # Save all comments to a JSON file
    with open("dao_all_comments_with_embeddings.json", "w") as f:
        json.dump(all_comments, f, indent=4)

    print(f"Total comments collected: {len(all_comments)}")


if __name__ == "__main__":
    main()