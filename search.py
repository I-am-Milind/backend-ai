import os
import requests

BING_API_KEY = os.getenv("BING_API_KEY")

def bing_search(query: str):
    if not BING_API_KEY:
        return [], []

    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "mkt": "en-US", "count": 3}

    r = requests.get(url, headers=headers, params=params, timeout=10)
    data = r.json()

    answers = []
    sources = []

    for item in data.get("webPages", {}).get("value", []):
        answers.append(item["snippet"])
        sources.append(item["url"])

    return answers, sources


def wikipedia_search(query: str):
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
    r = requests.get(url, timeout=10)

    if r.status_code != 200:
        return None, None

    data = r.json()
    return data.get("extract"), data.get("content_urls", {}).get("desktop", {}).get("page")


def hybrid_search(query: str):
    answers, sources = bing_search(query)

    wiki_text, wiki_url = wikipedia_search(query)

    if wiki_text:
        answers.append(wiki_text)
        sources.append(wiki_url)

    return {
        "answer": " ".join(answers),
        "sources": list(set(sources))
    }
