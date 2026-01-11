import socket

def internet_available() -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except:
        return False


def needs_live_data(query: str) -> bool:
    keywords = [
        "today", "latest", "current", "now",
        "net worth", "price", "news", "update"
    ]
    q = query.lower()
    return any(k in q for k in keywords)
