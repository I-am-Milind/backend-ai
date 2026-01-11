import time
from search import web_search
from fact_memory import get_fact, save_fact, is_stale

REFRESH_INTERVAL = 3600  # every 1 hour

def refresh_facts():
    while True:
        time.sleep(REFRESH_INTERVAL)
        # You can later loop through popular keys
        # kept minimal to save API calls
