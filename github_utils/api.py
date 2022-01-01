import os
import time
from ghapi.all import GhApi

from fastcore.net import HTTP403ForbiddenError

API_KEY = os.environ.get("GH_API_KEY")

if API_KEY is not None:
    print(f"GH_API_KEY: {API_KEY[:4]}...")
else:
    print(f"GH_API_KEY: not found; using unauthenticated")

API_OBJECT = GhApi(token=API_KEY)

MAX_REQUESTS = 5000 if API_KEY is not None else 1000

api_requests = 0

DEFAULT_SLEEP_TIME_MINUTES = 5

DEFAULT_MAX_RETRIES = 20

def try_request(api_function, *args, max_retries=DEFAULT_MAX_RETRIES, sleep_time_seconds=60*DEFAULT_SLEEP_TIME_MINUTES, **kwargs):
    global api_requests

    retries = 0

    while retries <= max_retries:
        if api_requests >= MAX_REQUESTS - 1:
            print(f"hit max requests; sleeping for {sleep_time_seconds} seconds")
            time.sleep(sleep_time_seconds)
            api_requests = 0
        try:
            response = api_function(*args, **kwargs)
            api_requests += 1
            return response
        except HTTP403ForbiddenError as e:
            print(f"403 error; sleeping for {sleep_time_seconds} seconds")
            time.sleep(sleep_time_seconds)
            api_requests = 0
        except Exception as e:
            print(e)
        retries += 1
    return None


def paged_try_request(api_function, *args, page_size=100, max_items=None, max_retries=DEFAULT_MAX_RETRIES, sleep_time_seconds=60*DEFAULT_SLEEP_TIME_MINUTES, **kwargs):
    """
    api_function: (*args, **kwargs) -> Union[List[item], None], e.g. api.pulls.list_review_comments
    """
    node_ids = set()

    current_page = 1

    while True:
        if max_items and len(node_ids) >= max_items:
            break
        response = try_request(
            api_function,
            *args,
            # try_request args
            max_retries=max_retries, sleep_time_seconds=sleep_time_seconds,
            # pagination args for underlying api_function
            per_page=page_size, page=current_page,
            **kwargs
        )
        found_new = False
        if response is not None:
            for item in response:
                if max_items and len(node_ids) >= max_items:
                    break
                if item.node_id not in node_ids:
                    found_new = True
                    node_ids.add(item.node_id)
                    yield item
        if not found_new:
            break
        current_page += 1