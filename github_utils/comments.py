from typing import Union

from ghapi.core import GhApi
from .api import try_request, paged_try_request, API_OBJECT
from itertools import groupby

from collections import defaultdict

def get_review_comments(owner: str, repo: str, page_size: int=100, max_comments: Union[int, None]=None):
    current_page = 1
    urls = set()

    def request(api):
        return api.pulls.list_review_comments_for_repo(
            owner, repo, per_page=page_size, page=current_page
        )
    
    while True:
        if max_comments and len(urls) >= max_comments:
            break
        response = try_request(request)
        found_new = False
        if response is not None:
            for comment in response:
                if max_comments and len(urls) >= max_comments:
                    break
                if comment.url not in urls:
                    found_new = True
                    urls.add(comment.url)
                    yield comment
        if not found_new:
            break
        current_page += 1

def aggregate_comments(owner: str, repo: str, approximate_max_comments: Union[int, None]=None):
    comments_by_pull_request_id = {}
    num_comments = 0

    paths_by_commit = defaultdict(set)

    for seed_comment in paged_try_request(
        API_OBJECT.pulls.list_review_comments_for_repo, owner=owner, repo=repo,
        max_items=approximate_max_comments,
    ):
        try:
            pull_request_id = int(seed_comment.pull_request_url.split('/')[-1])
        except:
            print(f"could not extract PR id from {seed_comment.pull_request_url} for comment {seed_comment.node_id}")
            continue
        if pull_request_id in comments_by_pull_request_id:
            continue
        all_comments_for_pr = list(paged_try_request(
            API_OBJECT.pulls.list_review_comments,
            owner, repo, pull_request_id,
            # max_items=max_comments,
            sort='created_at',
            direction='asc',
        ))

        # group comments by (comment.path, comment.original_position) to effectively get the thread of comments
        comments_by_path_and_position = defaultdict(list)
        seed_comment_found = False
        for comment in all_comments_for_pr:
            comments_by_path_and_position[(comment.original_commit_id, comment.path, comment.original_position)].append(comment)
            if comment.node_id == seed_comment.node_id:
                seed_comment_found = True

            paths_by_commit[comment.original_commit_id].add(comment.path)
            paths_by_commit[comment.commit_id].add(comment.path)

        if not seed_comment_found:
            print(f"did not find comment with node id {seed_comment.node_id} in pr comments for {owner} {repo} {pull_request_id}")
        comments_by_pull_request_id[pull_request_id] = dict(comments_by_path_and_position)

        # check if we've found enough comments
        num_comments += len(all_comments_for_pr)
        if approximate_max_comments is not None and num_comments >= approximate_max_comments:
            break

    # convert defaultdict to dict
    paths_by_commit = dict(paths_by_commit)

    return {
        # Dict[pull_request_id: int, Dict[(original_commit_id: str, path: str, original_position: int), List[comment: str]]]
        'comments': comments_by_pull_request_id,
        # Dict[commit_id, [List[path: str]]]
        'paths_by_commit': paths_by_commit, 
        # int
        'num_comments': num_comments,
    }
