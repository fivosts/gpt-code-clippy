from typing import Union
from fastcore.imports import all_equal

from ghapi.core import GhApi
from .api import try_request, paged_try_request, API_OBJECT
from itertools import groupby

from collections import defaultdict

COMMENT_INDIVIDUAL_FIELDS = ['id', 'body', 'created_at', 'updated_at', 'html_url']

COMMENT_THREAD_FIELDS = ['diff_hunk', 'path', 'position', 'original_position', 'commit_id', 'original_commit_id', 'start_line', 'original_start_line', 'start_side', 'line', 'original_line', 'side']

def process_reactions(reactions):
    reactions = dict(reactions)
    del reactions['url']
    for k in list(reactions.keys()):
        if reactions[k] == 0:
            del reactions[k]
    return reactions

def serializable_comment(comment):
    specific_data = {
        key: comment[key]
        for key in COMMENT_INDIVIDUAL_FIELDS
    }
    common_data = {
        key: comment[key]
        for key in COMMENT_THREAD_FIELDS
    }
    return specific_data, common_data

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
    num_comments = 0

    paths_by_commit = defaultdict(set)

    pull_requests_crawled = set()
    pull_request_comment_threads = []

    for seed_comment in paged_try_request(
        API_OBJECT.pulls.list_review_comments_for_repo, owner=owner, repo=repo,
        max_items=approximate_max_comments,
    ):
        try:
            pull_request_id = int(seed_comment.pull_request_url.split('/')[-1])
        except:
            print(f"could not extract PR id from {seed_comment.pull_request_url} for comment {seed_comment.node_id}")
            continue
        if pull_request_id in pull_requests_crawled:
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

        comment_threads = []

        for tpl, comment_thread in comments_by_path_and_position.items():
            this_specific, this_common = zip(*[serializable_comment(comment) for comment in comment_thread])
            if not all(x == this_common[0] for x in this_common):
                print(f"not all comment details match for {owner} {repo} {tpl}")
                continue

            # contains metadata such as original_comment_id, path, original_position_id, comments
            thread_info = this_common[0]
            thread_info['comments'] = this_specific
            comment_threads.append(thread_info)
        
        pull_request_comment_threads.append({
            'pull_request_number': pull_request_id,
            'comment_threads': comment_threads,
        })
        pull_requests_crawled.add(pull_request_id)

        # check if we've found enough comments
        num_comments += len(all_comments_for_pr)
        if approximate_max_comments is not None and num_comments >= approximate_max_comments:
            break

    # convert defaultdict of sets to dict of lists
    commits_and_paths = [{
        'commit_id': commit,
        'paths': list(sorted(paths))
        }
        for commit, paths in paths_by_commit.items()
    ]

    return {
        'commits_and_paths': commits_and_paths, 
        'pull_request_comment_threads': pull_request_comment_threads,
        'num_comments': num_comments,
    }
