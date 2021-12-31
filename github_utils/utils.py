import subprocess
from typing import List

def get_output(command, cwd):
    return subprocess.run(command, shell=True, cwd=cwd, stdout=subprocess.PIPE).stdout.decode("utf-8")

def get_git_commits(repodir):
    lines = get_output(
        "git log --pretty=oneline", cwd=repodir,
    ).splitlines()
    hashes = [line.split()[0] for line in lines]
    return hashes

def get_git_date(repodir, commit=None):
    command = 'git show -s --format="%ci"'
    if commit:
        command += f" {commit}"
    line = get_output(command, repodir)
    return line.strip()


class GitRepo:
    def __init__(self, repodir):
        import git
        self.repodir = repodir
        self.repo = git.Repo(repodir)

    def get_file_at_commit(self, commit_sha, path):
        commit = self.repo.commit(commit_sha)
        blob = commit.tree / path
        return blob.data_stream.read().decode('utf-8')

    def get_file_before_and_after_commit(self, commit_sha, path):
        try:
            parent_sha = self.repo.commit(f'{commit_sha}~1').hexsha
            try:
                parent_file = self.get_file_at_commit(parent_sha, path)
            except KeyError as e:
                # file was introduced in commit_sha
                parent_file = ''
        except Exception as e:
            print(e)
            parent_sha = None
        
        child_file = self.get_file_at_commit(commit_sha, path)
        return {
            'path': path,
            'parent_commit': parent_sha,
            'parent_content': parent_file,
            'child_commit': commit_sha,
            'child_content': child_file,
        }

    def get_commit_info(self, commit_sha: str, paths_to_get_diffs_for: List[str]):
        commit = self.repo.commit(commit_sha)
        diffs = {
            path: self.repo.git.diff(f'{commit_sha}~1', commit_sha, path)
            for path in paths_to_get_diffs_for
        }
        return {
            'diffs': diffs,
            'authored_date': commit.authored_date,
            'committed_date': commit.committed_date,
        }