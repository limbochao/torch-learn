#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_BASE_URL = "https://api.gitcode.com/api/v5"


def die(message, code=1):
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(code)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a GitCode Pull Request and its linked issues."
    )
    parser.add_argument("--pr-url", required=True)
    parser.add_argument("--require-single-issue", action="store_true")
    parser.add_argument("--token-env", default="GITCODE_TOKEN")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def validate_env_name(name):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        die("--token-env must be an environment variable name, not a token value.")


def parse_pr_url(value):
    parsed = urllib.parse.urlparse(value.strip())
    if parsed.scheme != "https" or parsed.hostname not in {"gitcode.com", "www.gitcode.com"}:
        die("--pr-url must be an https://gitcode.com Pull Request URL.")
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) != 4 or parts[2] not in {"merge_requests", "pull", "pulls"}:
        die("--pr-url must use /<owner>/<repo>/merge_requests/<number> or /pulls/<number>.")
    owner, repo, _, number_text = parts
    if not number_text.isdigit() or int(number_text) <= 0:
        die("Pull Request number must be a positive integer.")
    return owner, repo.removesuffix(".git"), int(number_text)


def sanitized_error(error, token):
    raw = error.read().decode("utf-8", errors="replace")
    if token:
        raw = raw.replace(token, "<redacted>")
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except json.JSONDecodeError:
        return raw.strip() or str(error.reason)


def get_json(url, token, timeout):
    headers = {"Accept": "application/json", "User-Agent": "codex-gitcode-sync-pr/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        die(f"GitCode API returned HTTP {error.code}: {sanitized_error(error, token)}")
    except urllib.error.URLError as error:
        die(f"GitCode API request failed: {error.reason}")
    except json.JSONDecodeError:
        die("GitCode API returned a non-JSON success response.")


def unwrap_object(value):
    current = value
    for _ in range(3):
        if not isinstance(current, dict):
            break
        nested = current.get("data")
        if not isinstance(nested, dict):
            break
        current = nested
    return current if isinstance(current, dict) else {}


def unwrap_list(value):
    current = value
    for _ in range(4):
        if isinstance(current, list):
            return current
        if not isinstance(current, dict):
            return []
        for key in ("data", "issues", "items", "list", "records"):
            if key in current:
                current = current[key]
                break
        else:
            if any(key in current for key in ("number", "iid", "id", "title")):
                return [current]
            return []
    return current if isinstance(current, list) else []


def first_value(mapping, *keys):
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def extract_head(pr):
    head = pr.get("head")
    source_branch = first_value(pr, "source_branch", "head_ref", "head_branch")
    source_sha = first_value(pr, "head_sha", "source_sha")
    source_repo = None
    if isinstance(head, dict):
        source_branch = source_branch or first_value(head, "ref", "branch", "label")
        source_sha = source_sha or first_value(head, "sha", "commit_sha")
        repo = head.get("repo") or head.get("repository")
        if isinstance(repo, dict):
            source_repo = first_value(repo, "full_name", "path_with_namespace", "path")
    elif isinstance(head, str):
        source_branch = source_branch or head
    if isinstance(source_branch, str) and ":" in source_branch:
        source_branch = source_branch.split(":", 1)[1]
    return source_branch, source_sha, source_repo


def summarize_issue(issue):
    if not isinstance(issue, dict):
        return {"number": None, "title": None, "url": None}
    return {
        "number": first_value(issue, "number", "iid", "id"),
        "title": first_value(issue, "title", "name"),
        "url": first_value(issue, "html_url", "web_url", "url"),
    }


def output(summary, as_json):
    if as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print(f"PR: {summary['pr_url']}")
    print(f"title: {summary['title'] or '<missing>'}")
    print(f"source_branch: {summary['source_branch'] or '<missing>'}")
    print(f"source_sha: {summary['source_sha'] or '<missing>'}")
    print(f"linked_issue_count: {summary['linked_issue_count']}")
    for issue in summary["linked_issues"]:
        print(
            f"issue: {issue['number'] or '<missing>'} "
            f"{issue['title'] or '<missing>'} {issue['url'] or '<missing>'}"
        )


def main():
    args = parse_args()
    validate_env_name(args.token_env)
    if args.timeout <= 0:
        die("--timeout must be greater than zero.")
    owner, repo, number = parse_pr_url(args.pr_url)
    token = os.environ.get(args.token_env)
    base_url = args.base_url.rstrip("/")
    owner_path = urllib.parse.quote(owner, safe="")
    repo_path = urllib.parse.quote(repo, safe="")
    pr_api_url = f"{base_url}/repos/{owner_path}/{repo_path}/pulls/{number}"
    issues_api_url = f"{pr_api_url}/issues"
    pr = unwrap_object(get_json(pr_api_url, token, args.timeout))
    issues = [
        summarize_issue(item)
        for item in unwrap_list(get_json(issues_api_url, token, args.timeout))
    ]
    source_branch, source_sha, source_repo = extract_head(pr)
    summary = {
        "owner": owner,
        "repo": repo,
        "number": number,
        "pr_url": first_value(pr, "html_url", "web_url") or args.pr_url,
        "title": first_value(pr, "title", "name"),
        "state": first_value(pr, "state", "status"),
        "source_branch": source_branch,
        "source_sha": source_sha,
        "source_repo": source_repo,
        "linked_issue_count": len(issues),
        "linked_issues": issues,
        "single_issue": issues[0] if len(issues) == 1 else None,
    }
    output(summary, args.json)
    if args.require_single_issue and len(issues) != 1:
        print(
            f"Error: source PR must have exactly one linked issue; found {len(issues)}.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
