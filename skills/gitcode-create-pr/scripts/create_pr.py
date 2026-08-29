#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
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
        description="Create or update a GitCode Pull Request through API v5."
    )
    parser.add_argument("--owner", default="Ascend")
    parser.add_argument("--repo", default="pytorch")
    parser.add_argument("--title")
    parser.add_argument("--head")
    parser.add_argument("--base", default="master")
    parser.add_argument("--update-number", type=int)
    body_group = parser.add_mutually_exclusive_group()
    body_group.add_argument("--body")
    body_group.add_argument("--body-file")
    parser.add_argument("--fork-path")
    parser.add_argument("--milestone-number", type=int)
    parser.add_argument("--labels")
    parser.add_argument("--issue", help="Issue number(s), separated by commas")
    parser.add_argument("--assignees")
    parser.add_argument("--testers")
    parser.add_argument("--prune-source-branch", action="store_true")
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--squash", action="store_true")
    parser.add_argument("--squash-commit-message")
    parser.add_argument("--close-related-issue", action="store_true")
    parser.add_argument("--token-env", default="GITCODE_TOKEN")
    parser.add_argument("--no-git-credential", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def validate_env_name(name):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        die("--token-env must be an environment variable name, not a token value.")


def read_body(args):
    if args.body is not None:
        return args.body
    if args.body_file is None:
        return None
    path = Path(args.body_file)
    if not path.is_file():
        die(f"body file does not exist: {path}")
    body = path.read_text(encoding="utf-8")
    if not body.strip():
        die("PR body file must not be empty.")
    return body


def normalize_head(head, fork_path):
    head = head.strip()
    if not head:
        die("--head must not be empty.")
    if fork_path:
        parts = fork_path.strip().split("/")
        if len(parts) != 2 or not all(parts):
            die("--fork-path must use the owner/repo format.")
        if ":" not in head:
            head = f"{parts[0]}:{head}"
    elif ":" in head:
        die("cross-repository --head requires --fork-path owner/repo.")
    return head


def parse_issues(value):
    if value is None:
        return []
    issues = []
    for item in value.split(","):
        item = item.strip().lstrip("#")
        if not item.isdigit() or int(item) <= 0:
            die("--issue must contain positive issue numbers separated by commas.")
        issues.append(int(item))
    return list(dict.fromkeys(issues))


def add_boolean_options(payload, args):
    for key, enabled in {
        "prune_source_branch": args.prune_source_branch,
        "draft": args.draft,
        "squash": args.squash,
        "close_related_issue": args.close_related_issue,
    }.items():
        if enabled:
            payload[key] = True
    if args.squash_commit_message is not None:
        payload["squash_commit_message"] = args.squash_commit_message


def build_create_payload(args, body, head):
    payload = {"title": args.title, "head": head, "base": args.base}
    optional = {
        "body": body,
        "milestone_number": args.milestone_number,
        "labels": args.labels,
        "assignees": args.assignees,
        "testers": args.testers,
        "fork_path": args.fork_path,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    add_boolean_options(payload, args)
    return payload


def build_update_payload(args, body):
    optional = {
        "title": args.title,
        "body": body,
        "milestone_number": args.milestone_number,
        "labels": args.labels,
        "assignees": args.assignees,
        "testers": args.testers,
    }
    payload = {key: value for key, value in optional.items() if value is not None}
    add_boolean_options(payload, args)
    return payload


def git_credential_token(base_url):
    host = urllib.parse.urlparse(base_url).hostname or "gitcode.com"
    if host.startswith("api."):
        host = host[4:]
    credential_input = f"protocol=https\nhost={host}\n\n"
    try:
        result = subprocess.run(
            ["git", "credential", "fill"],
            input=credential_input,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    fields = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields.get("password")


def resolve_token(args):
    token = os.environ.get(args.token_env)
    if token:
        return token, args.token_env
    if not args.no_git_credential:
        token = git_credential_token(args.base_url)
        if token:
            return token, "git credential"
    die(
        f"environment variable {args.token_env} is not set and no GitCode credential is available."
    )


def sanitized_error(error, token):
    raw = error.read().decode("utf-8", errors="replace")
    if token:
        raw = raw.replace(token, "<redacted>")
    try:
        parsed = json.loads(raw)
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        return raw.strip() or error.reason


def request_json(url, method, token, timeout, payload=None):
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "codex-gitcode-create-pr/2.0",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        die(f"GitCode API returned HTTP {error.code}: {sanitized_error(error, token)}")
    except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
        reason = getattr(error, "reason", error)
        die(
            "GitCode API request failed: "
            f"{reason}. Query the PR list before retrying to avoid duplicate creation."
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        die("GitCode API returned a non-JSON success response.")


def link_and_verify_issues(base_url, owner, repo, number, issues, token, timeout):
    issues_url = f"{base_url.rstrip('/')}/repos/{owner}/{repo}/pulls/{number}/issues"
    request_json(issues_url, "POST", token, timeout, issues)
    linked = request_json(issues_url, "GET", token, timeout)
    linked_numbers = {str(item.get("number")) for item in linked}
    missing = [issue for issue in issues if str(issue) not in linked_numbers]
    if missing:
        die(
            f"PR #{number} exists, but linked Issue verification failed for: "
            + ", ".join(f"#{issue}" for issue in missing)
        )
    return sorted(int(number) for number in linked_numbers if number and number.isdigit())


def output(value, as_json):
    if as_json:
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return
    if value.get("dry_run"):
        print(f"{value['method']} {value['url']}")
        print(json.dumps(value["payload"], ensure_ascii=False, indent=2))
        if value.get("linked_issues"):
            print(f"POST {value['issues_url']}")
            print(json.dumps(value["linked_issues"], ensure_ascii=False))
            print(f"GET {value['issues_url']}")
        print(
            f"token_env={value['token_env']} has_token={str(value['has_token']).lower()} "
            f"git_credential_fallback={str(value['git_credential_fallback']).lower()}"
        )
        return
    action = "Updated" if value["action"] == "update" else "Created"
    number = value.get("number") or value.get("id") or "unknown"
    print(f"{action} GitCode PR #{number}: {value.get('html_url') or '<URL missing>'}")
    if value.get("linked_issues"):
        print("Linked Issues: " + ", ".join(f"#{issue}" for issue in value["linked_issues"]))


def main():
    args = parse_args()
    validate_env_name(args.token_env)
    if not args.owner.strip() or not args.repo.strip():
        die("--owner and --repo must not be empty.")
    if args.timeout <= 0:
        die("--timeout must be greater than zero.")
    if args.update_number is not None and args.update_number <= 0:
        die("--update-number must be a positive PR number.")
    if args.squash_commit_message is not None and not args.squash:
        die("--squash-commit-message requires --squash.")

    body = read_body(args)
    issues = parse_issues(args.issue)
    owner = urllib.parse.quote(args.owner.strip(), safe="")
    repo = urllib.parse.quote(args.repo.strip(), safe="")
    pulls_url = f"{args.base_url.rstrip('/')}/repos/{owner}/{repo}/pulls"

    if args.update_number is None:
        if not args.title or not args.title.strip() or not args.head or not args.head.strip():
            die("creating a PR requires --title and --head.")
        if not args.base.strip():
            die("--base must not be empty.")
        head = normalize_head(args.head, args.fork_path)
        payload = build_create_payload(args, body, head)
        url = pulls_url
        method = "POST"
        action = "create"
        number = None
    else:
        if args.head is not None or args.fork_path is not None:
            die("--head and --fork-path cannot be used with --update-number.")
        payload = build_update_payload(args, body)
        if not payload and not issues:
            die("updating a PR requires a changed field or --issue.")
        url = f"{pulls_url}/{args.update_number}"
        method = "PATCH" if payload else "GET"
        action = "update"
        number = args.update_number

    issues_url = f"{pulls_url}/{number or '<created-number>'}/issues"
    token_from_env = bool(os.environ.get(args.token_env))
    if args.dry_run:
        output(
            {
                "dry_run": True,
                "method": method,
                "url": url,
                "auth": "Authorization: Bearer <redacted>",
                "token_env": args.token_env,
                "has_token": token_from_env,
                "git_credential_fallback": not args.no_git_credential,
                "payload": payload,
                "linked_issues": issues,
                "issues_url": issues_url,
            },
            args.json,
        )
        return

    token, auth_source = resolve_token(args)
    result = request_json(url, method, token, args.timeout, payload if method != "GET" else None)
    number = number or result.get("number") or result.get("iid")
    if not number:
        die("GitCode API response did not include a PR number; query the PR list before retrying.")

    linked_issues = []
    if issues:
        linked_issues = link_and_verify_issues(
            args.base_url, owner, repo, number, issues, token, args.timeout
        )

    summary = {
        "action": action,
        "id": result.get("id"),
        "number": number,
        "state": result.get("state"),
        "title": result.get("title") or args.title,
        "html_url": result.get("html_url")
        or result.get("web_url")
        or f"https://gitcode.com/{args.owner}/{args.repo}/merge_requests/{number}",
        "linked_issues": linked_issues,
        "auth_source": auth_source,
    }
    output(summary, args.json)


if __name__ == "__main__":
    main()
