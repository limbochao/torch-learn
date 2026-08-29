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
        description="Comment on a GitCode Pull Request through API v5."
    )
    parser.add_argument("--pr-url", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--token-env", default="GITCODE_TOKEN")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
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


def request_json(url, token, timeout, method="GET", payload=None):
    headers = {
        "Accept": "application/json",
        "User-Agent": "codex-gitcode-comment-pr/1.0",
    }
    data = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
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
        for key in ("data", "comments", "items", "list", "records"):
            if key in current:
                current = current[key]
                break
        else:
            return []
    return current if isinstance(current, list) else []


def comment_body(comment):
    if not isinstance(comment, dict):
        return None
    for key in ("body", "note", "content"):
        value = comment.get(key)
        if isinstance(value, str):
            return value
    return None


def comment_summary(comment):
    item = unwrap_object(comment)
    return {
        "id": item.get("id") or item.get("comment_id"),
        "body": comment_body(item),
        "html_url": item.get("html_url") or item.get("web_url") or item.get("url"),
    }


def output(value, as_json):
    if as_json:
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return
    if value.get("dry_run"):
        print(f"GET {value['comments_url']}")
        print(f"POST {value['comments_url']}")
        print(json.dumps(value["payload"], ensure_ascii=False, indent=2))
        print(f"token_env={value['token_env']} has_token={str(value['has_token']).lower()}")
        return
    if value.get("skipped_duplicate"):
        print(f"Skipped duplicate GitCode PR comment: {value['pr_url']}")
        return
    comment_id = value.get("comment_id") or "<ID missing>"
    print(f"Created GitCode PR comment {comment_id}: {value['pr_url']}")


def main():
    args = parse_args()
    validate_env_name(args.token_env)
    if args.timeout <= 0:
        die("--timeout must be greater than zero.")
    if not args.body.strip():
        die("--body must not be empty.")
    owner, repo, number = parse_pr_url(args.pr_url)
    owner_path = urllib.parse.quote(owner, safe="")
    repo_path = urllib.parse.quote(repo, safe="")
    comments_url = (
        f"{args.base_url.rstrip('/')}/repos/{owner_path}/{repo_path}/pulls/{number}/comments"
    )
    token = os.environ.get(args.token_env)
    payload = {"body": args.body}
    if args.dry_run:
        output(
            {
                "dry_run": True,
                "comments_url": comments_url,
                "payload": payload,
                "token_env": args.token_env,
                "has_token": bool(token),
            },
            args.json,
        )
        return
    if not token:
        die(f"environment variable {args.token_env} is not set.")
    if not args.allow_duplicate:
        comments = unwrap_list(request_json(comments_url, token, args.timeout))
        for comment in comments:
            if comment_body(comment) == args.body:
                summary = comment_summary(comment)
                output(
                    {
                        "pr_url": args.pr_url,
                        "skipped_duplicate": True,
                        "comment_id": summary["id"],
                        "comment_url": summary["html_url"],
                    },
                    args.json,
                )
                return
    result = comment_summary(
        request_json(comments_url, token, args.timeout, method="POST", payload=payload)
    )
    output(
        {
            "pr_url": args.pr_url,
            "skipped_duplicate": False,
            "comment_id": result["id"],
            "comment_url": result["html_url"],
            "body": result["body"] or args.body,
        },
        args.json,
    )


if __name__ == "__main__":
    main()
