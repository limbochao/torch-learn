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
    parser = argparse.ArgumentParser(description="Create a GitCode Issue through API v5.")
    parser.add_argument("--owner", default="Ascend")
    parser.add_argument("--repo", default="pytorch")
    parser.add_argument("--title", required=True)
    body_group = parser.add_mutually_exclusive_group(required=True)
    body_group.add_argument("--body")
    body_group.add_argument("--body-file")
    parser.add_argument("--assignee")
    parser.add_argument("--milestone", type=int)
    parser.add_argument("--labels")
    parser.add_argument("--security-hole", choices=("true", "false"))
    parser.add_argument("--template-path")
    parser.add_argument("--issue-type")
    parser.add_argument("--issue-severity")
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
        body = args.body
    else:
        path = Path(args.body_file)
        if not path.is_file():
            die(f"body file does not exist: {path}")
        body = path.read_text(encoding="utf-8")
    if not body.strip():
        die("Issue body must not be empty.")
    return body


def build_payload(args, body):
    payload = {"repo": args.repo, "title": args.title, "body": body}
    optional = {
        "assignee": args.assignee,
        "milestone": args.milestone,
        "labels": args.labels,
        "security_hole": args.security_hole,
        "template_path": args.template_path,
        "issue_type": args.issue_type,
        "issue_severity": args.issue_severity,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
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


def output(value, as_json):
    if as_json:
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return
    if value.get("dry_run"):
        print(f"POST {value['url']}")
        print(json.dumps(value["payload"], ensure_ascii=False, indent=2))
        print(
            f"token_env={value['token_env']} has_token={str(value['has_token']).lower()} "
            f"git_credential_fallback={str(value['git_credential_fallback']).lower()}"
        )
        return
    number = value.get("number") or value.get("id") or "unknown"
    print(f"Created GitCode Issue #{number}: {value.get('html_url') or '<URL missing>'}")


def sanitized_error(error, token):
    raw = error.read().decode("utf-8", errors="replace")
    if token:
        raw = raw.replace(token, "<redacted>")
    try:
        parsed = json.loads(raw)
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        return raw.strip() or error.reason


def main():
    args = parse_args()
    validate_env_name(args.token_env)
    if not args.owner.strip() or not args.repo.strip() or not args.title.strip():
        die("--owner, --repo, and --title must not be empty.")
    if args.timeout <= 0:
        die("--timeout must be greater than zero.")

    body = read_body(args)
    payload = build_payload(args, body)
    owner = urllib.parse.quote(args.owner.strip(), safe="")
    url = f"{args.base_url.rstrip('/')}/repos/{owner}/issues"
    token_from_env = bool(os.environ.get(args.token_env))

    if args.dry_run:
        output(
            {
                "dry_run": True,
                "method": "POST",
                "url": url,
                "auth": "Authorization: Bearer <redacted>",
                "token_env": args.token_env,
                "has_token": token_from_env,
                "git_credential_fallback": not args.no_git_credential,
                "payload": payload,
            },
            args.json,
        )
        return

    token, auth_source = resolve_token(args)

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "codex-gitcode-create-issue/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw = response.read().decode("utf-8")
            result = json.loads(raw)
    except urllib.error.HTTPError as error:
        die(f"GitCode API returned HTTP {error.code}: {sanitized_error(error, token)}")
    except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
        reason = getattr(error, "reason", error)
        die(
            "GitCode API request failed: "
            f"{reason}. Search by title before retrying to avoid duplicate creation."
        )
    except json.JSONDecodeError:
        die("GitCode API returned a non-JSON success response.")

    summary = {
        "id": result.get("id"),
        "number": result.get("number"),
        "state": result.get("state"),
        "title": result.get("title"),
        "html_url": result.get("html_url"),
        "auth_source": auth_source,
    }
    output(summary, args.json)


if __name__ == "__main__":
    main()
