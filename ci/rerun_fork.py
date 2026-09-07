#!/usr/bin/env python3
"""
Tooling for Rerun's fork of Lance: the labels that make its patch set trackable,
and views over it. See the README for the model these follow.

Every PR into a release line from 10.0.0 on carries two labels:

* the release-line label for its base branch, ``lance fork <version>``, so a
  patch stays queryable by line even after it has been dropped from the branch;
* exactly one kind label, saying whether the patch has a way out of the fork.

Subcommands:

``check``
    Enforce both on one PR. Reads the GitHub event payload at
    ``$GITHUB_EVENT_PATH``, or ``--base-ref``/``--label`` when run by hand.
    This is what CI runs.
``audit``
    Apply the same rules to every PR already raised against a branch, for
    labelling a line cut before the check existed.
``view``
    Print the patch set of a branch as a timeline, with the kind of each change.
``selftest``
    Assert the rules against the cases they are meant to cover.

Only ``check`` runs in CI. The others read through ``gh`` and change nothing.
"""

import argparse
import json
import os
import re
import subprocess
import sys

KIND_LABELS = (
    "permanent fork change",
    "upstreamable change",
    "upstream backport",
)

# Column-width names for the same kinds, used by `view`.
KIND_SHORT = {
    "permanent fork change": "permanent",
    "upstreamable change": "upstreamable",
    "upstream backport": "backport",
}

# Our release branches are release-<major>.<minor>.<patch>, one per upstream
# release we rebase onto. The label names the line in full, so release-10.0.0
# and a later release-10.1.0 are told apart.
RELEASE_BRANCH = re.compile(r"^release-(\d+)\.(\d+)\.(\d+)$")

# Lines older than this predate the labels, and are not relabelled in
# retrospect. release-8.0.0 and release-9.0.0 are therefore never blocked.
MIN_VERSION = (10, 0, 0)

# Plain ANSI, so this needs no colour library. Pad text to width *before*
# painting it: the escapes have no width, and padding after would misalign.
ANSI = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
}
RESET = "\033[0m"

# GitHub's own state colours, so the output reads the same way the PR list does.
STATE_COLOUR = {"OPEN": "green", "MERGED": "magenta", "CLOSED": "red"}
KIND_COLOUR = {
    "permanent fork change": "yellow",
    "upstreamable change": "green",
    "upstream backport": "magenta",
}

_colour = False


def set_colour(mode):
    """Resolve --color, honouring NO_COLOR and whether we are on a terminal."""
    global _colour
    if mode == "always":
        _colour = True
    elif mode == "never":
        _colour = False
    else:
        _colour = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def paint(text, *names):
    names = [name for name in names if name]
    if not _colour or not names:
        return text
    return "".join(ANSI[name] for name in names) + text + RESET


def line_label(base_ref):
    """Return the required ``lance fork`` label for a base branch, or None if
    the branch is out of scope."""
    match = RELEASE_BRANCH.match(base_ref)
    if match is None:
        return None
    version = tuple(int(part) for part in match.groups())
    if version < MIN_VERSION:
        return None
    return "lance fork " + ".".join(str(part) for part in version)


def check(base_ref, labels):
    """Return a list of problems with this PR's labels; empty means it passes."""
    required = line_label(base_ref)
    if required is None:
        return []

    problems = []

    if required not in labels:
        problems.append(
            f"Add the release-line label '{required}' (base branch is '{base_ref}')."
        )

    kinds = [label for label in labels if label in KIND_LABELS]
    if not kinds:
        problems.append(
            "Add exactly one kind label: "
            + ", ".join(f"'{kind}'" for kind in KIND_LABELS)
            + "."
        )
    elif len(kinds) > 1:
        problems.append(
            f"The kind labels are mutually exclusive, but this PR has {len(kinds)}: "
            + ", ".join(f"'{kind}'" for kind in kinds)
            + "."
        )

    return problems


CASES = [
    # (base_ref, labels, expected number of problems)
    ("release-10.0.0", ["lance fork 10.0.0", "permanent fork change"], 0),
    ("release-10.0.0", ["lance fork 10.0.0", "upstreamable change"], 0),
    ("release-10.0.0", ["lance fork 10.0.0", "upstream backport"], 0),
    # Unrelated labels are applied automatically by other workflows.
    (
        "release-10.0.0",
        ["lance fork 10.0.0", "upstream backport", "documentation", "A-ci"],
        0,
    ),
    # Each line has its own label, including a later patch release.
    ("release-10.1.0", ["lance fork 10.1.0", "permanent fork change"], 0),
    ("release-11.0.0", ["lance fork 11.0.0", "upstreamable change"], 0),
    # Lines older than MIN_VERSION are out of scope, labels or not.
    ("release-9.0.0", [], 0),
    ("release-8.0.0", ["lance fork 8.0.0"], 0),
    ("release-2.0.0", [], 0),
    # So is anything that is not a release-<x>.<y>.<z> branch.
    ("main", [], 0),
    ("adam/some-branch", [], 0),
    ("release-10.0.0-rc1", [], 0),
    # Missing one label, or the other, or both.
    ("release-10.0.0", ["lance fork 10.0.0"], 1),
    ("release-10.0.0", ["permanent fork change"], 1),
    ("release-10.0.0", [], 2),
    # The line label has to match the base branch exactly.
    ("release-10.0.0", ["lance fork 10.1.0", "permanent fork change"], 1),
    ("release-10.1.0", ["lance fork 10.0.0", "permanent fork change"], 1),
    # Kinds are mutually exclusive.
    (
        "release-10.0.0",
        ["lance fork 10.0.0", "permanent fork change", "upstream backport"],
        1,
    ),
    ("release-10.0.0", list(KIND_LABELS), 2),
]


def selftest():
    for base_ref, labels, expected in CASES:
        problems = check(base_ref, labels)
        assert len(problems) == expected, (
            f"{base_ref} with {labels}: expected {expected} problem(s), "
            f"got {len(problems)}: {problems}"
        )
    print(f"{len(CASES)} cases passed")
    return 0


def list_prs(base_ref, state, limit, fields):
    """Return the PRs into base_ref, via gh. Read-only."""
    command = [
        "gh", "pr", "list",
        "--base", base_ref,
        "--state", state,
        "--limit", str(limit),
        "--json", ",".join(fields),
    ]  # fmt: skip
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        sys.exit("gh is not installed; this subcommand needs it to list PRs")
    except subprocess.CalledProcessError as error:
        sys.exit(f"gh failed: {error.stderr.strip()}")
    return json.loads(result.stdout)


def kinds_of(labels):
    return [label for label in labels if label in KIND_LABELS]


def last_commit_date(number):
    """The newest commit date on one PR, or "" if gh cannot say.

    Asked for per PR rather than in the bulk list: 'commits' carries an authors
    connection per commit, and requesting it for a page of PRs at once exceeds
    GitHub's GraphQL node limit.
    """
    command = ["gh", "pr", "view", str(number), "--json", "commits"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    dates = [
        commit["committedDate"]
        for commit in json.loads(result.stdout).get("commits") or []
        if commit.get("committedDate")
    ]
    # Take the max, not the last: a rebase can leave committer dates unordered.
    return max(dates) if dates else ""


def pr_timestamp(pull_request):
    """When a PR last moved: when it merged, or its newest commit if it did not.

    The full ISO timestamp, so PRs that land on the same day still order by the
    minute they landed. ISO-8601 in UTC sorts correctly as a plain string.
    """
    return pull_request.get("mergedAt") or pull_request.get("lastCommit") or ""


def pr_date(pull_request):
    """Just the day, for display."""
    return pr_timestamp(pull_request)[:10]


def run_check(args):
    if args.base_ref is not None:
        base_ref, labels = args.base_ref, args.label
    else:
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if not event_path:
            sys.exit("pass --base-ref, or run where $GITHUB_EVENT_PATH is set")
        with open(event_path) as event_file:
            pull_request = json.load(event_file)["pull_request"]
        base_ref = pull_request["base"]["ref"]
        labels = [label["name"] for label in pull_request["labels"]]

    problems = check(base_ref, labels)
    for problem in problems:
        print(f"::error::{problem}")
    if problems:
        print("See the 'Adding a new Rerun-only patch' section of README.md.")
        return 1

    if line_label(base_ref) is None:
        print(f"{base_ref} is not a labelled release line; nothing to check.")
    else:
        print(f"{base_ref}: labels OK ({', '.join(sorted(labels))})")
    return 0


def run_audit(args):
    required = line_label(args.branch)
    if required is None:
        print(f"{args.branch} is not a labelled release line; nothing to audit.")
        return 0

    pull_requests = list_prs(
        args.branch,
        args.state,
        args.limit,
        ["number", "title", "state", "baseRefName", "labels", "url", "mergedAt"],
    )
    for pull_request in pull_requests:
        if not pull_request.get("mergedAt"):
            pull_request["lastCommit"] = last_commit_date(pull_request["number"])
    # Newest first, on the full timestamp so same-day PRs order by the minute
    # they landed. The number only breaks an exact tie.
    pull_requests.sort(key=lambda pr: (pr_timestamp(pr), pr["number"]), reverse=True)
    print(f"{len(pull_requests)} PR(s) into {args.branch}, state={args.state}\n")

    failing = 0
    for pull_request in pull_requests:
        labels = [label["name"] for label in pull_request["labels"]]
        kinds = kinds_of(labels)
        state = pull_request["state"]
        if check(pull_request["baseRefName"], labels):
            failing += 1
            missing = []
            if required not in labels:
                missing.append(f"'{required}'")
            if not kinds:
                missing.append("a kind label")
            elif len(kinds) > 1:
                missing.append(f"only one of its {len(kinds)} kind labels")
            note = paint("needs " + " and ".join(missing), "yellow")
        else:
            note = paint("ok: " + ", ".join(kinds), "green")
        number = paint("#{:<4}".format(pull_request["number"]), "bold")
        state_cell = paint("{:<7}".format(state), STATE_COLOUR.get(state))
        date_cell = paint("{:<11}".format(pr_date(pull_request)), "dim")
        url = paint(pull_request["url"], "dim")
        print(
            f"  {number} {state_cell} {date_cell} {pull_request['title']}\n"
            f"        {note}\n"
            f"        {url}\n"
        )

    print(f"{len(pull_requests) - failing} labelled, {failing} to fix")
    if failing:
        print(
            "\nTo label one:\n"
            f"  gh pr edit <number> --add-label '{required}' --add-label '<kind>'\n"
            "where <kind> is one of: " + ", ".join(KIND_LABELS)
        )
    return 1 if failing else 0


def run_view(args):
    if line_label(args.branch) is None:
        print(f"{args.branch} is not a labelled release line.")
        return 0

    pull_requests = list_prs(
        args.branch,
        "merged",
        args.limit,
        ["number", "title", "mergedAt", "labels"],
    )
    # Oldest first, on the full timestamp: this is the order the patches replay
    # in on a rebase, which for two merged the same day is the order they merged.
    pull_requests.sort(key=lambda pr: (pr_timestamp(pr), pr["number"]))

    title_width = max(20, args.width - 40)
    print(f"\n{args.branch}: {len(pull_requests)} patches\n")
    print(f"  {'merged':<11} {'pr':<6} {'kind':<13} title")
    print(f"  {'-' * 11} {'-' * 6} {'-' * 13} {'-' * title_width}")

    counts = dict.fromkeys(KIND_LABELS, 0)
    unlabelled = 0
    for pull_request in pull_requests:
        kinds = kinds_of([label["name"] for label in pull_request["labels"]])
        if len(kinds) == 1:
            kind = paint("{:<13}".format(KIND_SHORT[kinds[0]]), KIND_COLOUR[kinds[0]])
            counts[kinds[0]] += 1
        elif not kinds:
            kind = paint("{:<13}".format("-"), "dim")
            unlabelled += 1
        else:
            kind = paint("{:<13}".format("AMBIGUOUS"), "red")
        merged = pr_date(pull_request) or "-"
        title = pull_request["title"]
        if len(title) > title_width:
            title = title[: title_width - 1] + "…"
        number = paint("#{:<5}".format(pull_request["number"]), "bold")
        print(f"  {paint('{:<11}'.format(merged), 'dim')} {number} {kind} {title}")

    summary = ", ".join(
        f"{counts[kind]} {KIND_SHORT[kind]}" for kind in KIND_LABELS if counts[kind]
    )
    print(f"\n  {summary or 'nothing classified'}")
    if unlabelled:
        print(f"  {unlabelled} unlabelled -- run 'audit {args.branch}' for what to add")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    check_parser = subcommands.add_parser(
        "check", help="enforce the labels on one PR (this is what CI runs)"
    )
    check_parser.add_argument("--base-ref", help="base branch of the PR")
    check_parser.add_argument(
        "--label", action="append", default=[], help="a label on the PR; repeatable"
    )
    check_parser.set_defaults(func=run_check)

    audit_parser = subcommands.add_parser(
        "audit", help="report PRs already raised against a branch that lack labels"
    )
    audit_parser.add_argument("branch", help="the base branch to audit")
    audit_parser.add_argument(
        "--state",
        default="all",
        choices=("all", "open", "closed", "merged"),
        help="which PRs to look at (default: all)",
    )
    audit_parser.add_argument(
        "--limit", type=int, default=200, help="how many PRs to fetch"
    )
    audit_parser.add_argument(
        "--color",
        default="auto",
        choices=("auto", "always", "never"),
        help="colour the output (default: auto, i.e. only on a terminal)",
    )
    audit_parser.set_defaults(func=run_audit)

    view_parser = subcommands.add_parser(
        "view", help="print a branch's patch set as a timeline, with kinds"
    )
    view_parser.add_argument("branch", help="the release branch to show")
    view_parser.add_argument(
        "--limit", type=int, default=200, help="how many PRs to fetch"
    )
    view_parser.add_argument(
        "--width", type=int, default=100, help="output width to wrap titles to"
    )
    view_parser.add_argument(
        "--color",
        default="auto",
        choices=("auto", "always", "never"),
        help="colour the output (default: auto, i.e. only on a terminal)",
    )
    view_parser.set_defaults(func=run_view)

    selftest_parser = subcommands.add_parser(
        "selftest", help="assert the rules against known cases"
    )
    selftest_parser.set_defaults(func=lambda args: selftest())

    args = parser.parse_args()
    set_colour(getattr(args, "color", "never"))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
