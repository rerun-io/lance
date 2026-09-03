# Rerun's fork of Lance

This is [Rerun](https://rerun.io)'s fork of [Lance](https://github.com/lance-format/lance).

It exists to carry a small set of **Rerun-only patches** on top of an official upstream Lance
release until those patches are upstreamed (or no longer needed). The goal of this document is to
make the fork **boring to maintain**: a predictable, repeatable process for adding patches and for
rebasing them onto each new upstream release.

For the upstream project README (what Lance is, how to use it), see
[the upstream repo](https://github.com/lance-format/lance/blob/main/README.md).

---

## Repository layout

Two git remotes:

| Remote     | URL                          | Role                                   |
|------------|------------------------------|----------------------------------------|
| `origin`   | `rerun-io/lance`             | Our fork. Where we push everything.    |
| `upstream` | `lance-format/lance`         | Upstream. Source of releases.          |

Set them up once in a fresh clone:

```sh
git clone git@github.com:rerun-io/lance.git
cd lance
git remote add upstream git@github.com:lance-format/lance.git
git fetch --all --tags
```

Key branches:

| Branch                   | Meaning                                                                     |
|--------------------------|-----------------------------------------------------------------------------|
| `origin/main`            | Mirror of upstream `main`. We do **not** put Rerun patches here.            |
| `origin/release-X.Y.Z`   | The thing we actually ship: upstream tag `vX.Y.Z` + our Rerun-only commits. |

Note the naming difference: our release branches are `release-X.Y.Z`, matching the upstream **tag**
we branch from. Upstream's own release branches are `release/vX.Y`. The hyphenated form is
deliberate — upstream's CI workflows key off their naming, so ours needed explicit branch filters
to run at all.

We do **not** publish to crates.io. Downstream (Rerun) consumes this fork as a **git dependency**
pinned to a `release-X.Y.Z` branch (or a commit SHA on it). There are therefore no Rerun-specific
version bumps and no Rerun git tags — the workspace `version` stays identical to upstream's.

---

## The model

```
        upstream v10.0.0 (tag on upstream)
              │
              ▼
   ┌────────────────────────────────┐
   │  origin/release-10.0.0         │  ← branch = upstream tag + our patches, in order
   │                                │
   │      permanent commits         │
   │               +                │
   │      upstreamable commits      │
   │               +                │
   │      upstream backports        │
   │                                │
   └────────────────────────────────┘
```

Every PR into a release branch carries a `vX fork` label naming the line it landed on — `v10 fork`
for `release-10.0.0` — so the fork's history stays queryable after a patch has been dropped and no
longer shows up in the diff below.

On top of that, every patch is one of three kinds, and carries exactly one of these labels:

- **`permanent fork change`** — no upstream path; Rerun-specific by construction. Leaves the fork
  when the need goes away.
- **`upstreamable change`** — belongs upstream, but is not there yet. Leaves when upstream takes it.
- **`upstream backport`** — already upstream, on a release later than ours. Leaves when we rebase
  onto a release that has it.

The permanent set is the one to keep small. The other two kinds each have an exit condition, so
they leave on their own.

Every change we carry, as a diff against the upstream release we are on:

<https://github.com/lance-format/lance/compare/v10.0.0...rerun-io:lance:release-10.0.0>

The complete set of Rerun-only commits on any release branch is exactly:

```sh
git log --oneline vX.Y.Z..origin/release-X.Y.Z
```

If that command shows a commit you don't recognize, something went wrong in a rebase. Every commit
should be a deliberate Rerun patch.

### Rules

- **Never rewrite `origin/main`.** It tracks upstream only.
- **Never force-push a `release-X.Y.Z` branch that Rerun is already pinned to** without coordinating
  — downstream builds pin to it. Cut a new branch or append instead (see below).

---

## Adding a new Rerun-only patch

Branch off the current release branch, do the work, open a PR into that release branch.

Guidelines:

- **Target the active `release-X.Y.Z` branch**, never `origin/main`.
- Fill in `.github/pull_request_template.md`.
- Title the PR per [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). The
  `PR Checks` workflow validates the title and body with commitlint.
- **Label the PR** with `vX fork` and exactly one of the three kinds below. The `Fork Labels`
  workflow fails the PR until both are set.
- In the PR body, record **provenance**: for a port or backport, the source commit SHA and PR, the
  release line it came from, and whether it applied unchanged or was adapted; for a fork-only
  change, say that it is fork-only. This is what lets us identify and delete the patch later.
- After merge, the commit becomes part of the `vX.Y.Z..origin/release-X.Y.Z` patch set that future
  rebases must carry.

---

## Cutting a new fork release when upstream releases

When upstream ships a new release `vNEW` (e.g. `v11.0.0`), rebase our patch set onto it.

### 1. Fetch upstream and confirm the target tag

```sh
git fetch upstream --tags
# Stable tags only: a bare 'sort -V | tail' surfaces beta and rc tags, which we do not branch from.
git tag --list 'v*' | grep -Ev 'alpha|beta|rc|hotfix' | sort -V | tail
```

### 2. Capture the current patch set

```sh
# OLD = the release we're currently on, e.g. v10.0.0
git log --oneline v10.0.0..origin/release-10.0.0    # eyeball the patches we're about to move
```

### 3. Create the new release branch and rebase the patches onto it

```sh
git switch -c release-11.0.0 v11.0.0                # start from the new upstream tag

# Replay our patches (everything that was on the old release branch but not in the old tag).
# --onto release-11.0.0 puts them on the new base; v10.0.0 is the old base they came from.
git rebase --onto release-11.0.0 v10.0.0 origin/release-10.0.0
```

Resolve conflicts commit-by-commit. For each conflict:

```sh
# edit files, then:
git add -A
git rebase --continue
# if a patch has been upstreamed and is now redundant:
git rebase --skip
```

When a patch was upstreamed in `vNEW`, **skip it** — that's the payoff for upstreaming. Check the
`release-9.0.0` variant of a patch before porting the `release-8.0.0` one, and so on: the closer
ancestor of the new tag usually applies more cleanly.

### 4. Verify

```sh
git log --oneline v11.0.0..HEAD                     # should be our patch set, minus any upstreamed ones
cargo fmt --all
cargo clippy --all --tests --benches -- -D warnings
cargo test --workspace                              # or at least the crates our patches touch
```

The workspace version should now read the upstream `vNEW` version (e.g. `11.0.0`) unchanged — we do
not bump it.

Update the two refs in this README's compare link, and the diagram's branch name, to `vNEW`.

### 5. Publish the new release branch

```sh
git push origin release-11.0.0
```

### 6. Point downstream at it

Update Rerun's `Cargo.toml` git dependency to the new branch (or a pinned SHA on it):

```toml
lance = { git = "https://github.com/rerun-io/lance.git", branch = "release-11.0.0" }
```

Pin to a **commit SHA** rather than the bare branch if you need reproducible builds that don't move
when the release branch gets a new patch appended.

### 7. Keeping `origin/main` current (optional housekeeping)

```sh
git fetch upstream
git push origin upstream/main:main                  # fast-forward our mirror of upstream main
```

---

## Quick reference

```sh
# What are our patches on the current release?
git log --oneline vX.Y.Z..origin/release-X.Y.Z

# Add a patch
git switch -c myname/my-fix origin/release-X.Y.Z && ... && gh pr create --base release-X.Y.Z --draft

# Rebase onto a new upstream release
git switch -c release-NEW vNEW
git rebase --onto release-NEW vOLD origin/release-OLD
git push origin release-NEW
```
