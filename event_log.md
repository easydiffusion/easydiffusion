# event_log.md

This folder has a project-wide event log: `event_log.txt` (next to this file, inside the
`easydiffusion` repo).

It is a single, shared record of everything notable that happens in the codebase — new features,
bugfixes, performance work, releases, and so on — across all three repos in this workspace
(`easydiffusion`, `sdkit`, `easydiffusion.github.io`).

## What a line looks like

    [2026-07-24] [SEV-2] [RELEASE] v4.5.1: startup time improved by removing unnecessary steps ... 

Each line has four parts:

1. **Date** in `YYYY-MM-DD` format — taken from the git commit date or the date in `CHANGES.md`.
2. **Severity tag** — `SEV-1` down to `SEV-5`. See below.
3. **Category label** — one fixed word in brackets. See below.
4. **One-line description** — plain language. Links (GitHub issues, PRs, wiki pages) go at the end.

## Severity levels

| Tag     | Meaning                                                                  |
|---------|--------------------------------------------------------------------------|
| SEV-1   | Project-impacting. Major releases, engine swaps, headline features.       |
| SEV-2   | Big user-facing additions: new model support, major features, big fixes.  |
| SEV-3   | Normal bugfixes, performance wins, and useful improvements.               |
| SEV-4   | Minor fixes, dependency bumps, small internal cleanups.                   |
| SEV-5   | Trivial stuff: styling, typos, tweaks only relevant to the team.          |

## Categories

`RELEASE`, `FEATURE`, `BUGFIX`, `PERF`, `UI`, `ENGINE`, `SECURITY`, `INSTALLER`, `INTERNAL`,
`DEPS`, `CI`, `DOCS`, `WIKI`.

Use one category per entry. If a change fits two (e.g. a UI bugfix), pick the one a reader would
care about most.

## How entries are ordered

Oldest first, newest last — like an append-only log. When the same change shipped in several
engine tracks (e.g. the 24 Jul 2026 startup-time work landed as v3.0.17, v3.5.15 and v4.5.1),
it gets **one** entry, not three.

## Adding a new entry

1. Pick the date of the change (commit date, or today for non-code events).
2. Pick the lowest severity that honestly describes the impact. When in doubt, go one level down.
3. Write one line. Plain sentences, no jargon, no filler. Add a link if the change has an
   issue, PR, or wiki page.
4. Append it at the bottom of `event_log.txt`, keeping the same format.

That's it — no build step, no tooling. The file is the log.
