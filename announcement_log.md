# Announcement Log

This folder keeps a record of every announcement made about Easy Diffusion, and where it was made.

## Files

- `announcement_log.txt` - the log itself. One line per announcement channel (the same announcement posted in two places gets two lines).

## How to read a log entry

Each line is written in a simple key=value format, so scripts can read it too. Fields are separated by ` | `:

```
date=2022-09-24 | channel=twitter | type=release | topics=... | msg=... | url=...
```

- **date** - when it was announced (`YYYY-MM-DD`, or `unknown` if not known yet).
- **channel** - where it was posted: `blog`, `twitter`, `reddit`, or `discord`.
- **type** - what kind of announcement: `launch`, `release`, `feature`, `article`, `notice`, `general`, or `unknown`.
- **topics** - comma-separated list of topics/features covered.
- **msg** - one line describing what was announced.
- **url** - link to the actual announcement. Discord posts have no public URL, so the channel is named instead (blog posts use their permalink).

## How to add a new entry

1. Open `announcement_log.txt`.
2. Add a new line in date order, following the format above.
3. If the announcement was posted in more than one place, add one line per place.
4. Keep each announcement on a single line (no line breaks inside an entry).

## Parsing example

Because every entry uses the same `key=value` fields, the log is easy to process, e.g.:

```python
for line in open("announcement_log.txt"):
    if line.startswith("#"):
        continue  # skip comments
    fields = dict(part.split("=", 1) for part in line.strip().split(" | "))
    print(fields["date"], fields["channel"], fields["type"])
```

## Announcement channels

| Channel | Link |
| --- | --- |
| Blog | https://easydiffusion.github.io/blog/ |
| Twitter / X | https://x.com/cmdr2 |
| Reddit | r/StableDiffusion and r/deepdream |
| Discord | Easy Diffusion server, #announcements channel (https://discord.com/invite/u9yhsFmEkB) |
