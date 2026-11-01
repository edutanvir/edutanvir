#!/usr/bin/env python3
"""
make_stats.py - builds two neon "hacker" SVG cards from the GitHub API:
    stats-card.svg   (stars, commits, PRs, issues, repos, followers)
    langs-card.svg   (top languages, animated bars)

Runs inside your GitHub Action (uses the built-in GITHUB_TOKEN), so it does not
depend on any public stats service that can go down.

Usage:  python3 scripts/make_stats.py [out_dir] [--demo]
"""
import html
import json
import os
import sys
import urllib.request

OUT = next((a for a in sys.argv[1:] if not a.startswith("--")), "dist")
DEMO = "--demo" in sys.argv
USER = os.environ.get("GITHUB_REPOSITORY_OWNER") or os.environ.get("GH_USER") or "edutanvir"
NAME = "Md Tanvir Ahmed"
TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
API = "https://api.github.com"
PALETTE = ["#39ff14", "#00e5ff", "#ff9f1c", "#ffe600", "#4da3ff", "#2ee6a6", "#ff6b35", "#00c853"]
esc = html.escape


def _request(url, body=None):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "profile-stats-card"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def gh(path):
    return _request(API + path)


def gql(query, variables):
    out = _request(API + "/graphql", {"query": query, "variables": variables})
    if "errors" in out and not out.get("data"):
        raise RuntimeError(out["errors"])
    return out["data"]


def fetch():
    if DEMO:
        return dict(stars=24, commits=612, prs=11, issues=4, repos=18, followers=137,
                    langs={"Python": 52000, "Jupyter Notebook": 31000, "SQL": 14000,
                           "HTML": 6000, "Shell": 3000, "Scala": 1500})
    user = gh(f"/users/{USER}")
    repos, page = [], 1
    while page <= 5:
        chunk = gh(f"/users/{USER}/repos?per_page=100&page={page}&type=owner")
        repos += chunk
        if len(chunk) < 100:
            break
        page += 1
    own = [r for r in repos if not r.get("fork")]
    langs = {}
    for r in own[:40]:
        try:
            for k, v in gh(f"/repos/{USER}/{r['name']}/languages").items():
                langs[k] = langs.get(k, 0) + v
        except Exception as e:                     # one bad repo should not break the card
            print("language lookup skipped:", r["name"], e)
    commits = prs = issues = None
    try:
        q = """query($login:String!){user(login:$login){contributionsCollection{
                 totalCommitContributions totalPullRequestContributions
                 totalIssueContributions restrictedContributionsCount}}}"""
        c = gql(q, {"login": USER})["user"]["contributionsCollection"]
        commits = c["totalCommitContributions"] + c["restrictedContributionsCount"]
        prs, issues = c["totalPullRequestContributions"], c["totalIssueContributions"]
    except Exception as e:
        print("contribution stats unavailable:", e)
    return dict(stars=sum(r.get("stargazers_count", 0) for r in own), commits=commits, prs=prs,
                issues=issues, repos=user.get("public_repos", len(own)),
                followers=user.get("followers", 0), langs=langs)


def fmt(v):
    return "-" if v is None else f"{v:,}"


CSS = """
  text{font-family:"DejaVu Sans Mono",Consolas,Menlo,"Liberation Mono",monospace}
  .t{font-size:15px;font-weight:700;fill:#39ff14}
  .r{opacity:0;animation:in .5s ease forwards}
  @keyframes in{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:translateX(0)}}
  .cur{animation:bl 1s steps(1) infinite}
  @keyframes bl{50%{opacity:0}}
"""

FRAME = ('<rect x="1" y="1" width="{w}" height="{h}" rx="12" fill="#0d1117" stroke="#39ff14" stroke-width="1.6">'
         '<animate attributeName="stroke" values="#39ff14;#00e5ff;#ff9f1c;#ffe600;#39ff14" dur="8s" '
         'repeatCount="indefinite"/></rect>')


def title(text):
    return (f'<text class="t" x="20" y="31">&gt; {esc(text)}</text>'
            f'<rect class="cur" x="{20 + (len(text) + 2) * 9.1:.0f}" y="19" width="8" height="15" fill="#39ff14"/>')


def stats_svg(d):
    W, H = 440, 190
    rows = [("[$]", "Stars earned", d["stars"], "#ffe600"),
            ("[%]", "Commits (last year)", d["commits"], "#39ff14"),
            ("[+]", "Pull requests", d["prs"], "#00e5ff"),
            ("[!]", "Issues opened", d["issues"], "#ff9f1c"),
            ("[#]", "Public repos", d["repos"], "#4da3ff"),
            ("[@]", "Followers", d["followers"], "#2ee6a6")]
    body = ""
    for i, (g, label, val, col) in enumerate(rows):
        y = 58 + i * 23
        body += (f'<g class="r" style="animation-delay:{.3 + i * .15:.2f}s">'
                 f'<text x="20" y="{y}" font-size="14" font-weight="700" fill="{col}">{esc(g)}</text>'
                 f'<text x="58" y="{y}" font-size="14" fill="#c9d1d9">{esc(label)}</text>'
                 f'<text x="420" y="{y}" font-size="14" font-weight="700" fill="{col}" text-anchor="end">{fmt(val)}</text>'
                 f'</g>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            f'<style>{CSS}</style>{FRAME.format(w=W - 2, h=H - 2)}{title(f"{USER}_stats")}{body}</svg>')


def langs_svg(d):
    W, H = 320, 190
    langs = sorted(d["langs"].items(), key=lambda kv: -kv[1])[:6]
    total = sum(d["langs"].values()) or 1
    body = ""
    if not langs:
        body = '<text x="20" y="70" font-size="13" fill="#8b949e">no language data yet</text>'
    top = langs[0][1] if langs else 1
    for i, (name, size) in enumerate(langs):
        y = 62 + i * 21
        col = PALETTE[i % len(PALETTE)]
        label = name if len(name) <= 13 else name[:12] + "~"
        bar = max(4, 120 * size / top)
        pct = 100 * size / total
        delay = .3 + i * .15
        body += (f'<g class="r" style="animation-delay:{delay:.2f}s">'
                 f'<text x="20" y="{y}" font-size="12" fill="#c9d1d9">{esc(label)}</text></g>'
                 f'<rect x="124" y="{y - 9}" width="120" height="8" rx="4" fill="#16241b"/>'
                 f'<rect class="bar" x="124" y="{y - 9}" width="{bar:.0f}" height="8" rx="4" fill="{col}" '
                 f'style="animation-delay:{delay:.2f}s"/>'
                 f'<g class="r" style="animation-delay:{delay:.2f}s">'
                 f'<text x="300" y="{y}" font-size="12" font-weight="700" fill="{col}" text-anchor="end">{pct:.1f}%</text></g>')
    extra = ('.bar{transform-box:fill-box;transform-origin:left center;transform:scaleX(0);'
             'animation:grow 1.1s cubic-bezier(.2,.8,.2,1) forwards}'
             '@keyframes grow{to{transform:scaleX(1)}}')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            f'<style>{CSS}{extra}</style>{FRAME.format(w=W - 2, h=H - 2)}{title("top_languages")}{body}</svg>')


def main():
    data = fetch()
    os.makedirs(OUT, exist_ok=True)
    for fname, svg in (("stats-card.svg", stats_svg(data)), ("langs-card.svg", langs_svg(data))):
        with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
            f.write(svg)
        print("wrote", os.path.join(OUT, fname))


if __name__ == "__main__":
    main()
