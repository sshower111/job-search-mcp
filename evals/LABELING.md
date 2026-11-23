# Labeling your own postings (about 30)

The 15 postings in `evals/postings/` are synthetic, and they were written by the same person who wrote
the baseline rules, so the baseline's score on them is optimistic. The useful number comes from your own
clipped postings. Keep that data out of git: `evals/private/` is git-ignored.

## 1. Make the private folder

```powershell
New-Item -ItemType Directory -Force evals\private
```

## 2. Choose about 30 postings

Pick them from your vault's `Jobs to tailor` folder (or older clippings) and include:

- about 15 you would apply to ("good") and about 15 you would skip ("bad")
- each track: support, integration, ai
- the hard cases: 3-5 years, "French is an asset", hybrid outside the GTA, "Senior" in the title
  but junior requirements, IT support roles with a "Support Engineer" title

Label each posting from your own judgement before you look at the tool's output.

## 3. Write `evals\private\labels.csv`

Use the same columns as `evals/labels.csv`. `file` is the file name inside your postings folder:

```csv
file,track,expected_fit,reasons
Junior AI Engineer - Some Company.md,ai,good,Entry level; GTA; Python + LLM work
Senior Support Engineer - Other Co.md,support,bad,5+ years required
```

- `track`: `support`, `integration`, `ai`, or `auto` (the best track is used)
- `expected_fit`: `good` or `bad`
- `reasons`: a few words on why. This helps when you go through the misses.

Save the file as UTF-8. Excel's "CSV UTF-8" option works.

## 4. Run it

Baseline only:

```powershell
jobagent eval --labels evals\private\labels.csv --postings-dir "C:\Users\Keith\vault\Jobs to tailor" --out evals\private\reports
```

With the LLM judge (start Ollama or LM Studio first and fill in the `[llm]` section of `jobagent.toml`):

```powershell
jobagent eval --llm --labels evals\private\labels.csv --postings-dir "C:\Users\Keith\vault\Jobs to tailor" --out evals\private\reports
```

## 5. Read the results

- Precision: of the postings the tool called good, how many you agree are good. Low precision means
  time spent on bad fits.
- Recall: of the postings you labeled good, how many the tool caught. Low recall means missed jobs.
- Go through the "Baseline misses" section. Then change `[matching]` in `jobagent.toml` (thresholds,
  excluded title phrases, locations) or the skills list in `src/jobsearch/skills.py`, and re-run.
  Change one thing at a time.
- If you publish numbers, publish only the aggregate metrics, never the postings, and say how many
  postings were labeled.
