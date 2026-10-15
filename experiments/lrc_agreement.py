"""
Inter-annotator agreement on word-level onsets in enhanced-LRC files.

Each song has >=2 annotator versions (same slug, " (N)" suffix). We parse the
per-word <mm:ss.xxx> timestamps, align the two word sequences (difflib on the
normalized word tokens), and for words both annotators labelled we compute the
absolute onset difference. This is the "noise floor": the disagreement between
two humans, which bounds how precisely any aligner can meaningfully be scored.

Output:
  - per-pair / per-song stats (n matched words, mean/median |diff|, %within tol)
  - overall macro (per-song average) and micro (pooled words) aggregation
  - CSV of per-song results
"""
import re
import os
import csv
import difflib
import statistics
from collections import defaultdict

# Single source of truth for LRC parsing lives in lrc_loader; the agreement
# analysis just adds token normalization on top of it.
from data.lrc_loader import parse_words, slug_of, is_unannotated as _loader_unannotated
from data.lrc_loader import _meta as _loader_meta

# Paths relative to the repo root (run: python -m experiments.lrc_agreement)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LRC_DIR = os.path.join(REPO_ROOT, "lrc_files")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
TOLERANCES = (0.03, 0.05, 0.1, 0.2)


def normalize(word):
    """Lowercase, strip surrounding punctuation/whitespace for matching."""
    w = word.strip().lower()
    w = re.sub(r"^[^\w']+|[^\w']+$", "", w)
    return w


def parse_lrc(path):
    """(normalized_word, onset_sec) list — wraps the shared loader parser."""
    out = []
    for w, o in parse_words(path, keep_case=True):
        nw = normalize(w)
        if nw:
            out.append((nw, o))
    return out


def is_unannotated(words):
    """words here is a [(word, onset)] list; delegate to loader logic."""
    return _loader_unannotated(words)


def file_hash(path):
    import hashlib
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def align_and_diff(wa, wb):
    """Align two (word, onset) lists by word tokens; return list of |onset diffs|
    for matched words, and counts."""
    tokens_a = [w for w, _ in wa]
    tokens_b = [w for w, _ in wb]
    sm = difflib.SequenceMatcher(a=tokens_a, b=tokens_b, autojunk=False)
    diffs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                oa = wa[i1 + k][1]
                ob = wb[j1 + k][1]
                diffs.append(abs(oa - ob))
    return diffs, len(tokens_a), len(tokens_b)


def stats_for(diffs):
    if not diffs:
        return None
    d = {
        "n_matched": len(diffs),
        "mean_ms": statistics.mean(diffs) * 1000,
        "median_ms": statistics.median(diffs) * 1000,
        "p90_ms": sorted(diffs)[int(0.9 * (len(diffs) - 1))] * 1000,
        "max_ms": max(diffs) * 1000,
    }
    for t in TOLERANCES:
        d[f"within_{int(t*1000)}ms"] = sum(1 for x in diffs if x <= t) / len(diffs)
    return d


def main():
    files = [f for f in os.listdir(LRC_DIR) if f.lower().endswith(".lrc")]
    groups = defaultdict(list)
    for f in files:
        groups[slug_of(f)].append(f)

    multi = {s: sorted(fs) for s, fs in groups.items() if len(fs) >= 2}

    per_song = []
    all_pooled = []  # pooled diffs for micro aggregation
    excluded = []    # (slug, reason)

    for slug, fs in sorted(multi.items()):
        parsed = {f: parse_lrc(os.path.join(LRC_DIR, f)) for f in fs}

        # 1) drop unannotated (all-zero timing) files
        annotated = [f for f in fs if not is_unannotated(parsed[f])]
        dropped_zero = [f for f in fs if f not in annotated]

        # 2) dedupe exact-identical files (byte-identical copies aren't a
        #    second independent annotation)
        seen_hash = {}
        distinct = []
        for f in annotated:
            h = file_hash(os.path.join(LRC_DIR, f))
            if h not in seen_hash:
                seen_hash[h] = f
                distinct.append(f)

        if len(distinct) < 2:
            reason = []
            if dropped_zero:
                reason.append(f"{len(dropped_zero)} unannotated (all-zero)")
            if len(annotated) - len(distinct) > 0:
                reason.append("remaining files byte-identical")
            if not reason:
                reason.append("only one usable version")
            excluded.append((slug, get_title(os.path.join(LRC_DIR, fs[0])),
                             "; ".join(reason)))
            continue

        title = get_title(os.path.join(LRC_DIR, distinct[0]))
        song_diffs = []
        n_words = {f: len(parsed[f]) for f in distinct}
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                d, na, nb = align_and_diff(parsed[distinct[i]], parsed[distinct[j]])
                song_diffs.extend(d)
        st = stats_for(song_diffs)
        if st is None:
            continue
        st["slug"] = slug
        st["title"] = title
        st["n_versions"] = len(distinct)
        st["word_counts"] = "/".join(str(n_words[f]) for f in distinct)
        # Classify: a pair is an INDEPENDENT re-timing only if the two annotators
        # actually re-entered word onsets. When one annotator started from the
        # other's file (or a shared auto-alignment) and only tweaked a few words,
        # the vast majority of word onsets are byte-identical. We treat >90% of
        # words having an *exactly* 0.000 s difference as "shared base" -> not an
        # independent re-annotation of word onsets.
        zero_frac = sum(1 for x in song_diffs if x == 0.0) / len(song_diffs)
        st["zero_frac"] = zero_frac
        st["shared_timing"] = zero_frac > 0.90
        st["_diffs"] = song_diffs
        note = []
        if dropped_zero:
            note.append("dropped unannotated ver")
        if st["shared_timing"]:
            note.append(f"{zero_frac*100:.0f}% word onsets identical (shared base)")
        st["note"] = "; ".join(note)
        per_song.append(st)
        all_pooled.extend(song_diffs)

    # Split into genuinely re-timed songs vs shared-timing (section-only) pairs.
    genuine = [s for s in per_song if not s["shared_timing"]]
    shared = [s for s in per_song if s["shared_timing"]]
    genuine_pooled = [x for s in genuine for x in s["_diffs"]]

    def macro(subset, key):
        return statistics.mean(s[key] for s in subset) if subset else float("nan")

    micro_all = stats_for(all_pooled)
    micro_genuine = stats_for(genuine_pooled)

    os.makedirs(OUT_DIR, exist_ok=True)
    # write CSV (all comparable songs, with classification flag)
    cols = ["slug", "title", "category", "n_versions", "word_counts", "n_matched",
            "median_ms", "mean_ms", "p90_ms", "max_ms",
            "within_30ms", "within_50ms", "within_100ms", "within_200ms",
            "zero_frac", "note"]
    csv_path = os.path.join(OUT_DIR, "inter_annotator_agreement.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        rows = ([dict(s, category="genuine_re-timing") for s in genuine] +
                [dict(s, category="shared_word_timing") for s in shared])
        for s in sorted(rows, key=lambda x: (x["category"], x["median_ms"])):
            w.writerow([s.get(c, "") for c in cols])

    # ---- report ----
    print(f"Multi-version song groups found: {len(multi)}")
    print(f"  usable (>=2 distinct, annotated): {len(per_song)}")
    print(f"    - genuine independent word re-timing: {len(genuine)}")
    print(f"    - shared word timing (only section markers differ): {len(shared)}")
    print(f"  excluded (data quality): {len(excluded)}")
    if excluded:
        for slug, title, reason in excluded:
            print(f"      · {title or slug}: {reason}")
    print()

    print("=== NOISE FLOOR (genuine re-timing songs only) ===")
    if genuine:
        print(f"Based on {len(genuine)} songs, {len(genuine_pooled)} word pairs.")
        print(f"Macro (per-song avg): median {macro(genuine,'median_ms'):.0f} ms | "
              f"mean {macro(genuine,'mean_ms'):.0f} ms | "
              f"<30ms {macro(genuine,'within_30ms')*100:.0f}% | "
              f"<50ms {macro(genuine,'within_50ms')*100:.0f}% | "
              f"<100ms {macro(genuine,'within_100ms')*100:.0f}%")
        print(f"Micro (pooled words): median {micro_genuine['median_ms']:.0f} ms | "
              f"mean {micro_genuine['mean_ms']:.0f} ms | "
              f"<30ms {micro_genuine['within_30ms']*100:.0f}% | "
              f"<50ms {micro_genuine['within_50ms']*100:.0f}% | "
              f"<100ms {micro_genuine['within_100ms']*100:.0f}%")
    print()

    print("=== Genuine re-timing songs, best agreement first (median |diff|) ===")
    hdr = f"{'title':<34}{'matched':>8}{'median':>8}{'mean':>7}{'p90':>7}{'<50ms':>7}{'<100ms':>8}"
    print(hdr)
    for s in sorted(genuine, key=lambda x: x["median_ms"]):
        t = (s['title'] or s['slug'])[:33]
        flag = " †" if "unannotated" in s.get("note", "") else ""
        print(f"{t:<34}{s['n_matched']:>8}"
              f"{s['median_ms']:>7.0f} {s['mean_ms']:>6.0f} {s['p90_ms']:>6.0f}"
              f"{s['within_50ms']*100:>6.0f}%{s['within_100ms']*100:>7.0f}%{flag}")
    print(f"  († = one unannotated placeholder version dropped before comparison)")

    print(f"\n=== Shared word-timing pairs (NOT a real noise floor — excluded above) ===")
    print("These annotator versions have byte-identical word onsets; only section")
    print("markers ([Verse]/[Chorus]/[Instrumental]) were edited. Listed for the record:")
    for s in sorted(shared, key=lambda x: (x['title'] or x['slug'])):
        print(f"  · {(s['title'] or s['slug'])[:50]} ({s['n_matched']} words)")

    print(f"\nCSV written: {csv_path}")


def get_title(path):
    ti, ar = _loader_meta(path)
    if ti and ar:
        return f"{ti} — {ar}"
    return ti or ar


if __name__ == "__main__":
    main()
