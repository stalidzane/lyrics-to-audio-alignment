"""
Build the three results tables from results/<dataset>/ (run eval/score.py first):

    python3 writing/make_results_table.py

writes
    writing/results_summary.tex    overview: each system on mix and vocals; per dataset
                                   PCO@0.3, corrected PCO@0.1 and the offset
    writing/results_full.tex       every system and input on every dataset, mean ± std across songs
                                   (landscape page: needs the LaTeX package rotating)
    writing/results_aligned_only.tex  results_full over the songs each system aligned (failed songs left out)
    writing/results_languages.tex  PCO@0.1, raw and corrected, per language (vocals);
                                   Whisper small also with its coverage
    writing/results_stems.tex      the 5 songs with studio stems: mix / Demucs vocals / studio stem
    writing/results_tables.pdf     the four tables compiled together, for checking (needs pdflatex)

All values are macro averages: computed per song, then averaged over songs. Songs a system
failed on count as 0, except in results_aligned_only.tex.
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = [("jamendo", "JamendoLyrics MultiLang", 79),
            ("jamendopp", "JamendoLyrics++", 80),
            ("company", "Rhythm-game songs", 80)]
SYSTEMS = [("whisper-small-mix", "Whisper small, mix"),
           ("whisper-small-vocals", "Whisper small, vocals"),
           ("whisper-large-v3-mix", "Whisper large-v3, mix"),
           ("whisper-large-v3-vocals", "Whisper large-v3, vocals"),
           ("mms-mix", "MMS, mix"),
           ("mms-vocals", "MMS, vocals"),
           ("hb-mix", "H\\&B, mix"),
           ("hb-vocals", "H\\&B, vocals"),
           ("mfa-mix", "MFA, mix"),
           ("mfa-vocals", "MFA, vocals")]
# dataset names on two lines, so a long name does not widen its columns
TWO_LINE = {"jamendo": r"JamendoLyrics\\MultiLang", "jamendopp": r"JamendoLyrics++\\(English)",
            "company": r"Rhythm-game\\songs"}
# summary table: one block per system, rows mix and vocals
SUMMARY_SYSTEMS = [("whisper-small", "Whisper small"), ("whisper-large-v3", "Whisper large-v3"),
                   ("mms", "MMS"), ("hb", "Huang \\& Benetos"), ("mfa", "MFA")]
# language table: vocals input (every forced aligner does best on vocals)
LANG_SYSTEMS = [("whisper-small-vocals", "Whisper small"), ("mms-vocals", "MMS"),
                ("hb-vocals", "H\\&B"), ("mfa-vocals", "MFA")]
LANG_NAMES = {"en": "English", "lv": "Latvian", "de": "German", "es": "Spanish", "fr": "French",
              "it": "Italian", "pt": "Portuguese", "sv": "Swedish", "ro": "Romanian"}
TOLS = ("0.05", "0.1", "0.3")
MIN_LANG = 3  # languages with fewer songs are grouped as "Other"
# one language order for both datasets (the rhythm-game songs' order, by song count), so rows can be compared
LANG_ORDER = ["en", "lv", "it", "de", "es", "fr", "pt", "ro", "sv"]


def lang_rank(lang):
    return LANG_ORDER.index(lang) if lang in LANG_ORDER else len(LANG_ORDER)
# two-line column heads, so the metric is named in the table itself
HEAD_PCO03 = r"\shortstack{PCO\\0.3\,s}"
HEAD_CORR01 = r"\shortstack{Corr.\\PCO\\0.1\,s}"
HEAD_PCO01 = r"\shortstack{PCO\\0.1\,s}"


def pct(x):
    return f"{100 * float(x):.1f}"


def bold_if(text, is_best):
    return rf"\textbf{{{text}}}" if is_best else text


def best_set(texts, lower=False):
    """the best of a list of formatted numbers (ties all count); compared as printed, so a tie
    in the table is a tie in the bolding"""
    vals = [float(t) for t in texts if t not in ("", "--")]
    if not vals:
        return set()
    b = min(vals) if lower else max(vals)
    return {t for t in texts if t not in ("", "--") and float(t) == b}


def pm(mean, std, is_best=False):
    """mean ± std in percent, std in small type; the mean in bold if it is the best in its column."""
    return bold_if(pct(mean), is_best) + r"{\tiny$\,\pm$" + pct(std) + "}"


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary(ds, name="summary.csv"):
    """rows of results/<ds>/<name> for all languages together, keyed by system"""
    return {r["system"]: r for r in read(os.path.join(ROOT, "results", ds, name))
            if r["language"] == "all"}


def summary_table():
    """one row per system and input; per dataset: PCO@0.3, PCO@0.1, corrected PCO@0.1, offset"""
    sums = {ds: summary(ds) for ds, _, _ in DATASETS}
    metrics = ("pco_0.3", "pco_0.1", "corr_0.1")
    k = len(metrics) + 1  # columns per dataset (+ offset)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.4pt}",
             r"\begin{tabular}{ll" + "r" * (k * len(DATASETS)) + "}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{\shortstack{{{TWO_LINE[ds]}}}}}" for ds, _, _ in DATASETS)
             + r"\\",
             "".join(rf"\cmidrule(lr){{{3 + k * i}-{2 + k * (i + 1)}}}" for i in range(len(DATASETS))),
             "System & Input & " + " & ".join(" & ".join((HEAD_PCO03, HEAD_PCO01, HEAD_CORR01)) + r" & \shortstack{Offset\\(s)}"
                                          for _ in DATASETS) + r"\\"]
    # best value per dataset and metric, separately among the mix rows and among the vocals rows (bold)
    best = {(ds, m, inp): best_set([pct(sums[ds][f"{b}-{inp}"][m]) for b, _ in SUMMARY_SYSTEMS
                                    if f"{b}-{inp}" in sums[ds]])
            for ds, _, _ in DATASETS for m in metrics for inp in ("mix", "vocals")}
    for base, name in SUMMARY_SYSTEMS:
        lines.append(r"\midrule")
        for j, inp in enumerate(("mix", "vocals")):
            cells = []
            for ds, _, _ in DATASETS:
                r = sums[ds].get(f"{base}-{inp}")
                cells += ["--"] * k if r is None else \
                    [bold_if(pct(r[m]), pct(r[m]) in best[(ds, m, inp)]) for m in metrics] \
                    + [f"${float(r['offset']):+.2f}$"]
            lines.append((name if j == 0 else "") + f" & {inp} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Overview of the results on the full mix and on the Demucs vocals. "
              r"PCO 0.3\,s and 0.1\,s: share of reference words (\%) with an onset error within 0.3 or 0.1\,s; "
              r"Corr.\ PCO 0.1\,s: share within 0.1\,s after the constant offset correction; "
              r"Offset: median signed error in seconds (negative = early). "
              r"Macro averages over songs; songs a system failed on count as 0 "
              r"(their number is given in Table~\ref{tab:results-full}). "
              r"MFA has no Latvian model and is evaluated on the 53 non-Latvian rhythm-game songs. "
              r"Bold: best value per dataset and metric, separately for the mix and for the vocals.}",
              r"\label{tab:results-summary}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def full_table(aligned_only=False):
    """aligned_only=True: same table over the songs each system aligned (summary_ok.csv)"""
    ncol = 6 + 2 * len(TOLS)
    # mean ± std does not fit a portrait page: sidewaystable (package rotating) puts it on its own landscape page
    lines = [r"\begin{sidewaystable}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{lrrrr" + "r" * (2 * len(TOLS)) + "}", r"\toprule",
             r" & & & & & \multicolumn{" + str(len(TOLS)) + r"}{c}{PCO (\%)} & \multicolumn{" + str(len(TOLS))
             + r"}{c}{Corrected PCO (\%)}\\",
             rf"\cmidrule(lr){{6-{5 + len(TOLS)}}}\cmidrule(lr){{{6 + len(TOLS)}-{5 + 2 * len(TOLS)}}}",
             "System & " + ("Songs" if aligned_only else "Fail") + r" & Cov.\ (\%) & Offset (s) & AAE (s) & "
             + " & ".join(f"{t}\\,s" for t in TOLS) + " & " + " & ".join(f"{t}\\,s" for t in TOLS) + r"\\"]
    for ds, dname, n in DATASETS:
        rows = summary(ds, "summary_ok.csv" if aligned_only else "summary.csv")
        lines += [r"\midrule", rf"\multicolumn{{{ncol - 1}}}{{l}}{{\textit{{{dname} ({n} songs)}}}}\\"]
        # best per column within this dataset, separately for mix and vocals (bold): highest PCO, lowest AAE
        best, best_aae = {}, {}
        for inp in ("mix", "vocals"):
            present = [rows[k] for k, _ in SYSTEMS if k in rows and k.endswith(f"-{inp}")]
            for m in [f"pco_{t}" for t in TOLS] + [f"corr_{t}" for t in TOLS]:
                best[(m, inp)] = best_set([pct(r[m]) for r in present])
            best_aae[inp] = best_set([f"{float(r['aae']):.2f}" for r in present], lower=True)
        for key, name in SYSTEMS:
            r = rows.get(key)
            if r is None:
                continue
            inp = key.rsplit("-", 1)[1]
            fail = r["n_songs"] if aligned_only else (r["n_failed"] if int(r["n_failed"]) else "--")
            aae = f"{float(r['aae']):.2f}"
            cells = [fail, pct(r["coverage"]), f"${float(r['offset']):+.2f}$", bold_if(aae, aae in best_aae[inp])]
            cells += [pm(r[f"pco_{t}"], r[f"pco_std_{t}"], pct(r[f"pco_{t}"]) in best[(f"pco_{t}", inp)]) for t in TOLS]
            cells += [pm(r[f"corr_{t}"], r[f"corr_std_{t}"], pct(r[f"corr_{t}"]) in best[(f"corr_{t}", inp)])
                      for t in TOLS]
            lines.append(f"{name} & " + " & ".join(cells) + r"\\")
    if aligned_only:
        lines += [r"\bottomrule", r"\end{tabular}",
                  r"\caption{The results of Table~\ref{tab:results-full} computed only over the songs each system "
                  r"aligned (Songs = number of songs aligned); mean $\pm$ standard deviation across those songs. "
                  r"The offset is unchanged, since failed songs contribute no words to it. Rows are identical to Table~\ref{tab:results-full} "
                  r"for systems without failed songs. Bold: best value per dataset and column, separately for the mix and for the vocals.}",
                  r"\label{tab:results-aligned}", r"\end{sidewaystable}"]
        return "\n".join(lines) + "\n"
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Results of all systems on the full mix and on the Demucs vocals: mean $\pm$ standard deviation "
              r"across songs. Macro averages: each metric is computed per song, then averaged over songs. "
              r"Fail: songs the system could not align at all; they count as 0. "
              r"Cov.: share of reference words that receive a time (Whisper: matched to a transcribed word). "
              r"Offset: median signed error of the words within 1\,s of the reference (negative = early). "
              r"AAE: average absolute error as in MIREX, the per-song mean absolute error averaged over songs; "
              r"words placed far from their reference make it large. "
              r"PCO: share of reference words with an onset error within the tolerance; unmatched words count as errors. "
              r"Corrected PCO: the same after subtracting the offset. H\&B = Huang \& Benetos. "
              r"Bold: best value per dataset and column, separately for the mix and for the vocals (highest PCO, lowest AAE). "
              r"MFA has no Latvian model, so its rhythm-game rows cover the 53 non-Latvian songs.}",
              r"\label{tab:results-full}", r"\end{sidewaystable}"]
    return "\n".join(lines) + "\n"


def language_table():
    # Whisper small also gets a coverage column: its weakness on Latvian is recognition (few words
    # matched), which PCO alone does not show; the forced aligners always cover 100 % of the words
    cov_sys = {"whisper-small-vocals"}
    width = {k: 3 if k in cov_sys else 2 for k, _ in LANG_SYSTEMS}
    ncol = 2 + sum(width.values())
    starts, col = [], 3
    for k, _ in LANG_SYSTEMS:
        starts.append(col)
        col += width[k]
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lr" + "".join("r" * width[k] for k, _ in LANG_SYSTEMS) + "}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{{width[k]}}}{{c}}{{{n}}}" for k, n in LANG_SYSTEMS) + r"\\",
             "".join(rf"\cmidrule(lr){{{s0}-{s0 + width[k] - 1}}}" for s0, (k, _) in zip(starts, LANG_SYSTEMS)),
             "Language & Songs & " + " & ".join(
                 (r"\shortstack{Cov.\\(\%)} & " if k in cov_sys else "") + HEAD_PCO01 + " & " + HEAD_CORR01
                 for k, _ in LANG_SYSTEMS) + r"\\"]
    others = {}
    for ds, dname in (("jamendo", "JamendoLyrics MultiLang"), ("company", "Rhythm-game songs")):
        songs = {k: read(os.path.join(ROOT, "results", ds, f"{k}.csv"))
                 for k, _ in LANG_SYSTEMS if os.path.exists(os.path.join(ROOT, "results", ds, f"{k}.csv"))}
        count = {}
        for r in read(os.path.join(ROOT, "data", ds, "index.csv")):
            count[r["language"]] = count.get(r["language"], 0) + 1
        main = sorted([l for l, c in count.items() if c >= MIN_LANG], key=lang_rank)
        other = sorted(l for l, c in count.items() if c < MIN_LANG)
        groups = [(LANG_NAMES.get(l, l), [l]) for l in main]
        if other:
            groups.append(("Other", other))
            others[dname] = ", ".join(LANG_NAMES.get(l, l) for l in other)
        lines += [r"\midrule", rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{dname}}}}}\\"]
        for name, langs in groups:
            cov_cells, cells = {}, []
            for k, _ in LANG_SYSTEMS:
                rows = [r for r in songs.get(k, []) if r["language"] in langs]
                if not rows:
                    cov_cells[k] = "--"
                    cells += ["--", "--"]
                    continue
                cov_cells[k] = pct(sum(float(r["coverage"]) for r in rows) / len(rows))
                cells.append(pct(sum(float(r["pco_0.1"]) for r in rows) / len(rows)))
                cells.append(pct(sum(float(r["corr_0.1"]) for r in rows) / len(rows)))
            # bold the best system per language, separately for each metric (coverage is not bolded)
            for start in (0, 1):
                b = best_set(cells[start::2])
                for i in range(start, len(cells), 2):
                    cells[i] = bold_if(cells[i], cells[i] in b)
            out = []
            for j, (k, _) in enumerate(LANG_SYSTEMS):
                out += ([cov_cells[k]] if k in cov_sys else []) + cells[2 * j:2 * j + 2]
            lines.append(f"{name} & {sum(count[l] for l in langs)} & " + " & ".join(out) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Results per language at the game's 0.1\,s window, Demucs vocals as input. Cov.: share of "
              r"reference words (\%) matched to a word transcribed by Whisper (the forced aligners give every word a time). "
              r"PCO 0.1\,s: share of reference words (\%) within 0.1\,s; Corr.\ PCO 0.1\,s: the same after the constant offset correction. Songs a system failed on count as 0; "
              r"-- = no model for this language (MFA, Latvian). Best system per language and metric in bold. Other: languages with fewer than three songs ("
              + "; ".join(f"{k}: {v}" for k, v in others.items()) + ").}",
              r"\label{tab:results-languages}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def stems_table():
    """5 songs with studio vocal stems: each system on the mix, on Demucs vocals and on the stem"""
    if not os.path.exists(os.path.join(ROOT, "results", "stems_iso", "summary.csv")):
        return None
    mix_dem = summary("stems")                 # mix and Demucs vocals
    iso = summary("stems_iso")                 # "vocals" = the aligned studio stem
    inputs = [("Full mix", mix_dem, "mix"), ("Demucs vocals", mix_dem, "vocals"), ("Studio stem", iso, "vocals")]
    metrics = ("pco_0.3", "pco_0.1", "corr_0.1")
    k = len(metrics)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{l" + "r" * (k * len(inputs)) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{n}}}" for n, _, _ in inputs) + r"\\",
             "".join(rf"\cmidrule(lr){{{2 + k * i}-{1 + k * (i + 1)}}}" for i in range(len(inputs))),
             "System & " + " & ".join(" & ".join((HEAD_PCO03, HEAD_PCO01, HEAD_CORR01)) for _ in inputs) + r"\\",
             r"\midrule"]
    fails = []
    for base, name in SUMMARY_SYSTEMS:
        vals = {}
        for label, rows, inp in inputs:
            r = rows.get(f"{base}-{inp}")
            vals[label] = r
            if r is not None and int(r["n_failed"]):
                fails.append(f"{name} on the {label.lower()} ({r['n_failed']} of {r['n_songs']} songs)")
        cells = []
        # best input per system and metric in bold
        best = {m: best_set([pct(v[m]) for v in vals.values() if v is not None]) for m in metrics}
        for label, _, _ in inputs:
            r = vals[label]
            cells += ["--"] * k if r is None else [bold_if(pct(r[m]), pct(r[m]) in best[m]) for m in metrics]
        lines.append(f"{name} & " + " & ".join(cells) + r"\\")
    fail_text = ("Failed songs (counted as 0): " + "; ".join(fails) + ". ") if fails else ""
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Five rhythm-game songs for which studio vocal stems were available: each system on the full mix, "
              r"on the vocals separated by Demucs and on the studio vocal stem (aligned to the mix by cross-correlation). "
              r"PCO 0.3\,s and 0.1\,s: share of reference words (\%) within 0.3 or 0.1\,s; Corr.\ PCO 0.1\,s: within 0.1\,s "
              r"after the constant offset correction. " + fail_text
              + r"Bold: best input per system and metric.}",
              r"\label{tab:results-stems}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    w = os.path.join(ROOT, "writing")
    files = {"results_summary.tex": summary_table(),
             "results_full.tex": full_table(),
             "results_aligned_only.tex": full_table(aligned_only=True),
             "results_languages.tex": language_table()}
    stems = stems_table()
    if stems:
        files["results_stems.tex"] = stems
    for name, text in files.items():
        with open(os.path.join(w, name), "w", encoding="utf-8") as f:
            f.write("% Generated by writing/make_results_table.py -- do not edit by hand.\n"
                    "% Preamble needs: \\usepackage{booktabs,rotating}\n" + text)
        print("wrote", os.path.join(w, name))
    # compile a preview in a temporary folder, so no .aux/.log files are left in writing/
    with tempfile.TemporaryDirectory() as tmp:
        for name in files:
            shutil.copy(os.path.join(w, name), tmp)
        with open(os.path.join(tmp, "results_tables.tex"), "w", encoding="utf-8") as f:
            f.write("\\documentclass[a4paper,11pt]{article}\n\\usepackage[T1]{fontenc}\n"
                    "\\usepackage[textwidth=14cm,textheight=23cm]{geometry}\n\\usepackage{booktabs,rotating}\n"
                    "\\pagestyle{empty}\n\\begin{document}\n"
                    + "\n\\clearpage\n".join(f"\\input{{{n}}}" for n in files) + "\n\\end{document}\n")
        try:
            for _ in range(2):
                subprocess.run(["pdflatex", "-interaction=nonstopmode", "results_tables.tex"], cwd=tmp,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            shutil.copy(os.path.join(tmp, "results_tables.pdf"), w)
            with open(os.path.join(tmp, "results_tables.log"), encoding="utf-8", errors="replace") as f:
                if "Overfull" in f.read():
                    print("WARNING: a table is wider than the text (Overfull \\hbox)")
            print("wrote", os.path.join(w, "results_tables.pdf"))
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("(pdflatex not available or failed: the .tex tables are written, the preview PDF is not)")


if __name__ == "__main__":
    main()
