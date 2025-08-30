
print(">>>  Static Scan <<<")  


import os, re, json, argparse
from datetime import datetime
from html import escape

#  Defaults 
DEFAULT_EXTS = (".py", ".php", ".js", ".java")
SKIP_DIRS = {"venv", ".git", "__pycache__", "node_modules", "dist", "build", ".idea", ".vscode"}
MAX_FILE_MB = 2
MAX_LINES_PER_FILE = 5000

# AI re-ranking settings (applied only to suspicious hits)
AI_THRESHOLD_DEFAULT = 0.60      # mark as likely-vuln if score >= threshold
AI_PER_FILE_DEFAULT = 10         # max suspicious lines per file to score
AI_TOTAL_DEFAULT = 300           # global cap to avoid long runs

# Rules & Keywords
# name -> (regex, severity, applies_to_exts or None for all)
RULES = {
    "Hardcoded Password": (r"(password\s*=\s*['\"].+['\"])", "High", None),
    "Eval Usage":         (r"\beval\s*\(", "High", None),
    "Exec Usage":         (r"\bexec\s*\(", "High", (".py", ".js", ".php", ".java")),
    "OS Command":         (r"(os\.system\s*\(|subprocess\.(Popen|call|run)\s*\()", "High", (".py",)),
    "Weak Crypto (md5/sha1)": (r"\b(md5|sha1)\s*\(", "Medium", None),
    "Insecure Deserialization": (r"(pickle\.load\s*\(|yaml\.load\s*\()", "High", (".py",)),
    # SQLi concatenation
    "SQLi (concat '+')":  (r"SELECT\s+.+\s+FROM\s+.+\s+WHERE\s+.+\+\s*\w+", "High", (".py", ".js", ".java")),
    "SQLi (concat '.')":  (r"SELECT\s+.+\s+FROM\s+.+\s+WHERE\s+.+['\"]\s*\.\s*\$?\w+", "High", (".php",)),
    # Browser sinks (rough)
    "XSS Sink":           (r"(document\.write\s*\(|innerHTML\s*=)", "Medium", (".js",)),
    "Open Redirect":      (r"(window\.location\s*=|document\.location\s*=)", "Medium", (".js", ".php")),
}

# keyword(lower) -> severity
KEYWORDS = {
    "apikey": "High",
    "api_key": "High",
    "secret": "High",
    "token": "Medium",
    "credentials": "Medium",
    "passwd": "Medium",
    "private_key": "High",
    "base64.b64decode": "Low",
    "input(": "Low",
}

#  Helpers 
def is_skipped_dir(path: str) -> bool:
    parts = set(os.path.normpath(path).split(os.sep))
    return any(skip in parts for skip in SKIP_DIRS)

def file_too_big(path: str, max_mb: int) -> bool:
    try:
        return (os.path.getsize(path) / (1024 * 1024)) > max_mb
    except Exception:
        return True

def iter_files(root: str, include_exts):
    for r, _, files in os.walk(root):
        if is_skipped_dir(r):
            continue
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if include_exts and ext not in include_exts:
                continue
            fpath = os.path.join(r, fname)
            yield fpath, ext

# Core scan (Rules + Keywords) 
def scan_file_rules_keywords(path: str, ext: str, max_lines: int):
    rule_hits, kw_hits = [], []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                if i > max_lines: break
                s = line.strip()
                # rules
                for name, (pattern, sev, applies) in RULES.items():
                    if applies and ext not in applies:
                        continue
                    if re.search(pattern, s, re.IGNORECASE):
                        rule_hits.append({"file": path, "line": i, "issue": name, "severity": sev, "code": s})
                # keywords
                low = s.lower()
                for kw, sev in KEYWORDS.items():
                    if kw in low:
                        kw_hits.append({"file": path, "line": i, "keyword": kw, "severity": sev, "code": s})
    except Exception:
        pass
    return rule_hits, kw_hits

def scan_tree_rules_keywords(target: str, include_exts, max_lines, max_file_mb):
    rules, kws = [], []
    if os.path.isdir(target):
        for fpath, ext in iter_files(target, include_exts):
            if file_too_big(fpath, max_file_mb): 
                continue
            r, k = scan_file_rules_keywords(fpath, ext, max_lines)
            rules.extend(r); kws.extend(k)
    else:
        ext = os.path.splitext(target)[1].lower()
        if include_exts and ext in include_exts and not file_too_big(target, max_file_mb):
            r, k = scan_file_rules_keywords(target, ext, max_lines)
            rules.extend(r); kws.extend(k)
    return rules, kws

#  AI Re-ranking (only suspicious hits) 
def ai_score_hits(rules, kws, threshold, per_file, total_cap):
    """
    Score already-flagged lines (rules+keywords) with CodeBERT.
    Returns: dict[(file,line)] -> score (0..1)
    """
    # Unique suspicious set
    suspicious = {}
    for h in rules:
        suspicious.setdefault(h["file"], set()).add(h["line"])
    for h in kws:
        suspicious.setdefault(h["file"], set()).add(h["line"])

    # Apply per-file and global caps
    targets = []
    running_total = 0
    for f, lines in suspicious.items():
        sel = sorted(list(lines))[:max(1, per_file)]
        for ln in sel:
            targets.append((f, ln))
            running_total += 1
            if running_total >= total_cap:
                break
        if running_total >= total_cap:
            break
    if not targets:
        return {}

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    except Exception:
        print("[!] AI dependencies missing/failed. Skipping AI re-ranking.")
        return {}

    scores = {}
    for f, ln in targets:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                for i, line in enumerate(fp, 1):
                    if i == ln:
                        code = line.strip()
                        if not code:
                            break
                        inp = tok(code, return_tensors="pt", truncation=True, max_length=128).to(device)
                        with torch.no_grad():
                            logits = model(**inp).logits
                            prob = float(logits.softmax(dim=-1)[0,1].item())
                        scores[(f, ln)] = prob
                        break
        except Exception:
            continue
    return scores

def apply_ai_to_hits(rules, kws, scores, threshold):
    """Annotate hits with ai_score; escalate some severities."""
    escalations = 0
    for h in rules:
        key = (h["file"], h["line"])
        if key in scores:
            h["ai_score"] = round(scores[key], 2)
            if h["severity"] == "Medium" and scores[key] >= threshold:
                h["severity"] = "High"
                h["ai_escalated"] = True
                escalations += 1
    for h in kws:
        key = (h["file"], h["line"])
        if key in scores:
            h["ai_score"] = round(scores[key], 2)
            if h["severity"] == "Medium" and scores[key] >= threshold:
                h["severity"] = "High"
                h["ai_escalated"] = True
                escalations += 1
    return escalations

#  Reports 
def write_html(path, rules, kws, ai_scored, escalations):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def table(title, rows, cols):
        html = [f"<h2>{title}</h2>"]
        if not rows:
            html.append("<p><em>No findings.</em></p>")
            return "\n".join(html)
        html.append("<table border='1' cellspacing='0' cellpadding='6' style='border-collapse:collapse;width:100%'>")
        html.append("<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
        for r in rows:
            html.append("<tr>" + "".join(f"<td>{escape(str(r.get(c,'')))}</td>" for c in cols) + "</tr>")
        html.append("</table>")
        return "\n".join(html)

    # Built normalized rows with capitalized keys + optional AI column
    def has_ai(rows): 
        return any("ai_score" in r for r in rows)

    has_ai_rules = has_ai(rules)
    has_ai_kws   = has_ai(kws)

    rule_cols = ["File","Line","Issue","Severity","Code"] + (["AI"] if has_ai_rules else [])
    kw_cols   = ["File","Line","Keyword","Severity","Code"] + (["AI"] if has_ai_kws else [])

    rules_tbl = []
    for r in rules:
        row = {
            "File": r.get("file",""),
            "Line": r.get("line",""),
            "Issue": r.get("issue",""),
            "Severity": r.get("severity",""),
            "Code": r.get("code",""),
        }
        if has_ai_rules:
            row["AI"] = r.get("ai_score","")
        rules_tbl.append(row)

    kws_tbl = []
    for r in kws:
        row = {
            "File": r.get("file",""),
            "Line": r.get("line",""),
            "Keyword": r.get("keyword",""),
            "Severity": r.get("severity",""),
            "Code": r.get("code",""),
        }
        if has_ai_kws:
            row["AI"] = r.get("ai_score","")
        kws_tbl.append(row)

    out = []
    out.append("<html><head><meta charset='utf-8'><title>SAST Report</title>")
    out.append("<style>body{font-family:Segoe UI,Arial,sans-serif} th{background:#f4f6f8;text-align:left}</style>")
    out.append("</head><body>")
    out.append(f"<h1>Static Application Security Testing Report</h1><p><b>Generated:</b> {now}</p>")
    out.append(f"<p><b>AI re-ranking:</b> {'ON' if ai_scored else 'OFF'}; Escalations: {escalations}</p>")
    out.append(table("Rule-based Findings", rules_tbl, rule_cols))
    out.append(table("Keyword Findings",    kws_tbl,   kw_cols))
    out.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path

def write_json(path, rules, kws, ai_scored, escalations):
    payload = {
        "generated_at": datetime.now().isoformat(),
        "ai_rerank": bool(ai_scored),
        "ai_escalations": escalations,
        "summary": {
            "rules": len(rules),
            "keywords": len(kws),
            "high_total": sum(1 for r in rules if r["severity"]=="High") +
                          sum(1 for k in kws   if k["severity"]=="High")
        },
        "findings": {
            "rules": rules,
            "keywords": kws
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path

def write_pdf(path, rules, kws, ai_scored, escalations):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception:
        print("[!] reportlab not installed. `pip install reportlab` to enable PDF output.")
        return None

    styles = getSampleStyleSheet()
    story = [Paragraph("<b>Static Application Security Testing Report</b>", styles["Title"]),
             Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["Normal"]),
             Paragraph(f"AI re-ranking: {'ON' if ai_scored else 'OFF'}; Escalations: {escalations}", styles["Normal"]),
             Spacer(1, 12)]

    def add_table(title, rows, cols):
        story.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
        if not rows:
            story.append(Paragraph("<i>No findings.</i>", styles["Normal"]))
            story.append(Spacer(1, 8))
            return
        data = [cols] + [[str(r.get(c,"")) for c in cols] for r in rows]
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(t); story.append(Spacer(1, 12))

    # Build normalized rows for PDF too
    def has_ai(rows): 
        return any("ai_score" in r for r in rows)

    has_ai_rules = has_ai(rules)
    has_ai_kws   = has_ai(kws)

    rule_cols = ["File","Line","Issue","Severity","Code"] + (["AI"] if has_ai_rules else [])
    kw_cols   = ["File","Line","Keyword","Severity","Code"] + (["AI"] if has_ai_kws else [])

    rules_tbl = []
    for r in rules:
        row = {
            "File": r.get("file",""),
            "Line": r.get("line",""),
            "Issue": r.get("issue",""),
            "Severity": r.get("severity",""),
            "Code": r.get("code",""),
        }
        if has_ai_rules:
            row["AI"] = r.get("ai_score","")
        rules_tbl.append(row)

    kws_tbl = []
    for r in kws:
        row = {
            "File": r.get("file",""),
            "Line": r.get("line",""),
            "Keyword": r.get("keyword",""),
            "Severity": r.get("severity",""),
            "Code": r.get("code",""),
        }
        if has_ai_kws:
            row["AI"] = r.get("ai_score","")
        kws_tbl.append(row)

    add_table("Rule-based Findings", rules_tbl, rule_cols)
    add_table("Keyword Findings",    kws_tbl,   kw_cols)

    doc = SimpleDocTemplate(path, pagesize=A4)
    doc.build(story)
    return path

# CLI 
def main():
    ap = argparse.ArgumentParser(description="Thesis SAST Scanner (AI re-ranking)")
    ap.add_argument("target", help="File or directory to scan")
    ap.add_argument("--ext", nargs="*", default=list(DEFAULT_EXTS), help="Extensions to include (e.g. --ext .php .py)")
    ap.add_argument("--max-file-mb", type=int, default=MAX_FILE_MB, help="Skip files larger than this (MB)")
    ap.add_argument("--max-lines", type=int, default=MAX_LINES_PER_FILE, help="Max lines per file to inspect")
    # outputs
    ap.add_argument("--html", help="Write HTML report to this path")
    ap.add_argument("--json", help="Write JSON report to this path")
    ap.add_argument("--pdf",  help="Write PDF report to this path (requires reportlab)")
    # AI re-ranking
    ap.add_argument("--ai", action="store_true", help="Enable AI re-ranking (CodeBERT) on suspicious hits only")
    ap.add_argument("--ai-threshold", type=float, default=AI_THRESHOLD_DEFAULT, help="AI threshold (default 0.60)")
    ap.add_argument("--ai-per-file", type=int, default=AI_PER_FILE_DEFAULT, help="Max suspicious lines per file to score")
    ap.add_argument("--ai-total", type=int, default=AI_TOTAL_DEFAULT, help="Max suspicious lines overall to score")
    args = ap.parse_args()

    include_exts = tuple(e.lower() if e.startswith(".") else "."+e.lower() for e in args.ext)
    max_lines = max(1, args.max_lines)
    max_file_mb = max(1, args.max_file_mb)

    # Scan rules + keywords 
    rules, kws = scan_tree_rules_keywords(args.target, include_exts, max_lines, max_file_mb)

    # AI re-ranking of flagged lines only
    ai_scored = False
    escalations = 0
    if args.ai and (rules or kws):
        print(f"[•] AI re-ranking enabled. Threshold={args.ai_threshold}  per_file={args.ai_per_file}  total={args.ai_total}")
        scores = ai_score_hits(rules, kws, args.ai_threshold, args.ai_per_file, args.ai_total)
        if scores:
            escalations = apply_ai_to_hits(rules, kws, scores, args.ai_threshold)
            ai_scored = True
        else:
            print("[i] No AI scores produced (deps missing or no suspicious lines).")

    # Outputs
    wrote_any = False
    if args.html:
        write_html(args.html if args.html.endswith(".html") else args.html + ".html", rules, kws, ai_scored, escalations)
        print(f"[✓] HTML report -> {args.html}")
        wrote_any = True
    if args.json:
        write_json(args.json if args.json.endswith(".json") else args.json + ".json", rules, kws, ai_scored, escalations)
        print(f"[✓] JSON report -> {args.json}")
        wrote_any = True
    if args.pdf:
        out_pdf = args.pdf if args.pdf.endswith(".pdf") else args.pdf + ".pdf"
        if write_pdf(out_pdf, rules, kws, ai_scored, escalations):
            print(f"[✓] PDF report  -> {out_pdf}")
            wrote_any = True
    if not wrote_any:
        write_html("sast_report.html", rules, kws, ai_scored, escalations)
        print("[i] No output flags given; wrote: sast_report.html")

    # Console summary + CI exit code
    high = sum(1 for r in rules if r["severity"] == "High") + sum(1 for k in kws if k["severity"] == "High")
    print(f"[summary] rules={len(rules)} keywords={len(kws)} escalations={escalations} high={high}")
    if high > 0:
        exit(2)
    else:
        exit(0)

if __name__ == "__main__":
    main()
