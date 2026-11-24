"""Send training completion notifications to Telegram using local .env credentials."""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_env(path=ROOT / ".env"):
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def send(text):
    env = load_env()
    token = os.environ.get("TG_BOT_TOKEN") or env.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID") or env.get("TG_CHAT_ID")
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception:
        return False


def format_benchmark(report_path):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    lines = [
        f"Готов прогон: {Path(report_path).parent.name}",
        f"train={report.get('train_feeds', report.get('train_feed_limit'))} | val={report.get('val_feeds')}",
        "",
    ]
    for row in report.get("models", []):
        holdout = row.get("holdout", {}).get("ndcg")
        full = row.get("val", {}).get("ndcg")
        secs = row.get("seconds")
        ram = row.get("peak_ram_mib")
        meta = []
        if secs is not None:
            meta.append(f"{secs:.0f}с")
        if ram is not None:
            meta.append(f"{ram / 1024:.1f} ГБ")
        suffix = f" ({', '.join(meta)})" if meta else ""
        if holdout is not None and full is not None:
            lines.append(f"• {row['model']}: holdout={holdout:.5f} | full={full:.5f}{suffix}")
    return "\n".join(lines)


def format_ranker(output_dir, report):
    return "\n".join([
        f"Готов CatBoost: {Path(output_dir).name} ({report.get('variant')})",
        f"val_ndcg={report.get('val_ndcg', 0):.5f} | @10={report.get('val_ndcg_at_10', 0):.5f}",
        f"trees={report.get('trees')} | train={report.get('train_feeds', report.get('train_feed_limit'))} | RAM={report.get('peak_ram_mib', 0) / 1024:.1f} ГБ",
    ])


def watch_report(report_path, interval=15):
    path = Path(report_path)
    while True:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("complete"):
                    send(format_benchmark(path))
                    return
            except Exception:
                pass
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    msg = sub.add_parser("send")
    msg.add_argument("text")
    watch = sub.add_parser("watch")
    watch.add_argument("--report", type=Path, required=True)
    watch.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()
    if args.cmd == "send":
        ok = send(args.text)
        print("OK" if ok else "FAILED")
    else:
        watch_report(args.report, args.interval)


if __name__ == "__main__":
    main()
