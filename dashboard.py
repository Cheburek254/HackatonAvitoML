"""Local desktop launcher for the CUDA ranking benchmark."""

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from uuid import uuid4

from accelerator import nvidia_gpus
from experiments import GPU_ABLATION

ROOT = Path(__file__).resolve().parent
BG, PANEL, BORDER = "#111820", "#1a2530", "#324251"
TEXT, MUTED, ACCENT = "#f2f5f8", "#aebdca", "#6ee7bc"
TUNING = re.compile(r"^(\S+) (?:trees|epoch)=(\d+) tuning_ndcg=([\d.]+)")
BATCH = re.compile(r"epoch=(\d+) batch=(\d+)/(\d+)")
TREE = re.compile(r"^(\d+):\s")


def command_for(data, output, gpu, models, iterations, epochs, train_feeds, val_feeds):
    return [sys.executable, "-u", "-X", "utf8", str(ROOT / "experiments.py"), "benchmark",
            "--data", str(data), "--output", str(output), "--gpu", gpu, "--models", *models,
            "--iterations", str(iterations), "--epochs", str(epochs), "--train-feeds", str(train_feeds),
            "--val-feeds", str(val_feeds), "--by-category"]


class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title("Avito ML · Эксперименты")
        root.configure(bg=BG)
        root.geometry("1180x800")
        root.minsize(970, 720)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.events = queue.Queue()
        self.process = None
        self.running = False
        self.cancelled = False
        self.report_path = None
        self.report_stamp = None
        self.watch_config = None
        self.watch_stamp = None
        self.watch_log = None
        self.watch_status = None
        self.watch_label = ""
        self.watch_offset = 0
        self.watch_active = False
        self.result_rows = {}
        self.models, self.completed, self.curve = [], set(), []
        self.gpus = {}
        self.data = tk.StringVar(value=str(ROOT / "runs" / "synthetic-v2" / "competition"))
        self.gpu = tk.StringVar()
        self.iterations, self.epochs = tk.StringVar(value="100"), tk.StringVar(value="5")
        self.train_feeds, self.val_feeds = tk.StringVar(value="0"), tk.StringVar(value="0")
        self.selection = {name: tk.BooleanVar(value=True) for name in GPU_ABLATION}
        self.status = tk.StringVar(value="Готово к запуску")
        self.sample = tk.StringVar()
        self.stage = tk.StringVar(value="")
        self.progress = tk.DoubleVar()
        self.model_progress = tk.DoubleVar()
        self.configure_styles()
        self.layout()
        self.refresh_gpus()
        root.after(100, self.poll)

    def configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TButton", background=PANEL, bordercolor=BORDER, padding=(10, 5))
        style.map("TButton", background=[("active", BORDER)], foreground=[("disabled", MUTED)])
        style.configure("Launch.TButton", background=ACCENT, foreground=BG, bordercolor=ACCENT, padding=(10, 9))
        style.map("Launch.TButton", background=[("disabled", BORDER), ("active", "#a3f3d7")])
        style.configure("TEntry", fieldbackground=PANEL, foreground=TEXT, bordercolor=BORDER, insertcolor=TEXT)
        style.configure("TCombobox", fieldbackground=PANEL, foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)])
        self.root.option_add("*TCombobox*Listbox.background", PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        style.configure("TCheckbutton", indicatorbackground=PANEL, indicatorforeground=BG)
        style.map(
            "TCheckbutton",
            background=[("active", BG)],
            indicatorbackground=[("selected", ACCENT), ("!selected", PANEL)],
            indicatorforeground=[("selected", BG), ("!selected", PANEL)],
            foreground=[("selected", TEXT), ("!selected", "#637587")],
        )
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                        bordercolor=BORDER, rowheight=25)
        style.configure("Treeview.Heading", background=BG, foreground=MUTED)
        style.map("Treeview", background=[("selected", BORDER)])
        style.configure("Horizontal.TProgressbar", troughcolor=PANEL, background=ACCENT, bordercolor=PANEL)

    def label(self, parent, text=None, *, variable=None, size=10, strong=False, color=TEXT):
        return tk.Label(parent, text=text, textvariable=variable, background=BG, foreground=color,
                        font=("Segoe UI", size, "bold" if strong else "normal"), anchor="w")

    def layout(self):
        outer = ttk.Frame(self.root, padding=(22, 15))
        outer.pack(fill="both", expand=True)
        heading = ttk.Frame(outer)
        heading.pack(fill="x", pady=(0, 13))
        self.label(heading, "Ранжирование лент", size=21, strong=True).pack(side="left")
        ttk.Button(heading, text="Открыть отчёт", command=self.open_report).pack(side="right")
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body, width=306)
        controls.pack(side="left", fill="y", padx=(0, 23))
        controls.pack_propagate(False)
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        self.label(controls, "Данные", strong=True).pack(fill="x")
        data_row = ttk.Frame(controls)
        data_row.pack(fill="x", pady=(4, 12))
        ttk.Entry(data_row, textvariable=self.data).pack(side="left", fill="x", expand=True)
        ttk.Button(data_row, text="…", command=self.pick_data, width=3).pack(side="left", padx=(5, 0))
        self.label(controls, "Дискретная видеокарта", strong=True).pack(fill="x")
        gpu_row = ttk.Frame(controls)
        gpu_row.pack(fill="x", pady=(4, 12))
        self.gpu_select = ttk.Combobox(gpu_row, textvariable=self.gpu, state="readonly", width=30)
        self.gpu_select.pack(side="left", fill="x", expand=True)
        ttk.Button(gpu_row, text="↻", command=self.refresh_gpus, width=3).pack(side="left", padx=(5, 0))
        budgets = ttk.Frame(controls)
        budgets.pack(fill="x", pady=(0, 10))
        for row, title, value in ((0, "Итерации CatBoost", self.iterations), (1, "Эпохи нейросети", self.epochs),
                                  (2, "Ленты train (0 = все)", self.train_feeds), (3, "Ленты val (0 = все)", self.val_feeds)):
            self.label(budgets, title).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(budgets, textvariable=value, width=9).grid(row=row, column=1, sticky="e", padx=(15, 0))
        budgets.columnconfigure(1, weight=1)
        self.label(controls, "Модели", strong=True).pack(fill="x", pady=(0, 4))
        for name, enabled in self.selection.items():
            ttk.Checkbutton(controls, text=name, variable=enabled).pack(anchor="w")
        actions = ttk.Frame(controls)
        actions.pack(fill="x", pady=(10, 0))
        self.start = ttk.Button(actions, text="Обучить и проверить", command=self.launch, style="Launch.TButton")
        self.start.pack(side="left", fill="x", expand=True)
        self.stop = ttk.Button(actions, text="Стоп", command=self.cancel, state="disabled")
        self.stop.pack(side="left", padx=(6, 0))

        self.label(right, variable=self.status, size=13, strong=True).pack(fill="x")
        self.label(right, variable=self.sample, color=MUTED).pack(fill="x")
        ttk.Progressbar(right, variable=self.progress, maximum=100).pack(fill="x", pady=(5, 8))
        self.label(right, variable=self.stage, color=MUTED).pack(fill="x")
        ttk.Progressbar(right, variable=self.model_progress, maximum=100).pack(fill="x", pady=(3, 9))
        self.label(right, "Полная NDCG · ранняя val", strong=True).pack(fill="x")
        self.chart = tk.Canvas(right, height=145, background=PANEL, highlightthickness=0)
        self.chart.pack(fill="x", pady=(4, 9))
        self.chart.bind("<Configure>", lambda _: self.draw_curve())

        table_frame = ttk.Frame(right)
        table_frame.pack(fill="both", expand=True)
        columns = {"model": ("Модель", 155), "late": ("Поздняя val", 94), "all": ("Вся val", 75),
                   "delta": ("Δ к history", 84), "seconds": ("Время", 62), "ram": ("Пик RAM", 80)}
        self.table = ttk.Treeview(table_frame, columns=list(columns), show="headings", height=6)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for name, (title, width) in columns.items():
            self.table.heading(name, text=title)
            self.table.column(name, width=width, anchor="w" if name == "model" else "e")
        self.table.tag_configure("best", foreground=ACCENT)
        self.table.bind("<<TreeviewSelect>>", self.select_result)
        self.label(right, "Журнал", strong=True).pack(fill="x", pady=(9, 4))
        log_frame = ttk.Frame(right)
        log_frame.pack(fill="x")
        self.log = tk.Text(log_frame, height=5, bg=PANEL, fg=MUTED, font=("Consolas", 9), wrap="word",
                           state="disabled", padx=7, pady=5, borderwidth=0, highlightthickness=0)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    def refresh_gpus(self):
        self.gpus = {f"{card.index} · {card.name} · {card.memory}": card.uuid for card in nvidia_gpus()}
        self.gpu_select.configure(values=list(self.gpus))
        if self.gpu.get() not in self.gpus:
            self.gpu.set(next(iter(self.gpus), "NVIDIA GPU не найдена"))

    def pick_data(self):
        path = filedialog.askdirectory(initialdir=self.data.get())
        if path:
            self.data.set(path)

    def launch(self):
        try:
            data = Path(self.data.get()).expanduser().resolve()
            if not all((data / split / "impressions").is_dir() for split in ("train", "val")):
                raise ValueError("Выберите каталог с train/impressions и val/impressions.")
            gpu = self.gpus.get(self.gpu.get())
            if gpu is None:
                raise ValueError("Выберите дискретную NVIDIA GPU.")
            models = [name for name, enabled in self.selection.items() if enabled.get()]
            if not models:
                raise ValueError("Выберите хотя бы одну модель.")
            iterations, epochs, train_feeds, val_feeds = map(
                int, (self.iterations.get(), self.epochs.get(), self.train_feeds.get(), self.val_feeds.get())
            )
            if min(iterations, epochs) < 1 or min(train_feeds, val_feeds) < 0:
                raise ValueError("Бюджет обучения должен быть положительным, лимиты лент — неотрицательными.")
        except ValueError as error:
            messagebox.showerror("Параметры запуска", str(error), parent=self.root)
            return
        output = ROOT / "runs" / "dashboard" / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
        output.parent.mkdir(parents=True, exist_ok=True)
        self.report_path, self.report_stamp = output / "report.json", None
        self.models, self.completed, self.curve = models, set(), []
        self.budget = iterations, epochs
        self.running, self.cancelled = True, False
        self.progress.set(0)
        self.model_progress.set(0)
        self.table.delete(*self.table.get_children())
        self.start.configure(state="disabled")
        self.stop.configure(state="normal")
        self.status.set("Запуск CUDA")
        self.sample.set("")
        self.stage.set("")
        self.draw_curve()
        self.clear_log()
        self.append(str(output))
        command = command_for(data, output, gpu, models, iterations, epochs, train_feeds, val_feeds)
        try:
            self.process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
        except OSError as error:
            self.append(str(error))
            self.finished(1)
            return
        threading.Thread(target=self.read_worker, args=(self.process, output.with_suffix(".log")), daemon=True).start()

    def read_worker(self, process, log):
        file = None
        try:
            file = log.open("w", encoding="utf-8")
        except OSError as error:
            self.events.put(("line", f"Журнал в файле недоступен: {error}"))
        try:
            with process.stdout:
                for line in process.stdout:
                    self.events.put(("line", line.rstrip()))
                    if file is not None and not file.closed:
                        try:
                            file.write(line)
                            file.flush()
                        except OSError:
                            self.events.put(("line", "Запись журнала на диск прервана."))
                            try:
                                file.close()
                            except OSError:
                                pass
                            file = None
        finally:
            if file is not None:
                file.close()
            self.events.put(("done", process.wait()))

    def cancel(self):
        if not self.running or self.process is None or self.process.poll() is not None:
            return
        self.cancelled = True
        self.status.set("Останавливаю запуск")
        self.stop.configure(state="disabled")
        if os.name == "nt":
            subprocess.Popen(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.killpg(self.process.pid, signal.SIGTERM)

    def close(self):
        if self.running:
            if not messagebox.askyesno("Обучение ещё идёт", "Остановить запуск и закрыть окно?", parent=self.root):
                return
            self.cancel()
        self.root.destroy()

    def poll(self):
        if self.watch_config is not None:
            self.poll_watch()
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                self.on_line(value)
            else:
                self.finished(value)
        if self.report_path:
            try:
                stamp = self.report_path.stat().st_mtime_ns
                if self.report_stamp != stamp:
                    self.render_report(json.loads(self.report_path.read_text(encoding="utf-8")))
                    self.report_stamp = stamp
            except (OSError, ValueError):
                pass
        self.root.after(160, self.poll)

    def poll_watch(self):
        try:
            stamp = self.watch_config.stat().st_mtime_ns
            if stamp != self.watch_stamp:
                job = json.loads(self.watch_config.read_text(encoding="utf-8"))
                self.report_path, self.report_stamp = Path(job["report"]), None
                self.watch_log, self.watch_status = Path(job["log"]), Path(job["status"])
                self.watch_label = job.get("label", "")
                self.data.set(job["data"])
                self.models, self.completed, self.curve = job["models"], set(), []
                self.budget = job["iterations"], job["epochs"]
                self.watch_offset = 0
                self.watch_active = True
                self.progress.set(0)
                self.model_progress.set(0)
                self.status.set("Ожидание запуска")
                self.sample.set(self.watch_label)
                self.stage.set("")
                self.clear_log()
                self.table.delete(*self.table.get_children())
                self.draw_curve()
                self.start.configure(state="disabled")
                self.watch_stamp = stamp
            if self.watch_log and self.watch_log.is_file():
                with self.watch_log.open("rb") as log:
                    log.seek(self.watch_offset)
                    chunk = log.read()
                end = chunk.rfind(b"\n") + 1
                self.watch_offset += end
                for line in chunk[:end].decode("utf-8", errors="replace").splitlines():
                    self.on_line(line)
            if self.watch_active and self.watch_status.is_file():
                code = int(self.watch_status.read_text(encoding="utf-8").strip())
                self.watch_active = False
                self.finished(code)
        except (OSError, ValueError, KeyError):
            pass

    def on_line(self, line):
        self.append(line)
        if line.startswith("Starting "):
            self.curve = []
            self.draw_curve()
            self.model_progress.set(0)
            self.status.set(f"{line.removeprefix('Starting ')} · {len(self.completed) + 1} из {len(self.models)}")
            self.stage.set("Подготовка признаков и данных")
        elif match := TUNING.match(line):
            model, step, value = match.groups()
            self.curve.append((int(step), float(value)))
            self.draw_curve()
            self.stage.set(f"Проверка {model} · полная NDCG {float(value):.5f}")
            total = self.budget[1 if model == "dual-history" else 0]
            current = int(step) / total
            self.model_progress.set(100 * (current if model == "dual-history" else .75 + .25 * current))
        elif match := BATCH.search(line):
            epoch, batch, total = map(int, match.groups())
            self.stage.set(f"Эпоха {epoch}/{self.budget[1]} · батч {batch}/{total}")
            self.model_progress.set(100 * (epoch - 1 + batch / total) / self.budget[1])
        elif match := TREE.match(line):
            step = int(match[1]) + 1
            self.stage.set(f"Дерево {step} из {self.budget[0]}")
            self.model_progress.set(75 * step / self.budget[0])
        elif line.startswith("Finished "):
            self.completed.add(line.split()[1].rstrip(":"))
            self.progress.set(100 * len(self.completed) / len(self.models))
            self.model_progress.set(100)
        elif line.startswith("Fitting RRF"):
            self.status.set("Подбор веса RRF на ранней val")

    def finished(self, code):
        self.running = False
        self.start.configure(state="normal")
        self.stop.configure(state="disabled")
        self.status.set("Обучение остановлено" if self.cancelled else
                        "Готово · результаты отсортированы по поздней val" if code == 0 else
                        f"Запуск завершился с ошибкой ({code})")
        if code == 0:
            self.progress.set(100)
            self.model_progress.set(100)

    def draw_curve(self):
        chart = self.chart
        chart.delete("all")
        width, height = chart.winfo_width(), chart.winfo_height()
        if not self.curve or width < 100:
            chart.create_text(14, 18, text="На графике появятся замеры по деревьям или эпохам.", anchor="w", fill=MUTED)
            return
        start, end = self.curve[0][0], self.curve[-1][0]
        values = [value for _, value in self.curve]
        lo, hi = min(values) - .002, max(values) + .002
        for k in range(4):
            y = 14 + (height - 35) * k / 3
            chart.create_line(58, y, width - 16, y, fill=BORDER)
            chart.create_text(51, y, text=f"{hi - (hi - lo) * k / 3:.4f}", fill=MUTED, anchor="e")
        points = [(58 + (width - 74) * (step - start) / max(end - start, 1),
                   14 + (height - 35) * (hi - value) / (hi - lo)) for step, value in self.curve]
        if len(points) > 1:
            chart.create_line(*[coord for point in points for coord in point], fill=ACCENT, width=2)
        x, y = points[-1]
        chart.create_oval(x - 3, y - 3, x + 3, y + 3, fill=ACCENT, outline="")
        chart.create_text(58, height - 10, text=str(start), anchor="w", fill=MUTED)
        if end != start:
            chart.create_text(width - 16, height - 10, text=str(end), anchor="e", fill=MUTED)

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def append(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 1200:
            self.log.delete("1.0", "201.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def render_report(self, report):
        sizes = [self.watch_label] if self.watch_label else []
        for split in ("train", "val"):
            count = report.get(f"{split}_feeds")
            limit = report.get(f"{split}_feed_limit")
            if count is not None:
                sizes.append(f"{split} · {count:,}".replace(",", " "))
            elif limit is not None:
                sizes.append(f"{split} · лимит {limit:,}".replace(",", " ") if limit else f"{split} · все")
        self.sample.set("   |   ".join(sizes))
        self.table.delete(*self.table.get_children())
        rows = report["models"]
        self.result_rows = {row["model"]: row for row in rows}
        baseline = next((row["holdout"]["ndcg"] for row in rows if row["model"] == "history"), None)
        for index, row in enumerate(rows):
            late, full = row["holdout"]["ndcg"], row["val"]["ndcg"]
            delta = f"{late - baseline:+.5f}" if baseline is not None else "—"
            runtime = f"{row['seconds']:.1f} с" if "seconds" in row else "—"
            ram = f"{row['peak_ram_mib']:.0f} MiB" if row.get("peak_ram_mib") is not None else "—"
            self.table.insert("", "end", iid=row["model"], values=(row["model"], f"{late:.5f}", f"{full:.5f}", delta, runtime, ram),
                              tags=("best",) if index == 0 else ())

    def select_result(self, _):
        if not self.running and not self.watch_active and self.table.selection():
            row = self.result_rows[self.table.selection()[0]]
            self.curve = row.get("tuning_curve", [])
            self.stage.set(row["model"])
            self.draw_curve()

    def open_report(self):
        if self.running:
            return
        path = filedialog.askopenfilename(initialdir=str(ROOT / "runs"), filetypes=[("Отчёт JSON", "report.json")])
        if not path:
            return
        self.load_report(Path(path))

    def load_report(self, path):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            self.watch_label = ""
            self.render_report(report)
            self.status.set("Отчёт")
            stamp = path.stat().st_mtime_ns
        except (OSError, ValueError, KeyError) as error:
            messagebox.showerror("Отчёт не прочитан", str(error), parent=self.root)
            return
        self.curve = []
        self.stage.set("")
        self.draw_curve()
        if report["models"]:
            self.table.selection_set(report["models"][0]["model"])
        self.report_path, self.report_stamp = path, stamp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Open an existing benchmark report on launch")
    parser.add_argument("--watch", type=Path, help="Watch an externally launched experiment")
    options = parser.parse_args()
    window = tk.Tk()
    app = Dashboard(window)
    app.watch_config = options.watch
    if options.report:
        app.load_report(options.report)
    window.mainloop()
