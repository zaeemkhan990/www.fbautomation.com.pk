import sys
if len(sys.argv) > 1 and sys.argv[1] in ("--check-fw", "--transcribe"):
    import os as _os, faulthandler as _fh
    _fh.enable()  # a native crash now prints the place it happened instead of dying silently
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoids the Windows OpenMP crash
    _os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    _os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
if len(sys.argv) > 1 and sys.argv[1] == "--check-fw":  # used by  Captions → 🩺 Test captions
    try:
        import faster_whisper, ctranslate2
        print(f"Python {sys.version.split()[0]} | faster-whisper {faster_whisper.__version__}, ctranslate2 {ctranslate2.__version__}", flush=True)
    except Exception as _e:
        print("IMPORT FAILED: " + str(_e), flush=True)
        sys.exit(2)
    _lv = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 0
    print("speech library loaded OK", flush=True)
    print(f"Loading the '{sys.argv[2]}' model (the first time it downloads — this can take a few minutes) ...", flush=True)
    WhisperModel = faster_whisper.WhisperModel
    _md = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else ""
    _m = WhisperModel(_md if (_md and _os.path.isdir(_md)) else sys.argv[2], device="cpu",
                      compute_type="float32" if _lv == 1 else "int8", **({"cpu_threads": 1} if _lv else {}))
    print("MODEL OK", flush=True)
    import numpy as _np
    print("Running a short listening test ...", flush=True)
    _sg, _ = _m.transcribe(_np.zeros(32000, dtype="float32"), vad_filter=(_lv == 0), beam_size=1)
    list(_sg)
    print("LISTEN OK", flush=True)
    sys.exit(0)
if len(sys.argv) > 1 and sys.argv[1] == "--transcribe":  # speech engine runs in its own process, so a crash can never close the app
    import json
    _path, _size, _lang, _out = sys.argv[2:6]
    _mdir = sys.argv[6] if len(sys.argv) > 6 else ""
    _task = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] in ("transcribe", "translate") else "transcribe"
    _lv = int(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8].isdigit() else 0  # 0 normal, 1/2 = safe modes (for PCs where it crashed)
    _beam = {"Fast": 1, "Balanced": 3, "Best": 5}.get(sys.argv[9] if len(sys.argv) > 9 else "Fast", 1)
    import time as _time, threading as _th
    _t0 = _time.time()
    _cur = ["starting"]

    def _st(msg):
        _cur[0] = msg
        sys.stderr.write("STAGE " + msg + "\n")
        sys.stderr.flush()

    def _hb():  # a sign of life every 20 s while a long step is running
        while True:
            _time.sleep(20)
            sys.stderr.write("STAGE still working: %s (%ds)\n" % (_cur[0], _time.time() - _t0))
            sys.stderr.flush()
    _th.Thread(target=_hb, daemon=True).start()
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.stderr.write("No module named 'faster_whisper'\n")
        sys.exit(2)
    _kw = dict(device="cpu", compute_type="float32" if _lv == 1 else "int8", cpu_threads=1 if _lv else min(8, _os.cpu_count() or 4))
    _src = _mdir if (_mdir and _os.path.isdir(_mdir)) else _size
    try:  # model already downloaded -> never touch the network (a blocked / slow connection used to stall every video)
        _st(f"loading the '{_size}' model")
        _model = WhisperModel(_src, local_files_only=True, **_kw)
    except Exception:
        _st(f"downloading the '{_size}' model (first time only — can take several minutes)")
        _model = WhisperModel(_src, **_kw)
    _st("model ready")
    _t_load = _time.time() - _t0

    def _load_audio(p):
        try:
            import subprocess as _sp, numpy as _np
            _r = _sp.run(["ffmpeg", "-nostdin", "-v", "error", "-i", p, "-vn", "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, **({"creationflags": 0x08000000} if _os.name == "nt" else {}))
            if _r.returncode == 0 and _r.stdout:
                return _np.frombuffer(_r.stdout, dtype=_np.float32)
        except Exception:
            pass
        return p
    _audio = _load_audio(_path)

    def _run(vad, lg):
        segs, info = _model.transcribe(_audio, language=lg, task=_task,
                                       word_timestamps=True, vad_filter=vad, beam_size=(1 if _lv else _beam),
                                       condition_on_previous_text=False)
        return [{"start": s.start, "end": s.end, "text": s.text.strip(),
                 "words": [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])]} for s in segs], info
    _lg = None if _lang == "auto" else _lang
    _res = None
    if not _lv:
        try:
            _st("listening")
            _res = _run(True, _lg)
        except Exception:
            _res = None
    if not _res or not _res[0]:
        _st("listening again (without the voice filter)")
        _res = _run(False, _lg)  # singing / chanting (azan, nasheeds) is often cut away by the voice filter
    if not _res[0] and _lg:
        _st("listening again (detecting the language)")
        _res = _run(False, None)  # wrong language picked -> let it detect the language itself
    _segments, _info = _res
    with open(_out, "w", encoding="utf-8") as _f:
        json.dump({"language": _info.language, "prob": float(_info.language_probability), "segments": _segments}, _f)
    sys.stderr.write("TIMING load=%.1f listen=%.1f\n" % (_t_load, _time.time() - _t0 - _t_load))
    sys.exit(0)
import os, re, io, sys, csv, math, hashlib, collections, traceback, faulthandler, tempfile, webbrowser, json, time, base64, shutil, queue, threading, subprocess, tkinter as tk
import tkinter.font as tkfont
from concurrent.futures import ThreadPoolExecutor, as_completed
from tkinter import ttk, filedialog, messagebox, simpledialog
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import yt_dlp

APP, VER, TAGLINE = "VortexForge", "v6.1.2", "Pro Edition • Download • Caption • Create"
DEFAULT_THEME = "Aurora"
DEV_NAME, DEV_PHONE = "Zaeem Khan", "03282999175"
CREDIT = f"Developed by {DEV_NAME}  |  Contact: {DEV_PHONE}"
APPDIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_NAMES = ("profile.png", "profile.jpg", "profile.jpeg", "profile.webp", "me.png", "me.jpg")
TOKDIR, ACTIVE, SETTINGS = "tokens", "active_account.txt", "settings.json"
ERRLOG, CRASHLOG = "tikvortex_error.log", "tikvortex_crash.log"
STATE, RECORDS, HISTORY = "run_state.json", "history_log.json", "history.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly",
          "https://www.googleapis.com/auth/documents"]
DOC_FIELDS = "body(content(paragraph(elements(startIndex,textRun(content)))))"
VIDEO = re.compile(r'https?://\S*tiktok\.com/\S+/(?:video|photo)/\d+\S*|https?://v[mt]\.tiktok\.com/\S+')
PROFILE = re.compile(r'https?://(?:www\.)?tiktok\.com/@[\w.\-]+/?(?:\?\S*)?')
THEMES = {
    "Aurora":   dict(bg="#07141a", side="#040d11", card="#0d2029", card2="#14303d", line="#1f4a5c",
                     text="#e8fbff", muted="#86b3c2", accent="#14b8a6", accent2="#a78bfa",
                     ok="#4ade80", warn="#fbbf24", err="#fb7185", log="#05101a"),
    "Neon Pro": dict(bg="#0a0a0f", side="#050507", card="#12121a", card2="#1c1c28", line="#2e2e44",
                     text="#f4f4ff", muted="#8d8da8", accent="#e11d74", accent2="#22e5ff",
                     ok="#4ade80", warn="#fde047", err="#ff6b6b", log="#07070b"),
    "Ocean":    dict(bg="#07182b", side="#040f1d", card="#0c2440", card2="#123259", line="#1d4b80",
                     text="#e6f3ff", muted="#85a9cf", accent="#0284c7", accent2="#5eead4",
                     ok="#34d399", warn="#fbbf24", err="#f87171", log="#050f1c"),
    "Sunset":   dict(bg="#1f0f1a", side="#150912", card="#2c1624", card2="#402036", line="#6b3355",
                     text="#ffeef5", muted="#cfa0b8", accent="#e0446a", accent2="#ffb347",
                     ok="#6ee7a0", warn="#fde047", err="#ff7a7a", log="#150910"),
    "Graphite": dict(bg="#121212", side="#0b0b0b", card="#1b1b1b", card2="#262626", line="#3a3a3a",
                     text="#f2f2f2", muted="#9a9a9a", accent="#6366f1", accent2="#f5c518",
                     ok="#4ade80", warn="#fbbf24", err="#f87171", log="#0d0d0d"),
    "Rose Quartz": dict(bg="#fdf0f4", side="#ffffff", card="#ffffff", card2="#fbe7ee", line="#f0cbd8",
                     text="#3a1f2b", muted="#9b6b7f", accent="#d6336c", accent2="#7c3aed",
                     ok="#15803d", warn="#b45309", err="#dc2626", log="#fff7fa"),
    "Mint Light": dict(bg="#eaf7f1", side="#ffffff", card="#ffffff", card2="#eef9f4", line="#cfe8dc",
                     text="#12372a", muted="#5f8a77", accent="#059669", accent2="#0284c7",
                     ok="#15803d", warn="#b45309", err="#dc2626", log="#f4fbf8"),
    "Midnight": dict(bg="#0b0f1a", side="#070a12", card="#131a2b", card2="#1b2540", line="#2a3868",
                     text="#e6ecff", muted="#8794b8", accent="#7c5cff", accent2="#22d3ee",
                     ok="#34d399", warn="#fbbf24", err="#f87171", log="#080c16"),
    "Snow":     dict(bg="#e9edf7", side="#ffffff", card="#ffffff", card2="#f0f3fb", line="#d5dcee",
                     text="#1b2238", muted="#6b7694", accent="#4f46e5", accent2="#0284c7",
                     ok="#16a34a", warn="#d97706", err="#dc2626", log="#f6f8fd"),
    "Emerald":  dict(bg="#08130f", side="#050d0a", card="#0f1f19", card2="#16322a", line="#215244",
                     text="#e3fff3", muted="#7fb09d", accent="#10b981", accent2="#a3e635",
                     ok="#4ade80", warn="#facc15", err="#fb7185", log="#06100c"),
    "Amethyst": dict(bg="#150b24", side="#0e0618", card="#1f1236", card2="#2e1b4f", line="#4a2c80",
                     text="#f3e9ff", muted="#b39ad6", accent="#d946ef", accent2="#f59e0b",
                     ok="#5eead4", warn="#fcd34d", err="#fb7185", log="#0f0719"),
    "Ember":    dict(bg="#1a0f0b", side="#120a07", card="#26150f", card2="#3a2016", line="#5e3322",
                     text="#fff1e8", muted="#c9a08a", accent="#ff6b35", accent2="#ffc83d",
                     ok="#6ee7a0", warn="#fde047", err="#ff5c5c", log="#120905"),
}
NAV_GROUPS = (
    ("DOWNLOAD", (("dl", "⬇  Auto Downloader"), ("links", "🔗  Link Collector"), ("uni", "🌐  4K / Any Link"))),
    ("STUDIO", (("clip", "✂  Video Clipper"), ("smart", "🔥  Smart Clips"), ("caps", "💬  Captions"),
                ("brand", "🏷  Brand & Logo"), ("edit", "🎨  Editing"), ("thumb", "🖼  Thumbnails"))),
    ("MANAGE", (("clean", "🧹  Cleaner"), ("prog", "📊  Progress"), ("hist", "🕘  History"))),
    ("SYSTEM", (("acc", "👤  Accounts"), ("set", "⚙  Settings"))),
)
NAV = tuple(i for _, g in NAV_GROUPS for i in g)
WHATS_NEW = (
    ("⏳", "You can see what it is doing", "While captions are made, the log shows each step: loading / downloading the model, listening, listening again. A sign of life appears every 20 s."),
    ("🕌", "Azan / chanting / songs", "No more \"no speech found\" failures: the voice filter is retried without, and the language is re-detected if the chosen one finds nothing. Videos with really no speech are simply skipped."),
    ("🌍", "Urdu → English captions, automatically", "The language is detected for every video; Urdu, Hindi, Arabic... speech is turned into English captions (English stays English)."),
    ("⏭", "No double captions", "Videos that already have captions (a subtitle track, text already burned in, or captions this app added) are skipped. You can turn this off in Captions → Output."),
    ("💬", "Captions rebuilt", "A much more reliable caption engine: safer file handling for ffmpeg, a download mirror and offline model folder for the speech model, and one-click Install / repair speech engine."),
    ("🫧", "Caption shapes", "Rounded bubble, sticker, neon glow, hard shadow and box shapes, with their own colour."),
    ("🎬", "Caption animations", "Pop-in, bounce, fade and slide-up, plus italic and letter spacing. 8 new ready-made styles (Hormozi, Bubble, Pill karaoke, Sticker, Glow neon, Cinema, Hard shadow, Reels slide)."),
    ("🌍", "Translate to English", "Tick it in Captions → Speech and any language is captioned in English."),
    ("🔎", "Profile search", "Type in the box on the Auto Downloader tab to filter profiles instantly."),
    ("📌", "Always on top", "Settings → Appearance. New F1 actions: clear log, copy log, test captions."),
)
SCROLL_PAGES = ("clip", "smart", "caps", "brand", "edit", "thumb", "set")  # long forms: the whole page scrolls
STAT_PAGES = ("dl", "links", "uni", "prog", "hist")  # pages that show the 5 number cards
PAGE_TITLES = {"dl": "Auto Downloader", "uni": "4K / Any Link Downloader", "links": "Link Collector", "smart": "Smart Clips", "caps": "Captions", "thumb": "Thumbnail Maker", "clean": "Cleaner",
               "clip": "Video Clipper", "brand": "Brand & Logo", "edit": "Editing", "prog": "Progress",
               "hist": "History", "acc": "Accounts", "set": "Settings"}


def fmt(h=None):  # TikTok: prefer H.264 so Windows can show thumbnails
    f = f"[height<={h}]" if h else ""
    return f"bv*[vcodec^=avc1]{f}+ba/b[vcodec^=avc1]{f}/bv*{f}+ba/b{f}/b"


QUALITY = {"Best": fmt(), "1080p": fmt(1080), "720p": fmt(720), "480p": fmt(480)}
UNI_Q = {"Best": None, "4K (2160p)": 2160, "1440p": 1440, "1080p": 1080, "720p": 720, "480p": 480}


class QuietLog:
    def debug(self, m): pass
    def info(self, m): pass
    def warning(self, m): pass
    def error(self, m): pass


class NoSpeech(Exception):
    pass


class Cancelled(Exception):
    pass


def clean(s):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", s or "")
    return re.sub(r"\s+", " ", s).strip()[:120]


def vid_id(url):
    m = re.search(r'/(?:video|photo)/(\d+)', url)
    return m.group(1) if m else url


def open_path(p):
    try:
        if sys.platform.startswith("win"):
            os.startfile(p)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])
    except Exception:
        pass


def jload(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def jsave(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def mix(a, b, t):
    ra = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    rb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(x + (y - x) * t) for x, y in zip(ra, rb))


def rrect(c, x1, y1, x2, y2, r, **kw):
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2,
           x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, **kw)


class Btn(tk.Canvas):  # rounded pill button
    def __init__(self, parent, app, text, cmd, kind="ghost", surface="card", size=10, padx=16, pady=7):
        super().__init__(parent, highlightthickness=0, bd=0, cursor="hand2")
        self.app, self.cmd, self.kind, self.surface, self.text = app, cmd, kind, surface, text
        self.padx, self.pady, self.hover = padx, pady, False
        self.font = tkfont.Font(family="Segoe UI", size=size, weight="bold")
        self.bind("<Enter>", lambda e: self._hov(True))
        self.bind("<Leave>", lambda e: self._hov(False))
        self.bind("<Button-1>", lambda e: self.cmd())
        app.btns.append(self)
        self.draw()

    def _hov(self, v):
        self.hover = v
        self.draw()

    def set_text(self, t):
        self.text = t
        self.draw()

    def draw(self):
        p = self.app.pal
        w = self.font.measure(self.text) + self.padx * 2
        h = self.font.metrics("linespace") + self.pady * 2
        self.configure(width=w, height=h, bg=p[self.surface])
        base, fg = {"primary": (p["accent"], "#ffffff"), "danger": (p["err"], "#ffffff"),
                    "success": (p["ok"], "#06120d"), "ghost": (p["card2"], p["text"])}[self.kind]
        self.delete("all")
        rrect(self, 1, 1, w - 1, h - 1, min(h // 2, 13), fill=mix(base, "#ffffff", 0.16) if self.hover else base, outline="")
        self.create_text(w // 2, h // 2, text=self.text, fill=fg, font=self.font)


def photo_path():
    for n in PHOTO_NAMES:
        p = os.path.join(APPDIR, n)
        if os.path.exists(p):
            return p
    return None


def make_photo(size, shape="circle"):
    """Profile picture: 'square' = rounded square filled with the photo (face centred), 'circle' = round on black.
    Uses profile.png/jpg next to the script if present, otherwise the built-in picture."""
    path = photo_path()
    try:
        from PIL import Image, ImageDraw, ImageOps, ImageTk
        im = Image.open(path) if path else Image.open(io.BytesIO(base64.b64decode(PHOTO_B64)))
        im = im.convert("RGB")
        mask = Image.new("L", (size * 4, size * 4), 0)
        md = ImageDraw.Draw(mask)
        if shape == "square":
            sq = ImageOps.fit(im, (size, size), Image.LANCZOS, centering=(0.5, 0.2))
            md.rounded_rectangle((0, 0, size * 4 - 1, size * 4 - 1), radius=int(size * 4 * 0.14), fill=255)
        else:
            im.thumbnail((size - 10, size - 10))
            sq = Image.new("RGB", (size, size), (0, 0, 0))
            sq.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
            md.ellipse((0, 0, size * 4 - 1, size * 4 - 1), fill=255)
        mask = mask.resize((size, size), Image.LANCZOS)
        out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        out.paste(sq, (0, 0), mask)
        return ImageTk.PhotoImage(out)
    except ImportError:  # no Pillow: PNG only, not rounded
        try:
            img = tk.PhotoImage(file=path) if path else tk.PhotoImage(data=PHOTO_B64)
            k = max(1, -(-max(img.width(), img.height()) // size))
            return img.subsample(k)
        except Exception:
            return None
    except Exception:
        return None


class Avatar(tk.Canvas):
    def __init__(self, parent, app, size, surface="side", ring=True, animate=False, click=True, shape="circle"):
        super().__init__(parent, width=size, height=size, highlightthickness=0, bd=0)
        self.app, self.size, self.surface, self.ring, self.shape = app, size, surface, ring, shape
        self.phase = 0.0
        self.img = make_photo(size - 6, shape)
        if click:
            self.configure(cursor="hand2")
            self.bind("<Button-1>", lambda e: app.show_about())
        app.btns.append(self)
        self.draw()
        if animate:
            self._spin()

    def reload(self):
        self.img = make_photo(self.size - 6, self.shape)
        self.draw()

    def ring_color(self):
        p = self.app.pal
        return mix(p["accent"], p["accent2"], (math.sin(self.phase) + 1) / 2)

    def _spin(self):  # slowly pulsing colour ring
        try:
            self.phase += 0.18
            self.itemconfig("ring", outline=self.ring_color())
            self.after(70, self._spin)
        except tk.TclError:
            pass

    def draw(self):
        p, s = self.app.pal, self.size
        self.configure(bg=p.get(self.surface, self.surface))
        self.delete("all")
        sq = self.shape == "square"
        if self.img:
            self.create_image(s // 2, s // 2, image=self.img)
        elif sq:
            rrect(self, 4, 4, s - 4, s - 4, int(s * 0.14), fill=p["accent"], outline="")
            self.create_text(s // 2, s // 2, text="ZK", fill="#ffffff", font=("Segoe UI", max(10, s // 3), "bold"))
        else:
            self.create_oval(4, 4, s - 4, s - 4, fill=p["accent"], outline="")
            self.create_text(s // 2, s // 2, text="ZK", fill="#ffffff", font=("Segoe UI", max(10, s // 3), "bold"))
        if self.ring:
            if sq:
                rrect(self, 2, 2, s - 2, s - 2, int(s * 0.14) + 2, fill="", outline=self.ring_color(), width=3, tags="ring")
            else:
                self.create_oval(2, 2, s - 2, s - 2, outline=self.ring_color(), width=3, tags="ring")


class Check(tk.Canvas):  # clear tick box
    def __init__(self, parent, app, text, var, surface="card"):
        super().__init__(parent, highlightthickness=0, bd=0, cursor="hand2")
        self.app, self.var, self.text, self.surface = app, var, text, surface
        self.font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        var.trace_add("write", lambda *a: self.draw())
        self.bind("<Button-1>", lambda e: var.set(not var.get()))
        app.btns.append(self)
        self.draw()

    def draw(self):
        p = self.app.pal
        w, h, on = self.font.measure(self.text) + 44, 30, self.var.get()
        self.configure(width=w, height=h, bg=p[self.surface])
        self.delete("all")
        rrect(self, 3, 4, 25, 26, 6, fill=p["accent"] if on else p["card2"], outline=p["accent"] if on else p["muted"], width=2)
        if on:
            self.create_line(8, 15, 13, 21, 21, 9, fill="#ffffff", width=3, capstyle="round", joinstyle="round")
        self.create_text(36, h // 2, text=self.text, anchor="w", fill=p["text"], font=self.font)


BRAND_COLORS = {"White": "#ffffff", "Yellow": "#ffd400", "Cyan": "#22d3ee", "Pink": "#ff4fa3",
                "Orange": "#ff8a2b", "Black": "#000000"}
POSITIONS = ["Bottom-right", "Bottom-left", "Top-right", "Top-left", "Top-center", "Bottom-center",
             "Center"]
ASPECTS = ["Original", "9:16 Vertical", "1:1 Square", "16:9 Landscape"]
MAXH = ["Original", "1080", "720", "480"]
CRFS = ["High (CRF 18)", "Balanced (CRF 23)", "Small file (CRF 28)"]
OUTMODES = ["Edited folder", "Replace original"]
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


def nowin():
    return {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_type,width,height:format=duration", "-of", "json", path],
                       capture_output=True, text=True, **nowin())
    d = json.loads(r.stdout or "{}")
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        raise Exception("not a video file")
    try:
        dur = float(d.get("format", {}).get("duration") or 0)
    except ValueError:
        dur = 0.0
    return {"w": int(v["width"]), "h": int(v["height"]), "dur": dur,
            "audio": any(s.get("codec_type") == "audio" for s in d.get("streams", []))}


CAP_TAG = "VortexForge-captions"


def frame_has_text(f, W, H):
    """Caption-like text in one grey frame: bright strokes next to dark outlines, in wide rows that stack up like a line of text."""
    y0, y1 = int(H * 0.40), int(H * 0.95)
    need, run, best = W * 0.06, 0, 0
    for y in range(y0, y1):
        row = f[y * W:(y + 1) * W]
        c, xmin, xmax = 0, W, 0
        for x in range(3, W - 3):
            if row[x] >= 190 and (row[x - 3] <= 60 or row[x + 3] <= 60 or row[x - 2] <= 60 or row[x + 2] <= 60):
                c += 1
                if x < xmin:
                    xmin = x
                xmax = x
        if c >= need and (xmax - xmin) >= W * 0.30:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best >= 8


def burned_in_text(path, info, frames=8):
    dur = info.get("dur") or 0
    if dur < 1.5:
        return False
    W = 180
    H = max(64, int(round(W * info["h"] / info["w"] / 2)) * 2)
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-an", "-vf",
                        f"fps={frames / dur:.5f},scale={W}:{H}:flags=area,format=gray", "-frames:v", str(frames),
                        "-f", "rawvideo", "-"], capture_output=True, **nowin())
    raw, fsz = r.stdout or b"", W * H
    n = len(raw) // fsz
    if n < 3:
        return False
    hits = sum(1 for i in range(n) if frame_has_text(raw[i * fsz:(i + 1) * fsz], W, H))
    return hits >= max(3, int(n * 0.4))


def has_captions(path, info=None):
    """Reason text when the video already has captions (subtitle track, our own tag, or text burned in), else ''."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format_tags=comment", "-of", "json", path],
                           capture_output=True, text=True, **nowin())
        d = json.loads(r.stdout or "{}")
    except Exception:
        d = {}
    if any(s.get("codec_type") == "subtitle" for s in d.get("streams", [])):
        return "it has a subtitle track"
    if CAP_TAG in str((d.get("format") or {}).get("tags", {})):
        return "captions were already added by this app"
    try:
        return "it already shows text / captions on screen" if burned_in_text(path, info or probe(path)) else ""
    except Exception:
        return ""


def make_badge(text, fs, color, opacity, box):
    """Render the brand text to a transparent PNG (needs Pillow)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise Exception("Brand overlay needs Pillow: run  pip install pillow")
    font = None
    for n in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(n, fs)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    pad = int(fs * 0.45)
    W, H = (r - l) + pad * 2, (b - t) + pad * 2
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    a = int(255 * max(0.0, min(1.0, opacity)))
    rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    if box:
        d.rounded_rectangle((0, 0, W - 1, H - 1), radius=H // 2, fill=(0, 0, 0, int(a * 0.55)))
    else:
        d.text((pad - l + 2, pad - t + 2), text, font=font, fill=(0, 0, 0, int(a * 0.6)))
    d.text((pad - l, pad - t), text, font=font, fill=rgb + (a,))
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.close()
    img.save(f.name)
    return f.name


def pos_expr(pos, m):
    return {"Top-left": (f"{m}", f"{m}"), "Top-right": (f"W-w-{m}", f"{m}"),
            "Bottom-left": (f"{m}", f"H-h-{m}"), "Bottom-right": (f"W-w-{m}", f"H-h-{m}"),
            "Top-center": ("(W-w)/2", f"{m}"), "Bottom-center": ("(W-w)/2", f"H-h-{m}"),
            "Center": ("(W-w)/2", "(H-h)/2")}.get(pos, (f"W-w-{m}", f"H-h-{m}"))


def build_ffmpeg(src, dst, info, o, username, preview=False, pw=540, seg=None, part_text=None, ass_fn=None):
    """Returns (ffmpeg command, temp files) or (None, []) when there is nothing to edit.
    seg=(start, length) makes one clip of a long video; part_text adds a 'Part N' label."""
    w, h, dur = info["w"], info["h"], info["dur"]
    if seg:
        ts, te, dur_in = float(seg[0]), 0.0, float(seg[1])
    else:
        ts, te = max(0.0, float(o["trim_start"])), max(0.0, float(o["trim_end"]))
        dur_in = dur - ts - te
    username = (username or "").strip().lstrip("@")
    brand = bool(o["brand_on"]) and bool(username)
    logo = bool(o["logo_on"]) and os.path.exists(o["logo_path"] or "")
    tw, th = {"9:16 Vertical": (1080, 1920), "1:1 Square": (1080, 1080),
              "16:9 Landscape": (1920, 1080)}.get(o["aspect"], (None, None))
    maxh = int(o["maxh"]) if str(o["maxh"]).isdigit() else 0
    fade = bool(o.get("fade")) and bool(seg)
    if not seg and not preview and not (brand or logo or tw or (maxh and h > maxh) or ts or te or o["mute"] or ass_fn):
        return None, []
    if not seg and (ts or te) and dur_in <= 0.2:
        raise Exception("cut is longer than the video")
    parts, last = [], "0:v"

    def step(expr):
        nonlocal last
        nm = f"s{len(parts)}"
        parts.append(f"[{last}]{expr}[{nm}]")
        last = nm

    ow, oh = w, h
    if tw:
        parts += [f"[{last}]split[sa][sb]",
                  f"[sa]scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},boxblur=25:2[bg]",
                  f"[sb]scale={tw}:{th}:force_original_aspect_ratio=decrease[fg]",
                  "[bg][fg]overlay=(W-w)/2:(H-h)/2[asp]"]
        last, ow, oh = "asp", tw, th
    if maxh and oh > maxh:
        ow, oh = int(round(ow * maxh / oh / 2) * 2), maxh
        step(f"scale={ow}:{oh}")
    if not tw and (ow % 2 or oh % 2):
        ow, oh = ow - ow % 2, oh - oh % 2
        step(f"scale={ow}:{oh}")
    inputs, n_in, temps = ["-i", src], 1, []
    if ass_fn:  # captions are burned after the shape/size steps, so their size fits the final video
        ap = ass_fn(ow, oh, ts)
        temps.append(ap)
        step("ass=" + ass_arg(ap))

    def badge(text, fs, color, opacity, box, pos, margin_pct, tag):
        nonlocal last, n_in
        png = make_badge(text, fs, color, opacity, box)
        temps.append(png)
        inputs.extend(["-i", png])
        parts.append(f"[{n_in}:v]scale='min(iw,{int(ow * 0.92)})':-2,format=rgba[{tag}b]")
        n_in += 1
        x, y = pos_expr(pos, int(ow * margin_pct))
        parts.append(f"[{last}][{tag}b]overlay=x='{x}':y='{y}'[{tag}o]")
        last = f"{tag}o"

    if brand:
        text = (o["brand_text"] or "@{username}").replace("{username}", username)
        badge(text, max(14, int(oh * float(o["brand_size"]) / 100)), BRAND_COLORS.get(o["brand_color"], "#ffffff"),
              float(o["brand_opacity"]), bool(o["brand_box"]), o["brand_pos"], 0.03, "br")
    if logo:
        inputs += ["-i", o["logo_path"]]
        lw = max(16, int(ow * float(o["logo_size"]) / 100) // 2 * 2)
        parts.append(f"[{n_in}:v]scale={lw}:-2,format=rgba,colorchannelmixer=aa={float(o['logo_opacity']):g}[lg]")
        n_in += 1
        x, y = pos_expr(o["logo_pos"], int(ow * 0.03))
        parts.append(f"[{last}][lg]overlay=x='{x}':y='{y}'[ol]")
        last = "ol"
    if part_text:
        badge(part_text, max(14, int(oh * float(o.get("part_size", 5.0)) / 100)),
              BRAND_COLORS.get(o.get("part_color", "Yellow"), "#ffd400"), 0.95, True,
              o.get("part_pos", "Top-center"), 0.04, "pt")
    tail = f"fade=t=in:st=0:d=0.3,fade=t=out:st={max(0.0, dur_in - 0.3):.2f}:d=0.3," if fade else ""
    parts.append(f"[{last}]{tail}format=yuv420p" + (f",scale={pw}:-2" if preview else "") + "[vout]")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    seek = ts if ts else (1.0 if (preview and dur > 2) else 0.0)
    if seek:
        cmd += ["-ss", f"{seek:g}"]
    cmd += inputs
    if not preview and (seg or ts or te):
        cmd += ["-t", f"{dur_in:.3f}"]
    cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]"]
    if preview:
        cmd += ["-an", "-frames:v", "1", dst]
    else:
        if info["audio"] and not o["mute"]:
            cmd += ["-map", "0:a?", "-c:a", "aac", "-b:a", "160k"]
            if fade:
                cmd += ["-af", f"afade=t=in:st=0:d=0.3,afade=t=out:st={max(0.0, dur_in - 0.3):.2f}:d=0.3"]
        else:
            cmd += ["-an"]
        m = re.search(r"CRF (\d+)", str(o["crf"]))
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", m.group(1) if m else "23",
                "-movflags", "+faststart"] + (["-metadata", f"comment={CAP_TAG}"] if ass_fn else []) + [dst]
    return cmd, temps


def split_segments(total, length, min_last=10.0, overlap=0.0):
    """Cut points for a long video: [(start, length), ...]. A tiny last piece is merged into the one before."""
    segs, start = [], 0.0
    step = max(1.0, length - max(0.0, overlap))
    while start < total - 0.5:
        segs.append([start, min(length, total - start)])
        start += step
    if len(segs) > 1 and segs[-1][1] < min_last:
        segs.pop()
        segs[-1][1] = total - segs[-1][0]
    return [tuple(s) for s in segs]


def hms(sec):
    sec = int(round(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


class ScrollFrame(ttk.Frame):  # vertically scrollable area (mouse wheel works)
    def __init__(self, parent, app, style="Card.TFrame", bgkey="card"):
        super().__init__(parent, style=style)
        self.cv = tk.Canvas(self, highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.cv.yview)
        self.inner = ttk.Frame(self.cv, style=style)
        self.win = self.cv.create_window((0, 0), window=self.inner, anchor="nw")
        self.cv.configure(yscrollcommand=sb.set)
        self.cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda e: self.cv.configure(scrollregion=self.cv.bbox("all")))
        self.cv.bind("<Configure>", lambda e: self.cv.itemconfigure(self.win, width=e.width))
        app.reg.append((self.cv, {"bg": bgkey}))


# ======================= Link collector / Smart clips / Captions / Thumbnails / Cleaner helpers =======================
LC_ORDERS = ["Latest → Oldest", "Oldest → Latest"]
CAP_COLORS = {"White": "#ffffff", "Yellow": "#ffd400", "Cyan": "#22d3ee", "Pink": "#ff4fa3", "Orange": "#ff8a2b",
              "Green": "#39e75f", "Red": "#ff3b3b", "Purple": "#a855f7", "Black": "#000000", "Lime": "#b6ff00",
              "Gold": "#ffb800", "Navy": "#0b1e5b", "Grey": "#3a3a3a"}
CAP_LANGS = {"Auto-detect": None, "Urdu": "ur", "Hindi": "hi", "English": "en", "Arabic": "ar", "Punjabi": "pa",
             "Persian": "fa", "Bengali": "bn", "Turkish": "tr", "Indonesian": "id", "Spanish": "es", "French": "fr",
             "German": "de", "Portuguese": "pt", "Russian": "ru", "Italian": "it", "Japanese": "ja",
             "Korean": "ko", "Chinese": "zh"}
LANG_NAMES = {v: k for k, v in CAP_LANGS.items() if v}
CAP_MODELS = ["tiny", "base", "small", "medium"]
CAP_FONTS = ["Arial", "Arial Black", "Impact", "Segoe UI", "Segoe UI Black", "Bahnschrift", "Trebuchet MS", "Verdana", "Tahoma",
             "Calibri", "Georgia", "Times New Roman", "Comic Sans MS", "Jameel Noori Nastaleeq", "Noto Nastaliq Urdu", "Noto Sans Arabic"]
CAP_POS = ["Bottom", "Center", "Top"]
CAP_SHAPES = ["Outline", "Box", "Bubble (rounded)", "Sticker", "Glow", "Hard shadow"]
CAP_ANIMS = ["None", "Fade", "Pop-in", "Bounce", "Slide up"]
CAP_DEFAULTS = dict(cp_shape="Outline", cp_boxcolor="Black", cp_anim="None", cp_italic=False, cp_spacing=0.0, cp_box=False)
CRASH_CODES = (3221225477, 3221225501, 3221226505, -11, -4, -6, 132, 134, 139)
NET_ERR = re.compile(r"connect|resolve|timed? ?out|offline|max retries|ssl|proxy|network|unreachable|getaddrinfo|10060|10061|11001|"
                     r"forbidden|local cache|locate the file|snapshot", re.I)
NO_ASS_MSG = ("this ffmpeg has no caption (libass) support — install the full build:  winget install Gyan.FFmpeg  "
              "(then restart the app)")
SAMPLE_TEXT = {"ur": "یہ آپ کے کیپشن کا نمونہ ہے", "ar": "هذا مثال على شكل الترجمة", "hi": "यह आपके कैप्शन का नमूना है",
               "fa": "این نمونه ای از زیرنویس شماست", "en": "This is how your captions will look"}
CAP_STYLES = {
    "Classic": dict(cp_font="Arial", cp_size=5.5, cp_color="White", cp_outline="Black", cp_hl="Yellow", cp_outw=3.0,
                    cp_box=False, cp_bold=True, cp_upper=False, cp_wordhl=False, cp_chars=28),
    "TikTok pop": dict(cp_font="Arial", cp_size=7.0, cp_color="White", cp_outline="Black", cp_hl="Yellow", cp_outw=4.0,
                       cp_box=False, cp_bold=True, cp_upper=True, cp_wordhl=True, cp_chars=18),
    "Yellow bold": dict(cp_font="Arial", cp_size=6.0, cp_color="Yellow", cp_outline="Black", cp_hl="White", cp_outw=3.5,
                        cp_box=False, cp_bold=True, cp_upper=False, cp_wordhl=False, cp_chars=24),
    "Black box": dict(cp_font="Arial", cp_size=5.0, cp_color="White", cp_outline="Black", cp_hl="Yellow", cp_outw=3.0,
                      cp_box=True, cp_bold=True, cp_upper=False, cp_wordhl=False, cp_chars=26),
    "Neon": dict(cp_font="Arial", cp_size=6.5, cp_color="Cyan", cp_outline="Purple", cp_hl="Pink", cp_outw=4.0,
                 cp_box=False, cp_bold=True, cp_upper=True, cp_wordhl=True, cp_chars=18),
    "Minimal": dict(cp_font="Segoe UI", cp_size=4.5, cp_color="White", cp_outline="Black", cp_hl="White", cp_outw=1.5,
                    cp_box=False, cp_bold=False, cp_upper=False, cp_wordhl=False, cp_chars=34),
    "Hormozi": dict(cp_font="Impact", cp_size=8.0, cp_color="White", cp_outline="Black", cp_hl="Yellow", cp_outw=5.0,
                    cp_bold=False, cp_upper=True, cp_wordhl=True, cp_chars=16, cp_anim="Pop-in"),
    "Bubble": dict(cp_font="Arial", cp_size=5.5, cp_color="Black", cp_outline="Black", cp_hl="Red", cp_outw=0.0,
                   cp_bold=True, cp_upper=False, cp_wordhl=False, cp_chars=24, cp_shape="Bubble (rounded)",
                   cp_boxcolor="White", cp_anim="Pop-in"),
    "Pill karaoke": dict(cp_font="Arial", cp_size=5.5, cp_color="White", cp_outline="Black", cp_hl="Yellow", cp_outw=0.0,
                         cp_bold=True, cp_upper=False, cp_wordhl=True, cp_chars=22, cp_shape="Bubble (rounded)",
                         cp_boxcolor="Black", cp_anim="Fade"),
    "Sticker": dict(cp_font="Arial", cp_size=6.5, cp_color="Black", cp_outline="Black", cp_hl="Black", cp_outw=0.0,
                    cp_bold=True, cp_upper=True, cp_wordhl=False, cp_chars=18, cp_shape="Sticker",
                    cp_boxcolor="White", cp_anim="Bounce"),
    "Glow neon": dict(cp_font="Arial", cp_size=6.5, cp_color="White", cp_outline="Cyan", cp_hl="Pink", cp_outw=4.0,
                      cp_bold=True, cp_upper=True, cp_wordhl=True, cp_chars=18, cp_shape="Glow", cp_anim="Fade"),
    "Cinema": dict(cp_font="Times New Roman", cp_size=4.8, cp_color="White", cp_outline="Black", cp_hl="White", cp_outw=2.0,
                   cp_bold=False, cp_upper=False, cp_wordhl=False, cp_chars=34, cp_italic=True, cp_spacing=1.0, cp_anim="Fade"),
    "Hard shadow": dict(cp_font="Arial", cp_size=6.5, cp_color="Yellow", cp_outline="Black", cp_hl="White", cp_outw=3.0,
                        cp_bold=True, cp_upper=True, cp_wordhl=False, cp_chars=20, cp_shape="Hard shadow", cp_boxcolor="Black"),
    "Reels slide": dict(cp_font="Arial", cp_size=6.5, cp_color="White", cp_outline="Black", cp_hl="Green", cp_outw=4.0,
                        cp_bold=True, cp_upper=True, cp_wordhl=True, cp_chars=18, cp_anim="Slide up"),
}
THUMB_SIZES = {"YouTube 1280x720": (1280, 720), "TikTok cover 1080x1920": (1080, 1920),
               "Square 1080x1080": (1080, 1080), "Original": None}
THUMB_FONTS = {"Impact": ["impact.ttf"], "Arial Black": ["ariblk.ttf"], "Arial Bold": ["arialbd.ttf", "Arial Bold.ttf"],
               "Segoe UI Bold": ["segoeuib.ttf"], "Verdana Bold": ["verdanab.ttf"],
               "DejaVu Bold": ["DejaVuSans-Bold.ttf"]}


# ---------- Link collector ----------
def tt_profiles(lines):
    out = []
    for ln in lines:
        m = re.search(r"tiktok\.com/@([\w.\-]+)", ln) or re.fullmatch(r"\s*@?([\w.\-]+)\s*", ln)
        if m:
            out.append(m.group(1))
    return list(dict.fromkeys(out))


def tt_date(vid):
    """TikTok video ids hold the upload time in their upper bits."""
    try:
        v = int(vid)
        return time.strftime("%Y-%m-%d", time.localtime(v >> 32)) if v > 10 ** 15 else ""
    except (ValueError, OverflowError, OSError):
        return ""


def flat_items(entries, user):
    """yt-dlp flat playlist entries -> [{id, url, kind}] (kind video/picture from the url)."""
    out = []
    for e in entries or []:
        if not e:
            continue
        u = e.get("url") or e.get("webpage_url") or ""
        m = re.search(r"/(video|photo)/(\d+)", u)
        vid = e.get("id") or (m.group(2) if m else "")
        if not str(vid).isdigit():
            continue
        kind = "picture" if (m and m.group(1) == "photo") else "video"
        if not u.startswith("http") or not m:
            u = f"https://www.tiktok.com/@{user}/video/{vid}"
        out.append({"id": int(vid), "url": u, "kind": kind})
    return {x["id"]: x for x in out}.values()


def kind_from_info(info):
    """'video' when a real video stream exists, otherwise 'picture' (photo slideshow / audio only)."""
    fm = info.get("formats") or []
    return "video" if any((f.get("vcodec") or "none") != "none" or f.get("height") for f in fm) else "picture"


# ---------- Smart clips ----------
def energy_profile(path):
    """Loudness (dB) for every second of the video, using ffmpeg only."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "8000", "-af",
           "asetnsamples=n=8000:p=0,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
           "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", **nowin())
    vals = []
    for t, v in re.findall(r"pts_time:([\d.]+)\s+lavfi\.astats\.Overall\.RMS_level=(\S+)", r.stdout):
        try:
            db = float(v)
        except ValueError:
            db = -90.0
        vals.append((float(t), max(db, -90.0)))
    if not vals:
        raise Exception("could not read the sound of this video (no audio?)")
    n = int(vals[-1][0]) + 1
    out = [-90.0] * n
    for t, db in vals:
        out[min(n - 1, int(t))] = db
    return out


def pick_highlights(db, n, length, gap):
    """Best n windows of `length` seconds (loudest on average), at least `gap` seconds apart. [(start, score)]"""
    length = int(max(3, round(length)))
    if len(db) <= length:
        return [(0, sum(db) / max(1, len(db)))]
    ps = [0.0]
    for v in db:
        ps.append(ps[-1] + max(v, -60.0))
    scores = [(ps[i + length] - ps[i]) / length for i in range(len(db) - length + 1)]
    taken, res = [], []
    for i in sorted(range(len(scores)), key=lambda k: -scores[k]):
        if len(res) >= n:
            break
        if all(abs(i - j) >= length + gap for j in taken):
            taken.append(i)
            res.append((i, scores[i]))
    return res


# ---------- Captions ----------
def ass_color(hexc, alpha=0):
    h = hexc.lstrip("#")
    return "&H%02X%s%s%s" % (alpha, h[4:6].upper(), h[2:4].upper(), h[0:2].upper())


def ass_time(t):
    cs = int(round(max(0.0, t) * 100))
    h, r = divmod(cs, 360000)
    m, r = divmod(r, 6000)
    s, c = divmod(r, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def srt_time(t):
    ms = int(round(max(0.0, t) * 1000))
    h, r = divmod(ms, 3600000)
    m, r = divmod(r, 60000)
    s, ms = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def chunk_words(words, maxc=24, maxd=4.0, gap=0.8):
    lines, cur = [], []
    for wd in words:
        t = wd["word"].strip()
        if not t:
            continue
        if cur:
            n = sum(len(x["word"]) + 1 for x in cur) + len(t)
            if n > maxc or wd["start"] - cur[-1]["end"] > gap or wd["end"] - cur[0]["start"] > maxd:
                lines.append(cur)
                cur = []
        cur.append({**wd, "word": t})
    if cur:
        lines.append(cur)
    return lines


def wrap_text(text, maxc):
    lines, cur = [], ""
    for wd in text.split():
        if cur and len(cur) + 1 + len(wd) > maxc:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    if cur:
        lines.append(cur)
    return lines


def caption_lines(segments, chars, upper=False):
    """[(start, end, [(word, start, end), ...])] — one entry per caption line shown on screen."""
    out = []
    for sg in segments:
        words = [x for x in (sg.get("words") or []) if x["word"].strip()]
        if words:
            for ln in chunk_words(words, chars):
                out.append((ln[0]["start"], ln[-1]["end"], [(x["word"], x["start"], x["end"]) for x in ln]))
        else:
            lines = wrap_text(sg["text"], chars)
            total = sum(len(x) for x in lines) or 1
            t = sg["start"]
            for ln in lines:
                d = (sg["end"] - sg["start"]) * len(ln) / total
                ws = ln.split()
                per = d / max(1, len(ws))
                out.append((t, t + d, [(w, t + i * per, t + (i + 1) * per) for i, w in enumerate(ws)]))
                t += d
    fixed = []
    for i, (a, b, ws) in enumerate(out):
        nxt = out[i + 1][0] if i + 1 < len(out) else b + 1
        b = min(max(b + 0.12, a + 0.4), max(nxt, a + 0.4))
        fixed.append((a, b, [((w.upper() if upper else w), s, e) for w, s, e in ws]))
    return fixed


def srt_text(lines):
    return "\n".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{' '.join(w for w, _, _ in ws)}\n"
                     for i, (a, b, ws) in enumerate(lines, 1))


def shift_lines(lines, delta):
    """Move caption times by delta seconds (when the start of the video is cut); lines before 0 are dropped."""
    out = []
    for a, b, ws in lines:
        if b + delta <= 0.05:
            continue
        out.append((max(0.0, a + delta), b + delta, [(w, max(0.0, s0 + delta), e0 + delta) for w, s0, e0 in ws]))
    return out


def build_ass(lines, w, h, st):
    """st: font,size,color,outline,hl,outw,box,bold,wordhl,pos,margin  (+ shape,boxcolor,anim,italic,spacing).
    size and margin are % of the video height."""
    fs = max(12, int(h * float(st["size"]) / 100))
    prim = ass_color(CAP_COLORS.get(st["color"], "#ffffff"))
    hl = ass_color(CAP_COLORS.get(st["hl"], "#ffd400"))
    outc = ass_color(CAP_COLORS.get(st["outline"], "#000000"))
    bhex = CAP_COLORS.get(st.get("boxcolor") or "Black", "#000000")
    shape = st.get("shape") or "Outline"
    if shape == "Outline" and st.get("box"):
        shape = "Box"
    ow_ = float(st["outw"])
    glow = False
    if shape == "Box":
        border, outl, shadow, outcol, back = 3, max(2, int(fs * 0.22)), 0, ass_color(bhex, 0x40), ass_color(bhex, 0x40)
    elif shape.startswith("Bubble"):  # a thick round outline in one colour merges into a rounded bubble
        border, outl, shadow, outcol, back = 1, round(fs * 0.30, 1), 0, ass_color(bhex), ass_color(bhex)
    elif shape == "Sticker":
        border, outl, shadow, outcol, back = 1, round(fs * 0.20, 1), round(fs * 0.05, 1), ass_color(bhex), ass_color("#000000", 0x80)
    elif shape == "Glow":
        border, outl, shadow, outcol, back, glow = 1, max(1.0, ow_ * 0.4), 0, outc, ass_color("#000000", 0xFF), True
    elif shape == "Hard shadow":
        border, outl, shadow, outcol, back = 1, max(1.0, ow_ * 0.7), round(max(2.0, fs * 0.08), 1), outc, ass_color(bhex)
    else:
        border, outl, shadow, outcol, back = 1, ow_, 1, outc, ass_color("#000000", 0x90)
    align = {"Bottom": 2, "Center": 5, "Top": 8}.get(st["pos"], 2)
    mv = int(h * float(st["margin"]) / 100)
    first, second = (hl, prim) if st["wordhl"] else (prim, prim)
    ital = -1 if st.get("italic") else 0
    try:
        spacing = float(st.get("spacing") or 0)
    except (TypeError, ValueError):
        spacing = 0.0
    anim = st.get("anim") or "None"
    pre = ""
    if anim == "Fade":
        pre = "{\\fad(160,120)}"
    elif anim == "Pop-in":
        pre = "{\\fscx70\\fscy70\\t(0,130,\\fscx100\\fscy100)\\fad(60,0)}"
    elif anim == "Bounce":
        pre = "{\\fscx55\\fscy55\\t(0,110,\\fscx112\\fscy112)\\t(110,200,\\fscx100\\fscy100)}"
    elif anim == "Slide up":
        x, y = w // 2, {2: h - mv, 5: h // 2, 8: mv}[align]
        pre = "{\\an%d\\move(%d,%d,%d,%d,0,170)\\fad(80,0)}" % (align, x, y + int(h * 0.03), x, y)
    head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {w}\nPlayResY: {h}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,{st['font']},{fs},{first},{second},{outcol},{back},{-1 if st['bold'] else 0},{ital},0,0,100,100,{spacing},0,"
            f"{border},{outl},{shadow},{align},{int(w * 0.06)},{int(w * 0.06)},{mv},1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    ev_lines = []
    esc = lambda s: s.replace("{", "(").replace("}", ")").replace("\\", "")
    for a, b, ws in lines:
        if st["wordhl"]:
            spent, parts = 0, []
            for wd, s0, e0 in ws:
                if wd == "\n":
                    parts.append("\\N")
                    continue
                k = max(1, int(round((e0 - a) * 100)) - spent)
                spent += k
                parts.append("{\\k%d}%s" % (k, esc(wd)))
            text = " ".join(parts)
        else:
            text = " ".join("\\N" if wd == "\n" else esc(wd) for wd, _, _ in ws)
        t0, t1 = ass_time(a), ass_time(b)
        if glow:  # soft halo behind the sharp text
            ev_lines.append(f"Dialogue: 0,{t0},{t1},Default,,0,0,0,,{pre}{{\\1a&HFF&\\bord{fs * 0.16:.1f}\\blur{fs * 0.10:.1f}}}{text}")
        ev_lines.append(f"Dialogue: {1 if glow else 0},{t0},{t1},Default,,0,0,0,,{pre}{text}")
    return head + "\n".join(ev_lines) + "\n"


def ff_path(p):
    """Escape a file path for use inside an ffmpeg filter (works on Windows too)."""
    return p.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def make_ass_file(text):
    """Write a caption file into its own tiny temp folder with a plain name (no spaces / odd characters)."""
    d = tempfile.mkdtemp(prefix="tvcap_")
    p = os.path.join(d, "cap.ass")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


def drop_ass(p):
    d = os.path.dirname(p)
    if os.path.basename(d).startswith("tvcap_"):
        shutil.rmtree(d, ignore_errors=True)
    else:
        try:
            os.remove(p)
        except OSError:
            pass


def ass_arg(p):
    """Filter argument for a file from make_ass_file(): the bare file name — run ffmpeg with cwd = its folder.
    That avoids every Windows path-escaping problem. Adds the Windows fonts folder so fonts are always found."""
    arg = f"'{os.path.basename(p)}'"
    win = os.environ.get("WINDIR", "")
    fd = os.path.join(win, "Fonts") if win else ""
    if fd and os.path.isdir(fd):
        arg += f":fontsdir='{ff_path(fd)}'"
    return arg


_ASS_OK = None


def ffmpeg_has_ass():
    global _ASS_OK
    if _ASS_OK is None:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, **nowin()).stdout
            _ASS_OK = bool(re.search(r"\bass\b", out))
        except Exception:
            _ASS_OK = True  # cannot tell -> just try
    return _ASS_OK


# ---------- Thumbnails ----------
def load_font(name, size):
    from PIL import ImageFont
    for fn in THUMB_FONTS.get(name, []) + ["DejaVuSans-Bold.ttf", "arialbd.ttf"]:
        try:
            return ImageFont.truetype(fn, size)
        except Exception:
            pass
    return ImageFont.load_default()


def grab_frame(video, t, width=0):
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.close()
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, t):.2f}", "-i", video, "-frames:v", "1"]
    if width:
        cmd += ["-vf", f"scale={width}:-2"]
    r = subprocess.run(cmd + [f.name], capture_output=True, text=True, **nowin())
    if r.returncode != 0 or not os.path.getsize(f.name):
        raise Exception("could not read a frame from this video")
    return f.name


def best_frame_time(video, dur):
    """Sharp, well-lit frame: samples 12 points and keeps the best one."""
    from PIL import Image, ImageFilter, ImageStat
    best, best_t = -1.0, min(1.0, dur / 2)
    for i in range(12):
        t = dur * (0.08 + 0.84 * i / 11)
        try:
            p = grab_frame(video, t, 320)
        except Exception:
            continue
        im = Image.open(p).convert("L")
        sharp = ImageStat.Stat(im.filter(ImageFilter.FIND_EDGES)).var[0]
        mean = ImageStat.Stat(im).mean[0]
        os.remove(p)
        score = sharp * (1.0 if 45 < mean < 200 else 0.35)
        if score > best:
            best, best_t = score, t
    return best_t


def render_thumb(frame_path, o, user=""):
    from PIL import Image, ImageDraw, ImageOps
    im = Image.open(frame_path).convert("RGB")
    size = THUMB_SIZES.get(o["th_size_preset"])
    if size:
        im = ImageOps.fit(im, size, Image.LANCZOS, centering=(0.5, 0.4))
    W, H = im.size
    pos = o["th_pos"]
    text = (o["th_text"] or "").strip()
    fs = max(16, int(H * float(o["th_size"]) / 100))
    font = load_font(o["th_font"], fs)
    d0 = ImageDraw.Draw(im)
    maxw = int(W * 0.9)
    lines, cur = [], ""
    for wd in text.split():
        trial = (cur + " " + wd).strip()
        if cur and d0.textlength(trial, font=font) > maxw:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    if cur:
        lines.append(cur)
    lh = int(fs * 1.15)
    block = lh * len(lines)
    m = int(H * 0.06)
    y0 = {"Top": m, "Center": (H - block) // 2}.get(pos, H - block - m)
    if o["th_shade"] and lines:
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        top, bot = max(0, y0 - m), min(H, y0 + block + m)
        for yy in range(top, bot):
            a = int(170 * (1 - abs((yy - (top + bot) / 2) / max(1, (bot - top) / 2)) ** 2))
            sd.line([(0, yy), (W, yy)], fill=(0, 0, 0, a))
        im = Image.alpha_composite(im.convert("RGBA"), sh).convert("RGB")
    d = ImageDraw.Draw(im)
    col = CAP_COLORS.get(o["th_color"], "#ffffff")
    stc = CAP_COLORS.get(o["th_stroke"], "#000000")
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=font)
        d.text(((W - tw) / 2, y0 + i * lh), ln, font=font, fill=col, stroke_width=int(o["th_strokew"]), stroke_fill=stc)
    if o["th_brand"] and (user or "").strip():
        png = make_badge((o["brand_text"] or "@{username}").replace("{username}", user.strip().lstrip("@")),
                         max(14, int(H * float(o["brand_size"]) / 100)), BRAND_COLORS.get(o["brand_color"], "#ffffff"),
                         float(o["brand_opacity"]), bool(o["brand_box"]))
        b = Image.open(png).convert("RGBA")
        os.remove(png)
        if b.width > W * 0.92:
            b = b.resize((int(W * 0.92), int(b.height * W * 0.92 / b.width)))
        bx = {"Bottom-left": m, "Top-left": m, "Top-center": (W - b.width) // 2, "Bottom-center": (W - b.width) // 2,
              "Center": (W - b.width) // 2}.get(o["brand_pos"], W - b.width - m)
        by = {"Top-left": m, "Top-right": m, "Top-center": m, "Center": (H - b.height) // 2}.get(o["brand_pos"], H - b.height - m)
        im.paste(b, (bx, by), b)
    if o["th_brand"] and o["logo_on"] and os.path.exists(o["logo_path"] or ""):
        lg = Image.open(o["logo_path"]).convert("RGBA")
        lw = max(16, int(W * float(o["logo_size"]) / 100))
        lg = lg.resize((lw, int(lg.height * lw / lg.width)))
        lg.putalpha(lg.getchannel("A").point(lambda v: int(v * float(o["logo_opacity"]))))
        lx = {"Top-right": W - lw - m, "Bottom-right": W - lw - m, "Top-center": (W - lw) // 2,
              "Bottom-center": (W - lw) // 2, "Center": (W - lw) // 2}.get(o["logo_pos"], m)
        ly = {"Bottom-left": H - lg.height - m, "Bottom-right": H - lg.height - m, "Bottom-center": H - lg.height - m,
              "Center": (H - lg.height) // 2}.get(o["logo_pos"], m)
        im.paste(lg, (lx, ly), lg)
    return im


# ---------- Cleaner ----------
def partial_hash(path, size, chunk=2 * 1024 * 1024):
    h = hashlib.md5()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk * 2:
            f.seek(-chunk, 2)
            h.update(f.read(chunk))
    return h.hexdigest()


def scan_videos(roots, skip=("_Cleaner_Trash",)):
    files = []
    for root in roots:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in skip]
            for fn in fns:
                if fn.lower().endswith(VIDEO_EXT):
                    p = os.path.join(dp, fn)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    files.append({"path": p, "size": st.st_size, "mtime": st.st_mtime})
    return files


def find_clean(files, dups=True, old_days=0, big_mb=0, now=None):
    now = now or time.time()
    why = {}
    if dups:
        by = {}
        for f in files:
            if f["size"]:
                by.setdefault(f["size"], []).append(f)
        for size, grp in by.items():
            if len(grp) < 2:
                continue
            hs = {}
            for f in grp:
                try:
                    hs.setdefault(partial_hash(f["path"], size), []).append(f)
                except OSError:
                    pass
            for g in hs.values():
                if len(g) > 1:
                    g.sort(key=lambda x: x["mtime"])
                    for f in g[1:]:
                        why.setdefault(f["path"], []).append("Duplicate of " + os.path.basename(g[0]["path"])[:36])
    for f in files:
        if old_days and (now - f["mtime"]) / 86400 >= old_days:
            why.setdefault(f["path"], []).append(f"Older than {old_days} days")
        if big_mb and f["size"] >= big_mb * 1024 * 1024:
            why.setdefault(f["path"], []).append(f"Bigger than {big_mb} MB")
    return [{**f, "why": why[f["path"]], "age": int((now - f["mtime"]) / 86400)} for f in files if f["path"] in why]


class Nav(tk.Canvas):  # sidebar menu item: icon badge + label on a soft pill
    def __init__(self, parent, app, key, text):
        super().__init__(parent, height=42, highlightthickness=0, bd=0, cursor="hand2")
        self.app, self.key = app, key
        parts = text.split(None, 1)
        self.icon, self.text = (parts[0], parts[1]) if len(parts) == 2 else ("•", text)
        self.font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.ifont = tkfont.Font(family="Segoe UI Emoji", size=11)
        self.bind("<Button-1>", lambda e: app.show_page(key))
        self.bind("<Enter>", lambda e: app.set_hov(key))
        self.bind("<Leave>", lambda e: app.set_hov(None))
        self.bind("<Configure>", lambda e: self.draw())
        app.btns.append(self)

    def draw(self):
        a, p = self.app, self.app.pal
        act, hv = self.key == a.page_key, self.key == a.nav_hov
        w, h = self.winfo_width(), self.winfo_height()
        self.configure(bg=p["side"])
        self.delete("all")
        if w < 20:
            return
        if act:
            rrect(self, 6, 2, w - 6, h - 2, 14, fill=mix(p["side"], p["accent"], 0.24), outline="")
            rrect(self, 6, 10, 10, h - 10, 2, fill=p["accent2"], outline="")
        elif hv:
            rrect(self, 6, 2, w - 6, h - 2, 14, fill=p["card"], outline="")
        rrect(self, 20, 7, 48, h - 7, 9, fill=p["accent"] if act else p["card2"], outline="")
        self.create_text(34, h // 2, text=self.icon, fill="#ffffff" if act else p["text"], font=self.ifont)
        self.create_text(60, h // 2, text=self.text, anchor="w", fill=p["text"] if (act or hv) else p["muted"],
                         font=self.font)


class EditQueue(ttk.Frame):  # list of video files + buttons; used by the Brand, Editing and Captions tabs
    def __init__(self, parent, app, parts, run_text, rows=8, action=None, preview=True, show_user=True):
        super().__init__(parent, style="Card.TFrame")
        self.app, self.parts, self.items, self.action, self.show_user = app, parts, [], action, show_user
        self.run_text = run_text
        L, B = app.L, app.btn
        L(self, "QUEUE  (any video files)", "a").pack(anchor="w")
        cols = [("file", "File", 190 if not show_user else 170, "w")]
        if show_user:
            cols.append(("user", "Username", 100, "w"))
        cols.append(("status", "Status", 100, "center"))
        wrap, self.tree = app.make_tree(self, tuple(cols), height=rows)
        wrap.pack(fill="both", expand=True, pady=6)
        self.msg = L(self, "", "m")
        self.msg.configure(wraplength=360)
        self.msg.pack(anchor="w")
        if show_user:
            self.tree.bind("<Double-1>", self.edit_user)
            urow = app.F(self)
            urow.pack(fill="x", pady=(6, 0))
            L(urow, "Username for added files:").pack(side="left")
            ttk.Entry(urow, textvariable=app.ev["manual_user"], width=14).pack(side="left", padx=6)
        b1 = app.F(self)
        b1.pack(fill="x", pady=6)
        B(b1, "➕ Videos", self.add_files, size=9, padx=10, pady=5).pack(side="left")
        B(b1, "📁 Folder", self.add_folder, size=9, padx=10, pady=5).pack(side="left", padx=4)
        B(b1, "⬇ Downloaded", self.add_history, size=9, padx=10, pady=5).pack(side="left")
        B(b1, "🗑", self.clear, size=9, padx=10, pady=5).pack(side="right")
        b2 = app.F(self)
        b2.pack(fill="x")
        if preview:
            B(b2, "🔍 Preview", self.preview, size=9, padx=12, pady=6).pack(side="left")
        B(b2, run_text, lambda: app.thread(self.run), "primary").pack(side="right")

    def note(self, text, err=False):
        self.app.after(0, lambda: self.msg.configure(text=text, foreground=self.app.pal["err" if err else "accent2"]))

    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        for i, it in enumerate(self.items):
            vals = [os.path.basename(it["path"])]
            if self.show_user:
                vals.append(("@" + it["user"]) if it["user"] else "—")
            self.tree.insert("", "end", iid=str(i), values=tuple(vals + [it["status"]]))

    def _add(self, paths, users=None):
        have = {it["path"] for it in self.items}
        user0 = self.app.ev["manual_user"].get().strip().lstrip("@")
        added = 0
        for i, p in enumerate(paths):
            if p in have or not p.lower().endswith(VIDEO_EXT):
                continue
            self.items.append({"path": p, "user": users[i] if users else user0, "status": "Waiting"})
            have.add(p)
            added += 1
        self.refresh()
        self.note(f"{added} video(s) added — {len(self.items)} in the queue. Press  {self.run_text}  to start."
                  if added else "No new videos were added.")

    def add_files(self):
        self._add(list(filedialog.askopenfilenames(filetypes=[("Videos", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v")])))

    def add_folder(self):
        d = filedialog.askdirectory()
        if not d:
            return
        ps = []
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs if x not in ("Edited", "Captioned")]
            ps += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(VIDEO_EXT)]
        self._add(ps)

    def add_history(self):
        ps, us = [], []
        for r in self.app.records:
            f = r.get("file")
            if r.get("status") == "done" and f and os.path.exists(f) and f.lower().endswith(VIDEO_EXT):
                u = r.get("pname") or r.get("uploader") or ""
                ps.append(f)
                us.append("" if u.startswith("(") else u)
        self._add(ps, us)

    def clear(self):
        self.items.clear()
        self.refresh()
        self.note("Queue cleared.")

    def edit_user(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        it = self.items[int(iid)]
        u = simpledialog.askstring("Username", "Username for the brand overlay of this video:",
                                   initialvalue=it["user"], parent=self.app)
        if u is not None:
            it["user"] = u.strip().lstrip("@")
            self.refresh()

    def preview(self):
        a, sel = self.app, self.tree.selection()
        it = self.items[int(sel[0])] if sel else (self.items[0] if self.items else None)
        if not it:
            self.note("Add a video to the queue first.", True)
            return
        user = it["user"] or a.ev["manual_user"].get().strip().lstrip("@") or "username"

        def go():
            try:
                a.after(0, a.show_preview, a.edit_file(it["path"], user, preview=True, parts=self.parts))
            except Exception as e:
                self.note(f"Preview failed: {str(e)[:110]}", True)
        a.thread(go)

    def run(self):
        a = self.app
        if a.edit_busy:
            self.note("Another editing job is already running — wait for it or press Stop.", True)
            return
        items = [it for it in self.items if it["status"] != "✔ Done"]
        if not items:
            self.note("Nothing to do — add videos to the queue first.", True)
            return
        a.edit_busy, a.stop_flag = True, False
        a.after(0, a.set_bar, 0, len(items), "Working")
        a.after(0, a.set_state, "● Working on videos...")
        ok = bad = 0
        try:
            for n, it in enumerate(items, 1):
                if a.stop_flag:
                    break
                it["status"] = "Working..."
                a.after(0, self.refresh)
                self.note(f"Working {n}/{len(items)}:  {os.path.basename(it['path'])}")
                try:
                    out = self.action(it["path"], it["user"]) if self.action else a.edit_file(it["path"], it["user"], parts=self.parts)
                    it["status"] = "✔ Done" if out else "Nothing to do"
                    a.say(f"🎨 {os.path.basename(out or it['path'])}"
                          + (f"  (brand @{it['user']})" if out and it["user"] and "brand" in self.parts else ""), "ok")
                    ok += 1
                except Exception as e:
                    it["status"] = "✖ Failed"
                    bad += 1
                    if not a.stop_flag:
                        a.say(f"Failed ({os.path.basename(it['path'])}): {str(e)[:140]}", "err")
                        self.note(f"✖ {os.path.basename(it['path'])}: {str(e)[:150]}", True)
                        if self.action:
                            a.after(0, a.cap_fail_popup, str(e))
                a.after(0, self.refresh)
                a.after(0, a.tick)
        finally:
            a.edit_busy = False
            a.after(0, a.set_state, "● Idle")
            a.after(0, a.set_bar, None, None, "Stopped" if a.stop_flag else "Complete ✔")
            if not a.stop_flag:
                if not bad:
                    self.note(f"✔ Finished — {ok} done.")
                a.after(0, a.notify, f"Finished!\n\nDone: {ok}\nFailed: {bad}")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        ZOOM = 0.8  # full zoom-out: smaller = more zoomed out (0.7 / 0.8 / 0.9)
        self.tk.call("tk", "scaling", float(self.tk.call("tk", "scaling")) * ZOOM)
        self.title(f"{APP} {VER} — {TAGLINE}")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        ww, wh = min(1220, sw - 40), min(880, sh - 90)  # never bigger than the screen
        self.geometry(f"{ww}x{wh}+{(sw - ww) // 2}+{max(0, (sh - wh) // 2 - 20)}")
        self.minsize(min(1080, ww), min(760, wh))
        self.q = queue.Queue()
        self.loglines = collections.deque(maxlen=400)  # last messages
        self._cap_warned = False
        self.lock, self.clock = threading.Lock(), threading.Lock()
        self.creds = self.docs_api = self.drive_api = None
        self.docs = {}
        self.sel = {}
        self.row_order = []
        # settings variables
        self.links_n = tk.IntVar(value=3)
        self.quality = tk.StringVar(value="Best")
        self.fmt = tk.StringVar(value="MP4 Video")
        self.par = tk.IntVar(value=3)
        self.retry = tk.IntVar(value=2)
        self.skipdup = tk.BooleanVar(value=True)
        self.folder = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Downloads", "TikTok Downloads"))
        self.proxy = tk.StringVar(value="")
        self.cookies = tk.StringVar(value="")
        self.cbrowser = tk.StringVar(value="None")
        self.sound = tk.BooleanVar(value=True)
        self.popup = tk.BooleanVar(value=True)
        self.sched_on = tk.BooleanVar(value=False)
        self.sched_time = tk.StringVar(value="02:00")
        self.theme = tk.StringVar(value="Midnight")
        self.order = tk.StringVar(value="Top first")
        self.rate = tk.IntVar(value=0)
        self.after_act = tk.StringVar(value="Do nothing")
        self.clip_on = tk.BooleanVar(value=False)
        self.take, self.last_clip, self._disk_n = {}, "", 0
        self.cfg = {"links": self.links_n, "quality": self.quality, "fmt": self.fmt, "par": self.par,
                    "retry": self.retry, "skipdup": self.skipdup, "folder": self.folder, "proxy": self.proxy,
                    "cookies": self.cookies, "cbrowser": self.cbrowser, "sound": self.sound, "popup": self.popup,
                    "sched_on": self.sched_on, "sched_time": self.sched_time, "theme": self.theme,
                    "order": self.order, "rate": self.rate, "after_act": self.after_act, "clip_on": self.clip_on}
        self.ev = {
            "auto": tk.BooleanVar(value=False),
            "brand_on": tk.BooleanVar(value=True), "brand_text": tk.StringVar(value="@{username}"),
            "brand_pos": tk.StringVar(value="Bottom-right"), "brand_size": tk.DoubleVar(value=3.5),
            "brand_color": tk.StringVar(value="White"), "brand_opacity": tk.DoubleVar(value=0.85),
            "brand_box": tk.BooleanVar(value=True),
            "logo_on": tk.BooleanVar(value=False), "logo_path": tk.StringVar(value=""),
            "logo_pos": tk.StringVar(value="Top-left"), "logo_size": tk.DoubleVar(value=14.0),
            "logo_opacity": tk.DoubleVar(value=0.9),
            "aspect": tk.StringVar(value="Original"), "maxh": tk.StringVar(value="Original"),
            "trim_start": tk.DoubleVar(value=0.0), "trim_end": tk.DoubleVar(value=0.0),
            "mute": tk.BooleanVar(value=False),
            "auto_brand": tk.BooleanVar(value=False), "auto_cap": tk.BooleanVar(value=False), "keep_orig": tk.BooleanVar(value=False), "brand_prev_user": tk.StringVar(value="thewizardliz"),
            "brand_sample": tk.StringVar(value=""),
            "clip_src": tk.StringVar(value=""), "clip_min": tk.DoubleVar(value=1.0),
            "clip_minlast": tk.DoubleVar(value=10.0), "clip_overlap": tk.DoubleVar(value=0.0),
            "clip_vertical": tk.BooleanVar(value=True), "clip_fade": tk.BooleanVar(value=True),
            "clip_part": tk.BooleanVar(value=True), "clip_part_text": tk.StringVar(value="Part {n}"),
            "clip_part_pos": tk.StringVar(value="Top-center"), "clip_part_size": tk.DoubleVar(value=5.0),
            "clip_part_color": tk.StringVar(value="Yellow"), "clip_brand": tk.BooleanVar(value=False),
            "clip_user": tk.StringVar(value=""),
            "lc_order": tk.StringVar(value="Latest → Oldest"), "lc_max": tk.IntVar(value=0),
            "lc_deep": tk.BooleanVar(value=True), "lc_skipdone": tk.BooleanVar(value=True),
            "lc_doc": tk.StringVar(value="➕ New Google Doc"),
            "sc_src": tk.StringVar(value=""), "sc_count": tk.IntVar(value=5), "sc_len": tk.DoubleVar(value=30.0),
            "sc_gap": tk.DoubleVar(value=20.0), "sc_rank": tk.StringVar(value="Best first"),
            "sc_label": tk.BooleanVar(value=True), "sc_label_text": tk.StringVar(value="Highlight {n}"),
            "cp_lang": tk.StringVar(value="Auto-detect"), "cp_model": tk.StringVar(value="base"),
            "cp_font": tk.StringVar(value="Arial"), "cp_size": tk.DoubleVar(value=7.0),
            "cp_color": tk.StringVar(value="White"), "cp_outline": tk.StringVar(value="Black"),
            "cp_hl": tk.StringVar(value="Yellow"), "cp_outw": tk.DoubleVar(value=4.0),
            "cp_box": tk.BooleanVar(value=False), "cp_bold": tk.BooleanVar(value=True),
            "cp_upper": tk.BooleanVar(value=True), "cp_wordhl": tk.BooleanVar(value=True),
            "cp_chars": tk.IntVar(value=18), "cp_pos": tk.StringVar(value="Bottom"), "cp_margin": tk.DoubleVar(value=14.0),
            "cp_burn": tk.BooleanVar(value=True), "cp_srt": tk.BooleanVar(value=True),
            "cp_shape": tk.StringVar(value="Outline"), "cp_boxcolor": tk.StringVar(value="Black"),
            "cp_anim": tk.StringVar(value="None"), "cp_italic": tk.BooleanVar(value=False),
            "cp_spacing": tk.DoubleVar(value=0.0), "cp_translate": tk.BooleanVar(value=True), "cp_speed": tk.StringVar(value="Fast"), "cp_skip": tk.BooleanVar(value=True),
            "cp_mirror": tk.BooleanVar(value=False), "cp_modeldir": tk.StringVar(value=""), "cp_pyexe": tk.StringVar(value=""),
            "th_src": tk.StringVar(value=""), "th_time": tk.DoubleVar(value=1.0),
            "th_text": tk.StringVar(value="WATCH THIS!"), "th_font": tk.StringVar(value="Impact"),
            "th_size": tk.DoubleVar(value=11.0), "th_color": tk.StringVar(value="Yellow"),
            "th_stroke": tk.StringVar(value="Black"), "th_strokew": tk.IntVar(value=6),
            "th_pos": tk.StringVar(value="Bottom"), "th_shade": tk.BooleanVar(value=True),
            "th_size_preset": tk.StringVar(value="YouTube 1280x720"), "th_brand": tk.BooleanVar(value=False),
            "th_user": tk.StringVar(value=""), "th_fmt": tk.StringVar(value="PNG"),
            "cl_dups": tk.BooleanVar(value=True), "cl_old": tk.IntVar(value=0), "cl_big": tk.IntVar(value=0),
            "cl_safe": tk.BooleanVar(value=True),
            "crf": tk.StringVar(value="Balanced (CRF 23)"), "outmode": tk.StringVar(value="Replace original"),
            "manual_user": tk.StringVar(value=""),
        }
        for k, v in self.ev.items():
            self.cfg["e_" + k] = v
        self.procs, self.edit_busy, self.clip_busy = set(), False, False
        self.clip_dur, self.clip_outdir = 0.0, ""
        self._bp_after, self._bp_busy, self._bp_again, self._sample = None, False, False, None
        self.lc_busy, self.lc_res, self.lc_doc_map, self.cl_items, self.cl_sel = False, [], {}, [], {}
        self._wm, self._cp_after, self._cp_busy, self._cp_again = None, None, False, False
        self.wlock = threading.Lock()
        self._th_after, self._th_busy, self._th_again, self._th_cache, self._save_after = None, False, False, {}, None
        self.last_page = tk.StringVar(value="dl")
        self.mig = tk.BooleanVar(value=False)
        self.cfg["outmode_v2"] = self.mig
        self.cfg["last_page"] = self.last_page
        self.geom, self.ver_seen = tk.StringVar(value=""), tk.StringVar(value="")
        self.cfg["geom"], self.cfg["ver_seen"] = self.geom, self.ver_seen
        self.topmost = tk.BooleanVar(value=False)
        self.cfg["topmost"] = self.topmost
        self.defaults = {k: v.get() for k, v in self.cfg.items()}
        self.pause_ev = threading.Event()
        self.pause_ev.set()
        self.speeds, self.extra, self.t0 = {}, "", 0
        self.running = self.stop_flag = False
        self.pdone = self.ptotal = 0
        self.bar_text = ""
        self.last_sched = ""
        self.hist = jload(HISTORY, {})
        self.records = jload(RECORDS, [])
        self.done_ids = {vid_id(r["url"]) for r in self.records if r.get("status") == "done"}
        for k, v in jload(SETTINGS, {}).items():
            if k in self.cfg:
                try:
                    self.cfg[k].set(v)
                except Exception:
                    pass
        if not self.mig.get():  # one-time: edited videos now replace the originals (no separate Edited folder)
            self.ev["outmode"].set("Replace original")
            self.mig.set(True)
        self.upgraded = self.ver_seen.get() != VER  # first start of a new version -> new default look + What's new
        if self.ver_seen.get() == "" or self.theme.get() not in THEMES:  # only the very first start of v5+ picks the new default look
            self.theme.set(DEFAULT_THEME)
        self.restore_geometry()
        if self.ver_seen.get() == "v6.0.0":  # v6.1: Urdu etc. -> English automatically
            self.ev["cp_translate"].set(True)
            self.ev["cp_lang"].set("Auto-detect")
        self.email = self.read_active()
        self.uname = ""
        self.build_ui()
        self.apply_theme()
        self.refresh_history()
        self.load_cache()
        self.try_saved_login()
        self.after(100, self.pump)
        self.after(500, self.speed_loop)
        self.after(20000, self.sched_loop)
        self.after(2500, self.check_resume)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind_all("<MouseWheel>", self._wheel)
        self.bind_all("<F5>", lambda e: self.thread(self.load_docs))
        self.bind_all("<Control-r>", lambda e: self.thread(self.run))
        self.bind_all("<Control-p>", lambda e: self.toggle_pause())
        self.bind_all("<Control-period>", lambda e: self.stop())
        self.bind_all("<F1>", self.palette)
        self.bind_all("<F2>", self.toggle_sidebar)
        self.bind_all("<F3>", lambda e: self.next_theme())
        for _i, (_k, _t) in enumerate(NAV[:9]):
            self.bind_all(f"<Control-Key-{_i + 1}>", lambda e, k=_k: self.show_page(k))
        self.after(900, self.check_whats_new)
        self.topmost.trace_add("write", lambda *a: self.apply_topmost())
        self.apply_topmost()
        self.after(1000, self.clip_loop)
        self.update_disk()
        self.clip_estimate()
        self.lc_docs_refresh()
        self.after(1500, self.bp_update_now)
        self.after(2200, self.cp_update_now)
        for v in self.cfg.values():
            v.trace_add("write", lambda *a: self.save_later())
        if not shutil.which("ffmpeg"):
            self.say("ffmpeg not found — MP3 and 4K merging need it (run: winget install ffmpeg)", "err")

    # ---------- UI ----------
    def R(self, w, **spec):
        self.reg.append((w, spec))
        return w

    def F(self, parent, **kw):
        return ttk.Frame(parent, style="Card.TFrame", **kw)

    def L(self, parent, text, kind=""):
        return ttk.Label(parent, text=text, style={"": "C.TLabel", "a": "CA.TLabel", "m": "CM.TLabel"}[kind])

    def btn(self, parent, text, cmd, kind="ghost", surface="card", **kw):
        return Btn(parent, self, text, cmd, kind, surface, **kw)

    def col(self, parent):  # a plain column (the whole page scrolls, not each column)
        f = ttk.Frame(parent, style="Card.TFrame")
        f.inner = f
        return f

    def wrap_notes(self, w):  # long grey hints wrap inside their box instead of being cut off
        for c in w.winfo_children():
            if isinstance(c, ttk.Label) and str(c.cget("style")) == "CM.TLabel" and isinstance(c.master, ttk.LabelFrame):
                c.configure(wraplength=340)
            self.wrap_notes(c)

    def tree_menu(self, t):  # right-click / Ctrl+C copy for every list
        st = {"iid": None, "col": "#1"}
        m = tk.Menu(t, tearoff=0)

        def lines(ids):
            out = []
            for i in ids:
                vals = [str(v) for v in t.item(i)["values"]]
                out.append(next((v for v in vals if v.startswith("http")), None) or "\t".join(vals))
            return out

        def put(ls):
            if ls:
                self.clipboard_clear()
                self.clipboard_append("\n".join(ls))
                self.say(f"📋 Copied {len(ls)} item(s)", "ok")

        def cell():
            if st["iid"]:
                vals = [str(v) for v in t.item(st["iid"])["values"]]
                try:
                    idx = int(st["col"].lstrip("#")) - 1
                except ValueError:
                    idx = 0
                put([next((v for v in vals if v.startswith("http")), None) or vals[min(idx, len(vals) - 1)]])

        m.add_command(label="Copy link / cell", command=cell)
        m.add_command(label="Copy selected rows", command=lambda: put(lines(t.selection())))
        m.add_command(label="Copy all rows", command=lambda: put(lines(t.get_children())))
        m.add_command(label="Select all", command=lambda: t.selection_set(t.get_children()))

        def pop(e):
            iid = t.identify_row(e.y)
            if iid and iid not in t.selection():
                t.selection_set(iid)
            st["iid"], st["col"] = iid, t.identify_column(e.x)
            m.tk_popup(e.x_root, e.y_root)
        t.bind("<Button-3>", pop)
        t.bind("<Control-c>", lambda e: put(lines(t.selection())))
        t.bind("<Control-a>", lambda e: (t.selection_set(t.get_children()), "break")[1])

    def lc_copy_sel(self):
        urls = [self.lc_res[int(i)]["url"] for i in self.lctree.selection() if int(i) < len(self.lc_res)]
        if not urls:
            self.say("Select some rows first (click, or Ctrl+click / Shift+click), then Copy selected", "err")
            return
        self.clipboard_clear()
        self.clipboard_append("\n".join(urls))
        self.say(f"{len(urls)} link(s) copied", "ok")

    def make_tree(self, parent, cols, **kw):
        wrap = self.F(parent)
        t = ttk.Treeview(wrap, columns=[c[0] for c in cols], show="headings", **kw)
        for c, h, w, an in cols:
            t.heading(c, text=h)
            t.column(c, width=w, minwidth=w, anchor=an)  # minwidth -> columns never squash, scroll instead
        sb = ttk.Scrollbar(wrap, orient="vertical", command=t.yview)
        hb = ttk.Scrollbar(wrap, orient="horizontal", command=t.xview)
        t.configure(yscrollcommand=sb.set, xscrollcommand=hb.set)
        t.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        hb.grid(row=1, column=0, sticky="ew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        self.tree_menu(t)
        return wrap, t

    def nav_item(self, parent, key, text):
        n = Nav(parent, self, key, text)
        n.pack(fill="x", padx=6, pady=1)
        self.nav[key] = n

    def show_about(self):
        if getattr(self, "about_win", None) is not None and self.about_win.winfo_exists():
            self.about_win.lift()
            return
        p = self.pal
        w = self.about_win = tk.Toplevel(self)
        w.title(f"About — {APP}")
        w.configure(bg=p["bg"])
        w.resizable(False, False)
        w.transient(self)
        card = tk.Frame(w, bg=p["card"], padx=34, pady=26, highlightthickness=2, highlightbackground=p["accent"])
        card.pack(padx=18, pady=18)
        top = tk.Frame(card, bg="#000000")
        top.pack(fill="x")
        Avatar(top, self, 240, "#000000", animate=True, click=False, shape="square").pack(pady=16)
        kw = dict(bg=p["card"])
        tk.Label(card, text=DEV_NAME, font=("Segoe UI", 22, "bold"), fg=p["text"], **kw).pack(pady=(14, 0))
        tk.Label(card, text=f"Developer of {APP}", font=("Segoe UI", 11), fg=p["muted"], **kw).pack()
        tk.Label(card, text=f"📞  {DEV_PHONE}", font=("Segoe UI", 17, "bold"), fg=p["accent2"], **kw).pack(pady=(10, 14))
        row = tk.Frame(card, **kw)
        row.pack()
        Btn(row, self, "📋 Copy number", self.copy_phone, "ghost").pack(side="left", padx=4)
        Btn(row, self, "💬 WhatsApp", self.open_whatsapp, "success").pack(side="left", padx=4)
        Btn(row, self, "🖼 Change photo", self.change_photo, "ghost").pack(side="left", padx=4)
        tk.Label(card, text=f"{APP} {VER}  •  {TAGLINE}", font=("Segoe UI", 10), fg=p["muted"], **kw).pack(pady=(16, 0))
        tk.Label(card, text="F1 Search  •  F2 Sidebar  •  F3 Theme  •  Ctrl+1…9 Tabs  •  F5 Load  •  Ctrl+R Run  •  Ctrl+P Pause  •  Ctrl+. Stop",
                 font=("Segoe UI", 9), fg=p["muted"], **kw).pack(pady=(4, 0))

    def copy_phone(self):
        self.clipboard_clear()
        self.clipboard_append(DEV_PHONE)
        self.say(f"Number copied: {DEV_PHONE}", "ok")

    def open_whatsapp(self):
        webbrowser.open("https://wa.me/92" + DEV_PHONE.lstrip("0"))

    def change_photo(self):
        f = filedialog.askopenfilename(filetypes=[("Pictures", "*.png *.jpg *.jpeg *.webp")])
        if not f:
            return
        for n in PHOTO_NAMES:
            if n.startswith("profile") and os.path.exists(os.path.join(APPDIR, n)):
                os.remove(os.path.join(APPDIR, n))
        ext = os.path.splitext(f)[1].lower()
        try:
            from PIL import Image
            Image.open(f).convert("RGB").save(os.path.join(APPDIR, "profile.png"))
        except Exception:
            shutil.copyfile(f, os.path.join(APPDIR, "profile" + (ext if ext in (".png", ".jpg", ".jpeg", ".webp") else ".png")))
        for b in list(self.btns):
            if isinstance(b, Avatar):
                try:
                    b.reload()
                except tk.TclError:
                    pass
        self.say("Profile photo updated ✔", "ok")

    def paint_nav(self):
        for n in self.nav.values():
            n.draw()

    def show_page(self, key):
        for pg in self.pages.values():
            pg.pack_forget()
        self.pages[key].pack(fill="both", expand=True)
        if key in STAT_PAGES:
            self.stats_row.pack(fill="x", padx=20, pady=(4, 8), before=self.bar_row)
        else:
            self.stats_row.pack_forget()
        self.page_key = key
        self.last_page.set(key)
        self.page_title.config(text=PAGE_TITLES[key])
        self.paint_nav()

    def build_ui(self):
        self.reg, self.btns, self.nav, self.pages = [], [], {}, {}
        for cls in ("TEntry", "TCombobox", "TSpinbox"):
            self.option_add(f"*{cls}.font", "{Segoe UI} 11")
        self.option_add("*TCombobox*Listbox.font", "{Segoe UI} 11")
        self.pal = THEMES[self.theme.get()]
        self.page_key, self.nav_hov = "dl", None
        R, F, L, B = self.R, self.F, self.L, self.btn

        foot = R(tk.Frame(self), bg="side")
        foot.pack(side="bottom", fill="x")
        R(tk.Label(foot, text=CREDIT, font=("Segoe UI", 10, "bold"), pady=7), bg="side", fg="accent2").pack()

        # ----- sidebar -----
        sw = int(250 * max(1.0, self.winfo_fpixels("1i") / 96))  # grows with Windows zoom so text never cuts
        side = self.side = R(tk.Frame(self, width=sw), bg="side")
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        pic = R(tk.Frame(side), bg="#000000")  # black card: photo fills a rounded square, name below
        pic.pack(fill="x", padx=12, pady=(10, 6))
        Avatar(pic, self, min(sw - 60, 160), "#000000", animate=True, shape="square").pack(pady=(10, 4))
        R(tk.Label(pic, text=DEV_NAME, font=("Segoe UI", 13, "bold")), bg="#000000", fg="#ffffff").pack()
        R(tk.Label(pic, text=f"Developer of {APP}", font=("Segoe UI", 9), pady=0), bg="#000000", fg="#9aa4c4").pack(pady=(0, 8))
        box = R(tk.Frame(side, padx=10, pady=8, highlightthickness=2), bg="card2", highlightbackground="accent")  # info box
        box.pack(side="bottom", fill="x", padx=12, pady=10)
        top = R(tk.Frame(box), bg="card2")
        top.pack(fill="x")
        R(tk.Label(top, text="⬇", font=("Segoe UI", 15, "bold"), width=3), bg="accent", fg="#ffffff").pack(side="left")
        nm = R(tk.Frame(top), bg="card2")
        nm.pack(side="left", padx=10)
        R(tk.Label(nm, text=APP, font=("Segoe UI", 12, "bold"), anchor="w"), bg="card2", fg="text").pack(anchor="w")
        R(tk.Label(nm, text=VER, font=("Segoe UI", 9), anchor="w"), bg="card2", fg="muted").pack(anchor="w")
        R(tk.Frame(box, height=1), bg="line").pack(fill="x", pady=6)
        R(tk.Label(box, text=f"Developed by {DEV_NAME}", font=("Segoe UI", 9, "bold"), anchor="w"),
          bg="card2", fg="text").pack(fill="x")
        R(tk.Label(box, text=f"📞  {DEV_PHONE}", font=("Segoe UI", 10, "bold"), anchor="w"),
          bg="card2", fg="accent2").pack(fill="x", pady=(2, 0))
        B(box, "✨ What's new   •   F1 Search", self.show_whats_new, "ghost", "card2", size=9, padx=10, pady=4).pack(fill="x", pady=(8, 0))
        navf = ScrollFrame(side, self, style="Side.TFrame", bgkey="side")  # menu scrolls when there are many tabs
        navf.pack(fill="both", expand=True)
        for gname, items in NAV_GROUPS:
            R(tk.Label(navf.inner, text=gname, font=("Segoe UI", 8, "bold"), anchor="w"), bg="side", fg="muted").pack(
                fill="x", padx=20, pady=(12, 2))
            for key, text in items:
                self.nav_item(navf.inner, key, text)

        # ----- main area -----
        main = self.main = R(tk.Frame(self), bg="bg")
        main.pack(side="left", fill="both", expand=True)
        hd = R(tk.Frame(main), bg="bg")
        hd.pack(fill="x", padx=20, pady=(16, 6))
        self.page_title = R(tk.Label(hd, text="Downloader", font=("Segoe UI", 21, "bold"), anchor="w"), bg="bg", fg="text")
        self.page_title.pack(side="left")
        self.chip = R(tk.Label(hd, text="👤 Not logged in", font=("Segoe UI", 10, "bold"), padx=14, pady=6),
                      bg="card", fg="accent2")
        self.chip.pack(side="right")
        self.disk_lbl = R(tk.Label(hd, text="", font=("Segoe UI", 10, "bold"), padx=10), bg="bg", fg="muted")
        self.disk_lbl.pack(side="right")
        B(hd, "🔍  Search  F1", self.palette, "ghost", "bg", size=9, padx=12, pady=5).pack(side="right", padx=8)

        sr = R(tk.Frame(main), bg="bg")
        sr.pack(fill="x", padx=20, pady=(4, 8))
        self.stats_row = sr
        self.stats = {}
        for key, cap, col in (("docs", "DOCS", "accent"), ("prof", "PROFILES", "accent2"),
                              ("links", "TOTAL LINKS", "ok"), ("sel", "SELECTED", "warn"),
                              ("today", "DOWNLOADED TODAY", "accent")):
            c = R(tk.Frame(sr, highlightthickness=1), bg="card", highlightbackground="line")
            c.pack(side="left", fill="x", expand=True, padx=(0, 10))
            R(tk.Frame(c, height=3), bg=col).pack(side="top", fill="x")
            ct = R(tk.Frame(c), bg="card")
            ct.pack(side="top", fill="x", expand=True)
            self.stats[key] = R(tk.Label(ct, text="0", font=("Segoe UI", 22, "bold"), anchor="w", padx=12),
                                bg="card", fg="text")
            self.stats[key].pack(fill="x", pady=(6, 0))
            R(tk.Label(ct, text=cap, font=("Segoe UI", 8, "bold"), anchor="w", padx=12), bg="card", fg="muted").pack(fill="x", pady=(0, 8))

        bar = R(tk.Frame(main), bg="bg")
        bar.pack(fill="x", padx=20, pady=(0, 8))
        self.bar_row = bar
        self.state_lbl = R(tk.Label(bar, text="● Idle", font=("Segoe UI", 10, "bold")), bg="bg", fg="accent2")
        self.state_lbl.pack(side="left")
        B(bar, "■ Stop", self.stop, "danger", "bg").pack(side="right")
        self.pause_btn = B(bar, "⏸ Pause", self.toggle_pause, "ghost", "bg")
        self.pause_btn.pack(side="right", padx=8)
        self.run_btn = B(bar, "▶  RUN DOCS", lambda: self.thread(self.run), "primary", "bg", size=11, padx=24, pady=9)
        self.run_btn.pack(side="right")
        B(bar, "⟳ Load Docs", lambda: self.thread(self.load_docs), "ghost", "bg").pack(side="right", padx=8)
        B(bar, "↻ Resume last", lambda: self.thread(self.resume), "ghost", "bg").pack(side="right")
        B(bar, "📂 Folder", self.open_folder, "ghost", "bg").pack(side="right", padx=8)
        self.pbar = tk.Canvas(main, height=30, highlightthickness=0, bd=0)
        self.pbar.pack(fill="x", padx=20, pady=(0, 10))
        self.pbar.bind("<Configure>", lambda e: self.set_bar())
        self.msg_lbl = R(tk.Label(main, text="", font=("Segoe UI", 10, "bold"), anchor="w"), bg="bg", fg="muted")
        self.msg_lbl.pack(fill="x", padx=22, pady=(0, 6))

        host = R(tk.Frame(main), bg="bg")
        host.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        self.pbody = {}
        for key in PAGE_TITLES:
            if key in SCROLL_PAGES:
                sf = ScrollFrame(host, self)
                self.pages[key] = sf
                self.pbody[key] = ttk.Frame(sf.inner, style="Card.TFrame", padding=16)
                self.pbody[key].pack(fill="both", expand=True)
            else:
                self.pages[key] = self.pbody[key] = ttk.Frame(host, style="Card.TFrame", padding=16)

        # ----- Downloader -----
        t = self.pbody["dl"]
        row = F(t)
        row.pack(fill="x", pady=(0, 8))
        B(row, "Select All", lambda: self.set_all(True)).pack(side="left")
        B(row, "Deselect All", lambda: self.set_all(False)).pack(side="left", padx=8)
        self.dl_filter = tk.StringVar()
        L(row, "🔍").pack(side="left", padx=(10, 2))
        ttk.Entry(row, textvariable=self.dl_filter, width=24).pack(side="left")
        self.dl_filter.trace_add("write", lambda *a: self.filter_rows())
        self.cnt_lbl = L(row, "0 profiles selected", "a")
        self.cnt_lbl.pack(side="right")
        qa = F(t)
        qa.pack(fill="x", pady=(0, 6))
        L(qa, "After each download, also add:", "a").pack(side="left")
        Check(qa, self, "💬 Captions", self.ev["auto_cap"]).pack(side="left", padx=(12, 0))
        Check(qa, self, "🏷 Brand", self.ev["auto_brand"]).pack(side="left", padx=10)
        Check(qa, self, "🎨 Edit", self.ev["auto"]).pack(side="left")
        wrap, self.tree = self.make_tree(t, (("sel", "✔", 40, "center"), ("doc", "Doc", 300, "w"),
                                             ("profile", "Profile", 300, "w"), ("links", "Links", 70, "center"),
                                             ("take", "Take", 70, "center")), height=6)
        wrap.pack(fill="both", expand=True)
        self.tree.bind("<Button-1>", self.toggle)
        self.tree.bind("<Double-1>", self.set_take)
        lh = F(t)
        lh.pack(fill="x", pady=(10, 4))
        L(lh, "LIVE LOG", "a").pack(side="left")
        B(lh, "💾 Save log", self.save_log, size=9, padx=12, pady=4).pack(side="right")
        self.log = tk.Text(t, height=10, bd=0, font=("Consolas", 10), wrap="word", padx=10, pady=8)
        self.log.pack(fill="both", expand=True)

        # ----- 4K Downloader -----
        t = self.pbody["uni"]
        L(t, "Paste links — one per line", "a").pack(anchor="w")
        self.paste = tk.Text(t, height=6, bd=0, font=("Consolas", 10), padx=10, pady=8)
        self.paste.pack(fill="x", pady=8)
        opt = F(t)
        opt.pack(fill="x")
        self.u_fmt = tk.StringVar(value="MP4 Video")
        self.u_q = tk.StringVar(value="Best")
        L(opt, "Format:").pack(side="left")
        ttk.Combobox(opt, textvariable=self.u_fmt, values=["MP4 Video", "MP3 Audio"], state="readonly",
                     width=11).pack(side="left", padx=6)
        L(opt, "Quality:").pack(side="left", padx=(12, 0))
        ttk.Combobox(opt, textvariable=self.u_q, values=list(UNI_Q), state="readonly",
                     width=11).pack(side="left", padx=6)
        B(opt, "⬇ Download All", self.uni_start, "primary").pack(side="right")
        B(opt, "Clear", self.uni_clear).pack(side="right", padx=8)
        B(opt, "📋 Paste", self.uni_paste).pack(side="right")
        Check(opt, self, "Auto-capture copied links", self.clip_on).pack(side="right", padx=14)
        wrap, self.utree = self.make_tree(t, (("n", "#", 40, "center"), ("link", "Link", 380, "w"),
                                              ("title", "Title", 280, "w"), ("status", "Status", 100, "center"),
                                              ("pct", "Progress", 80, "center")), height=8)
        wrap.pack(fill="both", expand=True, pady=(10, 0))

        ev = self.ev

        def spin(par, key, lo, hi, inc):
            return ttk.Spinbox(par, from_=lo, to=hi, increment=inc, textvariable=ev[key], width=8)

        def combo(par, key, vals, w=16):
            return ttk.Combobox(par, textvariable=ev[key], values=vals, state="readonly", width=w)

        def fill(parent, rows, start=0):
            for r, (lab, w) in enumerate(rows, start):
                if lab:
                    L(parent, lab).grid(row=r, column=0, sticky="w", pady=4)
                    w.grid(row=r, column=1, sticky="w", padx=8)
                else:
                    w.grid(row=r, column=0, columnspan=2, sticky="w", pady=3)

        def boxer(inner):
            def ebox(title):
                b = ttk.LabelFrame(inner, text=f" {title} ", style="Card.TLabelframe", padding=10)
                b.pack(fill="x", padx=(0, 10), pady=(0, 10))
                return b
            return ebox

        # ----- Clipper -----
        t = self.pbody["clip"]
        top = F(t)
        top.pack(fill="x")
        L(top, "Long video — paste a link or choose a file:", "a").pack(anchor="w")
        row = F(top)
        row.pack(fill="x", pady=6)
        ttk.Entry(row, textvariable=ev["clip_src"]).pack(side="left", fill="x", expand=True)
        B(row, "📋 Paste", self.clip_paste, size=9, padx=12, pady=5).pack(side="left", padx=6)
        B(row, "📂 File", self.clip_browse, size=9, padx=12, pady=5).pack(side="left")
        self.clip_info = L(top, "", "m")
        self.clip_info.pack(anchor="w")
        body = F(t)
        body.pack(fill="both", expand=True, pady=(10, 0))
        csf = self.col(body)
        csf.pack(side="left", fill="both", expand=True, padx=(0, 12))
        cbox = boxer(csf.inner)
        cl = cbox("⏱ Clip length")
        fill(cl, [("Each clip (minutes):", spin(cl, "clip_min", 0.25, 120, 0.25)),
                  ("Skip tiny last clip (sec):", spin(cl, "clip_minlast", 0, 120, 5)),
                  ("Overlap (sec):", spin(cl, "clip_overlap", 0, 30, 1))])
        qk = F(cl)
        qk.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        for txt, val in (("30 sec", 0.5), ("1 min", 1), ("2 min", 2), ("3 min", 3), ("5 min", 5)):
            B(qk, txt, lambda v=val: ev["clip_min"].set(v), size=9, padx=10, pady=4).pack(side="left", padx=2)
        ce = cbox("✨ Light editing")
        fill(ce, [("", Check(ce, self, "Make vertical 9:16 (blurred sides)", ev["clip_vertical"])),
                  ("", Check(ce, self, "Smooth fade in / out", ev["clip_fade"])),
                  ("", Check(ce, self, "Write Part number on every clip", ev["clip_part"])),
                  ("Label text:", ttk.Entry(ce, textvariable=ev["clip_part_text"], width=18)),
                  ("Label position:", combo(ce, "clip_part_pos", POSITIONS)),
                  ("Label size (%):", spin(ce, "clip_part_size", 2, 12, 0.5)),
                  ("Label colour:", combo(ce, "clip_part_color", list(BRAND_COLORS), 10)),
                  ("", Check(ce, self, "Add brand + logo (from Brand tab)", ev["clip_brand"])),
                  ("Brand username:", ttk.Entry(ce, textvariable=ev["clip_user"], width=18))])
        L(ce, "{n} = part,  {total} = parts", "m").grid(row=9, column=0, columnspan=2, sticky="w")
        cright = F(body)
        cright.pack(side="left", fill="both", expand=True)
        L(cright, "CLIPS", "a").pack(anchor="w")
        wrap, self.ctree = self.make_tree(cright, (("part", "Part", 60, "center"), ("from", "From", 70, "center"),
                                                   ("to", "To", 70, "center"), ("len", "Length", 70, "center"),
                                                   ("status", "Status", 110, "center")), height=10)
        wrap.pack(fill="both", expand=True, pady=6)
        crow = F(cright)
        crow.pack(fill="x", pady=(6, 0))
        B(crow, "✂  MAKE CLIPS", lambda: self.thread(self.clip_make), "primary", size=11, padx=22, pady=9).pack(side="right")
        B(crow, "📂 Open folder", self.clip_open, size=9, padx=12, pady=6).pack(side="left")
        ev["clip_min"].trace_add("write", lambda *a: self.clip_estimate())

        # ----- Brand & Logo -----
        t = self.pbody["brand"]
        bsf = self.col(t)
        bsf.pack(side="left", fill="both", expand=True, padx=(0, 12))
        bbox = boxer(bsf.inner)
        ba = bbox("🤖 Automatic")
        fill(ba, [("", Check(ba, self, "Add brand to every downloaded video automatically", ev["auto_brand"]))])
        br = bbox("🏷 Brand text — username of the video's account")
        fill(br, [("", Check(br, self, "Add brand text", ev["brand_on"])),
                  ("Text:", ttk.Entry(br, textvariable=ev["brand_text"], width=24)),
                  ("Position:", combo(br, "brand_pos", POSITIONS)),
                  ("Size (% of height):", spin(br, "brand_size", 1.5, 9, 0.5)),
                  ("Colour:", combo(br, "brand_color", list(BRAND_COLORS), 10)),
                  ("Opacity:", spin(br, "brand_opacity", 0.3, 1.0, 0.05)),
                  ("", Check(br, self, "Dark rounded box behind text", ev["brand_box"]))])
        L(br, "{username} = account name", "m").grid(row=7, column=0, columnspan=2, sticky="w")
        lg = bbox("🖼 Logo image")
        fill(lg, [("", Check(lg, self, "Add my logo", ev["logo_on"])),
                  ("Logo file:", ttk.Entry(lg, textvariable=ev["logo_path"], width=24)),
                  ("Position:", combo(lg, "logo_pos", POSITIONS)),
                  ("Size (% of width):", spin(lg, "logo_size", 5, 40, 1)),
                  ("Opacity:", spin(lg, "logo_opacity", 0.3, 1.0, 0.05))])
        B(lg, "Browse", self.browse_logo, size=9, padx=12, pady=5).grid(row=2, column=2)
        bo = bbox("💾 Output")
        fill(bo, [("Quality / size:", combo(bo, "crf", CRFS)), ("Save as:", combo(bo, "outmode", OUTMODES)),
                  ("", Check(bo, self, "Keep the original in an _Originals folder", ev["keep_orig"]))])
        rsf = self.col(t)
        rsf.pack(side="left", fill="both", expand=True)
        rin = rsf.inner
        L(rin, "LIVE PREVIEW", "a").pack(anchor="w")
        self.bp_label = R(tk.Label(rin, bd=0, text="preview loads here", font=("Segoe UI", 9)), bg="card", fg="muted")
        self.bp_label.pack(pady=6)
        pr1 = F(rin)
        pr1.pack(fill="x")
        L(pr1, "Preview username:").pack(side="left")
        ttk.Entry(pr1, textvariable=ev["brand_prev_user"], width=14).pack(side="left", padx=6)
        pr2 = F(rin)
        pr2.pack(fill="x", pady=6)
        B(pr2, "🔄 Update", self.bp_update_now, size=9, padx=10, pady=5).pack(side="left")
        B(pr2, "🎞 Sample video", self.bp_pick_sample, size=9, padx=10, pady=5).pack(side="left", padx=4)
        B(pr2, "✖", lambda: ev["brand_sample"].set(""), size=9, padx=10, pady=5).pack(side="left")
        self.bq = EditQueue(rin, self, ("brand",), "▶ APPLY BRAND", rows=4)
        self.bq.pack(fill="x", pady=(14, 0))
        for k in ("brand_on", "brand_text", "brand_pos", "brand_size", "brand_color", "brand_opacity", "brand_box",
                  "logo_on", "logo_path", "logo_pos", "logo_size", "logo_opacity", "brand_prev_user", "brand_sample"):
            ev[k].trace_add("write", lambda *a: self.bp_later())

        # ----- Editing -----
        t = self.pbody["edit"]
        esf = self.col(t)
        esf.pack(side="left", fill="both", expand=True, padx=(0, 12))
        ebox = boxer(esf.inner)
        au2 = ebox("🤖 Automatic")
        fill(au2, [("", Check(au2, self, "Auto-edit every downloaded video", ev["auto"]))])
        ex = ebox("✂ Extras")
        fill(ex, [("Cut from start (sec):", spin(ex, "trim_start", 0, 600, 0.5)),
                  ("Cut from end (sec):", spin(ex, "trim_end", 0, 600, 0.5)),
                  ("Fit to shape:", combo(ex, "aspect", ASPECTS)),
                  ("", Check(ex, self, "Remove sound (mute)", ev["mute"]))])
        ao = ebox("💾 Output")
        fill(ao, [("Max height (px):", combo(ao, "maxh", MAXH, 10)),
                  ("Quality / size:", combo(ao, "crf", CRFS)),
                  ("Save as:", combo(ao, "outmode", OUTMODES)),
                  ("", Check(ao, self, "Keep the original in an _Originals folder", ev["keep_orig"]))])
        self.eq = EditQueue(t, self, ("extras",), "▶ START EDITING")
        self.eq.pack(side="left", fill="both", expand=True)

        # ----- Link Collector -----
        t = self.pbody["links"]
        L(t, "Profile links or @usernames, one per line:", "a").pack(anchor="w")
        self.lc_in = tk.Text(t, height=5, bd=0, font=("Consolas", 10), padx=10, pady=8)
        self.lc_in.pack(fill="x", pady=6)
        r1 = F(t)
        r1.pack(fill="x")
        B(r1, "📋 Paste", self.lc_paste, size=9, padx=12, pady=5).pack(side="left")
        B(r1, "📄 From my docs", self.lc_from_docs, size=9, padx=12, pady=5).pack(side="left", padx=6)
        B(r1, "🗑 Clear", lambda: self.lc_in.delete("1.0", "end"), size=9, padx=12, pady=5).pack(side="left")
        B(r1, "🔎  COLLECT LINKS", self.lc_start, "primary", size=11, padx=22, pady=8).pack(side="right")
        r2 = F(t)
        r2.pack(fill="x", pady=8)
        L(r2, "Order:").pack(side="left")
        combo(r2, "lc_order", LC_ORDERS, 16).pack(side="left", padx=6)
        L(r2, "Max per profile (0 = all):").pack(side="left", padx=(16, 0))
        spin(r2, "lc_max", 0, 5000, 10).pack(side="left", padx=6)
        Check(r2, self, "Remove picture posts", ev["lc_deep"]).pack(side="left", padx=(16, 0))
        Check(r2, self, "Hide already downloaded", ev["lc_skipdone"]).pack(side="left", padx=10)
        self.lc_sum = L(t, "", "m")
        self.lc_sum.pack(anchor="w", pady=(0, 6))
        r4 = F(t)
        r4.pack(fill="x", pady=(0, 8))
        L(r4, "Add to doc:").pack(side="left")
        self.lc_doc_cb = ttk.Combobox(r4, textvariable=ev["lc_doc"], values=["➕ New Google Doc"], state="readonly", width=34)
        self.lc_doc_cb.pack(side="left", padx=6)
        B(r4, "➕ Add to Google Doc", self.lc_to_doc, "success", size=10, padx=14, pady=6).pack(side="left")
        B(r4, "💾 Save TXT", self.lc_save, size=9, padx=12, pady=6).pack(side="right")
        B(r4, "📋 Copy selected", self.lc_copy_sel, size=9, padx=12, pady=6).pack(side="right", padx=6)
        B(r4, "📋 Copy all", self.lc_copy, size=9, padx=12, pady=6).pack(side="right")
        wrap, self.lctree = self.make_tree(t, (("profile", "Profile", 120, "w"), ("date", "Date", 90, "center"),
                                               ("type", "Type", 70, "center"), ("link", "Link", 470, "w")), height=9)
        wrap.pack(fill="both", expand=True)

        # ----- Smart Clips -----
        t = self.pbody["smart"]
        top = F(t)
        top.pack(fill="x")
        L(top, "Long video — link or file:", "a").pack(anchor="w")
        row = F(top)
        row.pack(fill="x", pady=6)
        ttk.Entry(row, textvariable=ev["sc_src"]).pack(side="left", fill="x", expand=True)
        B(row, "📋 Paste", self.sc_paste, size=9, padx=12, pady=5).pack(side="left", padx=6)
        B(row, "📂 File", self.sc_browse, size=9, padx=12, pady=5).pack(side="left")
        self.sc_info = L(top, "", "m")
        self.sc_info.pack(anchor="w")
        body = F(t)
        body.pack(fill="both", expand=True, pady=(10, 0))
        ssf = self.col(body)
        ssf.pack(side="left", fill="both", expand=True, padx=(0, 12))
        sbox = boxer(ssf.inner)
        h1 = sbox("🔥 Highlights")
        fill(h1, [("How many highlights:", spin(h1, "sc_count", 1, 30, 1)),
                  ("Length of each (sec):", spin(h1, "sc_len", 8, 180, 1)),
                  ("Minimum gap (sec):", spin(h1, "sc_gap", 0, 600, 5)),
                  ("Number clips by:", combo(h1, "sc_rank", ["Best first", "Time order"], 14))])
        h2 = sbox("✨ Light editing (same style as the Clipper tab)")
        fill(h2, [("", Check(h2, self, "Make vertical 9:16 (blurred sides)", ev["clip_vertical"])),
                  ("", Check(h2, self, "Smooth fade in / out", ev["clip_fade"])),
                  ("", Check(h2, self, "Write a label on every clip", ev["sc_label"])),
                  ("Label text:", ttk.Entry(h2, textvariable=ev["sc_label_text"], width=18)),
                  ("Label position:", combo(h2, "clip_part_pos", POSITIONS)),
                  ("Label size (%):", spin(h2, "clip_part_size", 2, 12, 0.5)),
                  ("Label colour:", combo(h2, "clip_part_color", list(BRAND_COLORS), 10)),
                  ("", Check(h2, self, "Add brand + logo (from Brand tab)", ev["clip_brand"])),
                  ("Brand username:", ttk.Entry(h2, textvariable=ev["clip_user"], width=18))])
        sright = F(body)
        sright.pack(side="left", fill="both", expand=True)
        L(sright, "HIGHLIGHTS", "a").pack(anchor="w")
        wrap, self.stree = self.make_tree(sright, (("n", "#", 40, "center"), ("from", "From", 70, "center"),
                                                   ("to", "To", 70, "center"), ("score", "Energy", 70, "center"),
                                                   ("status", "Status", 110, "center")), height=10)
        wrap.pack(fill="both", expand=True, pady=6)
        srow = F(sright)
        srow.pack(fill="x", pady=(6, 0))
        B(srow, "🔥 FIND & CUT", lambda: self.thread(self.sc_make), "primary", size=11, padx=22, pady=9).pack(side="right")
        B(srow, "📂 Open folder", self.sc_open, size=9, padx=12, pady=6).pack(side="left")

        # ----- Captions -----
        t = self.pbody["caps"]
        capsf = self.col(t)
        capsf.pack(side="left", fill="both", expand=True, padx=(0, 12))
        cpb = boxer(capsf.inner)
        ca = cpb("🤖 Automatic")
        fill(ca, [("", Check(ca, self, "Add captions to every download", ev["auto_cap"]))])
        B(ca, "🩺 Test captions", lambda: self.thread(self.cap_selftest), size=9, padx=12, pady=5).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        sp = cpb("🌐 Speech")
        fill(sp, [("Language:", combo(sp, "cp_lang", list(CAP_LANGS), 16)),
                  ("Speech model:", combo(sp, "cp_model", CAP_MODELS, 10))])
        Check(sp, self, "Translate speech to English captions", ev["cp_translate"]).grid(row=3, column=0, columnspan=2, sticky="w", pady=3)
        L(sp, "Speed:").grid(row=2, column=0, sticky="w", pady=4)
        combo(sp, "cp_speed", ["Fast", "Balanced", "Best"], 10).grid(row=2, column=1, sticky="w", padx=8)
        stl = cpb("🎨 Style")
        pf = F(stl)
        pf.grid(row=0, column=0, columnspan=2, sticky="w")
        for i, nm in enumerate(CAP_STYLES):
            B(pf, nm, lambda n=nm: self.cp_preset(n), size=9, padx=10, pady=4).grid(row=i // 3, column=i % 3, padx=3, pady=3)
        fill(stl, [("Font:", ttk.Combobox(stl, textvariable=ev["cp_font"], values=CAP_FONTS, width=22)),
                   ("Size (% of height):", spin(stl, "cp_size", 2, 12, 0.5)),
                   ("Text colour:", combo(stl, "cp_color", list(CAP_COLORS), 10)),
                   ("Outline colour:", combo(stl, "cp_outline", list(CAP_COLORS), 10)),
                   ("Highlight colour:", combo(stl, "cp_hl", list(CAP_COLORS), 10)),
                   ("Outline width:", spin(stl, "cp_outw", 0, 8, 0.5)),
                   ("Position:", combo(stl, "cp_pos", CAP_POS, 10)),
                   ("Distance from edge (%):", spin(stl, "cp_margin", 2, 40, 1)),
                   ("Max letters per line:", spin(stl, "cp_chars", 10, 60, 2)),
                   ("Shape:", combo(stl, "cp_shape", CAP_SHAPES, 18)),
                   ("Shape colour:", combo(stl, "cp_boxcolor", list(CAP_COLORS), 10)),
                   ("Animation:", combo(stl, "cp_anim", CAP_ANIMS, 12)),
                   ("Letter spacing:", spin(stl, "cp_spacing", 0, 10, 0.5)),
                   ("", Check(stl, self, "Highlight the spoken word", ev["cp_wordhl"])),
                   ("", Check(stl, self, "Dark box behind text", ev["cp_box"])),
                   ("", Check(stl, self, "Bold", ev["cp_bold"])),
                   ("", Check(stl, self, "Italic", ev["cp_italic"])),
                   ("", Check(stl, self, "CAPITAL LETTERS", ev["cp_upper"]))], 1)
        cpo = cpb("💾 Output")
        fill(cpo, [("", Check(cpo, self, "Burn captions into the video", ev["cp_burn"])),
                   ("", Check(cpo, self, "Also save a .srt subtitle file", ev["cp_srt"])),
                   ("Quality / size:", combo(cpo, "crf", CRFS)),
                   ("Save as:", combo(cpo, "outmode", OUTMODES)),
                   ("", Check(cpo, self, "Keep the original in an _Originals folder", ev["keep_orig"])),
                   ("", Check(cpo, self, "Skip videos that already have captions", ev["cp_skip"]))])
        cpr = self.col(t)
        cpr.pack(side="left", fill="both", expand=True)
        L(cpr.inner, "LIVE PREVIEW", "a").pack(anchor="w")
        self.cp_label = R(tk.Label(cpr.inner, bd=0, text="preview loads here", font=("Segoe UI", 9)), bg="card", fg="muted")
        self.cp_label.pack(pady=6)
        B(cpr.inner, "🔄 Update preview", self.cp_update_now, size=9, padx=10, pady=5).pack(anchor="w")
        self.cq = EditQueue(cpr.inner, self, (), "▶ ADD CAPTIONS", rows=4, action=self.caption_file, preview=False, show_user=False)
        self.cq.pack(fill="x", pady=(14, 0))
        for k in ("cp_lang", "cp_font", "cp_size", "cp_color", "cp_outline", "cp_hl", "cp_outw", "cp_pos", "cp_margin",
                  "cp_chars", "cp_wordhl", "cp_box", "cp_bold", "cp_upper", "cp_shape", "cp_boxcolor", "cp_anim",
                  "cp_italic", "cp_spacing"):
            ev[k].trace_add("write", lambda *a: self.cp_later())

        # ----- Thumbnails -----
        t = self.pbody["thumb"]
        thsf = self.col(t)
        thsf.pack(side="left", fill="both", expand=True, padx=(0, 12))
        tbx = boxer(thsf.inner)
        tv = tbx("🎞 Video & frame")
        fill(tv, [("Video file:", ttk.Entry(tv, textvariable=ev["th_src"], width=26)),
                  ("Frame at (sec):", spin(tv, "th_time", 0, 36000, 0.5))])
        B(tv, "Browse", self.th_browse, size=9, padx=12, pady=5).grid(row=0, column=2)
        B(tv, "🎯 Auto-pick best frame", self.th_auto, size=9, padx=12, pady=5).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        tt = tbx("🔤 Title text")
        fill(tt, [("Text:", ttk.Entry(tt, textvariable=ev["th_text"], width=28)),
                  ("Font:", combo(tt, "th_font", list(THUMB_FONTS), 16)),
                  ("Size (% of height):", spin(tt, "th_size", 3, 30, 0.5)),
                  ("Text colour:", combo(tt, "th_color", list(CAP_COLORS), 10)),
                  ("Outline colour:", combo(tt, "th_stroke", list(CAP_COLORS), 10)),
                  ("Outline width:", spin(tt, "th_strokew", 0, 20, 1)),
                  ("Position:", combo(tt, "th_pos", CAP_POS, 10)),
                  ("", Check(tt, self, "Dark shade behind text", ev["th_shade"]))])
        ts = tbx("📐 Size & brand")
        fill(ts, [("Picture size:", combo(ts, "th_size_preset", list(THUMB_SIZES), 22)),
                  ("File type:", combo(ts, "th_fmt", ["PNG", "JPG"], 8)),
                  ("", Check(ts, self, "Add brand + logo (from Brand tab)", ev["th_brand"])),
                  ("Brand username:", ttk.Entry(ts, textvariable=ev["th_user"], width=18))])
        thr = self.col(t)
        thr.pack(side="left", fill="both", expand=True)
        L(thr.inner, "PREVIEW", "a").pack(anchor="w")
        self.th_label = R(tk.Label(thr.inner, bd=0, text="choose a video to see the preview", font=("Segoe UI", 9)), bg="card", fg="muted")
        self.th_label.pack(pady=6)
        tr = F(thr.inner)
        tr.pack(fill="x")
        B(tr, "🔄 Update", self.th_update_now, size=9, padx=10, pady=5).pack(side="left")
        B(tr, "💾 SAVE THUMBNAIL", self.th_save, "primary", size=10, padx=16, pady=7).pack(side="right")
        for k in ("th_src", "th_time", "th_text", "th_font", "th_size", "th_color", "th_stroke", "th_strokew", "th_pos",
                  "th_shade", "th_size_preset", "th_brand", "th_user", "brand_text", "brand_pos", "brand_size", "brand_color",
                  "brand_opacity", "brand_box", "logo_on", "logo_path", "logo_pos", "logo_size", "logo_opacity"):
            ev[k].trace_add("write", lambda *a: self.th_later())

        # ----- Cleaner -----
        t = self.pbody["clean"]
        co = F(t)
        co.pack(fill="x", pady=8)
        Check(co, self, "Find duplicates", ev["cl_dups"]).pack(side="left")
        L(co, "Older than (days, 0 = ignore):").pack(side="left", padx=(14, 0))
        spin(co, "cl_old", 0, 3650, 5).pack(side="left", padx=6)
        L(co, "Bigger than (MB, 0 = ignore):").pack(side="left", padx=(14, 0))
        spin(co, "cl_big", 0, 100000, 50).pack(side="left", padx=6)
        B(co, "✨ Move Edited videos up", self.cl_tidy, size=9, padx=12, pady=5).pack(side="right")
        cb2 = F(t)
        cb2.pack(fill="x")
        B(cb2, "🔍 SCAN", lambda: self.thread(self.cl_scan), "primary", size=10, padx=18, pady=7).pack(side="left")
        B(cb2, "Select all", lambda: self.cl_all(True), size=9, padx=10, pady=5).pack(side="left", padx=6)
        B(cb2, "Deselect", lambda: self.cl_all(False), size=9, padx=10, pady=5).pack(side="left")
        Check(cb2, self, "Safe: move to _Cleaner_Trash", ev["cl_safe"]).pack(side="left", padx=14)
        B(cb2, "🗑 DELETE SELECTED", self.cl_delete, "danger", size=10, padx=14, pady=6).pack(side="right")
        B(cb2, "📂 Trash", self.cl_open_trash, size=9, padx=10, pady=6).pack(side="right", padx=6)
        self.cl_sum = L(t, "", "a")
        self.cl_sum.pack(anchor="w", pady=6)
        wrap, self.cltree = self.make_tree(t, (("sel", "✔", 40, "center"), ("file", "File", 260, "w"),
                                               ("size", "Size", 80, "center"), ("age", "Age (days)", 80, "center"),
                                               ("why", "Reason", 300, "w")), height=10)
        wrap.pack(fill="both", expand=True)
        self.cltree.bind("<Button-1>", self.cl_toggle)

        # ----- Progress -----
        t = self.pbody["prog"]
        self.psum = L(t, "", "a")
        self.psum.pack(anchor="w", pady=(0, 8))
        wrap, self.ptree = self.make_tree(t, (("doc", "Doc", 200, "w"), ("profile", "Profile", 190, "w"),
                                              ("down", "Downloaded", 90, "center"), ("rem", "Remaining", 90, "center"),
                                              ("tot", "Total", 70, "center"), ("bar", "Progress", 190, "w")))
        wrap.pack(fill="both", expand=True)

        # ----- History -----
        t = self.pbody["hist"]
        hb = F(t)
        hb.pack(fill="x", pady=(0, 8))
        self.hfilter = tk.StringVar(value="All")
        cb = ttk.Combobox(hb, textvariable=self.hfilter, values=["All", "Done", "Failed"], state="readonly", width=8)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_history())
        B(hb, "📂 Open folder", self.h_open).pack(side="left", padx=8)
        B(hb, "▶ Play", self.h_play).pack(side="left")
        B(hb, "↻ Retry failed", lambda: self.thread(self.retry_failed)).pack(side="left", padx=8)
        B(hb, "⬇ Export CSV", self.export_csv).pack(side="left")
        B(hb, "Clear history", self.clear_history, "danger").pack(side="right")
        wrap, self.htree = self.make_tree(t, (("time", "Time", 130, "w"), ("status", "Status", 70, "center"),
                                              ("doc", "Doc", 150, "w"), ("profile", "Profile", 130, "w"),
                                              ("name", "Video", 380, "w")))
        wrap.pack(fill="both", expand=True)
        self.htree.bind("<Double-1>", lambda e: self.h_play())

        # ----- Accounts -----
        t = self.pbody["acc"]
        card = R(tk.Frame(t, padx=18, pady=16), bg="card2")
        card.pack(fill="x")
        self.avatar = R(tk.Label(card, text="?", width=3, font=("Segoe UI", 26, "bold")), bg="accent", fg="#ffffff")
        self.avatar.pack(side="left", padx=(0, 18))
        info = R(tk.Frame(card), bg="card2")
        info.pack(side="left")
        self.acc_name = R(tk.Label(info, text="Not logged in", font=("Segoe UI", 18, "bold"), anchor="w"), bg="card2", fg="text")
        self.acc_name.pack(anchor="w")
        self.acc_mail = R(tk.Label(info, text="", font=("Segoe UI", 10), anchor="w"), bg="card2", fg="muted")
        self.acc_mail.pack(anchor="w")
        self.acc_state = R(tk.Label(info, text="", font=("Segoe UI", 10, "bold"), anchor="w"), bg="card2", fg="ok")
        self.acc_state.pack(anchor="w", pady=(4, 0))
        ab = F(t)
        ab.pack(fill="x", pady=12)
        B(ab, "➕ Login / Add account", self.login, "primary").pack(side="left")
        B(ab, "✔ Use selected", self.use_selected).pack(side="left", padx=8)
        B(ab, "🗑 Remove selected", self.remove_selected).pack(side="left")
        B(ab, "Logout", self.logout, "danger").pack(side="right")
        L(t, "SAVED ACCOUNTS", "a").pack(anchor="w")
        wrap, self.atree = self.make_tree(t, (("act", "", 40, "center"), ("email", "Gmail", 420, "w"),
                                              ("status", "Status", 120, "w")), height=5)
        wrap.pack(fill="x", pady=6)

        # ----- Settings -----
        t = self.pbody["set"]
        SL, SR = F(t), F(t)
        SL.pack(side="left", fill="both", expand=True, padx=(0, 10), anchor="n")
        SR.pack(side="left", fill="both", expand=True, anchor="n")

        def box(parent, title):
            b = ttk.LabelFrame(parent, text=f" {title} ", style="Card.TLabelframe", padding=10)
            b.pack(fill="x", pady=(0, 10))
            return b
        dl = box(SL, "Download")
        rows = (("Links per profile:", ttk.Spinbox(dl, from_=1, to=500, textvariable=self.links_n, width=8)),
                ("Quality (TikTok):", ttk.Combobox(dl, textvariable=self.quality, values=list(QUALITY),
                                                   state="readonly", width=10)),
                ("Format:", ttk.Combobox(dl, textvariable=self.fmt, values=["MP4 Video", "MP3 Audio"],
                                         state="readonly", width=10)),
                ("Parallel downloads:", ttk.Spinbox(dl, from_=1, to=5, textvariable=self.par, width=8)),
                ("Auto-retry times:", ttk.Spinbox(dl, from_=0, to=5, textvariable=self.retry, width=8)),
                ("Download order:", ttk.Combobox(dl, textvariable=self.order, values=["Top first", "Bottom first"],
                                                state="readonly", width=12)),
                ("Speed limit (MB/s, 0 = off):", ttk.Spinbox(dl, from_=0, to=500, textvariable=self.rate, width=8)),
                ("", Check(dl, self, "Skip videos already downloaded", self.skipdup)),
                ("Save folder:", ttk.Entry(dl, textvariable=self.folder, width=30)))
        for r, (lab, w) in enumerate(rows):
            L(dl, lab).grid(row=r, column=0, sticky="w", pady=4)
            w.grid(row=r, column=1, sticky="w", padx=8)
        B(dl, "Browse", self.browse, size=9, padx=12, pady=5).grid(row=8, column=2)
        nw = box(SL, "Network (optional)")
        L(nw, "Proxy:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(nw, textvariable=self.proxy, width=30).grid(row=0, column=1, padx=8)
        L(nw, "Cookies file:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(nw, textvariable=self.cookies, width=30).grid(row=1, column=1, padx=8)
        B(nw, "Browse", self.browse_cookies, size=9, padx=12, pady=5).grid(row=1, column=2)
        L(nw, "Browser cookies:").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Combobox(nw, textvariable=self.cbrowser, values=["None", "chrome", "edge", "firefox"],
                     state="readonly", width=10).grid(row=2, column=1, sticky="w", padx=8)
        au = box(SR, "Automation & Alerts")
        Check(au, self, "Play sound when finished", self.sound).grid(row=0, column=0, columnspan=2, sticky="w", pady=3)
        Check(au, self, "Show message when finished", self.popup).grid(row=1, column=0, columnspan=2, sticky="w", pady=3)
        Check(au, self, "Auto-run all docs daily at (HH:MM):", self.sched_on).grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(au, textvariable=self.sched_time, width=8).grid(row=2, column=1, padx=8)
        L(au, "After finishing:").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(au, textvariable=self.after_act, values=["Do nothing", "Sleep PC", "Shutdown PC"],
                     state="readonly", width=13).grid(row=4, column=1, sticky="w", padx=8, pady=(10, 0))
        ap = box(SR, "Appearance")
        L(ap, "Theme:").grid(row=0, column=0)
        cb = ttk.Combobox(ap, textvariable=self.theme, values=list(THEMES), state="readonly", width=14)
        cb.grid(row=0, column=1, padx=8)
        cb.bind("<<ComboboxSelected>>", lambda e: self.apply_theme())
        B(ap, "🎨 Next", self.next_theme, size=9, padx=10, pady=4).grid(row=0, column=2)
        Check(ap, self, "Keep the window on top", self.topmost).grid(row=1, column=0, columnspan=3, sticky="w", pady=4)
        mt = box(SR, "Maintenance")
        B(mt, "⟳ Update yt-dlp", self.update_ytdlp).pack(side="left")
        bk = box(SR, "Backup & Restore")
        B(bk, "📤 Export settings", self.export_settings, size=9, padx=12, pady=5).pack(side="left")
        B(bk, "📥 Import settings", self.import_settings, size=9, padx=12, pady=5).pack(side="left", padx=8)
        sb = F(SR)
        sb.pack(fill="x", pady=(0, 10))
        B(sb, "💾 Save Settings", self.save_settings, "success").pack(side="left")
        B(sb, "Reset", self.reset_settings).pack(side="left", padx=8)
        abt = box(SR, "About")
        L(abt, f"{APP} {VER}  —  {TAGLINE}").pack(anchor="w")
        L(abt, CREDIT, "a").pack(anchor="w", pady=(4, 0))
        L(abt, "Shortcuts: F1 Search  •  F2 Sidebar  •  F3 Theme  •  Ctrl+1…9 Tabs  •  F5 Load  •  Ctrl+R Run  •  Ctrl+P Pause  •  Ctrl+. Stop", "m").pack(anchor="w", pady=(4, 0))
        self.wrap_notes(self)
        self.show_page(self.last_page.get() if self.last_page.get() in PAGE_TITLES else "dl")

    def apply_theme(self):
        p = self.pal = THEMES[self.theme.get()]
        st = ttk.Style(self)
        st.theme_use("clam")
        self.configure(bg=p["bg"])
        st.configure(".", background=p["card"], foreground=p["text"], fieldbackground=p["card2"],
                     font=("Segoe UI", 10), bordercolor=p["line"], lightcolor=p["card"], darkcolor=p["card"],
                     troughcolor=p["card2"])
        st.configure("Card.TFrame", background=p["card"])
        st.configure("Side.TFrame", background=p["side"])
        for name, fg, font in (("C.TLabel", p["text"], ("Segoe UI", 11)),
                               ("CA.TLabel", p["accent2"], ("Segoe UI", 11, "bold")),
                               ("CM.TLabel", p["muted"], ("Segoe UI", 10))):
            st.configure(name, background=p["card"], foreground=fg, font=font)
        st.configure("C.TCheckbutton", background=p["card"], foreground=p["text"])
        st.map("C.TCheckbutton", background=[("active", p["card"])], foreground=[("active", p["text"])])
        st.configure("Card.TLabelframe", background=p["card"], bordercolor=p["line"], relief="solid")
        st.configure("Card.TLabelframe.Label", background=p["card"], foreground=p["accent2"], font=("Segoe UI", 11, "bold"))
        st.configure("Treeview", background=p["card2"], fieldbackground=p["card2"], foreground=p["text"],
                     rowheight=30, borderwidth=0, font=("Segoe UI", 10))
        st.map("Treeview", background=[("selected", p["accent"])], foreground=[("selected", "#ffffff")])
        st.configure("Treeview.Heading", background=p["line"], foreground=p["text"], font=("Segoe UI", 10, "bold"), relief="flat")
        st.map("Treeview.Heading", background=[("active", p["accent"])])
        for w in ("TEntry", "TCombobox", "TSpinbox"):
            st.configure(w, fieldbackground=p["card2"], foreground=p["text"], insertcolor=p["text"],
                         bordercolor=p["line"], arrowcolor=p["text"], background=p["card2"])
        st.map("TCombobox", fieldbackground=[("readonly", p["card2"])], foreground=[("readonly", p["text"])],
               selectbackground=[("readonly", p["card2"])], selectforeground=[("readonly", p["text"])])
        st.configure("Vertical.TScrollbar", background=p["line"], troughcolor=p["card"], arrowcolor=p["text"],
                     bordercolor=p["card"])
        self.option_add("*TCombobox*Listbox.background", p["card2"])
        self.option_add("*TCombobox*Listbox.foreground", p["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", p["accent"])
        for w, spec in self.reg:
            w.configure(**{k: p.get(v, v) for k, v in spec.items()})
        for b in list(self.btns):
            try:
                b.draw()
            except tk.TclError:
                self.btns.remove(b)
        for w in (self.log, self.paste):
            w.configure(bg=p["log"], fg=p["text"], insertbackground=p["text"])
        self.log.tag_config("ok", foreground=p["ok"])
        self.log.tag_config("err", foreground=p["err"])
        self.log.tag_config("info", foreground=p["accent2"])
        self.paint_nav()
        self.set_bar()

    def set_bar(self, done=None, total=None, text=None):
        if done is not None:
            self.pdone = done
        if total is not None:
            self.ptotal = total
        if text is not None:
            self.bar_text = text
        c, p = self.pbar, self.pal
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        c.configure(bg=p["bg"])
        if w < 20:
            return
        rrect(c, 1, 1, w - 1, h - 1, h // 2, fill=p["card2"], outline=p["line"])
        pct = self.pdone / self.ptotal if self.ptotal else 0
        fw = int(w * min(pct, 1))
        if fw > 0:
            rrect(c, 1, 1, max(fw, h), h - 1, h // 2, fill=p["accent"], outline="")
            c.create_line(h // 2, 5, max(fw, h) - h // 2, 5, fill=mix(p["accent"], "#ffffff", 0.35), width=2)
        parts = [f"{int(pct * 100)}%   {self.pdone} / {self.ptotal} videos", self.bar_text, self.extra]
        c.create_text(w // 2, h // 2, text="   •   ".join(x for x in parts if x),
                      fill="#ffffff" if pct > 0.5 else p["text"], font=("Segoe UI", 10, "bold"))

    def set_state(self, text):
        self.state_lbl.config(text=text)

    def tick(self):
        self.set_bar(self.pdone + 1)

    # ---------- helpers ----------
    def thread(self, fn):
        def go():
            try:
                fn()
            except Exception as e:
                self.log_error(traceback.format_exc())
                self.say(f"⚠ Unexpected error: {str(e)[:120]} (details in {ERRLOG})", "err")
        threading.Thread(target=go, daemon=True).start()

    def say(self, msg, tag=None):
        if msg.strip():
            self.loglines.append((time.strftime("%H:%M:%S"), tag or "", msg.strip().splitlines()[-1][:300]))
        self.q.put((msg, tag))

    def pump(self):
        while not self.q.empty():
            msg, tag = self.q.get()
            if tag and tag.startswith("\x00toast:"):
                self.toast(msg, tag.split(":", 1)[1])
                continue
            self.log.insert("end", msg + "\n", tag)
            self.log.see("end")
            if msg.strip():
                self.msg_lbl.config(text=msg.strip().splitlines()[-1][:170], fg=self.pal.get(
                    {"err": "err", "ok": "ok", "info": "accent2"}.get(tag, "muted")))
        self.after(100, self.pump)

    def browse(self):
        f = filedialog.askdirectory()
        if f:
            self.folder.set(f)

    def browse_cookies(self):
        f = filedialog.askopenfilename(filetypes=[("Cookies", "*.txt"), ("All", "*.*")])
        if f:
            self.cookies.set(f)

    def on_close(self):
        if self.running and not messagebox.askyesno(APP, "Downloads are running.\nExit anyway? (You can resume next time.)"):
            return
        try:
            self.geom.set(self.geometry())
        except tk.TclError:
            pass
        self.save_settings(silent=True)
        self.stop_flag = True
        self.pause_ev.set()
        self.destroy()

    def save_settings(self, silent=False):
        d = {}
        for k, v in self.cfg.items():
            try:
                d[k] = v.get()
            except tk.TclError:
                d[k] = self.defaults.get(k)
        jsave(SETTINGS, d)
        if not silent:
            self.say("Settings saved ✔", "ok")

    def save_later(self):  # every change is saved automatically a moment after you make it
        if self._save_after:
            self.after_cancel(self._save_after)
        self._save_after = self.after(800, lambda: self.save_settings(silent=True))

    def reset_settings(self):
        for k, v in self.cfg.items():
            v.set(self.defaults[k])
        self.apply_theme()

    def update_ytdlp(self):
        def go():
            self.say("Updating yt-dlp...", "info")
            try:
                r = subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp"],
                                   capture_output=True, text=True, timeout=300)
                last = (r.stdout.strip().splitlines() or [""])[-1][:100]
                self.say(("Updated ✔ — please restart the app. " if r.returncode == 0 else "Update failed: ") + last,
                         "ok" if r.returncode == 0 else "err")
            except Exception as e:
                self.say(f"Update failed: {e}", "err")
        self.thread(go)

    # ---------- docs list ----------
    def toggle(self, event):
        iid = self.tree.identify_row(event.y)
        if iid:
            self.sel[iid] = not self.sel.get(iid, False)
            self.tree.set(iid, "sel", "☑" if self.sel[iid] else "☐")
            self.count()

    def set_all(self, val):
        for iid in self.tree.get_children():
            self.sel[iid] = val
            self.tree.set(iid, "sel", "☑" if val else "☐")
        self.count()

    def count(self):
        n = sum(self.sel.values())
        self.cnt_lbl.config(text=f"{n} profiles selected")
        ds = list(self.docs.values())
        self.stats["docs"].config(text=str(len(ds)))
        self.stats["prof"].config(text=str(sum(len(d["profiles"]) for d in ds)))
        self.stats["links"].config(text=str(sum(len(p["videos"]) for d in ds for p in d["profiles"])))
        self.stats["sel"].config(text=str(n))

    def add_row(self, iid, vals):
        self.tree.insert("", "end", iid=iid, values=tuple(vals) + ("auto",))
        self.row_order.append(iid)
        self.sel[iid] = True
        self.count()

    def clear_rows(self):
        self.tree.delete(*self.row_order)  # also removes rows hidden by the search box
        self.tree.delete(*self.tree.get_children())
        self.row_order.clear()
        if self.dl_filter.get():
            self.dl_filter.set("")
        self.sel.clear()
        self.take.clear()
        self.count()

    # ---------- Google login / accounts ----------
    def read_active(self):
        try:
            with open(ACTIVE, encoding="utf-8") as f:
                return f.read().strip() or None
        except Exception:
            return None

    def cache_path(self):
        return f"docs_cache_{self.email}.json" if self.email else "docs_cache.json"

    def activate(self, creds):
        self.creds = creds
        self.docs_api = build("docs", "v1", credentials=creds, cache_discovery=False)
        self.drive_api = build("drive", "v3", credentials=creds, cache_discovery=False)
        u = self.drive_api.about().get(fields="user(emailAddress,displayName)").execute()["user"]
        self.email, self.uname = u["emailAddress"], u.get("displayName", "")
        os.makedirs(TOKDIR, exist_ok=True)
        with open(os.path.join(TOKDIR, self.email + ".json"), "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        with open(ACTIVE, "w", encoding="utf-8") as f:
            f.write(self.email)
        self.after(0, self.show_account)
        self.say(f"Account active: {self.email}", "ok")

    def show_account(self):
        em = self.email if self.docs_api else None
        self.chip.config(text=f"👤 {em}" if em else "👤 Not logged in")
        self.avatar.config(text=em[0].upper() if em else "?")
        self.acc_name.config(text=(self.uname or em) if em else "Not logged in")
        self.acc_mail.config(text=em or "Click 'Login / Add account' to connect a Gmail")
        self.acc_state.config(text="● Connected  —  this account is running" if em else "● Disconnected")
        self.atree.delete(*self.atree.get_children())
        if os.path.isdir(TOKDIR):
            for fn in sorted(os.listdir(TOKDIR)):
                if fn.endswith(".json"):
                    e = fn[:-5]
                    on = em == e
                    self.atree.insert("", "end", iid=e, values=("●" if on else "", e, "Active" if on else "Saved"))

    def use_account(self, email, legacy=False):
        try:
            path = "token.json" if legacy else os.path.join(TOKDIR, email + ".json")
            creds = Credentials.from_authorized_user_file(path, SCOPES)
            if not creds.valid and creds.refresh_token:
                creds.refresh(Request())
            self.activate(creds)
            self.load_docs()
        except Exception as e:
            self.say(f"Login failed: {str(e)[:100]}", "err")

    def try_saved_login(self):
        if self.email and os.path.exists(os.path.join(TOKDIR, self.email + ".json")):
            self.thread(lambda: self.use_account(self.email))
        elif os.path.exists("token.json"):
            self.thread(lambda: self.use_account(None, legacy=True))
        self.show_account()

    def login(self):
        def go():
            try:
                if not os.path.exists("credentials.json"):
                    self.say("credentials.json not found next to tikflow.py", "err")
                    return
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                self.activate(flow.run_local_server(port=0, prompt="select_account"))
                self.load_docs()
            except Exception as e:
                self.say(f"Login error: {e}", "err")
        self.thread(go)

    def use_selected(self):
        sel = self.atree.selection()
        if sel:
            self.thread(lambda: self.use_account(sel[0]))

    def remove_selected(self):
        for e in self.atree.selection():
            try:
                os.remove(os.path.join(TOKDIR, e + ".json"))
            except OSError:
                pass
            if e == self.email:
                self.logout()
        self.show_account()

    def logout(self):
        for pth in (os.path.join(TOKDIR, f"{self.email}.json") if self.email else "", "token.json", ACTIVE):
            if pth and os.path.exists(pth):
                os.remove(pth)
        self.creds = self.docs_api = self.drive_api = None
        self.email, self.uname = None, ""
        self.docs = {}
        self.clear_rows()
        self.refresh_progress()
        self.show_account()
        self.say("Logged out")

    # ---------- Docs ----------
    def parse_doc(self, doc):
        profiles, cur = [], None
        for el in doc.get("body", {}).get("content", []):
            p = el.get("paragraph")
            if not p:
                continue
            text = "".join(r["textRun"]["content"] for r in p["elements"] if "textRun" in r)
            base = p["elements"][0].get("startIndex", 0)
            v = VIDEO.search(text)
            if v:
                if cur is None:
                    cur = {"name": "(no profile)", "url": "", "videos": []}
                    profiles.append(cur)
                url = v.group(0).strip()
                cur["videos"].append({"url": url, "start": base + v.start(), "end": base + v.start() + len(url)})
                continue
            m = PROFILE.search(text)
            if m:
                url = m.group(0).strip()
                cur = {"name": url.split("@")[1].split("/")[0].split("?")[0], "url": url, "videos": []}
                profiles.append(cur)
        return profiles

    def save_cache(self):
        jsave(self.cache_path(), self.docs)

    def load_cache(self):
        data = jload(self.cache_path(), None)
        if not data:
            return
        self.docs = data
        for did, d in self.docs.items():
            for i, p in enumerate(d["profiles"]):
                self.add_row(f"{did}|{i}", ("☑", d["title"], "@" + p["name"], len(p["videos"])))
        self.refresh_progress()
        self.say("Docs shown from last session. Click Load Docs to refresh.", "info")

    def load_docs(self):
        if not self.docs_api:
            self.say("Please login first (Accounts tab)", "err")
            return
        t0 = time.time()
        self.say("Loading Google Docs...", "info")
        self.docs = {}
        self.after(0, self.clear_rows)
        files = self.drive_api.files().list(
            q="mimeType='application/vnd.google-apps.document' and trashed=false",
            pageSize=200, fields="files(id,name)").execute().get("files", [])
        if self.creds.expired and self.creds.refresh_token:
            self.creds.refresh(Request())
        local = threading.local()

        def fetch(f):
            if not hasattr(local, "api"):
                local.api = build("docs", "v1", credentials=self.creds, cache_discovery=False)
            doc = local.api.documents().get(documentId=f["id"], fields=DOC_FIELDS).execute()
            return f, self.parse_doc(doc)

        with ThreadPoolExecutor(max_workers=10) as ex:
            for fut in as_completed([ex.submit(fetch, f) for f in files]):
                try:
                    f, profs = fut.result()
                except Exception as e:
                    self.say(f"Could not read a doc: {str(e)[:80]}", "err")
                    continue
                if not profs:
                    continue
                self.docs[f["id"]] = {"title": f["name"], "profiles": profs}
                for i, p in enumerate(profs):
                    self.after(0, lambda f=f, p=p, i=i: self.add_row(
                        f"{f['id']}|{i}", ("☑", f["name"], "@" + p["name"], len(p["videos"]))))
        self.after(0, self.refresh_progress)
        self.after(0, self.lc_docs_refresh)
        self.save_cache()
        self.say(f"Loaded {len(self.docs)} doc(s) in {time.time() - t0:.1f}s", "ok")

    # ---------- progress tab ----------
    def prow(self, did, idx):
        p = self.docs[did]["profiles"][idx]
        rem = len(p["videos"])
        down = self.hist.get(f"{did}|{p['name']}", 0)
        tot = down + rem
        pct = down * 100 // tot if tot else 0
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10) + f"  {pct}%"
        return (self.docs[did]["title"], "@" + p["name"], down, rem, tot, bar)

    def psummary(self):
        d = r = 0
        for did, doc in list(self.docs.items()):
            for idx in range(len(doc["profiles"])):
                v = self.prow(did, idx)
                d += v[2]
                r += v[3]
        self.psum.config(text=f"Downloaded so far: {d}      Remaining: {r}      Total: {d + r}")

    def refresh_progress(self):
        self.ptree.delete(*self.ptree.get_children())
        for did, doc in list(self.docs.items()):
            for idx in range(len(doc["profiles"])):
                self.ptree.insert("", "end", iid=f"{did}|{idx}", values=self.prow(did, idx))
        self.psummary()

    def update_prow(self, did, idx):
        iid = f"{did}|{idx}"
        if self.ptree.exists(iid):
            self.ptree.item(iid, values=self.prow(did, idx))
        self.psummary()

    # ---------- history tab ----------
    def refresh_history(self):
        self.htree.delete(*self.htree.get_children())
        today = time.strftime("%Y-%m-%d")
        self.stats["today"].config(text=str(sum(1 for r in self.records if r.get("status") == "done"
                                                  and str(r.get("t", "")).startswith(today))))
        flt = self.hfilter.get().lower()
        for i in range(len(self.records) - 1, -1, -1):
            r = self.records[i]
            if flt != "all" and r["status"] != flt:
                continue
            self.htree.insert("", "end", iid=str(i), values=(
                r["t"], "✔ Done" if r["status"] == "done" else "✖ Failed",
                r.get("title") or "4K Downloader", r.get("pname") or "-", r.get("name") or r["url"]))

    def h_rec(self):
        s = self.htree.selection()
        return self.records[int(s[0])] if s else None

    def h_open(self):
        r = self.h_rec()
        if r:
            open_path(r["out"] if os.path.isdir(r.get("out") or "") else os.path.dirname(r.get("file") or ""))

    def h_play(self):
        r = self.h_rec()
        if r and r.get("file") and os.path.exists(r["file"]):
            open_path(r["file"])
        elif r:
            self.say("File not found on disk (moved or deleted?)", "err")

    def clear_history(self):
        if messagebox.askyesno(APP, "Clear the whole history?\n(Videos already downloaded may be downloaded again.)"):
            self.records, self.done_ids = [], set()
            jsave(RECORDS, [])
            self.refresh_history()

    def retry_failed(self):
        failed = [r for r in self.records if r["status"] == "failed"]
        if not failed:
            self.say("No failed videos in history", "info")
            return
        jobs = []
        for r in failed:
            j = {k: r.get(k) for k in ("did", "title", "pname", "url", "out", "fmt", "quality", "uni", "h")}
            j.update(id=len(jobs) + 1, n=len(jobs) + 1, status="pending", removed=False)
            jobs.append(j)
        self.say(f"Retrying {len(jobs)} failed video(s)...", "info")
        self.execute(jobs)

    # ---------- 4K / universal tab ----------
    def uni_paste(self):
        try:
            self.paste.insert("end", self.clipboard_get().strip() + "\n")
        except tk.TclError:
            pass

    def uni_clear(self):
        self.paste.delete("1.0", "end")
        self.utree.delete(*self.utree.get_children())

    def uni_start(self):
        links = list(dict.fromkeys(re.findall(r'https?://\S+', self.paste.get("1.0", "end"))))
        if not links:
            self.say("Paste at least one link first", "err")
            return
        if self.running:
            self.say("Another task is already running", "err")
            return
        out = os.path.join(self.folder.get(), "4K Downloader")
        self.utree.delete(*self.utree.get_children())
        jobs = []
        for i, u in enumerate(links, 1):
            jobs.append(dict(id=i, n=i, did=None, title="", pname="", url=u, out=out, uni=True,
                             fmt="MP3" if self.u_fmt.get().startswith("MP3") else "MP4",
                             quality="Best", h=UNI_Q[self.u_q.get()], status="pending", removed=False))
            self.utree.insert("", "end", iid=f"u{i}", values=(i, u, "", "Waiting", ""))
        self.thread(lambda: self.execute(jobs))

    def uni_set(self, job, title=None, status=None, pct=None):
        if not job.get("uni"):
            return
        iid = f"u{job['id']}"
        if self.utree.exists(iid):
            for col, val in (("title", title), ("status", status), ("pct", pct)):
                if val is not None:
                    self.utree.set(iid, col, val)

    # ---------- run control ----------
    def set_take(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        n = simpledialog.askinteger("Links for this profile",
                                    f"How many links to download from {self.tree.set(iid, 'profile')}?\n(0 = use the global setting)",
                                    parent=self, minvalue=0, maxvalue=500)
        if n is None:
            return
        if n:
            self.take[iid] = n
        else:
            self.take.pop(iid, None)
        self.tree.set(iid, "take", n if n else "auto")

    def open_folder(self):
        p = self.folder.get()
        os.makedirs(p, exist_ok=True)
        open_path(p)

    def save_log(self):
        f = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="tikvortex_log.txt",
                                         filetypes=[("Text", "*.txt")])
        if f:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(self.log.get("1.0", "end"))
            self.say(f"Log saved: {f}", "ok")

    def export_csv(self):
        if not self.records:
            self.say("History is empty", "err")
            return
        f = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="tikvortex_history.csv",
                                         filetypes=[("CSV (Excel)", "*.csv")])
        if not f:
            return
        with open(f, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["Time", "Status", "Doc", "Profile", "Video", "Link", "File"])
            for r in self.records:
                w.writerow([r.get("t"), r.get("status"), r.get("title") or "4K Downloader", r.get("pname") or "",
                            r.get("name") or "", r.get("url"), r.get("file") or ""])
        self.say(f"History exported: {f}", "ok")

    def clip_loop(self):  # auto-capture copied links into the 4K tab
        if self.clip_on.get():
            try:
                txt = self.clipboard_get().strip()
            except tk.TclError:
                txt = ""
            if txt and txt != self.last_clip:
                self.last_clip = txt
                if re.fullmatch(r"https?://\S+", txt) and txt not in self.paste.get("1.0", "end"):
                    self.paste.insert("end", txt + "\n")
                    self.say("📋 Link captured from clipboard", "info")
        self.after(1000, self.clip_loop)

    def update_disk(self):
        path = self.folder.get() or os.path.expanduser("~")
        while path and not os.path.exists(path):
            nxt = os.path.dirname(path)
            if nxt == path:
                break
            path = nxt
        try:
            free = shutil.disk_usage(path).free / 1024 ** 3
            self.disk_lbl.config(text=f"💾 {free:,.1f} GB free", fg=self.pal["err"] if free < 2 else self.pal["muted"])
        except Exception:
            pass

    def finish_ui(self, msg):
        self.notify(msg)
        self.power_action()

    def power_action(self):
        act = self.after_act.get()
        if act == "Do nothing":
            return
        if not sys.platform.startswith("win"):
            self.say("Sleep / Shutdown after finishing works on Windows only", "err")
            return
        if act == "Shutdown PC":
            subprocess.Popen(["shutdown", "/s", "/t", "60"])
            if not messagebox.askokcancel(APP, "All done. The PC will shut down in 60 seconds.\n\nClick Cancel to stop it."):
                subprocess.Popen(["shutdown", "/a"])
        elif messagebox.askokcancel(APP, "All done. The PC will go to sleep now.\n\nClick Cancel to stay awake."):
            subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])

    def _wheel(self, e):
        try:
            w = self.winfo_containing(e.x_root, e.y_root)
        except Exception:
            return
        while w is not None:
            if isinstance(w, (ttk.Treeview, tk.Text, tk.Listbox)):
                return
            if isinstance(w, ScrollFrame):
                w.cv.yview_scroll(int(-e.delta / 120) or (-1 if e.delta > 0 else 1), "units")
                return
            w = getattr(w, "master", None)

    def eo(self):  # editing options (safe against half-typed numbers)
        o = {}
        for k, v in self.ev.items():
            try:
                o[k] = v.get()
            except tk.TclError:
                o[k] = self.defaults.get("e_" + k)
        return o

    def browse_logo(self):
        f = filedialog.askopenfilename(filetypes=[("Pictures", "*.png *.jpg *.jpeg *.webp")])
        if f:
            self.ev["logo_path"].set(f)

    def set_hov(self, key):
        self.nav_hov = key
        self.paint_nav()

    def show_preview(self, png):
        w = tk.Toplevel(self)
        w.title("Preview")
        w.configure(bg=self.pal["bg"])
        w.img = tk.PhotoImage(file=png)
        tk.Label(w, image=w.img, bg=self.pal["bg"]).pack(padx=14, pady=14)
        tk.Label(w, text="Preview frame — the final video keeps full quality", bg=self.pal["bg"],
                 fg=self.pal["muted"], font=("Segoe UI", 9)).pack(pady=(0, 12))

    # ----- Brand tab: live preview -----
    def bp_later(self):
        if self._bp_after:
            self.after_cancel(self._bp_after)
        self._bp_after = self.after(700, self.bp_update_now)

    def bp_pick_sample(self):
        f = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v")])
        if f:
            self.ev["brand_sample"].set(f)

    def bp_update_now(self):
        self._bp_after = None
        if self._bp_busy:
            self._bp_again = True
            return
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            self.bp_label.config(text="Install ffmpeg (winget install ffmpeg)\nto see the live preview", image="")
            return
        self._bp_busy = True
        self.thread(self._bp_work)

    def _bp_work(self):
        try:
            user = self.ev["brand_prev_user"].get().strip().lstrip("@") or "username"
            sample = self.ev["brand_sample"].get()
            if not (sample and os.path.exists(sample)):
                sample = self.sample_png()
            self.after(0, self.bp_show, self.edit_file(sample, user, preview=True, parts=("brand",), pw=190))
        except Exception as e:
            self.after(0, lambda m=str(e)[:90]: self.bp_label.config(text=f"Preview: {m}", image=""))
        finally:
            self._bp_busy = False
            if self._bp_again:
                self._bp_again = False
                self.after(0, self.bp_update_now)

    def bp_show(self, png):
        img = tk.PhotoImage(file=png)
        self.bp_label.config(image=img, text="")
        self.bp_label.image = img
        try:
            os.remove(png)
        except OSError:
            pass

    def sample_png(self):
        if self._sample and os.path.exists(self._sample):
            return self._sample
        from PIL import Image, ImageDraw
        w, h = 360, 640
        im = Image.new("RGB", (w, h))
        d = ImageDraw.Draw(im)
        for y in range(h):
            c = int(30 + 90 * y / h)
            d.line([(0, y), (w, y)], fill=(c, 40 + c // 3, 150 + c // 2))
        d.ellipse((115, 200, 245, 330), fill=(255, 255, 255))
        d.rectangle((70, 360, 290, 460), fill=(255, 255, 255))
        d.text((135, 520), "SAMPLE VIDEO", fill=(255, 255, 255))
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.close()
        im.save(f.name)
        self._sample = f.name
        return f.name

    # ----- Clipper tab -----
    def clip_paste(self):
        try:
            self.ev["clip_src"].set(self.clipboard_get().strip())
        except tk.TclError:
            pass
        self.clip_probe_async()

    def clip_browse(self):
        f = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v")])
        if f:
            self.ev["clip_src"].set(f)
            self.clip_probe_async()

    def clip_probe_async(self):
        src = self.ev["clip_src"].get().strip().strip('"')
        self.clip_dur = 0.0
        if os.path.exists(src) and shutil.which("ffprobe"):
            def go():
                try:
                    self.clip_dur = probe(src)["dur"]
                except Exception:
                    self.clip_dur = 0.0
                self.after(0, self.clip_estimate)
            self.thread(go)
        else:
            self.clip_estimate()

    def clip_estimate(self):
        try:
            length = max(5.0, float(self.ev["clip_min"].get()) * 60)
            last = float(self.ev["clip_minlast"].get())
            ov = float(self.ev["clip_overlap"].get())
        except (tk.TclError, ValueError):
            return
        if self.clip_dur > 0:
            n = len(split_segments(self.clip_dur, length, last, ov))
            self.clip_info.config(text=f"Video length {hms(self.clip_dur)}   →   about {n} clip(s) of {hms(length)}")
        else:
            self.clip_info.config(text=f"Each clip will be {hms(length)} long. A link is downloaded first, then cut.")

    def clip_open(self):
        d = self.clip_outdir or os.path.join(self.folder.get(), "Clipper")
        os.makedirs(d, exist_ok=True)
        open_path(d)

    def clip_fetch(self, url):
        base = self.ydl_base()
        with yt_dlp.YoutubeDL({**base, "skip_download": True}) as y:
            info = y.extract_info(url, download=False)
        title = clean(info.get("title") or info.get("id") or "video")
        out = os.path.join(self.folder.get(), "Clipper", "_source")
        os.makedirs(out, exist_ok=True)

        def hk(d):
            if self.stop_flag:
                raise Cancelled()
            if d.get("status") == "downloading":
                tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                if tot:
                    self.after(0, self.set_bar, None, None, f"Downloading {int(d.get('downloaded_bytes', 0) * 100 / tot)}%")
        opts = {**base, "format": "bv*[height<=1080]+ba/b[height<=1080]/b", "merge_output_format": "mp4/mkv",
                "outtmpl": os.path.join(out, title.replace("%", "%%") + ".%(ext)s"), "progress_hooks": [hk]}
        self.after(0, self.set_bar, 0, 1, "Downloading the long video")
        with yt_dlp.YoutubeDL(opts) as y:
            res = y.extract_info(url, download=True)
        rd = res.get("requested_downloads") or []
        return (rd[0]["filepath"] if rd else os.path.join(out, title + ".mp4")), title

    def clip_rows(self, segs):
        self.ctree.delete(*self.ctree.get_children())
        for i, (st, ln) in enumerate(segs, 1):
            self.ctree.insert("", "end", iid=str(i), values=(i, hms(st), hms(st + ln), hms(ln), "Waiting"))

    def clip_status(self, i, text):
        if self.ctree.exists(str(i)):
            self.ctree.set(str(i), "status", text)

    def clip_make(self):
        if self.clip_busy:
            self.say("Clipper is already running", "err")
            return
        src = self.ev["clip_src"].get().strip().strip('"')
        if not src:
            self.say("Paste a long video link or choose a file first", "err")
            return
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            self.say("ffmpeg is required (run: winget install ffmpeg)", "err")
            return
        self.clip_busy, self.stop_flag = True, False
        done = 0
        try:
            o = self.eo()
            self.after(0, self.set_state, "● Clipper working...")
            if re.match(r"https?://", src):
                path, title = self.clip_fetch(src)
            elif os.path.exists(src):
                path, title = src, clean(os.path.splitext(os.path.basename(src))[0])
            else:
                raise Exception("file not found")
            info = probe(path)
            self.clip_dur = info["dur"]
            length = max(5.0, float(o["clip_min"]) * 60)
            segs = split_segments(info["dur"], length, float(o["clip_minlast"]), float(o["clip_overlap"]))
            if not segs:
                raise Exception("the video is too short")
            outdir = self.clip_outdir = os.path.join(self.folder.get(), "Clipper", title or "video")
            os.makedirs(outdir, exist_ok=True)
            self.after(0, self.clip_rows, segs)
            self.after(0, self.set_bar, 0, len(segs), "Making clips")
            self.after(0, self.clip_estimate)
            self.say(f"\n✂ {title}: {hms(info['dur'])} → {len(segs)} clips of {hms(length)}", "info")
            user = o["clip_user"].strip().lstrip("@")
            tmpl = o["clip_part_text"] or "Part {n}"
            if "{n}" not in tmpl:
                tmpl += " {n}"
            oo = dict(o)
            oo.update(brand_on=bool(o["clip_brand"]) and bool(user), logo_on=bool(o["clip_brand"]) and bool(o["logo_on"]),
                      aspect="9:16 Vertical" if o["clip_vertical"] else "Original", maxh="Original", mute=False,
                      fade=bool(o["clip_fade"]), part_pos=o["clip_part_pos"], part_size=o["clip_part_size"],
                      part_color=o["clip_part_color"])
            for i, (st, ln) in enumerate(segs, 1):
                if self.stop_flag:
                    break
                self.after(0, self.clip_status, i, "Working...")
                pt = tmpl.replace("{n}", str(i)).replace("{total}", str(len(segs))) if o["clip_part"] else None
                dst = os.path.join(outdir, f"{title} - Part {i}.mp4")
                cmd, temps = build_ffmpeg(path, dst, info, oo, user, seg=(st, ln), part_text=pt)
                try:
                    self.run_ffmpeg(cmd)
                finally:
                    for f in temps:
                        try:
                            os.remove(f)
                        except OSError:
                            pass
                done += 1
                self.after(0, self.clip_status, i, "✔ Done")
                self.after(0, self.tick)
                self.say(f"   ✔ Part {i}/{len(segs)}   {hms(st)} → {hms(st + ln)}", "ok")
            if not self.stop_flag:
                self.say(f"✂ {done} clips saved in: {outdir}", "ok")
                self.after(0, self.set_bar, None, None, "Clips ready ✔")
                self.after(0, self.notify, f"Clips ready!\n\n{done} clips saved in:\n{outdir}")
        except Cancelled:
            self.say("Clipper stopped", "err")
        except Exception as e:
            self.say(f"Clipper error: {str(e)[:140]}", "err")
        finally:
            self.clip_busy = False
            self.after(0, self.set_state, "● Idle")

    # ================= Link Collector =================
    def lc_paste(self):
        try:
            self.lc_in.insert("end", self.clipboard_get().strip() + "\n")
        except tk.TclError:
            pass

    def lc_from_docs(self):
        users = [p["name"] for d in self.docs.values() for p in d["profiles"] if p.get("url")]
        if not users:
            self.say("No profile links found in your docs — load docs first", "err")
            return
        self.lc_in.insert("end", "\n".join("https://www.tiktok.com/@" + u for u in dict.fromkeys(users)) + "\n")

    def lc_docs_refresh(self):
        self.lc_doc_map = {"➕ New Google Doc": None}
        for did, d in self.docs.items():
            self.lc_doc_map[f"{d['title']}  [{did[-5:]}]"] = did
        self.lc_doc_cb["values"] = list(self.lc_doc_map)
        if self.ev["lc_doc"].get() not in self.lc_doc_map:
            self.ev["lc_doc"].set("➕ New Google Doc")

    def lc_start(self):
        if self.lc_busy:
            self.say("Link collector is already running", "err")
            return
        users = tt_profiles(self.lc_in.get("1.0", "end").splitlines())
        if not users:
            self.say("Paste at least one TikTok profile link first", "err")
            return
        o = self.eo()
        self.thread(lambda: self.lc_work(users, o))

    def lc_kind(self, url):
        try:
            with yt_dlp.YoutubeDL({**self.ydl_base(), "skip_download": True}) as y:
                return kind_from_info(y.extract_info(url, download=False))
        except Exception as e:
            m = str(e).lower()
            return "picture" if ("photo" in m or "image" in m or "slideshow" in m) else "unknown"

    def lc_work(self, users, o):
        self.lc_busy, self.stop_flag = True, False
        latest = str(o["lc_order"]).startswith("Latest")
        maxn = int(o["lc_max"] or 0)
        rows, st = [], {"found": 0, "pics": 0, "done": 0}
        try:
            for user in users:
                if self.stop_flag:
                    break
                self.say(f"🔗 @{user}: reading the whole profile ...", "info")
                self.after(0, self.set_state, f"● Collecting @{user}")
                try:
                    opts = {**self.ydl_base(), "extract_flat": "in_playlist", "skip_download": True, "noplaylist": False}
                    with yt_dlp.YoutubeDL(opts) as y:
                        info = y.extract_info(f"https://www.tiktok.com/@{user}", download=False)
                    items = list(flat_items(info.get("entries"), user))
                except Exception as e:
                    self.say(f"@{user}: could not read the profile ({str(e)[:90]}). "
                             "Tip: Settings → Network → Browser cookies.", "err")
                    continue
                st["found"] += len(items)
                items.sort(key=lambda x: x["id"], reverse=latest)  # video ids grow with time
                cand = [x for x in items if x["kind"] != "picture"]
                st["pics"] += len(items) - len(cand)
                if o["lc_skipdone"]:
                    keep = [x for x in cand if str(x["id"]) not in self.done_ids]
                    st["done"] += len(cand) - len(keep)
                    cand = keep
                good, i = [], 0
                self.after(0, self.set_bar, 0, len(cand), f"@{user}")
                while i < len(cand) and not self.stop_flag and (not maxn or len(good) < maxn):
                    batch = cand[i:i + (12 if o["lc_deep"] else len(cand))]
                    i += len(batch)
                    if o["lc_deep"]:
                        with ThreadPoolExecutor(max_workers=6) as ex:
                            kinds = list(ex.map(lambda x: self.lc_kind(x["url"]), batch))
                    else:
                        kinds = ["video"] * len(batch)
                    for x, k in zip(batch, kinds):
                        if k == "picture":
                            st["pics"] += 1
                        else:
                            good.append({**x, "kind": k})
                    self.after(0, self.set_bar, i, len(cand), None)
                if maxn:
                    good = good[:maxn]
                rows += [{**x, "user": user} for x in good]
                self.say(f"   @{user}: {len(good)} video links ready", "ok")
        except Exception as e:
            self.say(f"Link collector error: {str(e)[:120]}", "err")
        finally:
            self.lc_res = rows
            self.after(0, self.lc_show, rows, st)
            self.lc_busy = False
            self.after(0, self.set_state, "● Idle")

    def lc_show(self, rows, st):
        self.lctree.delete(*self.lctree.get_children())
        for i, r in enumerate(rows):
            self.lctree.insert("", "end", iid=str(i), values=(
                "@" + r["user"], tt_date(r["id"]), "video" if r["kind"] == "video" else "check?", r["url"]))
        self.lc_sum.config(text=f"Found {st['found']}   •   picture posts removed {st['pics']}   •   "
                                f"already downloaded {st['done']}   •   READY {len(rows)}")
        self.set_bar(None, None, "Links ready ✔" if not self.stop_flag else "Stopped")

    def lc_copy(self):
        self.clipboard_clear()
        self.clipboard_append("\n".join(r["url"] for r in self.lc_res))
        self.say(f"{len(self.lc_res)} links copied", "ok")

    def lc_save(self):
        if not self.lc_res:
            self.say("Collect links first", "err")
            return
        f = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="tiktok_links.txt", filetypes=[("Text", "*.txt")])
        if f:
            with open(f, "w", encoding="utf-8") as fh:
                for u in dict.fromkeys(r["user"] for r in self.lc_res):
                    fh.write(f"https://www.tiktok.com/@{u}\n" + "\n".join(r["url"] for r in self.lc_res if r["user"] == u) + "\n\n")
            self.say(f"Saved: {f}", "ok")

    def lc_to_doc(self):
        if not self.lc_res:
            self.say("Collect links first", "err")
            return
        if not self.docs_api:
            self.say("Please login first (Accounts tab)", "err")
            return
        did = self.lc_doc_map.get(self.ev["lc_doc"].get())
        rows = list(self.lc_res)
        self.thread(lambda: self._lc_to_doc(did, rows))

    def _lc_to_doc(self, did, rows):
        try:
            existing = set()
            if did is None:
                did = self.docs_api.documents().create(
                    body={"title": "TikTok links " + time.strftime("%Y-%m-%d %H:%M")}).execute()["documentId"]
            else:
                fresh = self.parse_doc(self.docs_api.documents().get(documentId=did).execute())
                existing = {v["url"] for pr in fresh for v in pr["videos"]}
            by = {}
            for r in rows:
                if r["url"] not in existing:
                    by.setdefault(r["user"], []).append(r["url"])
            if not by:
                self.say("Nothing new to add — these links are already in that doc", "info")
                return
            text = "".join(f"https://www.tiktok.com/@{u}\n" + "\n".join(urls) + "\n\n" for u, urls in by.items())
            self.docs_api.documents().batchUpdate(documentId=did, body={"requests": [
                {"insertText": {"text": text, "endOfSegmentLocation": {}}}]}).execute()
            self.say(f"✔ {sum(len(v) for v in by.values())} links added to the Google Doc", "ok")
            self.load_docs()
        except Exception as e:
            self.say(f"Could not write to the doc: {str(e)[:110]}", "err")

    # ================= Smart Clips =================
    def sc_paste(self):
        try:
            self.ev["sc_src"].set(self.clipboard_get().strip())
        except tk.TclError:
            pass

    def sc_browse(self):
        f = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v")])
        if f:
            self.ev["sc_src"].set(f)

    def sc_open(self):
        d = self.clip_outdir or os.path.join(self.folder.get(), "Clipper")
        os.makedirs(d, exist_ok=True)
        open_path(d)

    def cut_one(self, path, info, dst, st, ln, o, user, ptext):
        oo = dict(o)
        oo.update(brand_on=bool(o["clip_brand"]) and bool(user), logo_on=bool(o["clip_brand"]) and bool(o["logo_on"]),
                  aspect="9:16 Vertical" if o["clip_vertical"] else "Original", maxh="Original", mute=False,
                  fade=bool(o["clip_fade"]), part_pos=o["clip_part_pos"], part_size=o["clip_part_size"],
                  part_color=o["clip_part_color"])
        cmd, temps = build_ffmpeg(path, dst, info, oo, user, seg=(st, ln), part_text=ptext)
        try:
            self.run_ffmpeg(cmd)
        finally:
            for f in temps:
                try:
                    os.remove(f)
                except OSError:
                    pass

    def sc_make(self):
        if self.clip_busy:
            self.say("Clipper is already running", "err")
            return
        src = self.ev["sc_src"].get().strip().strip('"')
        if not src:
            self.say("Paste a long video link or choose a file first", "err")
            return
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            self.say("ffmpeg is required (run: winget install ffmpeg)", "err")
            return
        self.clip_busy, self.stop_flag = True, False
        done = 0
        try:
            o = self.eo()
            self.after(0, self.set_state, "● Smart Clips working...")
            if re.match(r"https?://", src):
                path, title = self.clip_fetch(src)
            elif os.path.exists(src):
                path, title = src, clean(os.path.splitext(os.path.basename(src))[0])
            else:
                raise Exception("file not found")
            info = probe(path)
            self.after(0, self.set_bar, 0, 1, "Listening to the sound...")
            db = energy_profile(path)
            length = min(float(o["sc_len"]), info["dur"])
            picks = pick_highlights(db, int(o["sc_count"]), length, float(o["sc_gap"]))
            segs = [(float(s), min(length, info["dur"] - s), sc) for s, sc in picks]
            if o["sc_rank"] == "Time order":
                segs.sort(key=lambda x: x[0])
            outdir = self.clip_outdir = os.path.join(self.folder.get(), "Clipper", (title or "video") + " - highlights")
            os.makedirs(outdir, exist_ok=True)
            self.after(0, self.sc_rows, segs)
            self.after(0, self.set_bar, 0, len(segs), "Cutting highlights")
            self.say(f"\n🔥 {title}: {len(segs)} highlights of {hms(length)}", "info")
            user = o["clip_user"].strip().lstrip("@")
            tmpl = o["sc_label_text"] or "Highlight {n}"
            if "{n}" not in tmpl:
                tmpl += " {n}"
            for i, (st, ln, sc) in enumerate(segs, 1):
                if self.stop_flag:
                    break
                self.after(0, self.sc_status, i, "Working...")
                pt = tmpl.replace("{n}", str(i)).replace("{total}", str(len(segs))) if o["sc_label"] else None
                self.cut_one(path, info, os.path.join(outdir, f"{title} - Highlight {i}.mp4"), st, ln, o, user, pt)
                done += 1
                self.after(0, self.sc_status, i, "✔ Done")
                self.after(0, self.tick)
                self.say(f"   ✔ Highlight {i}   {hms(st)} → {hms(st + ln)}", "ok")
            if not self.stop_flag:
                self.after(0, self.set_bar, None, None, "Highlights ready ✔")
                self.after(0, self.notify, f"Highlights ready!\n\n{done} clips saved in:\n{outdir}")
        except Cancelled:
            self.say("Smart Clips stopped", "err")
        except Exception as e:
            self.say(f"Smart Clips error: {str(e)[:140]}", "err")
        finally:
            self.clip_busy = False
            self.after(0, self.set_state, "● Idle")

    def sc_rows(self, segs):
        self.stree.delete(*self.stree.get_children())
        for i, (st, ln, sc) in enumerate(segs, 1):
            self.stree.insert("", "end", iid=str(i), values=(i, hms(st), hms(st + ln), f"{sc:.0f} dB", "Waiting"))

    def sc_status(self, i, text):
        if self.stree.exists(str(i)):
            self.stree.set(str(i), "status", text)

    # ================= Captions =================
    def cap_style(self, o):
        return dict(font=o["cp_font"], size=o["cp_size"], color=o["cp_color"], outline=o["cp_outline"], hl=o["cp_hl"],
                    outw=o["cp_outw"], box=o["cp_box"], bold=o["cp_bold"], wordhl=o["cp_wordhl"], pos=o["cp_pos"],
                    margin=o["cp_margin"], shape=o["cp_shape"], boxcolor=o["cp_boxcolor"], anim=o["cp_anim"],
                    italic=o["cp_italic"], spacing=o["cp_spacing"])

    def cp_preset(self, name):
        for k, v in {**CAP_DEFAULTS, **CAP_STYLES[name]}.items():
            self.ev[k].set(v)
        self.say(f"Caption style: {name}", "info")

    def whisper_model(self, size):
        if self._wm and self._wm[0] == size:
            return self._wm[1]
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise Exception("Captions need:  pip install faster-whisper")
        self.say(f"Loading speech model '{size}' (the first time it downloads)...", "info")
        model = WhisperModel(size, device="cpu", compute_type="int8")
        self._wm = (size, model)
        return model

    def hf_env(self, mirror=None, level=0):
        env = dict(os.environ)
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        if mirror:
            env["HF_ENDPOINT"] = mirror
        if level:  # safe mode: generic CPU instructions + one thread (fixes crashes on older / unusual CPUs)
            env["CT2_FORCE_CPU_ISA"] = "GENERIC"
            env["OMP_NUM_THREADS"] = "1"
        return env

    def transcribe(self, path, o):
        """Listen to a video (in a separate process). Returns (caption lines, language code)."""
        with self.wlock:
            code = CAP_LANGS.get(o["cp_lang"])
            task = "translate" if o.get("cp_translate") else "transcribe"
            if task == "translate" and code == "en":
                code = None  # detect the real spoken language (Urdu etc.) and translate it to English
            mdir = (o.get("cp_modeldir") or "").strip()
            tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            tmp.close()
            base = self.speech_cmd(o)
            base += ["--transcribe", path, str(o["cp_model"]), code or "auto", tmp.name, mdir, task]
            speed = str(o.get("cp_speed") or "Fast")
            self.say(f"👂 Listening: {os.path.basename(path)[:50]}", "info")
            mirrors = ["https://hf-mirror.com", None] if (o.get("cp_mirror") or getattr(self, "_mirror_pref", None)) \
                else [None, "https://hf-mirror.com"]
            level, mi = getattr(self, "_asr_level", 0), 0  # level 1/2 = safe modes, remembered once one works
            err, rc = "", 1
            try:
                for _try in range(6):
                    p = subprocess.Popen(base + [str(level), speed], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                                         errors="replace", env=self.hf_env(mirrors[mi], level), **nowin())
                    self.procs.add(p)
                    tail, last_say, last_pct = collections.deque(maxlen=80), 0.0, -10
                    try:
                        for line in p.stderr:
                            line = line.strip()
                            if not line:
                                continue
                            tail.append(line[:300])
                            if line.startswith("STAGE "):
                                self.say("   ⏳ " + line[6:], "info")
                            else:
                                pm = re.search(r"(\d{1,3})%\|", line)
                                if pm and (int(pm.group(1)) >= last_pct + 10 or time.time() - last_say > 15):
                                    last_pct, last_say = int(pm.group(1)), time.time()
                                    self.say(f"   ⬇ downloading the speech model {pm.group(1)}%", "info")
                        p.wait()
                    finally:
                        self.procs.discard(p)
                    err = "\n".join(tail)
                    rc = p.returncode
                    if self.stop_flag:
                        raise Cancelled()
                    if rc == 0 and os.path.getsize(tmp.name):
                        self._asr_level = level
                        if mirrors[mi]:
                            self._mirror_pref = mirrors[mi]
                        tm = re.search(r"TIMING load=([\d.]+) listen=([\d.]+)", err)
                        if tm:
                            self.say(f"⏱ model {tm.group(1)}s  •  listening {tm.group(2)}s", "info")
                        break
                    if rc in CRASH_CODES and level < 2:
                        level += 1
                        self.say(f"⚙ The speech engine crashed (code {rc}) — retrying in safe mode {level} ...", "info")
                        continue
                    if mi + 1 < len(mirrors) and not mdir and NET_ERR.search(err):
                        mi += 1
                        self.say("🌐 The speech model could not be downloaded — trying another download server ...", "info")
                        continue
                    break
                if rc != 0 or not os.path.getsize(tmp.name):
                    self.log_error(f"speech engine failed (code {rc}, safe level {level}):\n{err[-2000:]}")
                    last = (err.strip().splitlines() or [""])[-1][:140]
                    if "No module named 'faster_whisper'" in err:
                        raise Exception("the speech engine is not installed — press  ⬇ Install / repair speech engine  in the Captions tab")
                    if rc in CRASH_CODES:
                        where = " | ".join(x.strip() for x in err.strip().splitlines()[-3:])[:220]
                        hint = ("Most likely an old Visual C++ runtime: press  🛠 Update Visual C++ runtime  in the Captions tab, "
                                "then try again")
                        raise Exception(f"the speech engine crashes on this PC (code {rc}) [{where}]. {hint}")
                    if NET_ERR.search(err):
                        raise Exception("could not download the speech model (internet / site blocked?). Try a VPN, or set an "
                                        "'Offline model folder' in Captions → Speech")
                    raise Exception(last or f"the speech engine stopped (code {rc})")
                with open(tmp.name, encoding="utf-8") as fh:
                    data = json.load(fh)
            finally:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass
        lang = data["language"]
        self.say(f"🌐 {LANG_NAMES.get(lang, lang)} ({data['prob'] * 100:.0f}%)" + (" → English" if task == "translate" else ""), "ok")
        if not data["segments"]:
            raise NoSpeech("no speech found in this video")
        return caption_lines(data["segments"], int(o["cp_chars"]), bool(o["cp_upper"])), lang

    def keep_original(self, src):
        """Put the original video into an _Originals folder (used when 'keep originals' is ticked)."""
        if os.path.exists(src):
            d = os.path.join(os.path.dirname(src), "_Originals")
            os.makedirs(d, exist_ok=True)
            shutil.move(src, os.path.join(d, os.path.basename(src)))

    def cl_tidy(self):
        def go():
            root, moved = self.folder.get(), 0
            trash = os.path.join(root, "_Cleaner_Trash", time.strftime("%Y%m%d_%H%M%S") + "_originals")
            for dp, dns, fns in os.walk(root):
                if os.path.basename(dp) not in ("Edited", "Captioned") or "_Cleaner_Trash" in dp:
                    continue
                parent = os.path.dirname(dp)
                for fn in fns:
                    stem, ext = os.path.splitext(fn)
                    if ext.lower() not in VIDEO_EXT + (".srt",):
                        continue
                    if stem.endswith(" (edited)"):
                        stem = stem[:-9]
                    target = os.path.join(parent, stem + ext)
                    if ext.lower() in VIDEO_EXT:  # the original with the same name moves to the trash (not deleted)
                        for e2 in VIDEO_EXT:
                            o = os.path.join(parent, stem + e2)
                            if os.path.exists(o):
                                os.makedirs(trash, exist_ok=True)
                                shutil.move(o, os.path.join(trash, os.path.basename(o)))
                    elif os.path.exists(target):
                        os.remove(target)
                    shutil.move(os.path.join(dp, fn), target)
                    moved += 1
            for dp, dns, fns in os.walk(root, topdown=False):
                if os.path.basename(dp) in ("Edited", "Captioned") and "_Cleaner_Trash" not in dp and not os.listdir(dp):
                    os.rmdir(dp)
            self.say(f"✨ {moved} edited file(s) moved up into their main folders. Originals are in "
                     f"_Cleaner_Trash (you can delete them from there).", "ok")
            self.update_disk()
        self.thread(go)

    def report_callback_exception(self, exc, val, tb):
        self.log_error("".join(traceback.format_exception(exc, val, tb)))
        self.say(f"⚠ Error: {val}", "err")

    def log_error(self, text):
        try:
            with open(ERRLOG, "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
        except OSError:
            pass

    def caption_file(self, path, user=""):
        o = self.eo()
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise Exception("ffmpeg is required (run: winget install ffmpeg)")
        path = os.path.abspath(path)
        info = probe(path)
        if o.get("cp_skip"):
            why = has_captions(path, info)
            if why:
                self.say(f"⏭ Skipped {os.path.basename(path)} — {why}", "info")
                return None
        try:
            lines, lang = self.transcribe(path, o)
        except NoSpeech:
            self.say(f"⏭ No speech in {os.path.basename(path)[:50]} — no captions needed", "info")
            return None
        folder, base = os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]
        replace = o["outmode"] == "Replace original"
        out_dir = folder if (replace or os.path.basename(folder) == "Captioned") else os.path.join(folder, "Captioned")
        os.makedirs(out_dir, exist_ok=True)
        srt_path = os.path.join(out_dir, base + ".srt")
        if o["cp_srt"] or not o["cp_burn"]:
            with open(srt_path, "w", encoding="utf-8") as fh:
                fh.write(srt_text(lines))
        if not o["cp_burn"]:
            return srt_path
        if not ffmpeg_has_ass():
            raise Exception(NO_ASS_MSG)
        assp = make_ass_file(build_ass(lines, info["w"], info["h"], self.cap_style(o)))
        tmp = os.path.join(out_dir, base + (".captioning.mp4" if replace else ".mp4"))
        final = os.path.join(out_dir, base + ".mp4")
        m = re.search(r"CRF (\d+)", str(o["crf"]))
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-vf", "ass=" + ass_arg(assp),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", m.group(1) if m else "23", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-metadata", f"comment={CAP_TAG}", tmp]
        try:
            self.run_ffmpeg(cmd, cwd=os.path.dirname(assp))
            if tmp != final:
                if o.get("keep_orig"):
                    self.keep_original(path)
                os.replace(tmp, final)
                if os.path.abspath(final) != os.path.abspath(path) and os.path.exists(path):
                    os.remove(path)
            return final
        except Exception:
            if tmp != final and os.path.exists(tmp):
                os.remove(tmp)
            raise
        finally:
            drop_ass(assp)

    def cp_later(self):
        if self._cp_after:
            self.after_cancel(self._cp_after)
        self._cp_after = self.after(700, self.cp_update_now)

    def cp_update_now(self):
        self._cp_after = None
        if self._cp_busy:
            self._cp_again = True
            return
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            self.cp_label.config(text="Install ffmpeg (winget install ffmpeg)\nto see the live preview", image="")
            return
        self._cp_busy = True
        self.thread(self._cp_work)

    def _cp_work(self):
        out = None
        try:
            o = self.eo()
            code = CAP_LANGS.get(o["cp_lang"]) or "en"
            words = SAMPLE_TEXT.get(code, SAMPLE_TEXT["en"]).split()
            per = 2.4 / max(1, len(words))
            seg = [{"start": 0.3, "end": 2.7, "text": " ".join(words),
                    "words": [{"start": 0.3 + i * per, "end": 0.3 + (i + 1) * per, "word": w} for i, w in enumerate(words)]}]
            merged = []
            for a_, b_, ws_ in caption_lines(seg, int(o["cp_chars"]), bool(o["cp_upper"])):
                if merged:
                    merged.append(("\n", 0, 0))
                merged += ws_
            lines = [(0.3, 2.9, merged)]
            assp = make_ass_file(build_ass(lines, 360, 640, self.cap_style(o)))
            out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            out.close()
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-t", "3", "-i", os.path.abspath(self.sample_png()),
                                "-ss", "1.5", "-vf", "ass=" + ass_arg(assp) + ",scale=190:-2", "-frames:v", "1", out.name],
                               capture_output=True, text=True, cwd=os.path.dirname(assp), **nowin())
            drop_ass(assp)
            if r.returncode != 0:
                raise Exception((r.stderr or "preview failed").strip().splitlines()[-1][:90])
            self.after(0, self.cp_show, out.name)
        except Exception as e:
            self.after(0, lambda m=str(e)[:90]: self.cp_label.config(text=f"Preview: {m}", image=""))
        finally:
            self._cp_busy = False
            if self._cp_again:
                self._cp_again = False
                self.after(0, self.cp_update_now)

    def cp_show(self, png):
        img = tk.PhotoImage(file=png)
        self.cp_label.config(image=img, text="")
        self.cp_label.image = img
        try:
            os.remove(png)
        except OSError:
            pass

    # ================= Thumbnails =================
    def th_browse(self):
        f = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v")])
        if f:
            self.ev["th_src"].set(f)

    def th_auto(self):
        src = self.ev["th_src"].get().strip().strip('"')
        if not os.path.exists(src):
            self.say("Choose a video first", "err")
            return

        def go():
            try:
                self.say("🎯 Looking for the best frame...", "info")
                t = best_frame_time(src, probe(src)["dur"])
                self.after(0, self.ev["th_time"].set, round(t, 1))
            except Exception as e:
                self.say(f"Auto-pick failed: {str(e)[:100]}", "err")
        self.thread(go)

    def th_later(self):
        if self._th_after:
            self.after_cancel(self._th_after)
        self._th_after = self.after(700, self.th_update_now)

    def th_update_now(self):
        self._th_after = None
        if self._th_busy:
            self._th_again = True
            return
        self._th_busy = True
        o = self.eo()
        self.thread(lambda: self._th_work(o))

    def th_frame(self, o):
        src = str(o["th_src"]).strip().strip('"')
        if not os.path.exists(src):
            raise Exception("choose a video first")
        key = (src, round(float(o["th_time"]), 1))
        if key not in self._th_cache or not os.path.exists(self._th_cache[key]):
            self._th_cache[key] = grab_frame(src, key[1])
        return self._th_cache[key]

    def _th_work(self, o):
        try:
            im = render_thumb(self.th_frame(o), o, o["th_user"])
            im.thumbnail((300, 420))
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            f.close()
            im.save(f.name)
            self.after(0, self.th_show, f.name)
        except ImportError:
            self.after(0, lambda: self.th_label.config(text="Needs:  pip install pillow", image=""))
        except Exception as e:
            self.after(0, lambda m=str(e)[:90]: self.th_label.config(text=f"Preview: {m}", image=""))
        finally:
            self._th_busy = False
            if self._th_again:
                self._th_again = False
                self.after(0, self.th_update_now)

    def th_show(self, png):
        img = tk.PhotoImage(file=png)
        self.th_label.config(image=img, text="")
        self.th_label.image = img
        try:
            os.remove(png)
        except OSError:
            pass

    def th_save(self):
        o = self.eo()
        ext = ".jpg" if o["th_fmt"] == "JPG" else ".png"
        d = os.path.join(self.folder.get(), "Thumbnails")
        os.makedirs(d, exist_ok=True)
        base = os.path.splitext(os.path.basename(str(o["th_src"])))[0] or "thumbnail"
        f = filedialog.asksaveasfilename(defaultextension=ext, initialdir=d, initialfile=base + ext,
                                         filetypes=[("Picture", "*" + ext)])
        if not f:
            return

        def go():
            try:
                im = render_thumb(self.th_frame(o), o, o["th_user"])
                im.save(f, quality=95) if ext == ".jpg" else im.save(f)
                self.say(f"🖼 Thumbnail saved: {f}", "ok")
            except Exception as e:
                self.say(f"Thumbnail failed: {str(e)[:110]}", "err")
        self.thread(go)

    # ================= Cleaner =================
    def cl_scan(self):
        root = self.folder.get()
        if not os.path.isdir(root):
            self.say("Save folder does not exist yet", "err")
            return
        o = self.eo()
        self.say("🧹 Scanning videos...", "info")
        files = scan_videos([root])
        items = find_clean(files, bool(o["cl_dups"]), int(o["cl_old"] or 0), int(o["cl_big"] or 0))
        self.after(0, self.cl_show, items, len(files))

    def cl_show(self, items, total):
        self.cl_items = items
        self.cl_sel = {}
        self.cltree.delete(*self.cltree.get_children())
        root = self.folder.get()
        for i, it in enumerate(items):
            on = any(w.startswith("Duplicate") for w in it["why"])
            self.cl_sel[str(i)] = on
            self.cltree.insert("", "end", iid=str(i), values=(
                "☑" if on else "☐", os.path.relpath(it["path"], root), f"{it['size'] / 1048576:.1f} MB", it["age"], " • ".join(it["why"])))
        self.cl_count()
        self.say(f"🧹 Scanned {total} videos — {len(items)} can be cleaned", "ok")

    def cl_count(self):
        sel = [self.cl_items[int(i)] for i, v in self.cl_sel.items() if v]
        self.cl_sum.config(text=f"{len(self.cl_items)} files found   •   {len(sel)} selected   •   "
                                f"frees {sum(x['size'] for x in sel) / 1073741824:.2f} GB")

    def cl_toggle(self, event):
        iid = self.cltree.identify_row(event.y)
        if iid:
            self.cl_sel[iid] = not self.cl_sel.get(iid, False)
            self.cltree.set(iid, "sel", "☑" if self.cl_sel[iid] else "☐")
            self.cl_count()

    def cl_all(self, val):
        for iid in self.cltree.get_children():
            self.cl_sel[iid] = val
            self.cltree.set(iid, "sel", "☑" if val else "☐")
        self.cl_count()

    def cl_open_trash(self):
        d = os.path.join(self.folder.get(), "_Cleaner_Trash")
        os.makedirs(d, exist_ok=True)
        open_path(d)

    def cl_delete(self):
        sel = [self.cl_items[int(i)] for i, v in self.cl_sel.items() if v]
        if not sel:
            self.say("Nothing selected", "err")
            return
        safe = bool(self.ev["cl_safe"].get())
        gb = sum(x["size"] for x in sel) / 1073741824
        if not messagebox.askyesno(APP, f"{'Move' if safe else 'PERMANENTLY DELETE'} {len(sel)} video(s) ({gb:.2f} GB)"
                                        f"{' to _Cleaner_Trash' if safe else ''}?"):
            return
        trash = os.path.join(self.folder.get(), "_Cleaner_Trash", time.strftime("%Y%m%d_%H%M%S"))
        n = 0
        for it in sel:
            try:
                if safe:
                    os.makedirs(trash, exist_ok=True)
                    shutil.move(it["path"], os.path.join(trash, os.path.basename(it["path"])))
                else:
                    os.remove(it["path"])
                n += 1
            except OSError as e:
                self.say(f"Could not remove {os.path.basename(it['path'])}: {e}", "err")
        self.say(f"🧹 {n} video(s) cleaned — {gb:.2f} GB freed", "ok")
        self.update_disk()
        self.thread(self.cl_scan)

    # ================= v5: search, toasts, backup, window memory =================
    def restore_geometry(self):
        m = re.match(r"^(\d+)x(\d+)\+(-?\d+)\+(-?\d+)$", self.geom.get() or "")
        if not m:
            return
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h, x, y = (int(g) for g in m.groups())
        w, h = max(min(w, sw - 20), 800), max(min(h, sh - 60), 600)
        x, y = max(0, min(x, sw - w)), max(0, min(y, sh - h - 40))
        self.geometry(f"{w}x{h}+{x}+{y}")

    def toast(self, msg, kind="ok"):
        p = self.pal
        col = {"ok": p["ok"], "err": p["err"]}.get(kind, p["accent2"])
        old = getattr(self, "_toast", None)
        if old is not None:
            try:
                old.destroy()
            except tk.TclError:
                pass
        t = self._toast = tk.Label(self, text=msg, font=("Segoe UI", 11, "bold"), bg=p["card2"], fg=col,
                                   padx=18, pady=10, highlightthickness=2, highlightbackground=col)
        t.place(relx=1.0, rely=1.0, x=-24, y=-52, anchor="se")
        t.lift()
        self.after(3500, lambda: self._toast_off(t))

    def _toast_off(self, t):
        try:
            t.destroy()
        except tk.TclError:
            pass

    def toast_safe(self, msg, kind="ok"):  # can be called from any thread
        self.q.put((msg, "\x00toast:" + kind))

    def toggle_sidebar(self, event=None):
        if self.side.winfo_ismapped():
            self.side.pack_forget()
        else:
            self.side.pack(side="left", fill="y", before=self.main)

    def set_theme(self, name):
        if name in THEMES:
            self.theme.set(name)
            self.apply_theme()
            self.toast(f"Theme: {name}", "info")

    def next_theme(self):
        names = list(THEMES)
        cur = self.theme.get()
        self.set_theme(names[(names.index(cur) + 1) % len(names)] if cur in names else DEFAULT_THEME)

    def palette_items(self):
        items = [(f"Go to  {PAGE_TITLES[k]}", lambda k=k: self.show_page(k)) for k in PAGE_TITLES]
        items += [("Run docs", lambda: self.thread(self.run)),
                  ("Load docs", lambda: self.thread(self.load_docs)),
                  ("Resume last run", lambda: self.thread(self.resume)),
                  ("Pause / continue", self.toggle_pause),
                  ("Stop", self.stop),
                  ("Open download folder", self.open_folder),
                  ("Save log", self.save_log),
                  ("Update yt-dlp", self.update_ytdlp),
                  ("Hide / show sidebar", self.toggle_sidebar),
                  ("Test captions (find the problem)", lambda: self.thread(self.cap_selftest)),
                  ("Install / repair speech engine", lambda: self.thread(self.cap_install)),
                  ("Build Python 3.12 speech environment", lambda: self.thread(self.cap_setup312)),
                  ("Update Visual C++ runtime", lambda: self.thread(self.cap_vcredist)),
                  ("Clear log", self.clear_log),
                  ("Copy log", self.copy_log),
                  ("Keep window on top  (on / off)", lambda: self.topmost.set(not self.topmost.get())),
                  ("Next theme", self.next_theme),
                  ("What's new", self.show_whats_new),
                  ("About / developer", self.show_about),
                  ("Export settings", self.export_settings),
                  ("Import settings", self.import_settings)]
        items += [(f"Theme:  {n}", lambda n=n: self.set_theme(n)) for n in THEMES]
        return items

    def palette(self, event=None):
        old = getattr(self, "_pal_win", None)
        if old is not None and old.winfo_exists():
            old.destroy()
            return
        p = self.pal
        w = self._pal_win = tk.Toplevel(self)
        w.overrideredirect(True)
        w.configure(bg=p["accent"])
        ww = 540
        w.geometry(f"{ww}x360+{self.winfo_rootx() + max(0, (self.winfo_width() - ww) // 2)}+{self.winfo_rooty() + 90}")
        inner = tk.Frame(w, bg=p["card"])
        inner.pack(fill="both", expand=True, padx=2, pady=2)
        tk.Label(inner, text="🔍  Search tabs & actions   (Esc to close)", font=("Segoe UI", 9, "bold"),
                 bg=p["card"], fg=p["muted"], anchor="w").pack(fill="x", padx=16, pady=(12, 0))
        var = tk.StringVar()
        ent = tk.Entry(inner, textvariable=var, font=("Segoe UI", 14), bg=p["card2"], fg=p["text"],
                       insertbackground=p["text"], bd=0, relief="flat")
        ent.pack(fill="x", padx=14, pady=(6, 8), ipady=8)
        lb = tk.Listbox(inner, font=("Segoe UI", 12), bg=p["card"], fg=p["text"], selectbackground=p["accent"],
                        selectforeground="#ffffff", bd=0, highlightthickness=0, activestyle="none")
        lb.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        cmds, shown = self.palette_items(), []

        def refresh(*_):
            terms = var.get().lower().split()
            lb.delete(0, "end")
            shown.clear()
            for label, fn in cmds:
                if all(t in label.lower() for t in terms):
                    shown.append(fn)
                    lb.insert("end", "   " + label)
            if shown:
                lb.selection_set(0)

        def go(event=None):
            sel = lb.curselection()
            fn = shown[sel[0] if sel else 0] if shown else None
            w.destroy()
            if fn:
                self.after(10, fn)
            return "break"

        def move(d):
            if not shown:
                return "break"
            sel = lb.curselection()
            i = max(0, min(len(shown) - 1, (sel[0] if sel else 0) + d))
            lb.selection_clear(0, "end")
            lb.selection_set(i)
            lb.see(i)
            return "break"

        var.trace_add("write", refresh)
        for wd in (ent, lb):
            wd.bind("<Return>", go)
            wd.bind("<Escape>", lambda e: w.destroy())
        lb.bind("<Double-Button-1>", go)
        ent.bind("<Down>", lambda e: move(1))
        ent.bind("<Up>", lambda e: move(-1))
        refresh()
        w.after(60, ent.focus_force)

    def check_whats_new(self):
        if self.upgraded:
            self.upgraded = False
            self.show_whats_new()
        self.ver_seen.set(VER)

    def show_whats_new(self):
        old = getattr(self, "_wn_win", None)
        if old is not None and old.winfo_exists():
            old.lift()
            return
        p = self.pal
        w = self._wn_win = tk.Toplevel(self)
        w.title(f"What's new — {APP} {VER}")
        w.configure(bg=p["bg"])
        w.resizable(False, False)
        w.transient(self)
        card = tk.Frame(w, bg=p["card"], padx=30, pady=24, highlightthickness=2, highlightbackground=p["accent"])
        card.pack(padx=18, pady=18)
        tk.Label(card, text=f"✨  {APP} {VER}", font=("Segoe UI", 20, "bold"), bg=p["card"], fg=p["text"]).pack(anchor="w")
        tk.Label(card, text=TAGLINE, font=("Segoe UI", 10), bg=p["card"], fg=p["accent2"]).pack(anchor="w", pady=(0, 12))
        for ic, head, body in WHATS_NEW:
            row = tk.Frame(card, bg=p["card"])
            row.pack(fill="x", pady=5)
            tk.Label(row, text=ic, font=("Segoe UI Emoji", 16), bg=p["accent"], fg="#ffffff", width=3).pack(side="left", anchor="n")
            tx = tk.Frame(row, bg=p["card"])
            tx.pack(side="left", padx=12)
            tk.Label(tx, text=head, font=("Segoe UI", 11, "bold"), bg=p["card"], fg=p["text"], anchor="w").pack(fill="x")
            tk.Label(tx, text=body, font=("Segoe UI", 10), bg=p["card"], fg=p["muted"], anchor="w",
                     justify="left", wraplength=430).pack(fill="x")
        Btn(card, self, "Got it", w.destroy, "primary", "card", size=11, padx=30, pady=8).pack(pady=(16, 0))

    def export_settings(self):
        f = filedialog.asksaveasfilename(defaultextension=".json", initialfile="tikvortex_settings.json",
                                         filetypes=[("Settings", "*.json")])
        if not f:
            return
        self.save_settings(silent=True)
        try:
            shutil.copyfile(SETTINGS, f)
            self.toast("Settings exported ✔")
        except Exception as ex:
            messagebox.showerror(APP, f"Could not export:\n{ex}")

    def import_settings(self):
        f = filedialog.askopenfilename(filetypes=[("Settings", "*.json")])
        if not f:
            return
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            if not isinstance(d, dict):
                raise ValueError("not a settings file")
        except Exception as ex:
            messagebox.showerror(APP, f"Could not read that file:\n{ex}")
            return
        n = 0
        for k, v in d.items():
            if k in self.cfg and k not in ("geom", "ver_seen"):
                try:
                    self.cfg[k].set(v)
                    n += 1
                except Exception:
                    pass
        if self.theme.get() not in THEMES:
            self.theme.set(DEFAULT_THEME)
        self.apply_theme()
        self.save_settings(silent=True)
        self.toast(f"Imported {n} settings ✔")

    def cap_fail_popup(self, msg):
        self.toast(f"Captions failed: {msg[:70]}", "err")
        self.log_error("captions: " + msg)
        if self._cap_warned:
            return
        self._cap_warned = True
        messagebox.showwarning(APP, f"Captions could not be added:\n\n{msg[:300]}")

    def speech_py(self):
        exe = (self.ev["cp_pyexe"].get() or "").strip()
        return exe if (exe and os.path.exists(exe)) else sys.executable

    def speech_cmd(self, o):
        """Command that starts the speech-engine process: the app's own Python, or the one chosen for speech."""
        exe = (o.get("cp_pyexe") or "").strip()
        if exe and os.path.exists(exe) and not getattr(sys, "frozen", False):
            return [exe, os.path.abspath(__file__)]
        return [sys.executable] + ([] if getattr(sys, "frozen", False) else [os.path.abspath(__file__)])

    def browse_pyexe(self):
        f = filedialog.askopenfilename(title="python.exe to use for the speech engine", filetypes=[("python.exe", "python*.exe"), ("All", "*.*")])
        if f:
            self.ev["cp_pyexe"].set(f)

    def cap_setup312(self):
        """Python 3.14 crashes the speech engine (ctranslate2). Build a private Python 3.12 environment and use it for speech only."""
        env_dir = os.path.join(APPDIR, "speech_env")
        vpy = os.path.join(env_dir, "Scripts", "python.exe")

        def run(cmd, label):
            self.say(f"🛠 {label} ...", "info")
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", **nowin())
                self.procs.add(p)
                for line in p.stdout:
                    line = line.strip()
                    if line:
                        self.say("   " + line[:150], "info")
                p.wait()
                self.procs.discard(p)
                return p.returncode
            except Exception as e:
                self.say(f"   could not run {cmd[0]}: {str(e)[:100]}", "err")
                return -1

        def find312():
            try:
                r = subprocess.run(["py", "-3.12", "-c", "import sys;print(sys.executable)"], capture_output=True, text=True, **nowin())
                return r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
            except Exception:
                return ""
        if not os.path.exists(vpy):
            exe = find312()
            if not exe:
                run(["winget", "install", "-e", "--id", "Python.Python.3.12", "--silent",
                     "--accept-package-agreements", "--accept-source-agreements"], "Installing Python 3.12 (a Windows prompt may appear — press Yes)")
                exe = find312()
            if not exe:
                self.say("✖ Python 3.12 is not available. Install it from python.org/downloads (version 3.12.x, tick 'Add python.exe to PATH'), "
                         "then press this button again.", "err")
                return
            run([exe, "-m", "venv", env_dir], "Creating the speech environment")
        if not os.path.exists(vpy):
            self.say("✖ Could not create the speech environment.", "err")
            return
        if run([vpy, "-m", "pip", "install", "--upgrade", "faster-whisper", "numpy"], "Installing the speech engine (a few minutes)") != 0:
            self.say("✖ Install failed — check the internet and press the button again.", "err")
            return
        self.after(0, self.ev["cp_pyexe"].set, vpy)
        self._asr_level = 0
        self.say("✔ Speech environment ready. Now press 🩺 Test captions.", "ok")
        self.toast_safe("Python 3.12 speech environment ready ✔", "ok")

    def cap_vcredist(self):
        """An old Microsoft Visual C++ runtime makes the speech engine crash with 'access violation' — install the latest."""
        self.say("🛠 Installing the latest Microsoft Visual C++ runtime (a Windows prompt may appear — press Yes) ...", "info")
        rc = -1
        try:
            p = subprocess.Popen(["winget", "install", "-e", "--id", "Microsoft.VCRedist.2015+.x64", "--silent",
                                  "--accept-package-agreements", "--accept-source-agreements"],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", **nowin())
            self.procs.add(p)
            for line in p.stdout:
                line = line.strip()
                if line:
                    self.say("   " + line[:150], "info")
            p.wait()
            self.procs.discard(p)
            rc = p.returncode
        except Exception as e:
            self.say(f"   could not run winget: {str(e)[:100]}", "err")
        if rc == 0:
            self.say("✔ Visual C++ runtime updated. Now press 🩺 Test captions.", "ok")
            self.toast_safe("Visual C++ runtime updated ✔", "ok")
        elif rc in (-1978335189, 2316632107):
            self.say("ℹ winget says the Visual C++ runtime is already the newest one — press 🩺 Test captions and send me the report.", "info")
        else:
            self.say("⚠ winget could not do it — opening the Microsoft download page. Install it, then press 🩺 Test captions.", "err")
            webbrowser.open("https://aka.ms/vs/17/release/vc_redist.x64.exe")

    def cap_install(self):
        if getattr(sys, "frozen", False):
            self.say("This build cannot install packages by itself — install faster-whisper in the Python it was built with.", "err")
            return
        self.say("⬇ Installing the speech engine (faster-whisper + pillow) — this can take a few minutes ...", "info")
        cmd = [self.speech_py(), "-m", "pip", "install", "--upgrade", "faster-whisper", "numpy"]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", **nowin())
            self.procs.add(p)
            for line in p.stdout:
                line = line.strip()
                if line:
                    self.say("   " + line[:150], "info")
            p.wait()
            self.procs.discard(p)
            if p.returncode == 0:
                self.say("✔ Speech engine installed. Now press 🩺 Test captions.", "ok")
                self.toast_safe("Speech engine installed ✔", "ok")
            else:
                self.say(f"✖ Install failed (code {p.returncode}). Check the internet and try again.", "err")
        except Exception as e:
            self.say(f"✖ Could not start pip: {str(e)[:120]}", "err")

    def browse_modeldir(self):
        f = filedialog.askdirectory(title="Folder that contains the downloaded faster-whisper model (model.bin ...)")
        if f:
            self.ev["cp_modeldir"].set(f)

    def apply_topmost(self):
        try:
            self.attributes("-topmost", bool(self.topmost.get()))
        except tk.TclError:
            pass

    def clear_log(self):
        self.log.delete("1.0", "end")

    def copy_log(self):
        self.clipboard_clear()
        self.clipboard_append(self.log.get("1.0", "end"))
        self.toast("Log copied ✔")

    def filter_rows(self):  # the search box on the Auto Downloader tab
        q = self.dl_filter.get().strip().lower()
        shown, i = set(self.tree.get_children()), 0
        for iid in self.row_order:
            vals = self.tree.item(iid, "values")
            if q in " ".join(str(v) for v in vals[1:3]).lower():
                self.tree.move(iid, "", i)
                i += 1
            elif iid in shown:
                self.tree.detach(iid)

    def cap_selftest(self):
        steps, detail = [], []

        def ok(t):
            steps.append("✔ " + t)
            self.say("✔ " + t, "ok")

        def bad(t):
            steps.append("✖ " + t)
            self.say("✖ " + t, "err")
        o = self.eo()
        self.say("🩺 Testing captions on this PC ...", "info")
        ff = shutil.which("ffmpeg")
        if not ff:
            bad("ffmpeg not found  →  run:  winget install Gyan.FFmpeg   (then restart the app)")
        else:
            ok("ffmpeg found")
            try:
                flt = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True, **nowin()).stdout
                (ok if re.search(r"\bass\b", flt) else bad)("ffmpeg has the subtitle (ass) filter" if re.search(r"\bass\b", flt)
                                                           else "this ffmpeg has no subtitle filter  →  install the full build: winget install Gyan.FFmpeg")
            except Exception as e:
                bad(f"could not read ffmpeg filters: {e}")
        try:
            import PIL
            ok(f"Pillow {PIL.__version__}")
        except ImportError:
            bad("Pillow missing  →  pip install pillow")
        py = sys.executable
        self.say(f"   Python used by this app: {py}", "info")
        try:
            import importlib.metadata as _im, platform as _pf
            vs = []
            for pk in ("faster-whisper", "ctranslate2", "av", "numpy", "onnxruntime", "tokenizers", "huggingface-hub"):
                try:
                    vs.append(f"{pk} {_im.version(pk)}")
                except Exception:
                    vs.append(f"{pk} (not installed)")
            info_line = f"Python {sys.version.split()[0]} {_pf.architecture()[0]} on {_pf.platform()}  |  " + ", ".join(vs)
        except Exception as _e:
            info_line = f"(could not read versions: {_e})"
        detail.append(info_line)
        self.say("   " + info_line[:150], "info")
        if os.name == "nt":
            try:
                r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                    "(Get-Item $env:WINDIR\\System32\\msvcp140.dll).VersionInfo.ProductVersion"],
                                   capture_output=True, text=True, timeout=25, **nowin())
                vtxt = (r.stdout or "").strip().splitlines()[-1].strip() if (r.stdout or "").strip() else ""
                vm = re.match(r"(\d+)\.(\d+)", vtxt)
                detail.append("Visual C++ runtime (msvcp140.dll): " + (vtxt or "not found"))
                self.say("   Visual C++ runtime: " + (vtxt or "not found"), "info")
                if not vm:
                    bad("Visual C++ runtime not found  →  press  🛠 Update Visual C++ runtime")
                elif (int(vm.group(1)), int(vm.group(2))) < (14, 40):
                    bad(f"Visual C++ runtime is OLD ({vtxt}) — this is the usual cause of the speech crash  →  "
                        "press  🛠 Update Visual C++ runtime, then test again")
                else:
                    ok(f"Visual C++ runtime is up to date ({vtxt})")
            except Exception:
                pass
        try:
            import socket as _sk
            _sk.create_connection(("huggingface.co", 443), 6).close()
            ok("huggingface.co reachable — the speech model can download")
        except Exception:
            self.say("⚠ huggingface.co is not reachable — a download mirror is tried automatically "
                     "(or set an offline model folder in Captions → Speech)", "info")
        base_cmd = self.speech_cmd(o) + [
            "--check-fw", str(o["cp_model"]), (o.get("cp_modeldir") or "").strip()]
        try:
            for level in (0, 1, 2):
                if level:
                    self.say(f"   ⚙ Trying safe mode {level} (generic CPU mode) ...", "info")
                p = subprocess.Popen(base_cmd + [str(level)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     errors="replace",
                                     env=self.hf_env("https://hf-mirror.com" if o.get("cp_mirror") else None, level), **nowin())
                self.procs.add(p)
                last = ""
                for line in p.stdout:
                    line = line.strip()
                    if line:
                        last = line
                        detail.append(line[:220])
                        self.say("   " + line[:150], "info")
                p.wait()
                self.procs.discard(p)
                if p.returncode == 0:
                    ok(f"speech engine works and the '{o['cp_model']}' model is ready" + (f"  (safe mode {level})" if level else ""))
                    if level:
                        self._asr_level = level
                        ok("this PC needs safe mode — captions will use it automatically from now on")
                    else:
                        self._asr_level = 0
                    break
                if "IMPORT FAILED" in last or "No module named" in last:
                    bad(f"faster-whisper is not installed for THIS Python  →  press  ⬇ Install / repair speech engine  "
                        f"(or run:  \"{py}\" -m pip install faster-whisper)")
                    break
                if p.returncode in CRASH_CODES:
                    self.say(f"   💥 crashed (code {p.returncode})", "info")
                    detail.append(f"--- crashed at safe level {level}, code {p.returncode} ---")
                    if level == 2:
                        bad(f"the speech engine crashes on this PC even in safe mode (code {p.returncode}) — "
                            "copy the lines above and send them to the developer")
                    continue
                bad(f"speech engine problem (code {p.returncode}): {last[:120]}  — if it says network/connection, "
                    "check the internet (the model downloads once)")
                break
        except Exception as e:
            bad(f"could not start the speech engine: {str(e)[:100]}")
        if ff:
            d = tempfile.mkdtemp()
            try:
                vid, out = (os.path.join(d, n) for n in ("t.mp4", "o.mp4"))
                subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=360x640:d=2", "-pix_fmt", "yuv420p", vid],
                               check=True, capture_output=True, **nowin())
                ass = make_ass_file(build_ass([(0.2, 1.8, [("Test", 0.2, 0.8), ("caption", 0.8, 1.8)])], 360, 640, self.cap_style(o)))
                r = subprocess.run([ff, "-y", "-v", "error", "-i", vid, "-vf", "ass=" + ass_arg(ass), out],
                                   capture_output=True, text=True, cwd=os.path.dirname(ass), **nowin())
                drop_ass(ass)
                if r.returncode == 0 and os.path.exists(out):
                    ok("captions can be burned into a video on this PC")
                else:
                    bad("burning failed: " + ((r.stderr or "").strip().splitlines() or ["?"])[-1][:140])
            except Exception as e:
                bad(f"burn test failed: {str(e)[:120]}")
            finally:
                shutil.rmtree(d, ignore_errors=True)
        bads = [x for x in steps if x.startswith("✖")]
        report = f"{APP} {VER} captions test\n" + "\n".join(steps) + "\n\n--- details ---\n" + "\n".join(detail)
        try:
            with open("captions_test.txt", "w", encoding="utf-8") as fh:
                fh.write(report)
        except OSError:
            pass

        def show():
            try:
                self.clipboard_clear()
                self.clipboard_append(report)
            except tk.TclError:
                pass
            messagebox.showinfo("Captions test", "\n".join(steps) + (
                "\n\nEverything is OK — captions should work." if not bads else "\n\nFix the ✖ items above, then test again.")
                + "\n\n📋 The full report is copied — paste it (Ctrl+V) in a message to the developer.\n"
                  "(also saved as captions_test.txt)\n\n" + "\n".join(detail)[-900:])
        self.after(0, show)

    def run_ffmpeg(self, cmd, cwd=None):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             errors="replace", cwd=cwd, **nowin())
        self.procs.add(p)
        try:
            _, err = p.communicate()
        finally:
            self.procs.discard(p)
        if self.stop_flag:
            raise Cancelled()
        if p.returncode != 0:
            self.log_error("ffmpeg failed:\n" + (err or "")[-1500:])
            raise Exception(((err or "").strip().splitlines() or ["ffmpeg failed"])[-1][:160])

    def edit_file(self, src, username, preview=False, parts=("brand", "extras", "captions"), pw=540, cap_lines=None, outmode=None):
        """Apply the editing options to one video. Returns the new file path (or None if nothing to do)."""
        o = self.eo()
        if outmode:
            o["outmode"] = outmode
        if "brand" not in parts:
            o["brand_on"] = o["logo_on"] = False
        if "extras" not in parts:
            o.update(trim_start=0.0, trim_end=0.0, aspect="Original", maxh="Original", mute=False)
        if "captions" not in parts:
            cap_lines = None
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise Exception("ffmpeg is required (run: winget install ffmpeg)")
        src = os.path.abspath(src)
        info = probe(src)
        base, folder = os.path.splitext(os.path.basename(src))[0], os.path.dirname(src)
        if preview:
            tmp = final = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        elif o["outmode"] == "Replace original":
            tmp, final = os.path.join(folder, base + ".editing.mp4"), os.path.join(folder, base + ".mp4")
        else:
            od, nm = (folder, base + " (edited).mp4") if os.path.basename(folder) == "Edited" \
                else (os.path.join(folder, "Edited"), base + ".mp4")
            os.makedirs(od, exist_ok=True)
            tmp = final = os.path.join(od, nm)
        ass_fn, cap_dir = None, []
        if cap_lines and not preview:
            if not ffmpeg_has_ass():
                raise Exception(NO_ASS_MSG)
            style = self.cap_style(o)

            def ass_fn(ow, oh, offset):
                ls = shift_lines(cap_lines, -offset) if offset else cap_lines
                ap = make_ass_file(build_ass(ls, ow, oh, style))
                cap_dir[:] = [os.path.dirname(ap)]
                return ap
        cmd, temps = build_ffmpeg(src, tmp, info, o, username, preview, pw=pw, ass_fn=ass_fn)
        try:
            if cmd is None:
                return None
            self.run_ffmpeg(cmd, cwd=cap_dir[0] if cap_dir else None)
            if tmp != final:
                if o.get("keep_orig"):
                    self.keep_original(src)
                os.replace(tmp, final)
                if os.path.abspath(final) != os.path.abspath(src) and os.path.exists(src):
                    os.remove(src)
            if cap_lines and o["cp_srt"]:
                with open(os.path.splitext(final)[0] + ".srt", "w", encoding="utf-8") as fh:
                    fh.write(srt_text(shift_lines(cap_lines, -float(o["trim_start"])) if o["trim_start"] else cap_lines))
            return final
        except Exception:
            if tmp != final and os.path.exists(tmp):
                os.remove(tmp)
            raise
        finally:
            for f in temps:
                if f.endswith(".ass"):
                    drop_ass(f)
                    continue
                try:
                    os.remove(f)
                except OSError:
                    pass

    def stop(self):
        self.stop_flag = True
        self.pause_ev.set()
        for pr in list(self.procs):
            try:
                pr.terminate()
            except Exception:
                pass
        self.say("Stopping...", "err")

    def toggle_pause(self):
        if not self.running:
            return
        if self.pause_ev.is_set():
            self.pause_ev.clear()
            self.pause_btn.set_text("▶ Continue")
            self.set_state("⏸ Paused")
            self.say("Paused", "info")
        else:
            self.pause_ev.set()
            self.pause_btn.set_text("⏸ Pause")
            self.set_state(f"● Running  •  {self.email or ''}")
            self.say("Continued", "ok")

    def speed_loop(self):
        if self.running:
            parts = []
            sp = sum(self.speeds.values())
            if sp:
                parts.append(f"{sp / 1048576:.1f} MB/s")
            if self.pdone and self.ptotal > self.pdone:
                eta = int((time.time() - self.t0) / self.pdone * (self.ptotal - self.pdone))
                parts.append(f"ETA {eta // 60:02d}:{eta % 60:02d}")
            self.extra = "   ".join(parts)
        else:
            self.extra = ""
        self._disk_n -= 1
        if self._disk_n <= 0:
            self._disk_n = 10
            self.update_disk()
        self.set_bar()
        self.after(500, self.speed_loop)

    def beep(self):
        try:
            import winsound
            winsound.Beep(880, 180)
            winsound.Beep(1175, 300)
        except Exception:
            self.bell()

    def notify(self, msg):
        if self.sound.get():
            self.beep()
        self.toast_safe("✔ " + (msg.strip().splitlines() or [""])[0][:90], "ok")
        if self.popup.get():
            messagebox.showinfo(APP, msg)

    def sched_loop(self):
        now = time.strftime("%H:%M")
        key = time.strftime("%Y-%m-%d ") + now
        if self.sched_on.get() and now == self.sched_time.get().strip() and self.last_sched != key \
                and not self.running and self.docs_api:
            self.last_sched = key
            self.say("⏰ Scheduled run starting...", "info")
            self.thread(lambda: (self.load_docs(), self.run(all_profiles=True)))
        self.after(20000, self.sched_loop)

    # ---------- resume after crash ----------
    def check_resume(self):
        st = jload(STATE, None)
        if not st:
            return
        jobs = st.get("jobs", [])
        pend = [j for j in jobs if j["status"] == "pending"]
        unremoved = [j for j in jobs if j.get("did") and j["status"] in ("done", "skipped") and not j.get("removed")]
        if not pend and not unremoved:
            try:
                os.remove(STATE)
            except OSError:
                pass
            return
        if not pend:
            self.thread(lambda: self.resume(st))
            return
        r = messagebox.askyesnocancel(
            "Resume downloads?",
            f"Your last session was interrupted (power cut or app closed).\n\n"
            f"{len(pend)} video(s) are still waiting.\n\n"
            "Yes  =  Resume where it stopped\n"
            "No  =  Start new (discard the waiting list)\n"
            "Cancel  =  Decide later")
        if r is True:
            self.thread(lambda: self.resume(st))
        elif r is False:
            self.thread(lambda: self.resume(st, drop=True))

    def resume(self, st=None, drop=False):
        st = st or jload(STATE, None)
        if not st:
            self.say("Nothing to resume", "info")
            return
        jobs = st["jobs"]
        for _ in range(50):  # wait for the saved login to come up
            if self.docs_api or not any(j.get("did") for j in jobs):
                break
            time.sleep(0.5)
        if any(j.get("did") for j in jobs) and not self.docs_api:
            self.say("Please login first (Accounts tab), then click ↻ Resume last", "err")
            return
        if drop:
            for j in jobs:
                if j["status"] == "pending":
                    j["status"] = "dropped"
        else:
            self.say(f"Resuming {sum(j['status'] == 'pending' for j in jobs)} video(s)...", "info")
        self.execute(jobs)

    # ---------- engine ----------
    def run(self, all_profiles=False):
        if self.running:
            return
        if not self.docs_api:
            self.say("Please login first (Accounts tab)", "err")
            return
        if not self.docs:
            self.say("Load docs first", "err")
            return
        if all_profiles:
            chosen = [f"{d}|{i}" for d, doc in self.docs.items() for i in range(len(doc["profiles"]))]
        else:
            chosen = [i for i, v in self.sel.items() if v]
        if not chosen:
            self.say("Select at least one profile first", "err")
            return
        n = self.links_n.get()
        mode = "MP3" if self.fmt.get().startswith("MP3") else "MP4"
        jobs = []
        for iid in sorted(chosen, key=lambda s: (s.split("|")[0], int(s.split("|")[1]))):
            did, i = iid.split("|")
            d = self.docs.get(did)
            if not d or int(i) >= len(d["profiles"]):
                continue
            p = d["profiles"][int(i)]
            tn = self.take.get(iid) or n
            vids = p["videos"] if self.order.get() == "Top first" else p["videos"][::-1]
            take = vids[:tn]
            out = os.path.join(self.folder.get(), clean(d["title"]).rstrip(". ") or "Doc")
            self.say(f"\nProfile: @{p['name']}   ({d['title']})", "info")
            self.say(f"  Total links: {len(p['videos'])}  |  Selected: {len(take)}  |  Remaining: {len(p['videos']) - len(take)}")
            self.say(f"  Folder: {out}")
            for v in take:
                k = len(jobs) + 1
                jobs.append(dict(id=k, n=k, did=did, title=d["title"], pname=p["name"], url=v["url"], out=out,
                                 fmt=mode, quality=self.quality.get(), uni=False, h=None,
                                 status="pending", removed=False))
        if not jobs:
            self.say("No links to download", "err")
            return
        self.execute(jobs)

    def save_state(self, jobs):
        with self.lock:
            jsave(STATE, {"email": self.email, "jobs": jobs})

    def execute(self, jobs):
        if self.running:
            self.say("Another task is already running", "err")
            return
        self.running, self.stop_flag = True, False
        self.pause_ev.set()
        self.speeds, self.t0 = {}, time.time()
        todo = [j for j in jobs if j["status"] == "pending"]
        cnt = {"ok": 0, "skip": 0, "fail": 0}
        self.after(0, self.set_bar, 0, len(todo), "Starting")
        self.after(0, self.set_state, f"● Running  •  {self.email or 'no account'}")
        self.save_state(jobs)
        try:
            if todo:
                with ThreadPoolExecutor(max_workers=max(1, self.par.get())) as ex:
                    list(ex.map(lambda j: self.process(j, jobs, cnt), todo))
            self.finish_docs(jobs)
        except Exception as e:
            self.say(f"Error: {e}", "err")
        finally:
            left = any(j["status"] == "pending" for j in jobs)
            unrem = any(j.get("did") and j["status"] in ("done", "skipped") and not j.get("removed") for j in jobs)
            if left or unrem:
                self.save_state(jobs)
            else:
                try:
                    os.remove(STATE)
                except OSError:
                    pass
            self.running = False
            self.save_cache()
            self.after(0, self.refresh_progress)
            self.after(0, self.set_state, "● Idle")
            self.after(0, lambda: self.pause_btn.set_text("⏸ Pause"))
            if self.stop_flag:
                self.say("Stopped — click ↻ Resume last to continue later.", "err")
                self.after(0, self.set_bar, None, None, "Stopped")
            else:
                self.after(0, self.set_bar, None, None, "Completed ✔")
                if todo:
                    self.say("\nAll done ✔", "ok")
                    self.after(0, self.finish_ui, f"Download complete!\n\nDownloaded: {cnt['ok']}\n"
                                               f"Skipped (already had): {cnt['skip']}\nFailed: {cnt['fail']}")

    def process(self, job, jobs, cnt):
        while not self.pause_ev.is_set() and not self.stop_flag:
            time.sleep(0.3)
        if self.stop_flag:
            return
        n = job["n"]
        self.say(f"[{n}] 🔗 {job['url']}")
        self.after(0, self.uni_set, job, None, "Downloading", None)
        if job.get("did") and self.skipdup.get() and vid_id(job["url"]) in self.done_ids:
            job["status"] = "skipped"
            with self.clock:
                cnt["skip"] += 1
            self.say(f"[{n}]    ⏭ Skipped — already downloaded before", "info")
            self.finish_job(job, jobs)
            return
        err = ""
        tries = self.retry.get()
        for attempt in range(tries + 1):
            if self.stop_flag:
                return
            try:
                name, path = self.download(job)
                aparts = tuple(pn for pn, k in (("brand", "auto_brand"), ("extras", "auto"), ("captions", "auto_cap"))
                               if self.ev[k].get())
                if aparts and job["fmt"] != "MP3":
                    try:
                        user = job["pname"] if job.get("pname") and not job["pname"].startswith("(") else job.get("uploader", "")
                        cap_lines = None
                        if "captions" in aparts and self.ev["cp_skip"].get():
                            why = has_captions(path)
                            if why:
                                self.say(f"[{n}]    ⏭ No captions added — {why}", "info")
                                aparts = tuple(x for x in aparts if x != "captions")
                        if "captions" in aparts:
                            try:
                                cap_lines, lang = self.transcribe(path, self.eo())
                                self.say(f"[{n}]    💬 Captions ready ({LANG_NAMES.get(lang, lang)})", "ok")
                            except Cancelled:
                                raise
                            except NoSpeech:
                                self.say(f"[{n}]    ⏭ No speech in this video — no captions needed", "info")
                            except Exception as e:
                                self.say(f"[{n}]    ⚠ No captions: {str(e)[:80]}", "err")
                                self.after(0, self.cap_fail_popup, str(e))
                        use = tuple(x for x in aparts if x != "captions" or cap_lines)
                        newp = self.edit_file(path, user, parts=use, cap_lines=cap_lines, outmode="Replace original") if use else None
                        if newp:
                            path = newp
                            self.say(f"[{n}]    🎨 Edited" + (f" — brand @{user}" if user and "brand" in use and self.ev["brand_on"].get() else "")
                                     + (" + captions" if cap_lines else ""), "ok")
                    except Cancelled:
                        raise
                    except Exception as e:
                        self.say(f"[{n}]    ⚠ Edit failed: {str(e)[:80]} — original kept", "err")
                job["status"] = "done"
                with self.clock:
                    cnt["ok"] += 1
                self.say(f"[{n}]    ✔ Downloaded", "ok")
                self.add_record(job, "done", name, path)
                self.finish_job(job, jobs)
                return
            except Cancelled:
                return
            except Exception as e:
                if self.stop_flag:
                    return
                err = str(e).replace("\n", " ")[:90]
                if attempt < tries:
                    self.say(f"[{n}]    ↻ Retry {attempt + 1}/{tries} ({err})", "err")
                    time.sleep(2)
        job["status"] = "failed"
        with self.clock:
            cnt["fail"] += 1
        self.say(f"[{n}]    ✖ FAILED ({err})", "err")
        self.add_record(job, "failed", job["url"], "")
        self.finish_job(job, jobs)

    def finish_job(self, job, jobs):
        self.save_state(jobs)
        self.speeds.pop(job["id"], None)
        self.after(0, self.tick)
        self.after(0, self.ui_done, job)
        txt = {"done": "✔ Done", "skipped": "Skipped", "failed": "✖ Failed"}.get(job["status"], "")
        self.after(0, self.uni_set, job, None, txt, "100%" if job["status"] == "done" else "")

    def ui_done(self, job):
        did = job.get("did")
        if not did:
            return
        if job["status"] == "done":
            k = f"{did}|{job['pname']}"
            self.hist[k] = self.hist.get(k, 0) + 1
            jsave(HISTORY, self.hist)
        d = self.docs.get(did)
        if d and job["status"] in ("done", "skipped"):
            for idx, p in enumerate(d["profiles"]):
                if p["name"] == job["pname"]:
                    p["videos"] = [v for v in p["videos"] if v["url"] != job["url"]]
                    self.update_prow(did, idx)
        self.count()

    def add_record(self, job, status, name, path):
        rec = {k: job.get(k) for k in ("did", "title", "pname", "url", "out", "fmt", "quality", "uni", "h", "uploader")}
        rec.update(t=time.strftime("%Y-%m-%d %H:%M"), status=status, name=name, file=path)
        with self.clock:
            if status == "done":
                self.records = [r for r in self.records if not (r["url"] == rec["url"] and r["status"] == "failed")]
                self.done_ids.add(vid_id(rec["url"]))
            self.records = (self.records + [rec])[-5000:]
            jsave(RECORDS, self.records)
        self.after(0, self.refresh_history)

    def finish_docs(self, jobs):
        by = {}
        for j in jobs:
            if j.get("did") and j["status"] in ("done", "skipped") and not j.get("removed"):
                by.setdefault(j["did"], []).append(j)
        if not by:
            return
        if not self.docs_api:
            self.say("Not logged in — links stay in the docs and are removed next time", "err")
            return
        for did, js in by.items():
            try:
                fresh = self.parse_doc(self.docs_api.documents().get(documentId=did).execute())
                pos = {x["url"]: x for pr in fresh for x in pr["videos"]}
                found = [pos[j["url"]] for j in js if j["url"] in pos]
                if found:
                    reqs = [{"deleteContentRange": {"range": {"startIndex": x["start"], "endIndex": x["end"]}}}
                            for x in sorted(found, key=lambda x: -x["start"])]
                    self.docs_api.documents().batchUpdate(documentId=did, body={"requests": reqs}).execute()
                for j in js:
                    j["removed"] = True
                self.say(f"Removed {len(found)} link(s) from doc: {js[0]['title']}", "ok")
            except Exception as e:
                self.say(f"Could not edit doc {js[0]['title']}: {str(e)[:80]}", "err")
        self.save_state(jobs)

    # ---------- downloading ----------
    def ydl_base(self):
        o = {"quiet": True, "noprogress": True, "logger": QuietLog(), "noplaylist": True}
        if self.proxy.get().strip():
            o["proxy"] = self.proxy.get().strip()
        if self.cookies.get().strip() and os.path.exists(self.cookies.get().strip()):
            o["cookiefile"] = self.cookies.get().strip()
        elif self.cbrowser.get() != "None":
            o["cookiesfrombrowser"] = (self.cbrowser.get(),)
        if self.rate.get() > 0:
            o["ratelimit"] = self.rate.get() * 1024 * 1024
        return o

    def fmt_sel(self, job):
        if job["fmt"] == "MP3":
            return "ba/b"
        if job.get("uni"):
            f = f"[height<={job['h']}]" if job.get("h") else ""
            return f"bv*{f}+ba/b{f}/b"
        return QUALITY.get(job.get("quality"), QUALITY["Best"])

    def hook(self, d, job):
        while not self.pause_ev.is_set() and not self.stop_flag:
            time.sleep(0.3)
        if self.stop_flag:
            raise Cancelled()
        if d.get("status") == "downloading":
            self.speeds[job["id"]] = d.get("speed") or 0
            now = time.time()
            if job.get("uni") and now - job.get("_t", 0) > 0.4:
                job["_t"] = now
                tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                pct = f"{int(d.get('downloaded_bytes', 0) * 100 / tot)}%" if tot else "..."
                self.after(0, self.uni_set, job, None, None, pct)

    def download(self, job):
        n, mp3 = job["n"], job["fmt"] == "MP3"
        if (mp3 or job.get("h") and job["h"] > 720) and not shutil.which("ffmpeg"):
            raise Exception("ffmpeg is required for MP3 / 4K (run: winget install ffmpeg)")
        base = self.ydl_base()
        with yt_dlp.YoutubeDL({**base, "skip_download": True}) as y:
            info = y.extract_info(job["url"], download=False)
        is_tt = "tiktok" in (info.get("extractor_key") or "").lower()
        job["uploader"] = info.get("uploader") or info.get("channel") or info.get("uploader_id") or ""
        name = clean((info.get("description") if is_tt else info.get("title")) or info.get("title")) or info["id"]
        self.say(f"[{n}]    # {name}", "info")
        self.after(0, self.uni_set, job, name, None, None)
        out = job["out"]
        os.makedirs(out, exist_ok=True)
        ext = "mp3" if mp3 else "mp4"
        if os.path.exists(os.path.join(out, f"{name}.{ext}")):
            name = f"{name} {info['id']}"
        opts = {**base, "format": self.fmt_sel(job), "merge_output_format": "mp4/mkv",
                "outtmpl": os.path.join(out, name.replace("%", "%%") + ".%(ext)s"),
                "progress_hooks": [lambda d: self.hook(d, job)]}
        if mp3:
            opts["postprocessors"] = [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}]
        elif shutil.which("ffmpeg") and not job.get("uni"):  # cover picture so the file shows a thumbnail
            opts["writethumbnail"] = True
            opts["postprocessors"] = [{"key": "FFmpegThumbnailsConvertor", "format": "jpg", "when": "before_dl"},
                                      {"key": "EmbedThumbnail"}]
        with yt_dlp.YoutubeDL(opts) as y:
            res = y.extract_info(job["url"], download=True)
        rd = res.get("requested_downloads") or []
        path = rd[0].get("filepath") if rd else os.path.join(out, f"{name}.{ext}")
        return name, path


PHOTO_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAASwAAAEsCAIAAAD2HxkiAAEAAElEQVR42nz9168tS5ofiH1fmPTLbr+Pub5uObZR02gMR4ReBAgC9CZgqD9N7w"
    "IGehrOYEaCKA7Joelms9ndZbqqrr/Hbrts+oyITw9rZUZE7sOp7qprzt5rpYmIz/0M4n/9zwgBgBCAAIAIAQCBAAAQABDB+Q96f0vg/Yeo/1cI"
    "QIffPv4sIh1/wPlx+4lEH/y+499j/+FA7j/0n06Hn0NwfgzRXhw6l0rOv6Ent0D9/9iPJMTDddDo9od7JnL/gIgQ8UNP58mDG564/XX0/vT4Oh"
    "DQv+HjDfQ/Qe5jZgTkPefh2o+PGKG/9Q/dlPOTxwvrn5XzE6NfwMO/QOrv+EP3TmSvxF7u4TsOf6HDVfmv/kPPy76/w8NBeyHDW+5/3V017pKg"
    "8UXYSxzd5uGukD708v0FdvxJZ/k5l9pfFwHh6FUa0d84Eg3r7bAnj99LBAyR3FXQ74MPLElEGP492Z8n+sAdHC7ouDbsFwIN90zuteHxWQAQun"
    "sMjv9Eo6cyvHEcloddVjS8MySCwxHhLlw8vhHsNybauxtub7gpOp5hx51BeFgd5B5Ehxs83iLrlxCRPSSGR8QOrxSPbw28qyNEJOd52jvF/geP"
    "f9OvhePX9L9uH2P/Ac51ovtanceI/iHV/xYis2vY/TE8vrXDPwxbAO01H1fUIQIQIiC6u2z4K7mf6j5/NrwbQABi3q4g5+kgHveHPcwOX2Yv+3"
    "CQonea9i9tuGUaB6PDEzi8fnsYHX6J+hsn59vQOZ76GyFgzD48HE5iBGDDDkTvtD/sZRrvKrI/gkD9Ax5W7+EOaDiEwF7s4XX1/yUnmtpzjABG"
    "4ZKeLEHyng+5N3u4tv4xkxOAji+ZnM/A/lkyAjREOCyx498TkQGi/jNoCEnorAB/25DdIOg+lv49D1dC9mE6l2UPeQToTxRynurx/wkIwBz/hg"
    "jQHFcHOPdGw8+b45dQv0fJ+WQcVsLhuR9/cnif/QYbrpTIf8vOHh+ixOFy0Dk60eZM7vtDPy0hsh8/fCs5b51sFD4uk0NEsfHJi872UvsHjuhs"
    "3P4BEJI9vdBfvDSc1cenc4gDRMOrRXDPvuGXEQ+x202uvPPDXVD4NB8EOIaMIfE7XoDdvIfc6BjVAL0UYJySOcfiEInQfUJ2E/Xr4xA5+81M/r"
    "ruP4JoiATgHsVOdmBP7eNbMDQcmUBoCAnpEPkMIDmrAIeo56zZw/lB0CeCNlMYQsywbOyuPxxjw1qxGwEPkZ76/Izsru1XvT2SvNR+uEy7wsge"
    "oej8Ctn4j/SB097JSMGJBs6GQ++A9BJyIDL91iKwZ4y9fWdb0ej3yS8Ijhsf8UkCTc4Jf8xKDlsOj4cljOM2wWG3DamTTdyd457cMNOH4n7Tox"
    "NT6bhs4LhEht8+/PnTpe+Ej2F/IhAyJ5khHPaVXwO4i+G4PI+hAbH/jcOVHw5qJCQcjmj7sImIyMCoFDlcNh1+kQAIiQ4XQ17SS32+gngsveyS"
    "6Ndevy/IrwiJ7KFhY+FxH+H45Xt7360m+usxhy2F/enQpwzH1GpYFMdbPjyqfosT0OiRwnFhYP8WYFgEwyk3RG8aao/RqnWCAIG3IfpwYtySz+"
    "Ze1C/1w2WP6gYn6enPLu987Y+oIaq4mWu/xcnPRIeLOyyU4UP7pNoWTsdEo4+jhsir65yUFvuV28fZfj0iAIA5hnZnyQyJGbmRYrgRGtbA8WUd"
    "T0uk4QBEN8VG9yCk4Qix1WC/nI9L/HgEEtAxEiJ8sOwm8v6xT93InkPHtWgLAoRRyuns/+NCxP48dw6PfvM6Bx562/QYGI7ZzDEXp74XQMP+tC"
    "ntkOgebhic2tddU+h3AZzK+XApQ4w/FBY0LBe07Y1jzHSW7THcotdpOeYseDwmjzGV3IzSbW3Z6N2nFUPlZMOwkxK4tYc9RJEOS88mOX5Lazgn"
    "h01JblDtkwkC6uujYSHQsL2GFN5e/LAGvPwNnGNw6Kg4NYFzcJKtYIftgMNDGY68/pHh8eG4a9ev9MlrHg5V9Sh7Rn87Ibhdt+NmdH7SORSdEt"
    "vp19m2GA5dAzzknLb5x4YT/ri2vL3RhyYvS6FjfoDDLznXQEPvBw8BrS8X3U7CMSd22ylETjY1SlnpWI6jlzKR108kcnoI6HdcyS0Khy3kB5LD"
    "dqbD9hiCPXr7wl2jcEznnAh5LJnQOa5sMPdaIUNmgLYr450dXs9pyJsQbGXbf+Bw6B4DQ59UYB8W3fzZ6TvaXdpfsF/F9ucT9hU29m/Cnl1PY9"
    "sx7ztsALRF3DE/etokxqFph+jlq+AnNDiKETiUcjZpcktWdI4KchI6ezIdTw56Uooe2mrDqxnKF3IS32GBISISktuQPRz56FWg/mSA+rbnMewa"
    "1ncS0L9fJ7+wRfvx64fix2kPIx1bAsOCstvRqbztR/Y5LPb1SZ9rHl+esQETyStn/DVzyIZwuDVy66DhDOofK9GT7jm6vW+nP+HEJ/AehXMNTq"
    "fDOTeoT18ON3U4rdAedB8Y6njbzznBnfPUftH4pPKqq36PkBM37KtFdLNRG5+RvKeB7pbo6y2bQxxOH7RJtRvm3M41DfEEEenYcu7TRnIyR/Qa"
    "AqOcc9Qu8Ctb2xehIUNwdrPT5RmqU7fphTZPc6YRXoN/yNdt93soab1pkp9goNt8cK7Q1rnDVbB+aQydjP4EIxq3/AidFGdoaPeLG3FIzw6JAf"
    "bvD53OsB3x9T/svjC0iwj7oDbsIGejPOlleWWTux+c5ifZRJ+clqN9a9iXeOjsCrdJjqxPtbwO3micaQt58saH4zT9A+PX/oRFd1YCYJzIeNjg"
    "3gxsNFtBHNaecb7Vr/XJb2wTwmis1I8i3WoEvYaykxIM0RJHyR86Y0ocioghvA4nh9P7PAQpGhU4tnNo4wAgIrIhGT42H/19CwBExk0++/h5PO"
    "xoqLBtzw5xqMVwNGAYFiiO+ol9AX8IGX2m6p1Rx2vr3+DxwELGWH/96DznfvU7vXTy0lYvOSC3APey16GcsB1cdyI1fNkxN8bhEO8Xmm0Rk1tX"
    "e30XHEo/fwlQPzClobCmfgmjPaPdcXNfCPrDBOhxBt60q2849skgDpUGwgg6cFgtCMjsKnV6Q34QHifcNnkcMsxhDR6bfU4lTsZdgYiIT4b2+A"
    "RfMcJjjNu4YIEXaMcXeCxvyP1YspWTzXeGJYRkN9oo1KHTFaP+/45lBA6BaARusHsIjzF4yPXAT7/6FdZfJdpBJTK0ozdwD26LYek//VikHScT"
    "T+dviIBIQ28W/ZbUGAJgXw4DdybtXAz5BbMtHPDYpx0ah9hf8nEbou3UDB0TdJY4uqNQB1tAT0AJw5zAgQlQ/9ZcRMVxNu4FncPn9aXwsYF3iN"
    "Gs/4XjxLnvcLsdG6/V1vcE7Jirj4U4QhENexARgfntCvd4sdUBQ2/WjXbE7G4/GI+r0OlD2CktOq/jWLzT0DzyQSPD+eDdwyi9siv40NXGJ5gZ"
    "pzocwAVDPEFbGnrtSBogP2T6TURDIkvoIoiGYSwC2WzL9kDIW6vkZDb9Jh5hvtCvZd1ara9QySmGvUphaHOhP7mDoWoekigcWhJgp5swLsDxAN"
    "2wFcEHQUfo9Shty3F4r8xfhMPh6Ex47Wx31IXyGpJeVCIHVXC8aTx2y9GfL42yXLShDt1Jo4/1wnETEp0a2MZN9DJ16rct+bgBm7r6zSY0RHZL"
    "OuPvoUl2OA/wQ1CPp5uShntD6EdLfWpwXK3Ygwv61Uw+ds8FrLhnGwwpNo5hUNhXVoxGiSyNeig2iTr0XPpHxIbKwisVj0ESwV3zdMig+1YnOo"
    "3oHi8z7Ay0F+6MsL09hrbEJLvY3APB3552dINe6dMPPPpFPAyoyb5yJGeTOUMtN+of2wPkZJ8AAMLdG87ZNzTqnWPNn0LhcVxH6KAcnwyxLAJz"
    "KMrR20XohWdn+E/HMeRQXCLSkOYgOp1iQi/5dL7VtvGfQOK8bhxiP+IFQmR9kYSEbgFmcXTAnDCF/nlhhzJD7QMAYOiYlIKFIh0eGTqPf2iqQD"
    "+FtRdwuEcGCogIBAFF0ArsOsMNsMP5xAA0k4iciLTTFHBuvT+piNy/kh3xuMsHCUcoEhoh2exEgbx/tkMLskcpDB11wiHiDWWPRQoSIHmr6rh6"
    "+7TKTl1s9dKfA+huIXSaIoebN/aenQEwDimziywlu8oHmOnQrDge7ccbGdbhYZ3awO6GTeoHFUNz/fAjwsPYueAf/M/hb9FOANDbPsPr8JalM/"
    "mz+8tBYtunPCwZ6hes06AY1udTOPCQPtsTAVzM73BRx+XOBmSB3xV1F7v3wcPIn7xBhXPo2Mk4+mGavJ1INjM+FvbGbyH0k0S7SA7XYACIkwLS"
    "GiQwDEwdQBOgFkgFhAgqxBYRBEINsoWQkDEERbIl7k4pHcCr/1cfjUrHTrPf4yDbOvKGjW4zbEjn+oacU2X3ePhhSE8uouWwR4+rgcjYzYzuNy"
    "A9mWcfT3gEd/DuAk5HEHxCF0JlSzv0sK/gAt96/Cl6h9k4I/Oxpu50EiycF93eNAEAiL7XgOilgG6pjMepvB3c9eUpDOUnoteyGHgT9i7Az0mw"
    "X4rDMeagLLC/n+OnkrsNEd0jYtibA2gAXbT0EMVtg9eZ1ltwhtd+IjejQOaPCuw5ZmzINT6owQFsOFwMAANGH+DbSGgOtUAP9yYPAGCP5RAaDa"
    "CIIxiJOmCdQA3ECEWLMuTqlDrTF18IOoISzL4BsTZTTTzmpIi1htlFTOhApx0YyBAajj0u9IgrfYTAYe6Ito4aTil0CS2j3eDmd4RPl2yPtMIB"
    "ok+2uDwkuH1K7obZMa7ODr3I4SHYFYQwEEkOBUefJpFTFTqYhGOS664v8jviownmUKvRUAEiesff8GAPh5E43rA9KPo83Mezo72RAQ9im7hHNg"
    "A5BZsFtSBZGIF7hMCTowvJjf0DSgMdlskxPcAD08AFABxXwCGTRBrCIrpod6czh+AXDzancdNcBzyNXn/aZVP0v8/6EYL9wr49YuiYkhogJDAx"
    "V5qEAUQERVzTgH0AInOougNUAXaczB4iAJyJNmFtR8gBQq455S2ZRLCMm5okcNko4EZzaBhDpOY57R669EbNJUPOZEsSgAwxd10OQ00YFdFu9Y"
    "7+ROOwIN2mKOII0+wP76mHaqMDlbccg8MeQBeW7Ce6ZHO5vh1B/pn3JJlzmDPDnfZ0HNth/N/gJrl4Kg/kisOw0Q7X0cWL92nhMEh2+BYELmVn"
    "OCGEx7xzkxPEYe5MFnbjTO+cppuNHeN5YF9M25L1kGygN5GGIUk5MnUO5eOQ+IJN4NGSoHyE45DA2bLxCE90iDG2tHHBRMMxgaPz2SXB+DmFbZ"
    "j0v8vInwQQEAeDaLuTh464BjLASpIRdp0JBUDMWkVYGwEAAnQKZU6hAZ5hrkkwhIy356I4EbuaGINuKuoAW84YmgYAI16BUcgYCwIyVLVtpVgr"
    "5iiSX2SPH3Xb74uFgWBn4sZwJOogVCBc2s0YKuMMdv0h7PHkI9sipBG4l2jU2kN0CY0Di8VnEg1R1C6FY2sC/eO/D1X9T6Kd/x3HFM7Ltqem3Z"
    "t2QEdOzYUOXoRwnE5iXwsPOw0d/LVdEGRREzb02iYmuWeK7awTkQAyR3qbDy077gHEodN/DO5ujY22CuiDg4VVuNzGnhxEA3JsmPiQC0SyJLyh"
    "GAc/gR9XnjbB6ReQRVA4BBInOx/mDDiQeEb4B3I6Lmibef3BgTgeNcCItAcBKiBjAIiQAQnUHDUiSQYp7zSInRYMDICudNgBzGUVYGsIJShJzR"
    "biGeaoO+TsIiyXfBPBHkgvmFZa7bbFru0QiSMhdPMYjaratptOEjQKlDlPYyke1zXreHaZRKfh7uv1JGGTN7SsIZBQGwgNCBfg4EwRj22p4dAl"
    "4wR97Kc62ENL0Cdx42iqQuD1TMktwoZzdsAU4NAociATfr5H9hfJZTU4gBL7a/1hT8e3hkN88ZcQDj1DewD1gcHrNeAwhEanrYD9YeREpGMOCO"
    "4G8WCLNh0RXmdjeB/9ONItTxE+UNITuQNoooEtTR5Mdrh69CBa6CLCcECJjpjdY2DdE8CiU0vSALIdDh73KofnZo+LUbIFR8wJ2V4GHoAABMMp"
    "NDwfHCGXwQgwIXRkiCGloplIjYw08YBTqVipWU08kXAidKtxgXmjm73iheJTVpyJfaX5liIBrYR6LqtIbGa44bpuIGzLbVnuO9UyBgKNVl0UcA"
    "StpURSdbULmGac7cpGg0kkcoMRVPs1W07Df7TATU4v2+Qt/3Rv5oWJago1cQDUIBQIMoeOJOITjJ5NR9ByjslJtcBtJZH7qp8y0D0QIwIjMmjr"
    "SPunI2C9nTSii5hymg19yHFZGWDn/0eGG6BTufnJs1du+JxXRCQwQ+7uw5TIuXR3naOTCTtYe+o7x87jEUN73nt0wxEF5BDORoQUtLgcQmTj1z"
    "GcpeSDNoaWKXkETo8TaylLR46Tnzyhj84fgnZPOweni4OWBWZxD+ANqeCpqAK6dbG9KCcjPgxIbQNGMVRIFENzFhTTsJsHbYAmkEYGQgqhlZpN"
    "QmOQAMhobdi7vdjXxEhvGr5VARhldDlnmmAiSC/5ZiIqocu2UyRn1OVdueGchUGI1HZNKwSRqtOIczJKdXEgqmqfJfEiC/f5RkYcRci4PM/CvN"
    "iykJ1MIti8XbJVbqZbuCjNZKVnpQ41CsPDUoctMRrLSXiKAAhgDCGOlizaGZjpO/ajTtvoxEQbrg4DIXwiL2GLkR4DgLZD4A/enVjbv3gCt2ix"
    "v4VP5DnIP0WI3H6Ae6XkMmldgiqiD3Yipxo76A4Mhzn2oFtyiNMIgPjf/DNyxvv9tuvDqU3CvHfiRClyYH2W4YxOkkPOhjyGOmew5m36kTSN3U"
    "7uTNeBbJIZrtNdPX6qMMyxnI3taRO4r8WmZkjka7R4giLAhlaQYaANaSRzFeWfZmvBUaFIWTdhhSKoKGkoUIYzKbXBJCA0bco70oqDqowsWrHb"
    "l63SiMi4CKFCXTPGmUBQVU1R2zV69yNH0KYD0NOIxRK1Vhy0YHq7zy+X06oslaE0llkoQ8GEEHGc3D6ud3m5nCbTJJ4t5vsG92VltAIQ+y7RfL"
    "Yz87ftKbBQU7BWUQdSgdTA7ZHlF99k1TW8nsNQPA3p0xMiqM3YbA/KISmi2wlxetFkSysPweOtAkAwBhF9YQw74e5hyuhtP3Sw3MPcEsfgQZtq"
    "4jA4HZJedNaSk2CipbCiu+WcoaQzBjHC5p+jhJO8vzhIIb+5gTYrsZOAJ3BsF6jjDpkcORlnUDI0vXuea58R2unIoYvYM9fQPi0bLBEsq2LowT"
    "gyHdj/wpMmgXNZTqf3WPnbMwERAAyjTmtIZPuL6WPIzYaySHcZUzsVbiCqTIQIETUJr8uCFLK65YokGFEqSrA841th1IRT1VbGGKIIuUBkkmtU"
    "ecdmoBGbO8k6wTlnnAEZ3RIXoRSMKAv52eTk+nT2+LjK0niZxpxRGgZcBqtdke9ZLdlmt7mcJycxO0lwHUgjZkWlMqW1Xl3j7nO+fV0vbup5yN"
    "udDhsjc5MqkAfu8nFCA17DDcHNUZx8cOjqeGMgB+BhCVc4sNcQ0X/0NsKhxf70eg3MNoO8QhXR18tyWutHooLT7j2krZ5CFowkLyxXyk4aiZwy"
    "GJxqC535v1uFDQiXvm+Pbvd9WHPCkS1yA6LDUPbghJbz7460Bwkvm/zZhW+ZYSO1LWeYM+I6eUpFA1feIfJaPLc99sh96G4FOHqk7izQEcpCrx"
    "OAbvsefXKfjd+GgdIGL6Ltn56sb7uZwXYC5bYOv2vmOxUzRoIBRxIQBSyZBnrK68iUAKrRMGfYGVi3saAW2kejWwTDVC0w5iJRbdU1GxVnUD0w"
    "VWitCSmQkgEDZAxJgJICu7aeJ4lk5nyezbOYVHeynBVl/R9+/btKGcbY5XLy7OR5XTdd25zOJ53SrdEaW6CuqZu87qbT6ss0P5XbVTerTbZRyb"
    "6r7lVaY6RJoMWIkZ254ABN7Rvfxtbkbkn5oQzDjZTki34529ZhSTrF27HocyINoAP4dN89OlwFd3z3pMp0hNLQA+t6onfgDAHBTRTIT0v9OvHY"
    "aESHp/ABLjHiP/5nQ8B3sjVyR4XuvMCdQOCHiMlkS9AeYOS2bwcW1IckB9ErigdI0H+O+g+AAMbOB4c4at/ZWCjMA4k5wL6hl+Rjrkalt1vVkE"
    "EwxtDHyd3PZvmDnp6Fu6Km36zPWpIR7zgiAjDQAlWteYfcoOBIC9HMcdfpuqYg4AxVCwBkKlH8gGrPojmnNoxmmofQ3msD1OYAIDiQblXXLbNw"
    "mfIkEhK0MRqRLubpNJJGawYGOGuUfv3+sWq7smmncXwynwCY33//9stPnn90ff7jzbrtFGeoDEkZCCE2eROEodZGihCCaWGmqyp4rMOvivO1ml"
    "aQuFp5DsfFLaP8dNTXt3COSgtqGVqpxh3ckbssvJfoqSX2OkYeVQzGyHMnvxw2MzpFivGlNI9bHv35ILrqUtgf5OPW7BBdPcgX+IXuMLf2lBWP"
    "oOB//D+M7oKI0M7uLKzP6VQOfXp/COHJAJBLlkC/q0NkQTXozJrIk1wEv2+K41PSeT5DqoPkAB4dWoTTCrcNI3QHEOAVG94nII0kIAAASRtNz9"
    "LVPzrb3nfZVXD3UAS/2V52IJXGSjMDOmBGAkmmGall2L5It4IaIEXEuIhC3paNKjvsulYpBSoPKa/bhjHgyDWA6QopmEDSWkkOi1QupknEDemW"
    "GcOZkZyFkiVSVE09T2JjzHfv7nZlfb6YNUoR4NXZSdd1b27u/vTLTy4vzpRS+7zUyIXgXdsBsixN6qZlDFe78m6VC85kECqSGoIfqovfby8f1W"
    "JlUg3cCl6MB0SeSgjRB0gAw6tDR4/TP8SJ4MnqdxSGyJcL/NB/hkzI+3AEl9hk01uy8z5E+tDn9pIq/TzDAbMOXAby+jUe3sFpiNgUa4BHueMG"
    "NIj/+J99aOf3gABXfBYtdx/tgJy8VJW8w5Js07UfD1pErFt4e8BQpwVLjoowjfRvRlW2xVw5PWJfwa2H547kP57E12Fm9SHhnZ4uq/Vclv/oYh"
    "NGYg6rv75fflectEZ0CsG0E1lloos4IdCE7WLWMs6ByZa4Nq1Wbcgggy12eadV21adUlpVHLWUUZYkAEbrNgm54IikjKpjyU6mMXV1KDkDU5XV"
    "Ls/TONRdd7actW07TePv3z/sqzaS/H69nU/Tn3/2USR4WVbn8+z8ZNZpWq03+7xCIRhDAkziOI6jrlNtpwLJszRFGZRNV1Vt1XSdMq/qi/9493"
    "zVZmuYlBSAcXXqHK73mNs0zkH63p5L08EPjJgcOeQB7Wv7H/QEO+AXJDRCCXg9VAdY6ODI0eaAH9rkdh+i7TeRK+w7NI48WBc80UyFD1IK+4Aj"
    "HH1AJ1Ad0bTwdKZKHk3I5QKMwCvkxDp39uAl0K6C2pCj9HvYA5NbkqQnvjNKjC080KKKaMhDkHx4qiN57HfqXDS9Nwca2mM6YN0/PH88S7tdaf"
    "4/9y9vqwUwDQYiVs2DPUdUIKb4MBVVB4EUjINiUMr2UTc5RwOABhmIREPAJEWBbopaKcVN07YQCMaoVXUnAhFFnAue5/uNKalrBGN5UTaqS8Mg"
    "joKLsxMD8G61qlr1br2/X2+vl9PnV2c///hZLFjTNKdnk4DBzbubpu0MYNtpRgCMlXVzv9oGUkohqqZmjMehnE9SxrnkfHE6I6LrbitN9e3+4l"
    "V58qjiAoMWImOYA0MaNwxcHSAcnWXgC4QieNwgZ8zuymV7yP4nckwjITKnWCCfx+HO/i0qz8I7R9rgjvSaW4z1cbGPLv7g0+JpfEE/p5FMDqXd"
    "Hl6iT5bRSpjatAH8ete2sbxCYCgo+xIdB6lXH0PkSnyhD4YlB33vVGmOGg+54GkrHeAIl6EzkCAHB38YtZuhDwTu00GnOHSgxU+mJUOTGBE0B/"
    "XJZPts2r5f039YPVu1S85bYyARdYqVMuJEri6DW0alaipp6kRoKaBTbafUvmw0CoMMAOrmHXLJEMnQNEu6jpqu3e0K0jUHpXXXdVoTNU19eTL/"
    "48+upUh2u/3Drijr5ouXl9+9eR9LfrqYcaR1XnSd/id//OVnLy5DyTab/bro5klwMsuKPFeqQ87W2/J+V6z2FUP45Nm5lJwLoQwtlvNZEoMxHE"
    "Fp3VVtoVUUh4EM//7V/s9e6N/c5f/29fKmiTdmUmN2yE7pKFr7ITlDF3CKVt/a4CBCYXWM0IpboluxD+AbRBpjKsBDDuJw1rqqX67gvAGLhXSk"
    "WtD0+Mx+p5HT43G77G5JfGQrW+V+Gkk8kcfNHnSi0MHlETtK69BxWG9Jv3bEZkOrTYktYdHGG5cv68gdW6IKOrRNV7/Rheb1sZMsiu+oRkYO24"
    "QcmIQLgSLPmKH/PhuEhwIf7eSWwMkxCMnvPYwp40OZQUAAEVYCmpOg/WaVvt7Fq27OeacNBwYZr2fweIXfLvh92zHNUiAyKBkznJuibIhMlvD1"
    "ZotMlFWJDI0hbYxSXb1nQRDIII4Cxgy/e7iv6yaO0heXy9PJ5f/u5591bdN0XRqGcRDu86KqqyyOzpIgVPXnV8vv3j/82U8/Opsm291+vdnWVf"
    "n8ZHo5n1V5vtkV67wum67uuk51Ly4W89kEAAIZzKdZFIZKKSCTJDFo3dTNNEmjMCTETquu7er8/hfzYh53/+7m6uuNXLegWVyYwBFEHOk7edhe"
    "IgJg6LlWeBzm47n5RH+HLN4eR+r/g1SMLbTQwXOiR69xszAcmCDAEAyh38bwdyCNyDUDENZz8hhyWnDsTVyYFzkIclsVk5NhIfzj/6Ffpi5I5Y"
    "OT9v4bmKvhPgQq+oCMkctFcNoh3snmTFjclo3Xjx1ScfTYJqPij8Z4ZF+4wGvUjE/u0Tlrp4XkauyxGKqW9Kfp5lmivi+mr6oJEQPAjO8/DV9/"
    "JL/NRKs0NmyhICryreoaydRJZpCa1XpX11WaRoC42e6bps6rksjkeaF1lyQx55whm0+nQEBGpVHwxfOrq5NJEkqllAxFHMevXr/76vs3F4sMyU"
    "xC/tnlQitdaewQ79d7OKTKCJ9ezJNIci7vHjffv71Lsuzy9CSIAgOYTqdSBpvdzmhjjOm6jiOGUoSSCwYcUWsdJ7EMpFGKMWaYIEIy7a7i//bt"
    "5Hf5i8c2e1BJoaWjv4ZPnRq8Y/wDmHDv/bvkI+dfek15Grg16A3UepCMl+PRuMR30TBPJPae5qLgMB4cF50Re9ApdsaChg7Zw0V6kYsDREACI6"
    "wIjh0joq+VY7UFhuDngD/7oGhcQvNYXc7m086fjAS1HSAtET3lN4OTjjsFuYc+xtG4pPdf8VFobtegFxF3aZbu4nHfkYCuMyyV3ZLXmzp6X0gi"
    "Amx/nnz3i/S706SpdLhTF5WhqmpBP0KzzgI2SWNERGIMjOCw3+/jKJwkQVnuokAGQgikvCxO5nOl9IGWzRAvTudffvRskYRGNaatr89PGtX+qz"
    "//j6EUZ7PkfJ4WebXMYmNM3bbE+Hc/3Mwm6fnJjOlWEOlOlQRFtV9vtp99/Gw6yYIwrOqm6Mz9w/rwTDf7vGlayVkoRRTIOBBSsEMrQgjBOXLJ"
    "GXLJkAGIIJsl7f+R3Vw97v7t/adMZyuc7inSwC2+fXSMjRhTTmRwx8GevIerMAz+Rw7fgCO1YHRNDwD94RO6FFlHsoZchM2HhtiDvitau4VxXe"
    "p5QhwbBmjj9hNDM88jCvtEfICt+fJbthnlNBwRR6oqDk4J/UEcul/n9cRwwOAb/5Dr1TFcKhK5qFiXT4Vj7LjDzsJhjARP0YsOoscB56AlW5MT"
    "Jj1LIAQkRi0Afxnch1zflemqTi8nd/+H86+vw3XRUkWzfRtu8rqpS8F0LLs0IEIhuEgCs9uu7u7u0yQOpMiLYr3bciHatuOMNW0bhuE0S4EojU"
    "MBdL6chcwEYJ6dzQVnVxenm/3+f/yXf7Hd5Z++vPrk+vz+9mGSRh9fLd6/vZ0tZ2XdGkNJHL6/eWjbdjmfnC3n+T6XUkwmGTFed6qqmlVe5Y0+"
    "lKNFVRdVfTrLAsYQKItDBiaUPI0jo9RskgJQHAWc8SCKpWCAoLThCFLyd/X0f/7D9C9XV4WWDUUVBbb8QRyXie6YFS0Lte/142BP4HbQnzZcx6"
    "54ZEdlVmft8E8OeHjcLbWDKfJVIPoWbO/t4KjOjJu4H+p8OpsQB72NUW7ab2c8WjH1gYeEnaGO6fhOqoCOlj24sMF+hkieSBQ6DiQ9s6EvjPEo"
    "oeTIiXu5twcxJ1+V7JDZMnAlPNDVJrAxHwflGM94AH187uF+mNMItWMMF1JweG9GgzgTO4n0qOZFB//V2W/+wdkPbds9FIFh6baRq6Kjbn+S4W"
    "SSac3qqhTQNNU+ZnI5TVVTdV13c3sXx2EUBkrp+SQpyjoIpOS8beqT+STiEDDoqm3dNl88vwhQnZ2e1Ur9T//yL5ngf/zTTwOOm+0OjPriellX"
    "ZSBZFMrHzW69yau6BaD5JEvTjIjNZjOl9S6vDcN3D5ubdQ5MKAOAMJtkF6cnSBRwCIWQzGRJjAihFG1dFU1dlyUxVlVNloT73Z4hZJM0jCJDuu"
    "rwMlL/1xfvzqLqn998tK6pZUIDH6D7RAfg8hjuBh+gaODI3bEfhLv4C18uxne2GR3Cfae+x22hN9RCC4pBX13bUVJx8Ztoga6HatAcZfdopE/p"
    "QCXd2cCx9T4UrbZ14nkNEseP/qlvyIhWgRk9FtUAvhnQzwiuQ+SgIGQlrwaeLbr2DuQWykPsHL6Z7OjDoUn5DmQ4lq9zy1YcJJW8VvVoEOx4RI"
    "4/2bVIc60kBTOfhg83+kQR+z+//Ls/PXv9uO4qSkQ0X+2hbppZ3D1bBkKw1XrflNu2uKuKdRwg6S6JIyCIAjmZZDKQWmkppJTBYpYxgMUk+uTq"
    "bBrLqtg3dV0Vxck0XWbxNEvCJPnn//6v86b9/MXl9dn87bvbpqk/uVymkr16e3N1sVjtiv/0+x9bpaVgFyeL5XwWR0nVtLv9vqrq6SRbl/XNpk"
    "ImGEPOWRwE8yybJFEkWCTFNIk4mKqsqrLc7fakFZJpOoWMA+ed0nVTM8aR8f1u97jaAkIQp1kansGb63BTsmzfhS0JA/jkDaFVZbUPehCec94M"
    "Dh5pzu5AHJGOHTL7SDvO1eD1PRYHBwscrUpvwbtNuNEw3Q2nOOJmeCSmJygd59zAp1Rn5zFx/OifjqRgnaySvLGM2/73YXDojt7wyVTdI12hq2"
    "NCbh/JETzwK4InuDF3fyIQfmBQTKOngx+e5w6qpL4YCaJbU/SqZJfBTmuzB/F/++hXH01Wb25rhROUs9VOS6xOs0ZC/ubtm/XqQbDWqLyqirrp"
    "lovZYpJp1W52+7ZTjLGua9FQGsdpHBil0lhO0lBy2KzXRZ7Psvh0mp3P0yQQV5en//3/8pfboswieXU2B6XevL05X05enk1vbx9OllNA/Juv3y"
    "nCFxenZ8up4DzJJruy+vHNu7NZ9vziNG/V//rrb7XBxTR92OVl3c2yNIqCx9V6nxdGK9W1umtV20aSS844mSQKsiSuqlobjUIiF50hhrA8WQZR"
    "dP+wevXj6zSJZycnVN4s5KbQcaskALZHeqonHOIclOBl/qMawbXx8KUHwS/+nnqYfmCnujh98khJAwfkSUZqW3AW4u+40ICP93Z+HdFWdE8TNH"
    "/9o1/qIgKRuwl7xQDnjHD8DHr9Myeu+ohnd8eOBB1psEa0qDfPLwQtMh791lqPORvHMfSartALn6Cl6iPSE6FRsuewr62FPjt7NBw8UglP2ea2"
    "if4v17/6dPrw+r5TkDSUbbabmO8j3Ky367fvbznHxSwjrbpOxXF8spjWdTOfpFkckFGB5G1bdW3TtF0chYKD6jqGBEavH1ekzXKWnM7SiyxSdf"
    "X5J89+9c3rv/jdd1dnJ8s0Pl9M9ru96rqffnS52+yKsn5xff6rb9/xIIoFj+NwkiX3632Uplp1p7N0kqV3m+Kv/vAKOb88mb27X2+L+uJkprvu"
    "/d1DGgXzaTaNo0kcTJNQcnxcb7dV02qKo0gbQ4wDQ2ZMlsZpHCqitq7J6PPz0zCOvvvhzW6fz5aLWYox7b/bTgkCANaRtKuefDkbF2Tp5jcukw"
    "FHYGhyNZ0Y4qi1NsDKrACgo1zTD+asZA6iFbmwHUkiZIMSrK+yjI4gDbp2Q/0i64WoaOzoiR9q7YHXXOlrtMMmdBTvR9pHNs4gPvEwJkR/U7uQ"
    "bXR3Sw8vQJ907HyXcwy6REx6MqxF7y68QIyeYSmMEtHhYgdcgbW4sZAIcnNwl+icYMVB/f3F939/9uM3N7qmGOQy394K89C2+Wpb5mU9SZP5JN"
    "NGJ3EcBqJtGtV1ddPUTYVkpllUVdVuX6ZxmMZh2zZ1XQGZvCgkoziQSRTN0zDlGDHz7OpMA/7zv/xtq/SL02XbNGeL7O37+3kaXp/Nfnh1c3F5"
    "9tXr+1KZ01lWN23dqqquP/7o+TRLGUAUyJvN/m+/e9do+uz5RdN2bx52v/zkehYHbdtGoXx2seQIWqlICiJIk3gxzZ5fnl+fL+MwSLJsOsmuz0"
    "+WszTLkrbTANAofbfa/f6710LK51eXu6J+f3MXhtHnz7KUdd8W80pLdeAiPpG7trnIE+N0IkfCd5wBAfkEQkTHCgUsC5BGcu7OJh+EwBHQthJc"
    "vQA/IUVwz/B+kIBDdur4SiL4u9tVWXb4dYP2s58WD4Bnjh/9UytyiM7BRU7MQEfR8AOKWC51xEUawcjoz9OK7I2Bh56a5RlbW49hZ/V2Ipah7C"
    "ITYSTnZLsqI4XJ/tk7SkXgytggA2+O098RhzbC9iq6/yfz//h2F6waGaSnXXlL7X1n1MN6hxwZmLP5JAqDJInqsszzfLffSyk557vt7v5hpQ1d"
    "nCyF4G2nDrykMBBlVWutgUgrRVpdzaJlLNMomsynf/Hbb3/949vnyykyRsYssni93v7yJ8/vHzZBEHRE37y+/eT5Vdu2632RhMEXH19fnp5o1e"
    "12+dvH3ZuHLRf85x8/k5K3iv7k8xfn81RrnVd1FAZEVFV1KPk8S6aTuG3aOBTK6KKqd9u9AWoBvnt7v85r4pKQl62uO4rTLIqS1b5sGp2kSZZl"
    "N7d3Ra1+ch0w095WaWUEASNiaK1zPCaiQ3t7Uh/1Vlb99B9diC+CO7vvkdC9e4yjY3isHtD9fnSYd+jQ4f0DdzS+RHRNLNCPNeBFH+zFHazcai"
    "+06Yl3uFMWezkcP/qn/Q04XcPR145w0069PaqdnTCCrmObE7SOv8nQUSA8NMQYfkiV1ksL0av67TPpi2WnD+A2XUaiCIdRqNMq7e040NnWDJxb"
    "THEvUP8XyZ8LlT+ohQjjevu2q1Y8jLb7nZAi4Oz5xTIOBSI8PjzsiqLtFJcySZKH1VoKhkBvbx+VVoHgnDPGcLPfv3r3HoAE4jQJQ4afnM0+vp"
    "gzMKcX53/48eaH20cGcD5JOoNn87Sr6lkWL7Pw5nEbJem7h12WxpMovH3YLCbJly+vzs5OfvzxDal2Ppsgss+fn1/NJ4/rHQF9/vL6NIsed/vf"
    "v7pL4ygIxLv7VZbGWRzlRX7/8HixnHLOHjd7ZAiMBVH6h1c3X7+5abXe7vd3q906r7558/7t3SNjjIAR43WrAdjF5UWSJpvtLpV1qaNWQa2FAv"
    "mh3NLlPLnlG1rk2SCjhn7x7vryOfkQ+ey9wVblKNQ2SPaik0Chh+pA14112LouxMXRsh3RSp1aBl2hXByZwcOwOpGeLsbDJvRwls5HP1E6I9sR"
    "tfNPz0jMRW+6uw4HX0Zvwt8bRTswT+dZ+Y0me3gNxakzHhxlNB8evLo2FmgFd2ymYY2rB1thAAyhjlF/FP34U/nVRk9FEKribru6wXhWlIVqm/"
    "l0cn0yTSLZKfWwWr+/fZBBEEYhB+y6ru1apdqQc0DY7QvO+Xaft22XBGKexvM4vD5bMlKLWD47m+uuPTk7fX2/+/bNjdK0SOOm65rOvDifb7f5"
    "589PtVKPpTIGd2UVBsF8EkcBD4R49uzq8f5xkkTZJCvqTjAoqibLsuUk+eT5uQB4dftY1l2WRJEQRCZLk0hw0kpw/vGzC44IjCVJMptNylZtiy"
    "ZNwnkaSSE5lzII2q57fnkWBXK3L1a7/b4odnmhCVqlEeDq8pybmqO6yyWRrI3QA198kB9C13jetsytQaWj2WYR0o5ChdW4Q3Djox0b2HoHB+E+"
    "Dw3lFoXj+bs3j3KQ5Eg2LUSLaPXDjNcb6sOx69rjOF3gqLF7rAmH0E9jcJ/rF40uNw/QS+jtRvXJ+8cHhGPQji/K2n/B8ORxZGdln6+rKeC8DC"
    "eM++gN/2/sB1vwBiI+Dbk4kM9winWExZ+m/0l1nQmnATaP9+9UMC+rvCy21xdn18tpEop9UeVlc3N/HwThbDYvy9IQMcY2223bNgAYSPG42wou"
    "uOD7otznRad1lkQ3D6sk4F++uJAcA8lJhv+/v/j1LEsFQzDm/SafTZJn80wwmqXh1+/Xr+82Ly9O27YLJV9v8ySMXjy7aLvmD69ubzb7t7frSZ"
    "a2bReGUciRA+X7/Ksf379Z7df7AozhjNedIkOLNFxOknkWF3lRNvVyueCAXacet/ui6QTnaZpNpxPB+HIxm2YpApvPp8+uLhfzaRgEURQRwDRL"
    "7x7Wq8fHMBBnE4wk3RVhgLA3MXhdPfgg58HX10J/YvfE88qOlHtzHHJyWZeWDTTOMq1lD4zQZOh5o1qWK/btCCvIzp4g7+xPkcNzQE8E0haZ1N"
    "vLeKtakKuD6LgNPCE/2XMHfJPSwXloUBz2PCEcBQF0DhtLtEXwcOfkMyOGjpFjRGNZ9E6fFz3ptTF/i544xTk0MfJULi0G4/DYNZK5CO9TVr0v"
    "2UXcFptbSi9NU+/W95dnJxeLbJ5E7+4e8rr74c3bLE3PT852+zyv6khK0Kqua+SYpnFVVWEo1/vdDLOqaQDofDnd7PLLefzLl+dpJIt9cX55+s"
    "/+zV8vF9PZNLt/WD3sy1kWf3q+kKBPZunv366+f/94fTrv7Z5ZlqXPLk/BmH/3q2/fb/K27X7x6bNpEs6mJ/k+L+q2Verhcb2YT95u9lKIySS5"
    "fdwKLp6dziZZvN3mBdJ8ksyXczBkEItGSRl8dj6dTWc8DB8fVstp8rjN19tyvctbpYQQXHCllZSSC0mIf/Szzx8eV3cP6yiqf3EJGsWfv2MPOi"
    "tM7KCe4KnflzVkcV7aSF9izEy36sI+eZtGgA2rqndEwNi/c5SzbQ9y0FoahFlwLOeInr7cEeOFFiiOzj9bAuUg4grO7nDJv4fGjJMZOAhRz44W"
    "n5RqH0ImwRMyLaKvsYhOFut9tjdZGlBniANry1NPQ9e5CoYZ//Ege7IZ8ann6xONQ/8aWK/UwAR2IdZfTt9QeSPDiNrN+5xhkJarN5zBJ8/Onp"
    "0tHh5X63359vaWMfbi2bO6bba7XVlXkWQP200YhVrpxSTjnEeBTOK4abvFNPvp83OJ5uXp9M8+u4wF3N09Tifxq7v1t+8e/95PPrtfbb5/fz9J"
    "wpfL2dk0nk2S2235+mGPCGXdxlLMJsnF2UmWBPePq//3X/zd128fLufZf/mLz37y8vJxs3n//na9K89PF+tdwaT87Q/vgyC8PF9+9eN7TfTy6i"
    "zirCiqMJBnyzkwdve4AUNBIBgCZ8gIZCARNGNwe7/63XdvN0X18sUzZHy3zzf7vTFUVG0o+Lc/vPnt1z8sl7OPnl9roruHx2eTNgvxpsw2XToC"
    "ZOIwUfNcb91HTzgofbqSfG4ad6BA9BkWOSmTwyEcOOzHEImuuak/uHJLEacf4svekkudGlCWQ1fwmAOTPybFEUPY7imv3zuMKMjVcaJRn9hWbD"
    "bh9JFt4LvVHG1i0bO19QAUbo921CC2DUvycAJPOfDoC/M5U8ERAtD6U/Zudg6qBhGdTiqhY5JMGGCd8eqF/EaYYpqwr75/w+YfYXVf7defvXz2"
    "8fXZerv97vV7jbxT+vrquq6bvCiV1oJz1bW7okri+GSxYIx1XaeNjoJgmmazLJ5Ier5IX55OJNLqca0MLM7O/vLvvn9xdbHZl1/9+DZNwstpdj"
    "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"
    "E03nAg8TqqUl5r0peldYSnyco/Q6//7itM9H8y+FMhjiirLr+APjAzo4O5iXC0l6xIzjADcFyC+77syPiKvHmbNfl0ebieP+dwM8YqRjl6WN59"
    "+CTBwU+AHO86R+6g1yNwfGhdjq4D9vPUt+10F4AhG3zRiECiRqJZZNoqRy5mkVb73ScvLvM8X23LFrDRVSjlYjpZ7XZV3ezyfRBIQ7Db7yZZFk"
    "ZRXZUP653R6mQ+CcB8eTlr8+22qG5Wu6rRTPDPrq62RZdNJuuy2u5yDsAZJkn06ubh23f3WRzf76tda7qm+XufPv/4+qwsy+/fP57Mskiwv/nq"
    "x3frYjmJZ5NEI3vclUXddsbMsiTg+LOfvFQGvvvxTV5WNw+bKIrKopokCZOibhttKBA8DMR2l7+4OlVKZVKczabXl6e//ubHP//NV9MsCaS83+"
    "7iQD5PTu9WW2PMYjZZTNIwCMI4+ujq1CjDOBNCKq0ft0UYJchQMfrk408e17uibl5ON59NZ3+1uuJgFHHrCWpVShw9Expw+HbI5cpz+eQD+0PW"
    "dOlgxupTgnxuNrrK3UD0tMfiWTA4i8qiXmiY3/cKwOj5QA3ma/11E7qjDupVj/vEUoBnNuMQW8n+O7DGSD6AZrBzBN8Rzor2k6c2Cv7GJpf4SU"
    "47ylPidnyXfHW2Jw5K+ESIClyDGiAaS6CjK4LhaSwcLFSZMZrOk8IoM58kUN2fzZIkEK/e7lpNjTZxJOezWVU3ddMYMmVZn52d7otCSsERVVNH"
    "oUzCoK7Lkyz842ezbr/f5dWbh92+6RhnP31xnYTBb354+7gvmBDXi0kWRTfr/aubx1e394awqsskjsDov/fp8y9fXq4229e3K2Tw/nHzuM0l4y"
    "8vTpque9iVjNWxlItJerGcZllMxvyH33773dv7s5P5LAkmcSSEgCi63+7P5lOQ7OpkOonCu8dNGsrr8+XXr96rrr24OFnl1X//r/4ib9rlLDsw"
    "mS9Oz9/dPk7S6PpscbacgTbb3e5EIEnWVRUTYl2WedW9XeVMyDhOs8XJ21dvNMHFyWKe6k+nq7fF7F0dNxACGCd1Iwe9Qa7bjwNAcTGR1mfBZx"
    "OSQ1snd23aWDS4f6MvG+QIsDjoruEnPcdzR3Gih/2b3kvX87VwoT2e8/lg9WjALT1BWBfVkXkf+I0hn7fl+YD7BmOeZxzZwthWrORmtVYu7hjM"
    "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"
    "DB+6cekTzz+1vbPrOp9gnkRgRCKQqoYQN0PYjMMw3L138I1X3339wWar66/23Wc+9fHPfO5zB8en3/jSFy8vHc6Oryzclm6+8sUv3rx/8O/+hT"
    "93/fHHL1+6eGF/j842d9699eDw6Oknr+wkZTdL733XdU6oE9cJd77rnHPeua4T55mJYlRAo5J4HYd3XntrM7qtrWUvdHz/jozD7mrxla//QUe6"
    "Jbq7vXU2xp2Ll/7sz/z03ubw4be+PrzxnQ/cuPLih5+7cu2qc3L56tW9K1cX+/v99p5brvxy5RZL1/XSLYjZO3f33v3bhyeu627eO+y6fsvx3v"
    "b2zs7+hd0L9/pP3+YnN9RNycl18YDmS47OB76j7sXcdvGtGR/2Wy9nY0MEzDlA1rY3zwsz18PkGpt6MFltIpJoB2SNVGwyT7imCJdFmhOUbDPe"
    "T53BFvMPnpn3ypUM2Z4wFRLClSqVGy2kE6O+UKXOGxUy/4/qJbAAC9GGz3HVmuaTsb33GscSVQcWn6N21ZK4RYygUZTXfnA5ciP7kYZh8Yy6/f"
    "3u+N7DcPPew+uX92NE550XXix671yM7EW6ziUvYfqDJDFkmCJIY/Q+dCvqorp+c+nq1See2104OXznXbzx0psSPnz1+o0nn3jnzTcvrhZLLyJu"
    "CPij/9aP7g/Hr//6Lz988/WjO3f2ZLN46rGX3775+mtvfPzjHxnHuF6PmieuwsSkoBBIlRWsSjFkWoXzxCwivvNXH7vY9yeEvZPDo+H03vCth0"
    "/2i//9T//Q3aOTSKQsGvWZ61cuvf/Kzbc2fd9f+dALl65fozhIUo56v9rZdt6z9yxMInm8Sex44Xh56fLlzatvR3abzeb06Ozy0q/VHa7XcevG"
    "Q/dkyJfm4ueEiXbALIwCsFQH241DZWybBnjqSRq3TdtTMOM0oB3q2XKoAWca2FJ7aENhKBLgmfO25rDDNpTQBvwSs5+xFAvxgfNgsDXPA9yka5"
    "Qh3TxIznZ4GidUTY8zsFWeRbzNgzMm7CLQ7GWl7USPpPm2GK1Z2ipXuKIVEJ4bBqcfM5A7djdOdz/x2NGvXthebk43R6frzWazvej3tpYHx2eL"
    "RXd0duyFF92iiBAkq+ocAySRuk52nQ9w/dJt7V5/AkExRlzZ36Vx2Dy4QwcP/p3v/cit52+MUYlle9lf2Nm6enH31hd+VdfrvdXq4rOPg24Mw/"
    "pj3/vRGMPJ6do577wTSRMJZoAUDGIFEZGm7VWYRZwX38c4sO8v3nh8e38dhnG4cmU4W6uCVK93SebqgWRrdCSyd+WKIxXvxYnzPUH7xVK6nr2T"
    "ricCi0BcnuIB7DyYr1+/5phSTPdqu19puHd4ooiHW999X65HxEcMB4p40XQKpn3ZOAcLvMlON0CNYLG11DciRzJJmUYFmYYX3L5NaFln5zh9IG"
    "7ak8ZBYS5rwNzFYTOGU2PGhPJNegQzSEujddjppxHCGkWpRfRXahVZnxbM8zMqTm72mMZh1KQZquFzVFU5jL5gquyL2h02tcAg+BtlRN1y2RBT"
    "eYodSAtdidfwD/a/70PDF57edtvr8P6Do8OjY+9kd9VrGHZWi5MDWnhHBBIpVFpmx6Qk4nxHROLIdeS6brEV15sxjGHcDBd3VlCMF/dOj469k2"
    "euPt2ttsNm3feddB2Y5cIekhVQFarbq0XUOAyiMRKx9y7zeYHEU8sirwRSq/QgEu85qt/eHTZRxzGenmAMDpBsPCbnWMPgF91i2S12tr13i56d"
    "60mmPI2uZ2b2HbssT2fnpupF0+RTfNevtneW/d3jo8vb/Wop+uAQ2PDl5/4X/7GoHMkTGbN3ftcKE2BmVocBq1S7UtM/mJarDbwzpnRT5jQR2a"
    "WjUwZqIJZCzSA6P6+36Onqf50kNkVqUBcY2I5DihxHQBUP7oVrakbjciVjsy1W/WKJasoJalA5qPK8NpiN6VzoYsneyK1YQ2xr25szBHjZ/WxI"
    "LzfCPMxuoZi5OZq6p92p2CJS85YhBH2fntKdJ+nuN69s7/OFrUH5/Tt39/b2HOmq9zvLhfeOmSmqkKSIiPynCUAknqHKUy22tZTRScea9C3Lpd"
    "9e7EeFc8x6tuiFWIVBDI0xhqhh1KghaiQiFpAjx+LEO+n7jlmYyJEkMgxBMI4II/qeIKm/5b3XqKqhX20hqO+W4/FRGDYEOO985533rlt577tl"
    "73rHTA5BiNk56TrpehIvIuR9VikkaV5iWioxaSK3ies8c+cEhOt7209eu7haP/zmlZ/8n165LKwRYsuSyrm0AlLTQ7fxXWRqnIoFm5ueZhiikj"
    "AB2209N4grv84ENQMu1CkjGnHjxL1pUGzgWZA96kTfHCD2wuth9WL0iJ+KZ1PRBs0Lblsnpe3Bj/Dt1Wj7djiEhghpFLxcTShS73VoM3eqWrCo"
    "co243BA1jCfLTggJ3IbnzAS/0/8I4fhsa7z2A39o+/1xcdFv77x+5+FL791brVbXLu2fbMawuxNjnIR4IOLs65t47CwEZkCZiDoHdVlXPY4AxF"
    "EnQoCIY3IsDJBqgBIUQoBzkcR5ERJ2EmNghhN477xzTIU4M5Vw4pjE8M9ZxPedcoxYdEQ7vuv7vtNxiMOG4khQ0SBKAmZE1lG6TkSk69xySc4z"
    "C3We2CWwIiVBQtlomDT90ETMvL3oO9a9rcWLzz1+eeE/+PiHfumNj4ah84tk/JoEKDCy+SYGiA10oZxqNl/S4NMaxQYbKjxybWT3fhjPLpnCqI"
    "7XcJ7ekCDEkg1vVZNVM76rSplnoW12TlmDIerHZt+MxWtNbISx1VIEbpE4jfirIluKh4QNI+MRQ1Szx0yrxQpW50dZoxUyIanGR31uYgM+L/hr"
    "b5wzminN1DswFwxHMbxFL/xn3/PdB3Cq2Fvw7fsHDx4eXr+8/97dB8dOWBw05T2IyRViImFhQX6hQCLsAIhj50S7TscBMbBGIoZGlgTryUNoOG"
    "bXORFPrGAQaYyd994LE4QhTI7IOy/imF0SgqdmCamS99ngiygsvvdLIhEJzulyQXEkItYYh42GkaFp1bmuY+9dv3DLRdKgQjyJS0duOudZpCR8"
    "5LuAADHquFl5fnp3cWNr8dGPflDPTuXSpa9/ecd4P1FtEDTLQcIEiaQmKssyM8kizLjJQcqkaZvkVESjmNhgTaJ9Sa9tCq22bQMDngfYjBrq/N"
    "7UpE3LlxuhTqkdK5DTN+puswDAbY5i/RtRPfbclAvchKvAzAOmhocNgtP8N2QyWqYXVMlbm5MDRgPUYKPjrRKKSXwG+4fwI133taVUu23mStsw"
    "EpIRBEqCVw52xt0nr/qTw9PN9d3li9f3v/De8e721aNDf0vD2WbDzCJS/BvTRccle7sksJoQCROLBHJOqOt00ekwEHS66iApTR0LiD0LgVWVmS"
    "MoakQgSbQ10tQ49yTOeRHP7CgxElN5qJi4rUIUSVjYd86LG4KXqES6ICgp8c4u4kicAPjMItL34kR8R84n8QGJkGSiXF7kBVmrSsIUmTSSRhrH"
    "61v9C889sXP1MT55eHfc/vZdT6JqSAkGLmLVXtzUX+X9hFkb3Ib5GkioEZ8VCyKMh7ywXNo+bLkATVwlPjewnzsLGloTVea7OVOM2rpdm2QOzy"
    "Jba0EZ5gCkmbe5NlxadQu3e0mjfyUzhyhFvf2o6TssonErUWqujkyN1YENHg4NxIPrWUmNzDyr2mypQNxCSFoTpjmUQcx4cNa9tL7yg1fGjTqs"
    "Vh+7cfFoMw7DerX0ezur45PTZBoq/afMZixhFJL+cqXUZ/LeJ/goHPqOdApJJUbChGlO40CIYAbggKjMnRRwmlAaE/i0sp04cY5UEwWUfJcASQ"
    "BAQozUweFF7zqNEQAjScBVufMJViPeM7E44QTpoITunqo+4enuObnxoGmjAsh5/+Dh4cH9hx/6yFMXr1xaXLjcy/DFN5d3DhyLkjKxls3TzBvm"
    "/jIjQiuqy4LKnnp0xlFoAsF4MrArsRSxtJlsyRRNSCbvEkQzfrRtwpfjoXRObGIZKla0YZja5InKta7jsoyZbsQodtI/G680eWsmrrHB/Z/T0p"
    "1v7aIJAW2km80BDEPProV3C9imtnq1dFjTnwKz7dU27VnjvHzEqGMW2iTMMdAXbu/8xBNLP2x63+1vrz554+JXbh2drYdU2iqQxnW5py6OWHmi"
    "bhOpeE8EqJadKD3IrC5TRXm6SeoWFSTkHIgSkFAcExSqiJHEp+xKl4jCkrhPUx3sXEbfh3TECen0S+KJSGJquEJUdQyUgPvZjpzPOnDOdqMJKE"
    "4uZaqVnx6kadMQALJYvH/nweN7qyevXlxsbUu36HZ2v3m0jU10K4mwwSN45EOfxhJ6rh031zTVFAkyKjVLwS1fLSyHAg2DBm3X0MSuTAR6LVDP"
    "xjkFQ3uzxnQL27R9WNtHMWFQntqJCZpUmIkHmntE9cLI7byjdCX5fP6MicmmJlJj5rE3/JypXwnTymp3FfvAbIEBZmuPp/Ms8tqpaQ3ck4nYjo"
    "rqWEbLNgD+8p2tDfnlkrHouk13dW/n2ZPh2+/cufPwBINfj2PyESYjE2lMoWicog7z5wc7SQxCZknftCaAoXTZN5ca5oCKQicyiVBG1CuQkN4g"
    "FqFE0shHKBACuo6ISCPnHBkHRMohMNlgRb4npy5GAQMRvgNibvVDc3EvQgRmITdJPiTRqzLAjTNRCtAIgJ0MQfewefapx/qt7X57h72n6G9udo"
    "ilkJzTE8UMIsPznKNGGXrOGksmp3ZudOdqb2fDSTG3HKOT4bZYhOkIoiIbuAoGykzdiijZzMwniLhVp9pfbUDb8HN0bv5/qsmIatn81IRRcNub"
    "rRucjcaB+RFnF9cG5oM5XbX1bc2bnbAnp0mMajX3ZoNsogse4dmf8TS4aV3nAkj07SP/cNNd3u50e3c4ORkI13dXf+QTH/zlb77+5s33zqL6vl"
    "fV7CdMspnUa2EiRZKvCItCKeWjpSaqaqJq5pTy5EhOc1AhTpRMBcSDoxJDNTvyiNm5tA+CCTqF3rKQFgnfFKkuHSGSc5Q8EOLzfTXdN+GgmlwX"
    "9X49tQaTFSNnXaS/KFVY0OlsUZCEcXxsu9te7vjF0q92VZXG8Wj0WUrOJdsVNs61Cj+4kbBU52jJJyiMact7QsEWcdtiq0VtuQph0jZOPQgB7P"
    "QjC8bQsIrMsV3qJlTqt+2alLKsrFc+58sz03zy9m7E5lY1zy4yRwRbNW2FBVfRZhNXMWWPVBt91V5PW4tpVduFVtE+jWqbmW3kQCXKVAtSmRna"
    "S3HZN8pfjKkpa2PRZlGp1W4GEJPog3V/+7R7/KIO2ztd12GzGY5OPvmhD33shae/9uprIkLiRVqEhyScsKZ2TMpocs6ZGkxZkjpRU2IgEVQVSi"
    "7nupCmh6mqqpmyQ44JIo5YkoSK89/FpMqIJMyIpAye+HQi6VhPx24OkIFO371y+qOmumaSN0j+Xy8k5dzRXA1qJFUAqiDPm4cHdHjf7e/I1rbf"
    "u7g+OSbF4ZpN958t+qDd/Yptz+SymA7fBCC1DlCiZgQH2MjcCs6fOhE16iL9IVL3ZUhljnHJbylmnqZpb2o5GPOb/RWmBv7bZlFXUTgJldCsac"
    "Hx1Eswk1RGAXHk24o1QxTNtx2zlL8ZVSTDbMoAmki9gBEIzb5+NpUvStoLWvd9EyMw60dVkKw5XdFi+dEmxnHRmFr9afosjmkY+pcOtru49ovV"
    "YrFwGsPx8b1bd5979qk/8sOf3e67kAei0+6hWq+4RWE7sdhYJmNHuYw5l3YaJyJO8q/k5AlmZnYTT5EptyvzH8TMkmItKCpC5Bgp5qsakTALCZ"
    "vw2NRimf7rPOXbn+SJmDhKk8YEk/F+WjYgaEoVT8dPWsBQVfDpO2+5cZCtXXfxMeqWeve9DRav3pMUtDTJkCa20iRO4SImMd8dG6WG1Tc3oJe2"
    "HJ0cRzOfmpEzwhoHWtMsV2s3aiRGI6vCOXAUzfVo0xKp3Q5upoY2tC19Ce2bXFGg5xNVJtF65aeVbQJNf9Imt5VmcasAmLY+ZsbcTdseXTAhcT"
    "ARj1xQyIbuWLwddeFx9UA/gr/K1k0BS11lk74Io7ghJuDrhxcpRFXeuv7U3rXrO7s7D9555/133usWS+94GDa5nFQgxuQnLB0Rs2emNh9EpheH"
    "p/FbcrSWf0hIx5YklTaQtGmSVykoX1rzCpNyaQsh9S2n0kJyn8bczgsokwHmnHGf9Wgp2Tv/a0qkREoaGMoaiKIgchwpRtJAMRJzDHF8963lcr"
    "m49JjfuXh2+5asj+7FxRt3R3Josjd5VntYGxM3pmtu2XmYa0hQzU+Wt21ymgpcaQ6zIGuPwtR7mTwabPPWGvP4I44As7WbAB7NqXglCL4wOmuL"
    "X2zuxuxo40mOzjPOW4HIlKi9pgVccZ6T2SurPPPvnD4hMpi3zn/QUihsJ5etKCnfmCYYJWAYq5ghYptsYzTbKrcquilOsP0txepSdguRL95anW"
    "ivh/fQLS8898HHn3tyq5PXv/S1OAydYNhsKOezsWaluKb/AshVXGpjTE5nJspMjJox6FghRMJCJHnz40nsgVzVcs1JgXDK+Uwy7kgAVEk1L9F0"
    "L8K0phGnCmi6NPKEHXI9E7PrmAX5pCUGEOO0tSjFyBoQBsSAGDCGFOE2PLgrpwf99na3XLLrNvdvrfb33j5Z3D1hcmUS3OK4yqUKVO1GpU4qR5"
    "S5kJVXp9qMuHgGyaIw0CzTrJEUtigFGONrtqOWmcPUIXsEub0UtzS7x3Ky6eWNzXZKZ2sN071I6huOpqgrT+XcDdhgaPJSMNEzpYqg2bSxrhGu"
    "uoPkeGKuRnuunwxmiDE9bZSztfIMi9mSqwmLz6HWmtxRoyQ8B5M0bHWafUV5nxS8epdeP91Z6DqsB7fa2rm4/9iNx/Tk9Ojh4Wq5CJtNiIrKik"
    "1R9hEhJLGLRs2bZIXGMlJwBSWbndib0CTInubIDEnDwXT/Y0kOClF1gCCbEViVu46IONamHmY3/mkOziBO9l9VpJhEGOSfCJwjFhCDcnwSYiAo"
    "QsCUKqUx9p1//If/8M4nPuuuPkGLlVvtypWrLz9chqF3eXPXOS4oB1JOu1Gr0zADPdNuYOtVB6qSGDYOcHp2hTBBtt41CmuTTd9yp6huzSD7Vl"
    "fwt5GMTY+YSM455R5RvxYrqyfgHI7f4Iqnn64i0c4HVuR9o/FjNXqjUnuZtAfMERU8M5A0syGYBs0EMsB8ZpTv37VcQd1vGCVSqxly2AybeQhA"
    "Yyqt/WUnGM7c1+5tf/R6fzKM/bUr/sYT3c7ehaef3Nrfi1BoGEOQrotQR1BV0uiYwYKoykwKzgeREKULmKG1EmkMed6imrtfqaCJgVlQwKpc7i"
    "lpxkF5VJDa3mkOkf7rHGEadqlSOfrKZFlj2hKIidO/r7W7n99Ux6ya3zRVjCGdEqoKACKIse9cf+FxEIM4DiNv79D2/tduJu5PXk4tnotbktF5"
    "oyeXcHZJaXDTZJyqddQKEzHLn6Ca+NmeZLD0pxbPMLnkyRAezPvB3HKc6utRW4KPeLX/TUEoYko+bo+ehiDFsCJyG8iYVxTY3gMbGTi3Ny1wI9"
    "BF49s1JuhKmGDzkWDiL6ZwTGqkO/X51cnmVLqAM/AN0xlTznRYgBRml21mbp68+q/cXnY7e9L3iCCibm/30nPPOMcYh87ROA5aXtAYCapxRIyI"
    "kVMDQ7V+K0ogmQYSSM3J5tRS8CTvI9V8ooGQFriSTnPttB7KhyoXKkzRhBWPkv6rihiQV2Aua9Pcj1Jkab4ZEkMphLypxZA7ojHk+QQATu+Sxm"
    "ENaOIFu62dYfC//yZIIhpcbZUGTS+Fdc+3JlUqb5hiegUaGUnLULLI0klwbXZXrpBMskVk2RNNOdpYi7lCe2FgvZPUA3Ymnz9rSXAxoTbTIqun"
    "jrdS9GmeVyNQawwHpiK41cNwg2atHRGYUGLT9sT5wHgyMFITWmYvzzrl2tZNrSC5CnSxcYmUC1/rWixX3CqRqQj1cnRqewzzTNAHMLn4jfurB3"
    "Kx2/eBSOE0BpydRKj4hYs0jmPsIzEhKmlkxCkGHqwkvgegxI4ZpOS6MuBLC2MqbHRqV5ZqKc/H05uY16aqcxnoRKSECBLI9E+A3BMSV3VdIELI"
    "FzCtYXaJmZ/b6qrMiZRA7ISISAMgSe2tMeQzWFOPlFUhTG61o64TvyAS0rjcWr17N37rluQLYVKmY5L1o03XMxVfoZ5ZuDyzaZ6iUZainBXGNm"
    "rSAm2uCRseEluLqYmxzIUz2A7VYR0cKOTpIteBIWhPJFTLfbHqH7vLeBOtqFPGYFFSZ9aMJa4aA+18Js+WM1zE3GjqOSP4ntqDzDO1UhHRc2Mw"
    "mdWoxnMxExzB8oTLFzvDEde8i2lX1GLHxPm/q80UJ8F37nfvnK6e3TkbzkDSIaMINKqGYR28jGFk7+q0IxWKLErgGFkEGhGJJSVUExGTA4FZnA"
    "IQImXSfOpBQRrSOC5N5JQEIIxj6pWQd/mCnfd0zgO9qFg4yk5fIEZSnbracRI6ySSDTidz+rGm3ikTQphO18gg1YikdovpVIeK1xA9q3Cv40js"
    "2HWI0TvcPsT9YyXpqxnVJt/RrB1uNNgmQI/KN1iUYtW2Y90zNToGJmPJRhtwk14yi37K8lSd7lbWmWvvkQ3Dm0wjPWlSGg2CNUYYAHH1NcC3JP"
    "sypeXacUHzFloN9qxNWytwemSLkRovYmHCgWYGxClp3KRWTHgLM760StKpWmj9m/OcHZs6UFCzYKPTayQyZoFLcR2Wom4z6noz8mpAdMIpAAJh"
    "CIDi9DB0W3G5jMIiojEluUEjmBVMSkyACCszKwlrssmSgkWgOsG/pp1VU1fGgTW//WmLj0ohkAIdWJlAooE1sOsod8fTSadJUkMh1BG4piJQp8"
    "NBkxqAmIgdacz2auThHqkmZSkQoSBVxAiFxqgsSqTrY+eFaVtUsVlTlypgDepViVzNNOEmZKteS+t2rCXCE3ajZ3uVa/uJbCugqTUFg+os3Ceg"
    "sdeVIo7LZYnUzirNgVQnZGywhcymoYN2F7e1IlmmVL3lgMg3G1GLJ+VmgAhul6tlCMOmfBso4xRDDBsnZR4W176wcR4yUcuhhDVTmaqzQIXZ5i"
    "Ja4v7U5+X5+Mg8KZPIY4sFzHXcZfCabkgKkWy8Zcc0Bo7RMxHzenMybgJfeUw1RsCxqI4EsI4po4JY2bEqiKIIK2fTw0QrZ1ZA1bQminhYGFGR"
    "jMEQVXICmXZhYVaId+wceQ9mjpG6LkmT2HeT3QGcD1SAlECgQEnFo8j67/wqG6I5E2kkKKlSGFOZnb6VpN/B5pT9isRxJxQjAZEdL/uX7sQY2H"
    "nOO0etz0z2Hqz1vN4xGuE+T801NvEkaM6H2n3j5mZolmJzClfmUu3oEbW/j7nloTb1nTkJuZVd1jpxphSda59zBDiqld/Y+SY1Q3n1pk56HrKh"
    "JBtPF+46bq8qu5IzWO3T57pFsIYIoPZzph5KGROZXXSaSRTYE5gxa7xzI0Bow56aGthsiUCjPjCBBKYf7YTC6F5/2C87TuYE75xz0nvnBd7JcH"
    "SgCmaJqiBWpfT/iACJJMRIOlcVUCg0EhQadQyqEQkWXJZi/gSa/jkpQdUB3neu613fi/PsfHLZsnfkhEXgPYmUQ2yaShEkTf98lqeBkgMQqsjx"
    "LFoeJzNRDFmorXkggRigAcya3oeuj0Tx5DCOIYwbJaKuD9J1i14j/3e/z8S+TI6pJZ/kxyvzaTXbfbIK82E6LXMS53RVA4zgpD1Fa3nFbdAEN2"
    "+f7ck3Hf+8AaMRfOWPojMnERsh+ezAZuugywONEphN04zZriya5U+gdlMtT8COws0rXmxjuVpmrhO/CvnMPwI3rB4USKPN3kG5eHKrBmdjdITB"
    "XdGcddNCmuvPVLgXDWMKZUiTn4dOvTL5e3+woxG9AEEBFSIgxmGzlNgd3VqfHhOzKgWQiqTgXAVpDAqNIcQYNQZSpRgphlRnEhRhUMT8d2lM6z"
    "PP+pPbiUiI/WIpiwX7LnGlkFj3TkiEOw/viRQaEEcikBNiIh0JIzGT9+QcsyhPYIOpwiVEIiAZc5G6uAyNqjE5d5Hs/yRJeQDnyHcRGB7e2x7u"
    "9OORC6ddOFnpYTce/h//yfDrr3jpOWpNRuEmvMHyek1JZia8zK1Us6mQZpiw6mst1z82Xf8iO1FLhkJzWhUlE3OxpdSrGJpANDNO4MaIykWuAj"
    "N/YYPzrFIDcvzMn2uEqJO4usBVrSqdz/FYG7esKZhbxkYV4ExknGT0bqBRM2MI2aYSVZnbv3ng0rB9ubHMn4u4n9Z3bZzWnI9Hj1TZ3lqZxdFb"
    "97sA/YlnDpe8cbrxPPQ0rHD84OHpf/X6p5b7+x+8pKNKlqGpIsZ8uqfik4gV0Egl6CcGJAn11PfPs770/1HKXZCo4ly36NN9MY7jOGQ6Bmt0zr"
    "vOJ74M5xE/2DuOAeOgIfv3SZgYFGMSZ2cSbwLGFIVdas9Y9aAqa0wj+1R9RPLbS/evvyP/6b+87HDSCzaB33+g/8ur/J//8/5/+MJK+rLbzenn"
    "lUM/C5c/526iGeuSmosHN7mALYi7Kn9rHTvbvIudwvKl7MSrdjZhJcpIKng0L2vl9bWwd8Z5utT0h3s7/mrx8zTD71r8fuG9ocFqoyrWqi6BzY"
    "lnwDpco51mZToVIXmTLF/ldO0gsuEApzHgjI1gIAFsY0G4JXa312GUkQw3tVSZZkA8/usvPfath8s/+fTNZ1ahE3c0dF+6uff/feuz3z549sKD"
    "1/4w32PfQQcFJMszoYmCEgORS9M4IqUoLCrCpAyO021a0lQwN0QTRpRIY2SSuD5Lg3nE0QtAvNkMIBJE54kFEhnCznkdSA4BEdd3QhSGDfFAXZ"
    "8E3DnjJwvJ0w+pxA4Up3JI84Ew7eXJsgRmZYnKwvjt1+jzN5/5/L3H9n53s73qzoJ/eMykJAvkiQ/ZSw3qHOCcYrgOuED2PohzK7FM5bL9FhUo"
    "W2OZTRAv2X4fzwYlZALh+Zw9vUYmtkRwZdho2bqdc1si1z+A5wZ55JBQlCZPqZZrHg7b7lAdkcCoBYCir0wF+cSeSWZVQ5c34N1WuVNrh6JTAF"
    "rIaGHH2X+pfFcMaqF5dVviJoneCJ6KtgGwQw0z3EQrlDCu7MwfF/zyK/u//PKu72Ln3FkUGpg6EtFv3l0crONq2TExQp6REXFy/BAzKVyuJEkJ"
    "HANBWByF5PeT9PbnIUFqpaTwM8UwrjFSDGG9GV576+bNB0fXH7/2wnNPE1RCcMOaGSrCBEhwBGVy3eLVL3/j4ObbT3/gyd3HrrvVFi+2SFxefZ"
    "qTm+rxW/IATachAX6RQkWIQew6d3Qafvd1Lwu4zh2OW4fBkbBbKKU+LtsZGRs3g+EdkAXeNpIrg54u9LxGXpiNUDUuaHYYYR4cz1UqUPlPU34D"
    "VZ/TFBkEwwc2thueklXYDsp5dmJTddFyHS/WQTeBiL2JmKrXKtvZLz7dxnRfOTlNwhmbWSyTTSqtx79NtKdigjdt38Q1qJ42NPNNLpazwqlBjb"
    "tvcJbpk0hLDWmvIbVlZq1r7aRkJthrxH3M0ikRAvmgxKTSEzFFhO886N9/GJ+/rgmARFMTJOEvQIRJgQCWLExN84skxS4tp2T5TVgUZgqBxYdx"
    "M6zP4tnp+zdvfu3L33rnYH1wNvzYj3z/D//A94xQtwlElICF0BEi3fb27bffefOrX/NnRw95WKyWq529JD9NkvSie8s+w9Kwtrs3NH1kzV+2gF"
    "3fuW+/dfz12xfVT/1dRpo11vvDPAusVEj1ngirPysJ6tYTUGyrRcPd9BippaQRtxtnYwDgmhhrA5ZmIxRtqlBjK8fcFw6TkmTla1xF4MymCWIV"
    "rMzwfC6hpkG5YBIQ2jiAZtzZ0AYbKGOJjmjgWqgfpfKYeO4IMUpbtJEWdcNpV0w95nCeV27KcXO+MRNpbhEZdgg3EaPmBt4OPjFZG+vmC6KYq4"
    "nx/kn/jffwwvWgk19+KrpiMuIxCCIaI7E6cZTpRMzJxZu+RZGsfcuvpiIqC4tzZwFnh8cAffb7P/251c7Z2dotlg8fHO5td+IdRkgMIuK7buG8"
    "qu5fvvCH/thPre/fZ+HltSdluZ1unkwCZlaQS2NQIWsLg2TyDago+okZGlMBsujod18PB2edrJQgjyKLlVPJzPK4lZLMEBawGO25h2k2aAAaJz"
    "i3JIypuNNm88QMJcGFb8vE3Eb81pfP4kvPdyXmCQrnlHdk4MLWFg/y9hZa8k2RtEyou5UVjxI3aiDDu2isj3V9Yi5fr5z5RrPGJnuRK44JDVmZ"
    "J5uhPfjY7G72sm5icUxyYhkXao2rN73kaSA5mUOLyC1Jbc0QudTLVebB2Z8O1f6Ld/f+uJ6K6xhCRIiJG5EAFKSqyUwIkGosf2k2ACISC8Sl6h"
    "XJ75KzeCDeLZaruHtpSR5Mi+Vi98K+iAzD6clpWFzYBcAhOuZud8/v7jGjWyzGzbC6dq3f3qHOp2q6fmyX8iQ0u37z1S9ydWuCRChpfWLMX4d3"
    "EeE331oQllMJK9bDTTVenuuX0yiuTZe/EpnoUQAS1BsSDKghd/qkGHvqLQ3cDLmqnxHNCT0PIKFEbS5lNIoe+lEHINUrWAPBONdSmsVRc6EKeB"
    "t6ypirbaYmfQmsQDXp8qzutoPuqb7AjDdI1PBSW9OG6c42nnc2HRzYLhTydYDslcAk4oBbhRy4beoU9YgBAYGbOrltuppAmrZMafhbyN+ivHx4"
    "+f7J0bULrFGI07BhWsXCk+lT09NxzjFRgosmbUrKpgQI4qauOpMTUgizXyxWeyLLVVRVgooAkYXDsD49ptXWarFcbu3sLLZW7ITEUcf9cpuTNK"
    "dYpepDFiBmrmGNn66d7XIM5mOZGERd59+7e/zF91bUeehYmsjcCj7m8IJ6n5vedYvSPE9emg0oSq4Rz+bgBbGk3HrTYHysTRjJbILe1krmjWaT"
    "l1fn2FyGF0YGaa62ZSzB1ZJgaacT5UWms2fa9IsXt7Hxg43X1mDR2PrtqEa0VSkpm2FAgcmYSclc+TkpdfMcJVfghsGWi3LMr++N+bex85vyku"
    "y9toweTCDMdNbybH41+xsnLcE0TNRJWVAFGCzxzYOtV26zl3TrE+RTMLttQRRVkf55GtkVPX7ShMWoGjWRDlMCBRcEqBCzOOkWXbdausWSfAfv"
    "yInve4RRwrjoXNd3GgbdrEGAc3C5jCTO3fWsKp9MWhkRJ4ncltNR0nSw3Jpzf0iExC0k/ParwysPd9mX1qENSrHyJ8xn4GSsLph0xLC8sxrHbt"
    "9GtK61Ru3P81wR1LY9T44NmkWPljEz6hfPZF9ImLaN/SxcDYe2I9tyJmYDxFq9lsaxNCZ/st6CR1W3mF908ysp3N67MeODlj98QhTDrhlrEuLi"
    "LGJUh1W9IcByM0AWFpIzmow1u9qj64wDQmSiimYkRWMaNgNeOgc2BrUWKvt3pRaLkA5D/6X3t8Bg3zcSdJGU2IlsJBQQJxEmSNLtUUPI5t8slK"
    "aoGpNeTRgiCopMcC5pboLGoJHFee9Xu/uLnV3EOJyexBBiDBpGxFE1xhTtlKmJABE0WntNMlpN83vNjzmVqok8BeQhr7gYw79+uYN2wq333Aq7"
    "yjdtFhtVCYRdiDNRFxq46ISI4CYrur4v7dKYI4PsxSFNDHHOucdoEAxomDVkxXNcEGM8va1zH2TxZBn/X+u5K5/Cmw3JPD+rleYK322g91Mat3"
    "1WJvAP1bjWdoun6heMc8M+0+XlSk2rh3PtE8OMSdmqekw9SY+Wslc3by30J1ZVM4piA/A2kFg0CLemmpntQ9J9+e7Fw/W9nYWLef+Q1P+ckIEE"
    "QFlJsrRby0xFXELqEonmF0RynU6sHBWA+HEcIjRCCVGYHdGi67pFz12ncRw3a3JCzpF3gHLXs/MhBgdNTHtoZCLVOFWhyjkItpxXVEcTQhzr8d"
    "N3/v2H4Vff3iHnkBvaVQpTeIJgo8Nmg/MqglG2UUmGbcE8a4JxoQBoCXY3tQrzbDpsIMNGSmI9bc2RghaEgXnEYdE6mxuc/b2WOt/oS8tGAzpn"
    "5Zsa0i2QeqJE1Rtc8vRZ3yvKpZJa0NO0amrQt8EpGNUanffN5tstzHGEukEW5ZKZ3E49dLsVmcaeNSebJMrKtkCFzlczmnmMTS0wzXvVTopbm/"
    "UMRE7MDt96sP8HN6NPS4glEVqgyIcNT8FoIuRcBAIoKUcjYkyWiTRmYVFFjBqAQeMYVZ2PoEg0hghSZnJEnfBysVgsFgRV8QEynJ7pOFIEAA0B"
    "qsm5H+OoMWgYNYYYRg0hhhEhQKPGmCul5DzOK3Jy2oswM8StOv3VV+J3HuyI11yV2KG3UQ2DTKGZH7zOlF957DY70KaiibNiEpQCgZlgBmBcOV"
    "31RMHUWZjH5Lbo73qGcm0/EpUecV1A4FmDhc29rcm14drGKAVT6b9bcGj+nL71RFnLK5gfScpAtXbxLM9qhvGmme+PrWOBWtn6BHWFNW8Yod4s"
    "qWbGgZsh8GB4JW3Ik/V/4BxwxEp8z0M8rCWEzwnczFx5un8Ih/Wm+5XXtr7vqTNlJyIkQkoKlVJvpGmhKopxIUnkNU0LOJkUwck/RKpBNYm7eY"
    "xj0ChOhJQVvZPtrdVqe9kt+hhC0AiBjhsaAm0pi0vh9WmH1DASB84ew5BAU+ydqIhziaORGifp9sIkgAqnUQpE3Dhufu7rQrGrjeVZWIihzJuH"
    "q9Mmj6ZpD+OCKwYhsk3KRxDJzHJkMwTUyorAuZeES6GVEUc0H7dN26lihlerjaFpdjeLYp+3fGE0Ki24v5ZxID/Zs2alYTlPS4aTheebCdyUfm"
    "0IM3P7/ITXtRFKc29+UUJM5NOK/bf3g8JmYpNM11KJzxWiswBjrrfBssNym5U1l3sXepB1HvI51B3NaHgKYhL+tZuX/+LhOzf2XQhOJBQWDIlo"
    "QtmqEgiKjLVNiAqSFDMWFZqbjAKiEELy+CZShxdhVQZEw6JfLPq+X+04L0IDbTYayZFzCieduC7nJYlQzmaLiBEMHTcAsXiK7HwHaOollUNq4u"
    "WBckafW3j+xk39tTe2uGfV+s2D0YDb22Y9n0cE2txZGJlv7UXC0mQze7vNMrBinOTGtN/DNDUvF6pWmmUX8PzNsVgKuxoragwzOQGZi0sJh6r6"
    "Frs/gTJcBn62KsxmNI3e0GBe0ADQ+FEeBW7e3qnvRY2ZslqMGpQFqtW9JkuhaRKblhe4ZKrawrwYDKdLK824d6bBM7dNogEG2XtNebImtxjC7d"
    "Q3W4BrOI94vHe09dvv9H/uCo8bT2GovyFZeAsss8bsQFVJBECIqsRjGGnqV8aozKwhCrNLcYTeC1zfL1fLxXJrt1tuUQzEwQ3Bk/Q7F/xq6cRL"
    "BHU9M5MXMGuI0KgY4rgOm7XGKL6TriOAqVOAfaIGJ2WMQGOaxUE1gLaX8g+/JkenK7+FCPMGFreSsWsTzk0oJn3W1JxvSU2TILNAjCwcwvTLKu"
    "Wh0KjNF9SISLmmoLcMm3M0Q9hYFDZdjuRUkeKVrU7gc2NAJiMms6samGlTUz5h4cGhEuamPBYjzG7SfivjqkSQcSMgsvQQ6wC05vXkqWiWFs/I"
    "2pO6v6oXYXnotRi3qI/5TKKIHridf04TnZlIPE/EJrdH3QizQ6ruw8ZTWoPiitzQPI3+H7986Y988O6Wc5qtCUGmbSLtlVotZFm5q3FMZ2CIYQ"
    "whA82c875HiE6kE+e88ywORMMo683mwUG4+f7O3t7WpQv9ou939xMfJj48OnntzbgZ3M6W39lyO9u88LK1w4sFu57DQBrHs1Ppeq+ReyjDiUMk"
    "cqkeThb9FDkao6rz7t0H49//+jZ1vZkEP3KSgCaDF7XVN6k32XRTm6gyk/1JbTFDlRkPiymkMpznhu3MbLVTzAYwaCYI3HZjLMOp+F7FiuEs0K"
    "ZSQA0alS0+gmxwAzeQFd9qSciId8qpXUpYrtNxc7DktxJGtl4dyaip9LUsBVu9EtetCQacPG1EbHIkeVIRc92mJtkhqKH5nsNbtDGL1cM8rS/z"
    "qA1RpC5I1GER2azZ3K+2oypY1D6Jwx/cvvgbr937mY/pMTuX8U15yph6Gk7EtECgGhUUo4YYVTVqDOOoMfb9ksR3XScgXW/WB4c4WQ+npzqMDv"
    "DiJOJeCDu9393f2X36SRYJZ2caQxzjOAw4PAZDidhxv7PsL19yFy4cb9avfOObGIerj128cuOx5c5+Sjll71PPswycEwdOye8v8P/6Hf7OrZVb"
    "IcJQyKrKwUh+G9J8c0WrScilRqz+zZm7DZZNRG1G3+wqV24NxiVhzIPA9MpWatO0MidNUhHAMJlVzfZGRoL6GtRQ0CkcZpYDZhsX3N4JLd6ipt"
    "KwnQTWfM9JM2UMleXabaIRbU5cDfawTP4pvJML7mCmW5jHB5pLtZGswISHPmK6ZJIIGI9wHlb4j0lGtgck6tDeSKHQ9l7rGhcrjC+NZRA5hpL/"
    "By9d/vEX7kjX6Sa4FFDBeRCexDFZkAmeJnKIMYzDGDQenZzefP/uwwcnly/tX9nfX3WLcHx6fP9wc3RKwKLvtpaLRe+VxTONiJuT4Wj98BK5ft"
    "kPx8fcd357RYtuOD4dN8PAAqbh1n331vunY3zv9t0HIV64vOcd9733Xe+7jpzXnCmPqTNK0Kgs4ty9o/jffmmXvbNNZJMZQTZFz85Zm+/YRvCy"
    "CRSqeUnWxVuy5yfRIWf1ccHwWYA7yutayST1ForiGi8IzXZcXN75irut/ooWXVP95bVnaLwLbQGGuiqMooc8mshd686daaMrwzEZmarOqFYkXK"
    "cX8xF4neXAKkuqn9BUv9y0oax9uphPKnKbWuNiObBm92wb+2OiEIyTCvaArn89FyRbw5qqwmRmi0FoiLWJ7wSIi199f/833z7+qef1YHCCoFCZ"
    "gl3TsisxOIoIDQpGjGHYPDw8vndw8uD++r2bB9/4yut7W8tLy363X6gqgRZ93ztZ+tPVot9e9V4hMXrneNENEXpyOpxthvsHi9XKOacAi6wW3e"
    "Kxq9Q5DaG/e/+JP/T8peefOHh4fxjW3iUqagAvAM1yL9XSNgtwe07/py+4b9zacT2p1ms/G4v3JN4vdxqed8rz+lEbntTAH2Zm0UI/m3rIk0MX"
    "PFdlGwFmPgmNV71EGpZqEWR4TWa5FoljERLP20Q0O+On2hu2pTrr1hhDQJ0A+tq/wCyVtzHrs0ltgEnhZAuPYbKyanOpMvqG2tdiEtOSmvTSPN"
    "9ES5Ma9thD7fXQxCYjAsSo7xvvZ1PXohXgk/khjUq85XjnH4TnsK3GBFPpttM3OGkcFd3f/fqFH3h6dG6jSkKkiTFJHKGiKaMp/8CaZnqITFj4"
    "7vqli08//jh/wt17597hrQcOvLVcbvW9jwHHJ+5so2fj+uFhiCESthadX3Q4OnvvO+/urJaXrl7aW+1t71/oH7skuzuyteTewwmJG4bN1Ree76"
    "/sx7jZdxg3G9Wx/vCZT8xpEYKJ2Il3d07i3/m9HRLJLfwJODJRuGpp2njcmv2/UGDLY2Y0RoH5/KfaCFE36EpQYAOAR03baRz2hcNbxWc234Ix"
    "05xxI+Qo7Z/iYKuFZhMmVh1vaBg3FsvINXFwCgllqshT42HlFms/3dFMGnwjHbECt/YorWJQWNMPm+qOZtaVmuloUYkN1ZRrwhSXPa9KrJs7to"
    "WtJod4abuYk60sP5pwuWwxleURwXiBuQk7xaNMVClylcjh23dX37mDD1/gsCHWojHNiazMUpIpnDgIoOq7bmtLAHLeuW6x/ZGn19euhsO1U95Z"
    "LXe8X5K6ccTJGidr2gzMxN5JJxQjX9bl9tby2hW/t8PbS97ews528q6F4xPl4LZXbnepOjDBLxbMpNEzQ5zDZCJG6j6LkIYx0oXV8N9/futbt7"
    "bdChqt0hsAzkFEyeawcY3XNlxKi/qtoaAmRWQ6wcxtv6wYcOMqM27YkkkCWyjbjErUAGlU+FRNvp7FCNVhA8rrSWzd8GWBVCePOUGpJQ6yxfd6"
    "IxXLNzQ2UgJuZ52ANfGfT4yYqUaMBMmcFIZ5TmDGeVa/3R2rGr22oa29OEPQmAywhKy7t4zEi0/LGM+sUaNaUYw9kM2vlYdqFQdkGbXVBmXC7f"
    "I1HRi1O1i7CFLxHIYUce8m76XmwkJL5BKEO+76ricWjZpswKv9fnQ8Hm2GOB6PG/TdqvfOb3dbWx7knfN9R17EORcH7jxtb1PvcWEPDNIxnmx0"
    "GLnzzjvpIutZSjj0XS/iEEeCptGlcuRMhRPSQMSd51sH+t98YcWOyeh0YbP+KgmTzZzQNgQNT7VCSdprSG312Lx5bkZKLTefaxXCj5jytVF3PJ"
    "EjMKVYs6W6keVsT2p0nvrWVs2mdYgBSmZtbhqwTaJv5Y9NHt3cbvQVfnsuGeYRN6um4Un8qHwYtqHUNufT9LamXgzM0iwf28znLKcAaM2F5urG"
    "RX5Yk0ZapQzMwmdzaZ4JTZmsQLZui/OmkdFVVaPlXOdU4V+c0jg16hl6ijGwF3EUAxFiJssLgaFRssVeiVlcx4SUJKqimYIhcCsh9nQSxjGcHA"
    "9wstN1DJG+995xGBEZix6qLMxxJHJpPqlnJ3GzplUvXmTLc8/kkiZGE+QJ6d0Uj4IXSaNYkBItHX/zlnvprlA2OfI0kq2BtACdDzY3l6A2Nc/e"
    "j5oZ7Qw5VGf11WptQe5c062scoIbkUcrKLVzw8pGa1OcjL/OxlLUpDy2vXDzXjR+Whinbg6nm5lcvfngoLkl2da63GR5wPRYTBmKtt3VIHbR+L"
    "8qq75JgKpLtjxV2xWb9ZqprE+2ZxXZ67iBoTZuG5OYnMsXxhxFDp6Pdk2VxM0chdtVz5Mb2P7GGE9D1zMfDYN3ncQIsI5DJHI+QXyTfDa5lsSU"
    "LOycSw1T5hRd4cYwdICuw8nJAOYd3ztaxegkBVrEMXqh6CQQnQYMZ7To4NhtdbRwvHLUu7JtlcQ1OKcl8kFBISS4hgKR3BjD7ZMeyiIVHKQTCM"
    "N4tRmWGcNN2jvOg9IKGZ5MVPx8ki4Mnd6nWQOtNPFrA8XIsErDrhW3tfPBcpzYwrKZM5RBIVskR2MmLM4EwzTL5N1yYE6Zq2xVY36mfsRMk5bL"
    "b67TUTaoXDZWV0v150ZNh3r15WY8UGmOtvts8ujrtbEAAJtkD8wFD2g6dNn127pwuQaK29FoM1iotme7KzPNkcANZIDOO6LL/perF7l/yl4Iwy"
    "Z0rmOGglhYFTEFmkFy5OD0+dORmFTcABOcsALMyisZnWLhuyWP6/Fk3PBI48BOREQcCQcGBb/qadERE3vHq55WHS0X5CVHXqfQJc3+penBaQ4/"
    "I6KY7jwyRjjR041QYPL2BCPDTEGJbSsnYaW5ELe5d6b8mSxBzJXTPxGRjGw3f/FNNQurMyHTpiGr1ECDpS5LookQK8wb2JoGhYGbzLelewgDPT"
    "JE8VbBSZyQQWVvmpTh9r31jEdBPGfjTm5Aosymnmg7H/bmN4+CnO6qdQ2ZJk3jIi5dsimBokmNaSWjGaCTfRiTv+p8MHbZMqzlsg4pGzFPZk1R"
    "HTE1Mqkmp21uCUHz/NvHDTraEDF13o2bNfeeAWFiEY0x/XRKKiwKsGrdw2IKSGLnRaOm1ei9GwkDqXrHK5FRT5X8GDxIiB2xYxc0dEzSe1kqL6"
    "FLkqUnAUOdOBbHqmnVZa9g3uUEGibODaCqzOtRd7cWx6OQYkoUntrmVqcA0wC1uzkaCWfNibCKKrZTQZpTlKr3sCWYYG4xaIWgE59E2FjWUCUX"
    "TJXP0FwkmXIwTNIbC9kK6xyJt2lGsqlBC/LXXnXMJgbANz99423gBgRQbEslMqmYlpnbW3jR1qUFbJXmNTwC7UVqDiW3+pXZHWCCCpSFWDVE9T"
    "pd/d1sv1Y0ST2NB61K7IpdDHzeJ1IAUaaQg1FZtUStAmJMD0eOx8xzYaH16VnfdwmCyMyIgUVYXHYvlSmmsKoKUna1ACwk3vswjo4pCm9i3BB8"
    "z70wj9AYCeg67xx1npZ96PzQ99537EndOIh3vl+K70i4dimFRaUqi8XFHHXIMWITQoRzzh0NjuCYpeACG1Md2CYUtWp3NGoRM2qz2mXY9BELnm"
    "nYKXVAVsF75+JdK9g2sfrPKa0nUKeRQpqQZhsjgQZ7w8242aqoqwlhfqiZ5T0ZFEx71Fe8AxETCbFmUHYN8YM13JYMityMlGwfhlH7VU1fUbGX"
    "/abMW7mmXthuqlkmpoLJJqfyyI3iKed357blvM9qlTHzzm5TmIMbxl5VBVRmJBnUHtvlyzOaATWcVbPqz6InZgf13sdh3GyGRe9J4ZiFZbL6Mm"
    "f8RLIFINEmxDmoCnNaBU46H8V7F0MYhyFGPQ4YYwRxiOP+arGzvVTm4EWcT4pt7jrnO9/17D2Ly1+nRhJhAFIKDy2KeY1xjDRGdL0X1bORib05"
    "jKyuYSaeRsEAsWEmlfKlJLnwTGrF5RSxJgw036hpZjbMWKtTNikTZYfmNgoNVW/KzXDdOLJKF5yJNWerTAMQNn8Im/Sz1hswS/cr70cZTPtz04"
    "HS+G+MQDbq3tq4DHJ1LsJu1LymuORq93/kYJYbyAQzKaaYLpNDM9OCl8ysktrBhVbAc1H4OaUnGx9xEepa5FCpUk1roCBzTeuVq4jQsE/qxzwd"
    "8wkHjU44jvH0bOi8A0UnQoAQUjKuxd8QMzsBIhFnqr4qBE5k0XnmJRMQowJjxNnZOgx+e9HtLBe9k77rF8ulXyx83zvvpOvE+XTFZxFippgXXj"
    "m+AM4oC8UYsAnaLRbSOWI+HWapADXipeL4ZgJcNGbBpmmfy4RSGFU6idlmZ0W+TayoV0QznjPMKJPvWnjUsLbE+p2jipthx5V2QG0a3qZ9Z7By"
    "Jk/bCDmNDa5Br5VOkX9EPV0pWRUFbyLNzPia53iqSnwxdkQbr9NUmo/yACKHW1o3GdsWZU7nNkJBNoJOVMhCve0yWpqMLWJN2rjpa1rstFGzMZ"
    "e1ynyuAmJkt50V1dB0UhMR8SYQUsofkzhxkcdhXEdd9k5DEGESkcxEZAAiIiIJ98IszCRJ6+0cVB2ROOeEnDiWHHiBPSaod973vl+tur4TZnFO"
    "nIj37By7pPmUvAiZSXXSL2XHjkaNMQbFelTnXOdERUhkHbkhUHIFAFk5JdksZKuYgrGsFRk3QBbazITzsz5q4dxmiFzTI2HvbOW7YkNXKYsCLR"
    "qzaW6Y9rd5PahGGLG5ck7pAE06VClxc6Ryg2mrt+WyxfvzcXxVY1714WQHd80sbKJN2CByExnH0ydh2OBctMvR5FjXriksgYpRZE6MOhmsMtCW"
    "j2Z7XEaXOsMo8uycMkTTcr1jbhV71TFlGXhTj5uniEZuZI/TZiCDCkiERZ0TJd9hxXR8uj46Hbf6rks9fxGOSPQsJmLnCZnzKY4ZYBFxnigKi7"
    "DzzjnvnUj6ply3EGZmuK7zy5U4YSJxQgA7R8xZeJ3fKSUSlgxXVKiGGEIMIUTF2Xr0XdcvujQvUdBZSEWszjAt524+RuuQS5kWXF5sTQRq3YLl"
    "ezAmtpk7nhuLi7lqwpw859gj9lJph3d1rDJnflWobpFBGoOt9V+g2XmqwrkeimBm6ARV59JToXwSVm5SiQNjM4wsu82cEk/VmWWCXGBSnFJ7qY"
    "EMsI01poL1t/jRc/iKBiwxg6eW6WORKnJteE17kelft8/ZzJCsOgbNrMPeqK0Hn02IwcSZmvNHa/dU0pDBp06dE2EXmVwM42rZnZwNx2fr7dWi"
    "A6lCnJBG9h7MQBRxxORYnHBqqIoTEZ/OSWEnzjHgmFzfsTh2TpjZOSYSceIcpVFEguSI5IkTlCCEiKgkjlSJNCpi1DHo2Wbo+27Re2EWyajk42"
    "BARJjI1mwMN2SwTm20RzUnke2rTQ8PJiWCkUO7zvnE8wWoOYXRVDTVFMGzWUA15zYZ8rnbaBDybCGA5XpRTnC26mZYK1cTyg4zXqnsf+R4ZGv9"
    "EQKnP7pqGedc+gqtM113FN0gLHp08hpn1OPEsG5CfWm20xSeV6Xx1qPSXvgZJt0O05MzT3mKca3apzZbGE3GRk2XY8s3aO6I53TE7Z5TKs7W7Y"
    "1HG7IE6yFGBVRThSCMru+FeWdr0Xk5OT07PRvGcQwhRiBOFzNiFhHXdcwszku3YHHsOvE9u25qajhxnkEsIs5J17O4DLDHlPRCwq6b0oITXzhN"
    "FbIcWYEY4zCGs7MzL7zouqTXSQ9dwZsxEoK1CrYTinOnP9u7g5UrAQ100mAWGt1+QoGiYe7OoCKwKmKmWXzvOUNpYbxjpiouxgVuSIxlhzGHQ+"
    "N8shAstK0NNsoAMrCzWQkmkweIDT2OUVFG1qLPc/3WbLph5TtggwZhm1s6e63ZKp3xqC+UWlMC2lSX2QHbzHIMYCHDsawMKm8l7aSdq3aNTcTF"
    "I3R5Fto1ycepnf2WsoKZmBHjYxe7zmeutvOeCY6p65wwrXrnhTbjuB7DMAzDMI4KzVEw0+vlPAkrEUmXQMBMLK4jFhJHvmfXsXjmRAeeiMMaKb"
    "H00/MRly6u0HTDQw49jGFcr9frzfr0tO/81mrhneOUFTPlSYZYnp5W8m3rn6y3/+oTM6Nzroz2apktNM92iNGmawE5uQD0KLlkddg/gqXZ/Aar"
    "7IQ9rqxRtcbVU3MU1HXS0o0wx7vYcXHxHJaxsxFCQ4wDsCjcUh3HpUFX2cWNsydXmiXqKXthKxFmirxiY18im5vUNkTN2oA9GOvcbxYeCUOSLT"
    "MFVK5ee3K3X8v07pTsvfwOKNpRF0gL5Bn2Xmub42Y6MitGsrmeoWO8cLn/j//oY2GcAuKh3ndOuPPeCTuWrdVi0fthHE/ONqenm81mCJoZi0oU"
    "xkgsygKRMi0FCQAWIecTV5uFECMhEifKacriBidsaTlN0sJLQ3kgRg3juBnj2emZd25rtXSuE+ec8+kEZgJpVHYktj9YANWN+syiLVM4rHF2l9"
    "8paN9sy+Blc2moBFQuwYLnqJkMLmj29ppSQ02Ymugnk0xSSc4lpxsmUqoZRluR56P4mi2fiqx7p3kFC7iTvQEYzlxLsxIMNkCsNIa43PAt+7+d"
    "ezNZnhV4brXKcz77l1Z+SEsTJZ5Bg7hyZWsdMk9ptbsWN62gIqYyLV9Ytwc35BPO7CO2UQ2tFQpsLx654BKmsBn/4k8++6Erh3feplUuPZiZvf"
    "dR1TuXBJqLzgE4PdtERcz8S2ERT8RC4xjEORFGCp0BEhpUiHjCOCNGZtagxCpOSBxEFMqq7BxrTHZFJGyialqHqroZ48HB4apz29tbIk66jiW9"
    "IVzy7kNMsJmiNqnJkI0mFNSQY1rj2DQ+bMqZc9mPNAcAcpv8ef5FZzvXQnGL8sxcaKBATLMoiSZbvjnnpt/ArVR7ElyXbgFX5Zu519WmQdnCjf"
    "jO0xxHZ7SfbBa6PUGm1DpuvXMttIcsuX8mbDaCt2ngwK3ohVoOaNN7tkIym2o28yIBdgjbsIaqYK1IJaotKQcwgNsLP9mBcM5Q4UYu0yT6lmgt"
    "ZeI4YrXHf/aHr7//xtsOKk40qGMpN+d04yMOhLjwTlaL0/X69DSQOHFeVbuuc8Jd54kphNQnJSZWHbzvRDgOa1muANJkWY8bZgF1AmIhxDChnl"
    "XHWIZjiBpDUOBsvT58cLDVd9vbK3GOnRfnWThHZzCgJN4vFkJACm/jYku1Q3orebDiXKamqWUvcJVONmvJwaqarPMUNj9+Bhc1NO7yTkxNipwz"
    "zNyO5emRd3+uXvJzSUnGy1EL14ayiqKFbaT/hrNYX2YprSk6T/pFKzDCZGWHtUw3sgarectfj6lNZ8REsoaQ+WZVIizMkA92OznP5sU54mzxNV"
    "Wyc1GiU3O3BVp92hTAVlniLRujqOPYUN4sX6Gq50QYm+F7PrD9oce37959MA4Di6M02WNJ4fHeexH2znnfee+6TraX/bLjo+PDg+OjzTBuNhuN"
    "MYxjCGEchiGEcRxDGAMoAiGGdKdTDRrGmE+6GBNpO0SNUTUgDBqDqmqMcRzCuBnHYQxhvRmOHh7srBZ7e7vOd+K96/usJOc6ZyJx/95PP0se1V"
    "E/U3m09/oWIW80VWg4reZo4Yq71jKPbUzTmB19WVSRjxxhbpTYBmQ7vwG1ENQZk9H4s87VvdSk0BRZDpuMCFCDrz8/iC/vR/rUvqK0rbvL4iGm"
    "+VdtJU/6hFkTf1YcwADRGdV8icZ7zyUOCQxqpXel1dwWBsaPTVncUoyC9XKobWMJNsfUlAeYHCKwWPvKAKBJp2R0oWCm+eHaoNxQ/+ZsNOHvf/"
    "HyYuGHqKfHJ5e3+0mBWo9Y771wTI0Ozk1J8m48ePhgWK8vXthnhlfnASGKqkxwLM57IkAcOehm7fte8udWMFHcJJC9gxOCjhsSIWGoxhA0KbM3"
    "m/Xx8c7WcrlckYhznpwTzsFsWdHG7ESOB/mZH//AJ//56Ve/8VCWAkXNSOdzesF5STOvL0uCMlsVBjXwxCY+tEJ+uAnBs2oVlE4mGEZ8ipaAMA"
    "9dq8Uc2lQL41E2/jdDZzRUDFh8RbG4WePDbFI6rXH1TI+UjsEy/9gcx1VTWsi6bELr6fzcsk5tYPU7KI6HBglrhGJyrsdlInZhabncGj6pjmDb"
    "fQ4wTB5mmOu7EYq3uwTmzB6ROjQimw47U6SaZAZedd/34hXn3YUr10/fHLNqjJlESangJJiZJTphGYdI7ES6zi+8e3h08uDeOO7tLxd958WLS9"
    "QaEeecLBd9dD4yd51TRNd5iZGn9AZiwkbRdRyGzBzOQUI8atysB8Rhb2er6xYs4pxnJ/XJ56jEPKGI3fb2hf1/69NXvvqV+8IOc4pe02Mmxcy4"
    "NkOZGH2J1ASUGbG36UyySXxvXjOyA6nSGhCeicb5EcWeKSZtlYiKsWxd/U2afHEy8LlMI+Otb/0HIEgxP+RNwoMa4H7tcgCWIWdpHSXEo6zDif"
    "lgrudsaRPWNDa7i9uxEJdomFJntHxJm9htyY2YMwhhWE2T2JTbWDor4q8orSqOmEJxyZB+6zZuh5AtGcwmmqdw20345JN3P/PBrfUQnn7++a9/"
    "qZ92zZxsK8wEVo3MJOJHjM53RIGUONJytbzad5vN+uT44emJWy4Xi76X5HEFd97FEBaLvnNOVVS90+icZyhBWYSFdVj7wYOFvSfm9bABOKiGcb"
    "O7vbW7d0HEEYkIS55JSEXep+ErM6vyajcE/okX3d9eHI94THjEbE41g7cSqmKC5pe3yks3EHyb1lpfZGN3KsOEpvPIBdJdzQF1S7WFnrEck7mc"
    "8axHBLRtgGZozJaCaqtsnAOcWcUGlyAui0RhIvbUIFVmNyVDs2u7lq1brIHy2CERHjFbbHngsBEvmB8qUyQHGmqMWZZorWfz49oYmZpkqoogNm"
    "BXG2xVJf5pWmOu0QyLw5rCR85V/qWs71f3/uFP/vAzq9X22enptScff/nykydHt7a2lxg3ksQrGolZnAeixui9Z47MxJEjCyF6J52XxaJfb4aT"
    "s5Pj42PvOufYO9/3XQjDuPH9ou+7vhvEe/GdT1uPJOY3YcOjEoFkM4ybYej7bndn69Lli4vVMkXBEGVJzcSVYabk5Mi5XaqK5e4Q6MOPL3/owm"
    "/+zvGPnPEVxuYc1xUzduwUbW2EY8atbQyrFsiN5IQ1qqVzzdSZKqa07Cc8JWHGN7HCekuUaaAIVolONO/folXnnYP7s/XlGNovZl2lxhqRXRSA"
    "OSYwo3QyGWeJGY7NUAIWPsk4N923zqA2AY7rN2NR6FyYW2YEZyWgrRIoLysTmj3lrhcWVNW3NPmTbCD23AIs2okisyEYmmY7ncdUafpNKstLJ7"
    "+1O/z693z67wxjINIt75779Ofe+cW/+8zWUtMSzgnVqQZyUFCMIpInbKIEISLV6Hy3WC63t8aT45Ojk9OzM00pEd7JatEv+s53Xee9MPu0RlNN"
    "SEwiIUYC0vq8cvni3u5O13t2ntJZKZKXYp6buKkp7SbsDMZA7sI16Hjl+o0Xnlre+9Lff2Xnf32qe8yhXv5pNmR6lDeAGzDXud28Nspn6NomaK"
    "jC0611BiDltseZVFKN6L/AO4zQbmY4anXFhih+Dohqzxm2uzmoCYa2o70azJ5/0RM1ud9oGxh1Ilgd6HaJ1W5vrWqtScOQKuwxZn2Gxq/Ezeyu"
    "pX42RK9qmrVtEst+Zm6v3w3L2Zo0+Hw1hUpRnnE76kdC6xmoa34aO6pyvxtevfzgn1648eRHP/yhYbNeLLtxWD/z8e+++/XfPX7w6u6lC6SRzc"
    "0LCvGencQQicHO5cwlSnZejTE657dXy6WXGMMwhM1mDMM6xA1tXOy76DwRhMR7571zQs75rd2dixd3F4t+tVj0y6X4frpzuKQvTd5CBrHzWfEr"
    "Lm0zmpQowyauLu1efRwxbG1tP/fCR3/jV/+XT+z8i993fyrC8ZQESDWHZ6LjFdtsk9zLMH2O+ctc2+G1PqLzMEnbOC0XpHoAgRqYcMvRhTWM1q"
    "8YcycuU2trnGYoeMTSrVbw6f1hO7Pi88Vv+Sx+KvZKBW8JZufU8ZZoYQ9ZE73ZuCOLOnfC4BWD7xTKg4YekVUXVX1adNv1MCstFxTSDPj82Bdm"
    "gMkmddhAJU1hX9jlmE87mpektWu20NIKImAQOa8Hz5794oPTwx/4rh95/MZjBwcH6XvtO/fCj//My//z397VKM5LMvJB62iHHbn0UynEE8Xp9U"
    "iq0BiYnQgjrhaRdxQxpK65E/ZOhMk733VuuVoutrYW2zt+sfCdF+fBnp0Hs7Aj59I6nHrIIs5Ntya2FwNx7mwY/TMfWSwXw+nJ+mz4ro99+Cx2"
    "T+g7H/G/8Y3wE0IDaqpxVVeyoeQZFzwqCMqcPZw5JdOr36Q62fqrqitpriM19t8mObc6wKeveKr82Eogbcxp20tv9DGwtOvZC3IuPqPwGWDM5V"
    "Xvld5eb3sMc4xDsQA3udgNcamSn+aHNcM8VMuoJmNvNpdbyzxtao7ad0Xb4KnGz5a0wCbXrSVnGP5NY4Ys2pb888j5Ka4tK4xeofmG6n0b3N/Y"
    "/OZCb28if++nvseJMHNU0qjD2fryU89c+dQffvDFf/HYjRuIo7AQJB8gmLCDzhEkanS+S/MGxOgEIuKEoxBBKYzMxLLgCBFy4pxzwvC+c8zdol"
    "vs7Pt+wUIgp+xT3yWhtGmaAALEvhPnav8DyiASB40ijuI4bF3ef/oFjAPAJ6dnH/7QC49dvfTgND5z8ZsHuPK2fkporbOIchPxYLoI1LyhzBWg"
    "24TbnVMtczmCHiWW4SbQyZhFjRYc1WQMm5NCbJdwm7AJ62t9RMJRKT/PiUnqZAIWb2R+JzMlq/5kGOE89ySr/JtM9DCpKQ3rYhpUYDbsBNfxJR"
    "fQaRFew9KXef5zG4lcA59E1RxUye8sZdhwGM1Ep4mrwLlmOmYWEbR30zma+Ly0oZGngiiAF1fClx4bv7IOsrVafuyjHztbb8AJqoYIxM362T/0"
    "E3z9IycPHzjf5cDspKwWEWbnJDcrxQmnX+Ak43ROfOcXW6uu7336r++6Zb9YrZZbW33fLZfL5Wq52N5e7l/olivnvfiOnefsTyQWR+nOmY5X77"
    "OstJrYhVySDbFzcnyy7p//dNd3UVVBEfHxx6+9+OJHHz64dxbcR+T39vh9pVW+Mde3gRtdBqbEqelebaZ9bBJpp+ENFxLLdKPPKkmaVSkmMdY0"
    "WmmCOJrZGnhmXeS5PP8RfDKLHeQ2Mdg2hqp7snXvzFAO1nxYR4tSBTUw0pZJeV2qQExWohowDKPa5HM5kPahoOw3kxOpUgUa4g/y0IXtTRjWmM"
    "8FJmR+JxsaaeVB1eQZGNG52SdM8FvGi9vS6RHKYqBxiyBnKhvlBAKwvIhXnzr7l5Hc8fHR40888cRTT27GQdiRsIJiRIzad+6Fn/7Zdb8/np3m"
    "A0qESTiddOIk9SqdYyfMIuLFiTjnut51nfOd8971C9f14nvXL53vfOf75apbLLvFYrG97buecza1MDFpgMbkZ2JiFiERFsciyUzPNBHv07uvKk"
    "7OHtzbXP7gzhPPIQzIOFTpe/+Zz376+ORoHCNT/Lj7lY5OskHcgkuMzholVdD2EWY5Z8ZjVPXfNkTVZO5ldRbq+EDsfRNzEXL9A2D8Fu072+BD"
    "idFQ24xQhOZ6bja3oPzOiyEFVEUxmnS0aSHJNJ6GiQCwhoa6+Ig0nXpTZZCmpzBEY0xesdIvYrQJFQqdhNlsRkpcnAnTz5oWbl50SSBfEya5bK"
    "r16MajwOucVYulEE2Im/qtTttK+t1SUFW1Y8TnUAtMTEJpulD8N0TEqrS8RK89v/n5GM7AcrY++9iL33Vhb09VxbETIU4hRxqGcefi5Wd+8s+f"
    "BmgM7LqkkXEiznkR4QS2KMhZYXFeXPpP13V9t1j6rnPeOe+FmVik6/1y5Vfbfrnlu4VznYjPn0CYuwV3C/F+eo4ikrjtCbzvaqkDKJSdC8cHD2"
    "n/wsc/JxSJJb1a3rtxGF988UVxfrM+G6Jsx5uflH8qNIB8Zegin4mYuuLGcKfn97akmVI07zRxIwhr5lIpLdN4AVDXG9ujmBtQMKbkaD4vUUl3"
    "Lq7oNTa5T+3xPq0JNlHTU0OjqjDRtDcfPdsgZqku+wI/q0CH6pCoA86298r2Hj4dp3zO62fSUutHerSwzjQ3y6AdPCPTTJsmz8uILEYxezEaeS"
    "9z+wOWIT631a9hH2KuL6z9XSkfUgig5UX9zofGX4jDSYRAg7D/2Mc+7lzKH8x2CiUKEVEpDuuLT3/w0uf+1NHJKZGy9+k8JCJx4pwT7/IhyZxO"
    "w2STd74T53236BbLrlv4zot3zjlmzo7e9Ad5L10vrqv4sPSNO0/i8ystQiJIoaUi00sv4jyG0zsH691P/+Ri2WHKB02e4c1meObpp29cv3F6ek"
    "aIGywu4o1PdP+KWUFiJSKc98qaIoSm/8GmMz+RsB8VJ2kTC+0grX1pTALm1LGtuz2aAhlV6kiP7ItOkFEYlSzKPbAeQ+cMbFPnkev2nHOU8gUN"
    "M5YZSOxxi5mPlmZjTesFsbPQul2l8gyN6ghVwEumYGEzH0erLXhEkDi34LZzvFLY4haTuwLmKTU/jh0zNnfGhjdpc2Bn8j5uqxMoLa7Er34k/g"
    "Li2RjhnEDRLxdPPf1MCKFeU5CtfBFRieJw9tiLn7nw2T96fHSoCkq3NSfsnPNenBcRcY5yfcMijsWRZDK++M51C98tnF843yXNqTifuNpZB80g"
    "ccSp4uV0gE/brWNx2bkuUhoNADCsb7793up7/1c7Vx+LY1Al1WJQpqDx4sX95194/vD4IF3gN1g+hle+2/8rxwD7xkZbDiSAzj37rNEHWRXKXF"
    "tDdlbWbsaG5N7e5OtXf04oZxt1doCQ7Uf1KGeqeT7Wj1Ghvs1ktBFRAw27pRFXFmtQfkJiIYwFdo+KtLCnB1slbIUcoYHeT2yD+b2qUOsJZgvK"
    "26EW20+dEaFie5nOSWNs+GCNOG7JlPZ449kBeK44n5vVmOc6muoYr3DEXB8tr4ff+8DwT6OOmzGFiiHE0PWL/f2LMShzPguFJ75ZpBihYB3W1z"
    "/1Y7uf+unTo6P0a25ae4niJD5P8/LkQFicd33vvHfei0s9USfOs7h0lyNmCCsAKFEaxAt5z67L23MqDVUBzQ8/7cxRNYwY1+++/vrie/7opec/"
    "pOMGxFAlStmP+QEtFt0zzz633qzTF++EzrS/hm9/yv8zTwPIM6lxdDZQZ0PnahhtiYLKDSWkea+MHJ/F6Li5jeIxeXn1k7UH6ZT9bqNdzStUFx"
    "IacA4UJsx7bvBHm/5QsUMoZTXZowjmTtgCLsiSe8iO39GMgJoxqDUjVfti41A0AjojmLWipIK4Adfd02hmp6QgJuuEpnYMb7kKFSXAM7cKM1kX"
    "yFz61MidmvrbbgNCLCAG+ieHX35684shIoQwhiFFSMQYu65fbm1R6rdwjoKn1INhBkgRQazD+vr3/tj+p/7I2clxkm4mi0GCuwix8559l34jpZ"
    "42KNnexXsRL87l27wqadQYNcacOM8iXSddz+LTcYrkn2Ji58q+k/xNGiPi8N5rr7oXf/zaxz+jm3XSkZZJTrpHpud844knY4iq6XrJQnoSukt4"
    "7Xv9Lyz4CNQXe1LNxmkJQ03oZ/HuGBUwV0cYWrQI0N6LHtWtbgBPzI/uahfWUKmXrS5uau+Bz5WH81K4IFObotIY11sGUQ38QjkJrX3BnCOltV"
    "goxOUMNCcIW6Hp+Zlhfr5AFdYY3p2lYVmlKfPMpz9d99BITDFL46CGGMIGOiTNDwUbvErm+lD5bKnfRnPiybSgFXCs+szmnz0+/MYYnWoMIahm"
    "Mamq9otlv+iZIUJu2sgltxUYoBihqgDpuH7sUz+69cmfPDo8BCmJVPx4+redY+cTSS13NVnYeen61CN1fS/Opw+Zr3++k24pvktrL38zWZ9NlP"
    "FNnJQ6udyM4zsvv0wf/tHHP/tjOpwhyegKJse4MlX1woWLIjwMQ1SdQgHCWez38Pan3T/e5nvghXDuhFGr2nwEAiZNIGDn7Wxmb+dbGuUeiHMr"
    "zJxwFTNS45TOXZZKB8Rq0yy8GWVgzbCRTwWSBKrRt5ihZyzQxiJQy398DeiAFaSDUmjd/K2u4WRotbmtMm2+rLIkv0V8ms5Igwq2qogpE8fKYt"
    "OgSWoEKk2Nas2t4eY6MP0AqLByhuELTbq6SX86G8W3N1Qgs6FAyx4Pnxt+fnd4ZY2eEZITVVgAiLAwe++nhyOlr8WtQlUD2Cux0LC+9j0/4rru"
    "8Ev/Ymtr5XxPMYiwEiMqE7FzeSjMMo3dwQQWFXJMEORynkXEdwnohJJjlyIQU50jtqeu6cjUMLz7ykvy4R996od+Gps1sTT3FLIqagKws7Xtfa"
    "8a0y1TEYSZKW607/nup+Tnv4GffIBnhc6UeBbI1PjuS1JZ+vuUC5WsECpy+cfcmF3RKEvayjf1+jPQmKegTADnbklFgozz4lQ2dVehCXMtoqvJ"
    "djrDQW26tG3uVcwNNxgymRuh69FnYYaMNp8Q7c2rugFR8hYLOY5BTdpHrbxrtQmYTcPcQwvBsDZuqoKmvNPMxhBtYnyqHatOcxuWIaoQsQw9a/"
    "GB851bCEWAQO5S/NbH1v/D7vCdAT1pXoGqpX9OifG52YyqVGfYKaOuwCEzSjD/Rozrxz7+uSs//OdPA4XNacITJn9vMjKk5cXiRFy59nPptYiw"
    "sPOd+J6cI/E60fSq4DzPwyXVromJqNDh7Pjdl77pPvwjz/zYz2DYoGW6n0cbgtAvFl2/SKcis3jfiThidoyI3uP4u+WfXuNvKvd1So2ZEwFGO0"
    "JVEmNTqostHMbyVkIlmeY0wNbCUXPsOYmRMCM1tjwNPof7Q/W2mRtKyV9pgiqN2hVtL6+iFmpGm5ZN3j8SsdF0PiYDuGGEF7MLaB5Ro9xUxFwl"
    "sgbbaIpXtJmr5wlyszQIc92eC+pt0of1LQLgRkVVe7oTSZG52a8wh1VSpaMuFrj/ZPjtq+GrwxgGCCFMM/vckgWUyCnROIwnp5vp6MZU/hW9SJ"
    "UbqiqTsHDYnF78wMe63Yu3fvPn4vGdbmsHMeapmyqFAO/FSYok4Zp0ziwCaKIwTTu8Ejk4yTSN6e8mmjI0mKEhhnE4Obr1xmuLT/zkUz/6JzBu"
    "poJN66bOJT/QKIg5TzIBqMbcjgUo2VbZOQ4fl1/aozvf0c8oesYwawTaRidTIxadJJ5kEpUtd6W4I4wXuOp+cc7ZwnPHfIOOLVmis2RqLmoAno"
    "2o2rZ8/qm1qJ5rJ4LaiMFaqZpuksyHLTYnNX9pRkPUhAuUWQRoBgdukMrcUmtahls97c5JkLhQEyfLGc04Zw1e9ZzMk9leTvn8yIOaZmnjUwO1"
    "7kiwg/TMci1++cXh710dvzREjsgme40Koqm1ETXGGCMRosaz9ToN55g514OlLTS9XelciooIJeK4Od2+fO3Jn/wP5PEPrw8PMjE0sX2gya3bqM"
    "OISByYWRwIIAELSFRchiMWKUc+CghM0BjDMK7XJw/uv/Od11bf+zNP/eifpHEDOmd6mC6PYiQMIERN7DmopucnIh4QqCgAjVFJ2X9Qfv/73D+6"
    "LG+B+wkaX2DPM5DlrKeBGu4Hk7Xe4tONmpmpSR1oue8tfrQJd4D9tIazyDURvHEsMj0CR4ESrMmFc1xJv8yYJ74Xw7GhrZn2D1cWDxuoubFHFA"
    "pPBThZHEQukLVtYXEbcmTMK7X7w0b3ySQNX8LYD2eT3HMHeU5+b5YeP2Ima2n1aC3VyI0r7ojQ6b29+MZVvLY/vqQkg/YxDiwZ42tKXzA7YWYm"
    "7z2gx8cnqlNvrWo7JkUvN+mJqiABMVMY+uXqiR/72fc//4sPv/Jri8VCuo5UqfOUYl9KFLEi6c5y88p3VGBHqQEDkCo7RnGBQjWGGEIYxtOD+7"
    "du3bvyI3/hxqc+h2GdD9KU0dui7aZujxA07egxqvMeNZSw8YtJkgeRnsZuC+99kn/+prz4mn5m4AtEg0Dzztnovrko+Wvmrd2gyTRBSl+hpDjU"
    "BIbW+Yb2/WZq8petGtQEQXOT9WupRI1ApA0qaaPAiVprrvUMV5FeasxUr22xHaDSyq1J1wQX5eZhTcoGNcFXXNPKyDpq6ZxDqgm7MeyI8mxLGT"
    "od69b0OaE0qp2az59258T45qQGGqRp/icC9kTa6YPt4c398Mqevr2kExBt0DEBFIkohliMnEj8T5H0GIXIOTeG+ODBwTjGvmN7A2brDGODvkmF"
    "KQk71hhE5PEf+GPd3uWbv/oPu7OT5e4+YiD2kCTHyVhlQFUh4khkKsMFAMWYQwhFiqyQgBjHMI7jenNw+/2HZ/rET/+Hj33sk3G9zp9FTawW7B"
    "CB8i9P22aMse+7tOCiRieeCCl7W2SyBSuYdERHRE/Sl/fxxk35zG1+doN9gjKC6fWw5dMKSe2nm2uVBc2idNym7IZp9MctvHQ+zGCw5Y/xnLzX"
    "dBi5Sc5tV0f15pj49hqAwk1CWxFo81wy4q3L6By2vuiTW11JS1xu0KyzY4mLy8MiKbgtG2Gwa5VC3yQp1haM9fYbmQvPFcHV9De3R5SwTy0/33"
    "R3F2JHBK+HO+Nb++HbO+HtLj7wjokXAzpAhWOcmn0iTqtOQJiUmRWaOdikGje3b98+3YyLblFxSInCRU2KlzESE0BR4ZgBxebssY//oW734uu/"
    "+PfGh/f3Ll5iqORGHREp5WhRABoVQszekQiYqNjyAQojM0NEYxjOzjZn67tvv7HuLjz3b//li08/p+s1iRgfEepxWufQTQ56iFivB0rhbblzn6"
    "DgaVYhIEVO/GYiFZKRVwscvoBffhIXbuHZe/5jB3oNYMLApMS+7uSUg2Ysd6SJDsMEPbf0TUKlLtldlbgBuMznTTOiNM/RozX1hFqwQ0NFaahS"
    "U5Zsdb3O3kO2EQvw5wu1dsw3TdNmVfXMz8GPUJrBis1KPdxwW6vpkc/xQrgcD/M83xY6X3y9zBZ+Yxs7ZmfMv1MzgSJdkATcEcjhcGt4e398ZT"
    "+8ttCHCo0qKv2onLSTIERk2lq6BLatIjCLsJvgFKIa7927d7YeLuz2YrQYKB+L6wx0suHk10WZCCzM8ez08rMfWf7sf/rKP/27x3e+s33pcgfv"
    "XEfCIIohEOBENEbnhV0Xo7IOzCxM5HzUyMzsPDPpZr05W2/OTm+9+WbYe/Kjf/Yv71y6opt10rhN2QcoARPGAN/gyJh4M8ST07PNer2ztTWpSb"
    "SOwTSyk9RxAKmTZCCGum5Q8jh8hr94ffjacffcbXnxrj4ZaJsQGaOFEFKTHJgyxasNsXXmT0lAjZu03vGbNcaG0s0l0rTinDAzS5mdyGYXmugp"
    "nlWgXGHu4NmRY1sNXOaE51g1U0Vu4QENN6IlWj9Kl9e49wpBFxaMZRKoikvYIoOmY4FtMlmDxONCkKkYfy7GNZaaglz8VMZ7rUQC6oiE9WRHX9"
    "8fXtqLr3fxviNVcoF80MhEkuozO1EEUaKdZbRfylSJkxEB0CC+FxEmOrh/9/R0UM0OPvNYi7FaS0ehsl8xLXBhYQmb9dbexY//u//RG//65x5+"
    "87f3LlygHq7zPBlmYgwERWD1MQu+xVHXcYyUE9HGGMK4Plufnt1792158hMf/1P/28XWSsPA4spznLlZDVa7FoXpaY9jOD052QxrmfTlMI2Res"
    "QLT8psSSIeJlKWDS2J4s76W5f610/p8h3+4B154ZiuEJgxcjOlmx0eTfp7vaCg7S5UXls+NI1lbXI0cQtaZMwVxTOOzDy5sIXIGJCYAQeUME17"
    "36nG2PTPfU1MIGMFaWAeJvu4ZXvM7mbzgnASybPlidbkRJ4jrAjmKKu5oQ3sxppS8sNnG8eBUsejEQ9ICk5hJSIlR7IiHbbjWxfCyxfjy124TR"
    "oiuUguwBFUocLsnNM8gi9mS1UQpeOFOarWm0PxdUruUTvhg4f3Do7PxhhFXKaZGaMVjAObTTmg0307jwVAcRzZued+6i+8s7X74Cu/vLPv2XnG"
    "wBmUne0X6UhidpB0ejNCAFEM47g+G9bru++/v/2Rz33oj/373rPGkM7A82j6SSCI5uCog1caQzw6OtIweOdUlYicuPy6CSMmwSmcd6keU1UNcZ"
    "olcW6addujYkG3n5P3n6Lfv4tnb/NHH7inI61IR6Y4F0vUH4KN+QIVEYW282fHbM2pYF0xtTYjbh1rTX43z3I/c509TeCsHIFnyYLNqf0ICqE3"
    "JMXUsjPA+lRii2ERzKE8NWGXmmzx/KfBpmpa7ES91iHJIycfGKOK9lArYxjTiXFlgW0x3mQ2tV76JDsS0IIQe9y/OL56afz2VnzX02ZUGVUUXR"
    "IQA4gxikiWTecRnZp5K9Kbr1oSAXSycmY1StrunHfHhw8e3D9cr68svEv5TDXgtlxvZ9k/1TpmUZRMGqHhqR/5E/3+5du/98+2oN1ylX5kQnTO"
    "sfMJ3wYWAmL6JFFDGMf1sD4+PHh4cOX7f+a5H/uTjFEVLK72HNh2/I2jzmy+ZUONqpsh3rt7l5mdczEG73zNZ9Cct0bEqpEoN2nZJWi3nYQrEU"
    "VyIUon4Tp/6xpeOdYbt+SF2/LCWveJiCk1b6SWQhYwbYE0VMd5XCmJPCULcaEWzZ1QebSA8yyUmnfURKmj0mQKsxNNlTYVZ9xstLbJa4Zkngwr"
    "xTgp8QiccsNSLFE1aKO2W3u7PVepGkPsYB1NHh0/yn5SznVYW6Fww+22RhbDiwQTlDy48zi9oC9dii/vjq/7eBiUIvkBPWWLAEGhmjuNzjvLp8"
    "1dw+n+CaXauk2YXJFsU4I6lhiixuicOzs7unv37nrzzM4WpkFbvqVIMlfUIp+LLWfWbEpfqoCZWcf1te/+wW7v0u3f/Ee0PvXLLWYS6abLmgO7"
    "LMgOQRHDMI6bYX18eHw6XPvxf/+Z7/8RHjfZzQHYk6a5q8CEFE0/oU6+iBj1+HTz3ns3F4tFKoaZiaWfDA7MmKr0CBHjQCn0h3yjBhN75zTGMU"
    "a4BTPv4P2d8M7j9PsP+flb/MIDehK8JBo5nzzWlQZDnW9owQBX0ljb0jznaCP72GFaejO2U2s0rbNBQ4gmE30x52PUCJcZQwPwTTYgnWc6wHoA"
    "5yY/NjxDzA+k9oSazI0tY5Dn8a7NDdfezkv6Vptx1Vy1y2mZHpLmW1/cwu2r8dVL4duLeDuGcYSsgzABGMtlQydFG6DOdSySdh5xQjGpyijVmQ"
    "xbMDMKi5wJqZUSg2rcWm157zfD6c2b75yuPx7jwnspEIIpHiw9j0p7nGBzLbjHNNmIWYezSx/42Grvwtu/8g/Cw/dXexdIJMnHiEEhEAsQo2qM"
    "MQzh+P79DS+f/RP/4WMf/SQNZ5QpTzWY0vr17PerNo6uQnSgirt3H7715mur5YqYWVxUpTB0XUfgGAIzi3OcBvpNNlOJVSRhjqm2VDhxAGIMzE"
    "LOkyw6XT8WvnxV/uBYbtyiF+7KCwNfAJQpsJVZVbwFzgs2KmIq12Uox5S1saMBedr2PSpoqbGREs/o9phfyXg2RWtvbbD3ydwd/f8rACiry4Jz"
    "eS4Zs0AtqsMG0z3i9mrY3Pct+LA2BriqG0rJXEFr5irDxhXIpGCFEHFPZ/v06pXx67vhTQ7Hg8opPFGCCKY0FMmuulxMQFWXyyWxRNXeORbWqL"
    "D7TBK+QKcO2dRTYkka9RhHcb7zLmlGoOP9u++fnG4UO9OuqlVVz43BkY2XhlvVeLW7gZhFN2erS489+0f/g1tf+Fcnr39jsVz4ridQjh4kVeg4"
    "DMPZ+uTwcOfZ73r+R//0ztXrNJ6xd410y6gBDXjyUVNWwMo83nzr7Xt3bl3a34VqRkIwaQgsUmHKIGFW1cy1mU4uVU09AZmmGpnTA4QwxKjiRM"
    "RFXhCwj7cv6OsfoC/eoQ/elQ/cw3XwgihI4eyhVV3X7jvO4YcLYaPquc9LrppNyc6zC2u34HRrP3C6GFadDWamouZMa04S9rOTp8nFsTauitrA"
    "fBbIZJ3+aP/eXMuWTtQjWHU4N9ZogYW1dgXwCBxBufYpdUQsdLpH71zR71yIry3ivRCGQSVEESZghNFD5oAU1VRtQmPX9QARohArQBEJGVFkX0"
    "m/VZV5JluKiIHIk3G3aPreeuOlg4OjEC6odtnkwbmykjIxtE+1VTdVeiDX4GUS1jC6xerJH/1TD5544fbv/dJwcq9bbuXw66hRsT4+OBv52vf/"
    "8ae+/8e9dxSHbDs0LXwbm1NIylNSbWNwKzWGAuuNfvvbLw3rE76wR6TOyYTVlBgjAHEJe1PkcunykAWt4iQXkAoQRUCIRVx6P0NUACRw3hMooC"
    "N0GB/eoM9fl68e8bVb7mO38cGRttKPYzJAYfm81gZu7AM8U3ujduypTaSvnaBKdUKT4N125atv0dj/sgMEVmgj87Rbb3MaG0SVLfOMOoCb7lFp"
    "nYBsRHZbeNagGCtYIZqpFBuEZ7qzKD+Ki9XITEEMdkSesdnVNy/pqxfx5krvxnEzRJyJV/UpnD1qPq9UibJbPD3bVIpS1y1EOMbgnE+JYFFVRD"
    "rfUQwhRmGZykmUSZUwEYlCNY6FzgQosVONTvztm2++d/PmR194cqUQpZS8mQ9tIrH9D5NTYjKKcjYu2SC3Ih4Zh6sf/e7lpevv/PrPH7/3iltt"
    "x2EznJycnp1tP/vxj/7Yn9x+7AmMA4PZS7WqY35hmJubTVxLGz0IIr57//g7r77khMU5pHm9cFLNihONMYxD1/ep0HDea1QSJC+ykubqe4IgEl"
    "GEchqycnrwGENAumkTnAi0G0gQ4668s0/vPE1fuCMfus0vHNJjBE88NpYJc5EmrkknakYt1fLO5uBnMlYkNMstdy3Nn2h1rlPDYkZKqaHOjeh6"
    "WoITjsdXEXMr0Tb5tQUyA1iB1zwZE+2as6HzbMYf504xs2vAtlphW3OTZQmF3gYlR9QR6TLeuaivXgqv7Oj7giGAzyITO9WAOGhhh7DEGNnmAa"
    "R1QKIau65z3scQmUShiJzTWohUNcQ4tSxTuZXq2JhM8qB8n0mLnEuXGeRct9kcvfryN37wBz6tGpFisNJwECSTNKQgozQ3UKdmWzsbyhKmCokh"
    "57vjBweydeGZn/r3H779KjMPm83m8OCy909/6nOuk/XJsfddmoyeF1XAFmMoRmecwwUia29UY8S7N2/fu33Td52IqMZULKTugKbGMtG4WTvfcR"
    "YVESLAJOK8kxCDaomfzlRqhVJUYWHJXH1VVcB7X2ycJDygY+VODp/Wzz9BXzmQJ96Xj96l54KuiCLTpFDPPWriRnpq69epMKzG0TZyyBDvUZ8c"
    "T+5EngK8yKpBJ3hmPoPKpXUmI2urVPapb24gZm14CtmlNJW8c2d0EwqOquXMP1daBuZvqX9NCX+kNgUeLVQ8jz6zbIVAnhgLHOzH1y/GV7bDOx"
    "2OI2QDrxCoMqtGnVw2Osk4FBNvaFqZICBq9F3nnIOmyooUmqC76XHHdHSy1OAtAhMlQK8ijuPofUfEUYNMFw/J1gNikW9+7YsPD/70xb0rXPcz"
    "IAeZch7ZTNZONdoQM0SlyhsClOC9APzyS2/9yr/+7Rs3rn3P935i/8YLw2ZYgnaf6scQv/2tl5988tq1axePjzeq0TlHVSRMTVYqDFJwimvNNz"
    "U0YW9MNIz6ztvvHh0+cIlAlTECcN7FGIubkl03jGPnIc5DE+oSsdDBUPwpuX+dXQHCKcgmqibjZgwxkE54nkRvJFXZkBPGBX39qnvziK7eko/c"
    "xgdO9CIRE4Uazln0pbbGAFfoCprxd7rkS5ULGVBjEkzbeeBU/KGqUalEuKNhRZimM7U53Uy+GqjJWD9RXQs8laBqDtPmGtOe56a3nfvUeSimVd"
    "9t5OTtVWjGx641vYJZqSNCT8d7+p0L4eW9+LYb7wMUIWt4YoohdTs5xtRtKffv/BQcc9RIdU4AACLe+y6NGTTN/lKmkkJYQBQ1csLHF/mrgpmE"
    "JcYQYnAp5ZOSod6pqkzVFhMvl6u33njllVdeee7p65V0psiWCbK9qOr14TYnOj3UpJgbQ9zb3xpHvPLymw8P1h978aPjuLl778G9ew/u33/w8O"
    "HDk9P12enZSy+9dHhw/yd+4kf/xJ/849vbq2Gzds6dSxSsmV7WLWq93WRzDICzTXjzrbfGYZN2CgUciwiHEBP9NJVsCmVxYwgecL4j5fLZYcbd"
    "sFJ6UFRN3RrnOmBURdSQEDYAqUZxLt17WURBii6Alnznebn9rHz5vn/+Pfrw/XA1YEkAUZCyHiZhaWNVs9tbLf2mCRSDcM6E04rCp/d4muzPr/"
    "P2smplbw2kxWc9iZ0NmmL9ER6MNu3Mzhkws4RxUZa18JomjsdciJoPmYq/VLV1IHF0toc3ruDlvfCmD/dUx6gyUEeqRAqChozCgqalJVHzIKKM"
    "LTXPGjB9SHLOL/oeyIag8uJH1SyI4dyJwQTuTW+AczKOY4zRiZs+mBKEZJpWkzjniLnr+s1m/Y2vfvXHf+yHGEj3nNLFUiJSDLmjYAAAgkZJRE"
    "FUhpCkWAK2Zs0m7SbBYBS4cGHr13/td//JP/kXh4eHqmEc1sfHx3fu3D4+OR2GIcbYdWlPcYeHh//sF/7hP/y5n/vr/+Vf+8THP7xeD43gKQsK"
    "0cyu5vCJWQ8NR8dnb7/5RoJ5ZxgqMxFnJR/n4Wdaoq7rYgg6bJzvmKS8q6kCV4CYkzhJQc5L2jmd9wQ48SI0DBtWFSfE7MSBKIQgIkmynjdZ56"
    "I6iqdX3Veuum8Ni+t36fn3xhsP9aLSkoiIAkFLNtpUeJsWjc3zmfsRLa3CAOIqi7rGLTxCRmYFpdx6ZlGd9WwxjK2ElR7FeOA27qaBMcF2UK2A"
    "DDPwE8902uZWmuKcWdkTOaZxR29eotcvjK+s4i2NmwA5UyYSQiSK6W1VAEhHWUyvQoixQDgKb46LG4dIgb7rl8tlGMec/oaYdB7ihJRmHIHk2W"
    "GNCbM7DoMSee81C5dJ2KtqyoqvTD8okXTef+NrX7xz9/7eM1fzUpf89akpRcslzIZHpbkkc9aL7+6u/j9//x/9l3/jbx4cPvTeE6hLtTQAwHnX"
    "eUcanfedl6uXLz75+I133nrtP/8//Gf/u7/0l/6df/ePrzejRs1s71ZoAWtmBYx6uXqKAL577+GdW++mbSIV7XkvYRFHCkWcxENCquqcU1XVOE"
    "0l0hfhkgiJhaCsqiKCqGk5xxCY2Hdu2GyASCQhjsLCvgOBRRKVqpxkGtMtBEMUJ6Eb337Cv/uUX57wlTv6xK3w9DH2A68AJgoClJrv3B4DtrCF"
    "+ezPiLYqHjt1zOeZ7uc9+OcKEC7oJMfP/GylIlo+U7MvlhyA5mikc36Kuc2W53GJJi204kBLHhsAIgfqiLGtt67THzwdf+vG5tcuhFeweTAEBE"
    "1u3aAak6dbNV38qOkMZ7q0aEkwLs3yybUXNaxPj8I4eN85L4ZJLtNPkgxKMiHcU0EoBAzjBkTe+0wnIDQ6oVSWSVbIiDjVcHjw4MMf/cQHP/hc"
    "bmFk7UhW7TXkYTZjnzSPkPyabq0Wf/Ov/63/6v/+t5erRd+nWNDOe9/gPJDQhE41m253d3bGcf0vf+mX3n/v7g/+4Oe6zqmqyKPRuFykWCWrI2"
    "/3otAI+vzv/8Fv/fqvjONGVbe3dhPKMfe+pBoR0leSHPapjxXH4L1zOTyYnHMEhBhYOF0Ekj0M+UIxPnxw5+HDe853zvnpuhTTHqVTrVZoealE"
    "ybFW4DHSGIZeDy/ROzfkO9f49QUeQHigLaUFwMSaaTuYsdtmJ1e1trCVsZ8DzNM5QlR77pxP5Z4cT0yOn/3zxZoh53JIi9bI2OArEt8MBwsp+J"
    "yVokjluIKJ2RJ5s63dETvmcYfuPIZvPKO/97T+7t7wkgx3g2IIHCOIFKqxMFeIY062MPp3zmtIpnu2CUsDgdN5ktbS2frk4cO7Z6eHRFgsljJ5"
    "6tIFECBJI/g8dSARiTEO4yAi4nxF5DC3mYyJzykAnPPOuRjjsFnv7V/4zGe+r/M8OQ3qsMEEFFm+P+eRoIKYtrdWf/2/+Bv//f/7v9/Z2w0h5L"
    "c8nQMTtTmZhpyT1EPtvBdx4zgAWCy63/7t33n529/5wz/xY4tFn/SxxPwoeZMJOWiMhTjd4Bf/xa+8/O2vhRAAbG1vK9C8CUyY1A1iMrcYRMIh"
    "xjR3ImbVSJyHtNNrTt55jeHw4N69e7dVsbt7oesW6fmLSJJwxClZNamKdKJ6VkGyarqsRrhRRYGezy7wzRvy6jX3zkoOiWmghdISqNJiY+GbGV"
    "XPnSyzIAZjOSYjwuUWoMIzmUotWuH4mT8/E+1Wn2JJxUF17HKxQPOjfrSZVZma6LOyJchkvwN3oE4o7NG7N/Qrz+jvPRE+fzG+6sZ7Q4ibQFGh"
    "SfuUPAG58oTxaNa2As5fYLOPLccw1Lsfk7D0/bLrliEMx4cPjo8fatQU7ZDboZM4uJxvSY8m4lg47b01YoOmpIfpgSQGmogwS4xRNQ7r4dOf+b"
    "4rVy4Iw+JMLI9ZeEamZAKp6v7+9t/4L/7mf/v//G/39/fHMBKROG865yJMkpQmziUdufc+/VHe+xhjjHFnd/ub3/qDr3/tmz/5R36yX3QxarN9"
    "lJEvW2AQW4vQ+7ePfv4f/5MH92+nhb1abROQQP1MdgiMVCUmBFy+K2YaFWJWw6duqKhqusOrhuPjBw8e3BnGcWt7d3fnQtcv0txQxDFzQ/SqaG"
    "0Q5wXPTMIcFUUZz4DGGKIGOGa3pKPL8s4T/rXr7u0tOWJJq3GRbApy3iLbyOEekebLDb2YH+nooxkTtGKv8z8qi5CaXIiGHDEzED7KQcjN3Wuu"
    "ZuFS6XEyE4E9sWcad/S9x+JXnom/8fj4+Z3wuhsfBo3rwEOIGgMQKad0qUXuc5mY6ETHJSk7V8pfyd+WTKQZTtJKFhPklnJT+n61WG7FGA8P7x"
    "0d3FNF3y297xURqmlUodAYx6iRU1slFU6KNAfLqgJUQoMIJ8sEgZy4MYYYhtPT0yee/sAnPvFd0JCvedkzyFMXuolZTv8bgl6+vPv/+Dv/zd/6"
    "W/+3ixcvjjHk4xbV1DHNkkVcZgt774jJOZe+Ledd1DiMYf/C/pd+//df+tYrP/1Hf9r5amswt5rpK69eazBYCQB/41uv/8tf/OfQuNmcMvFyta"
    "VQl1IuUMPwSqlFNm0rtYtFVDVpvr3zaRIYNZydHj18cGcYh63t/e2dC953OWpKnPedaqzYN6r1x0SgTUacpNurEPYkiEOivxE06hApwMWIno6v"
    "yM0n/OvX3Vu77gERRu6DptXITGpKT64sMramHVvAssXYM9uucmEJlWPXHpScTsI/l+t/GFvhlPlV612zAicIscnQJZiox8acz00zzoE6Imzr+9"
    "fGLzw9/tqN8Xf2wysSHg5jHCONUUOIqqF2AohVJ85pxaDWz8FcSOGQ3FvkSjc1Jh0QCTGgzJKU/iyT9cD5xWJrudwC6fHxw7PTI0Lsut65LrUw"
    "Yxwx9c5mVgBOl/P6HyrN+HQuOvEErM9ORUSVfuiHfrj3Ew2JLX7BaNDzyUHjGC5f3vsn//if/Sd/+T/e29u1zLzpFIBzPr0AOesM8F0nTjrvmZ"
    "13ojGO4wjEMMbTs1Nx/Bu/8eu3b93/qZ/+KeYm9XmW6seG/QVohPzar3/+S1/4Hef44OCBOLdabUkWCSXxkBAj8a9YSsmFrHdVzVUokbDE1JAi"
    "HB09fPjwXhw3y9X2arXnfZ+Kz4xctV11YSDdZosujXNIRoJqVGg/SgVUpFWauwgR0Kg8qsSonR5doJs33OvX3Vu7/kgcBvSRFmncL1N7wfQT28"
    "QI6wGDZfzNFV4lAolpDqh3/PTPNrwXW4gSNfcdcIVAcn6Dm85Mk55iKcOpF+cX9OCx+PWnw28+Mfzm3via0+MIHiIrJMaE/MubV+mh6FR6zqg8"
    "5T1nzsFSdckJG3QTuDq1ZKpbpqhMIkDrVyiuX2wtt3aE5fDoweHBgxjGrsvBYKkV78RlmEoRVrRpPHVanc81SQXAycnRamvr6PDo05/+zI3Hr0"
    "2lIBUPU908y/xVdWd79a1vvfzv/YW/MI4b3/lU4E09TJec9ylQyXvvnIDI+y5pfbz3fd9t1meqMYRxs9mM43B6crLZbKLG3/2933r+uY9+5rOf"
    "HMfRyLksFb5cXrJs6PQs/vwv/PM3X3/ZeXd0dLDoF8vlMm9AIhMCrr6IKeAn86U0lr093b2ZBaoHB/cOjx7u7Fzc2d0X1znnATiRJD9q1Av5CE"
    "h41fyfDP+2sSup1qotRIaJs7AG+dRIHwMGuBDh4skFvPtE99Y1/s6e3GOhEcuAFUhSuI3xobeAisIDKTwWA9ptA7bthA8lmcEbUGxeZzAB19Tq"
    "cmDxDBO8HGzGnWSj/ZjQdXLqaSPM18cvXhz+wOlhjDrCpcCQVHOCWGM04WZTo6jiw0xTGQULIopY2lZ5665WoOwsKsT7lhGcb/VZcpXBgUntKZ"
    "3v+351fPhgM6yHcej9Iu8805WmlTZXDDqSqx1IiRMZ7KpaKsyT48OvfPnL3/09LzINhbI4eexhhBXEgAiz0P/4d//Hm++9u7+/N46h8xxDhFMn"
    "niQJUZM3vqBA8r1GWBS6Xq9BNIYwDOMwDOvNJoRwenr63Z/41I/8yB++d/vO7/32V77vB7777GxTAuUx79IU9ZXcuXf/rTde7/s+TWKc6/KIhY"
    "igLs8/UwuGmFlVUBCjeQRQqDxIwolFv+wXSxGfu2WEJFrKpieCZFh4FNflM8h1haGavAFCpKkpBSg0k6cyCQtcFfY5KUqhQundCERCQZUoEK9F"
    "fNz0frjO959031p3l2/Fp2/FZ+/pNaIOpIQolqkLK620M4zWGvxvQv9Ng2nHz/w5NIntRaXZ9OmAKcc29zbs2VfG4SasnsRzfIa/8Lz+xrXw1f"
    "3jz++F18YwDIFjxvoq0peJUrtzFpYR15Bk2IylPDfXtCPWwB0uQETjMgHb5jJz6tpxe8qny1v+RFMlEVXFyc7O/t7ehUkOyqXpwm0kQ85Xz/8j"
    "BWacrp9OXPrdZ2cnzOKcG4f4Qz/0Q8tF19QQYq4R0/1qtVz8wj/+53fef3+5tXz99ddWy6WmwI2sjFLOA5V88Lo8OId3npkQo8YQwhjGYbPZDG"
    "EAcHR88j2f/N6/+lf+L5cuXjo5OX71ldcee+zG1asXEw+K2IAHTd2lqsTyla9+65f+2S84JyGOR8eHq+XWhFrjLNMjFMFtmX3nvUFctkJPub0s"
    "AlUica5jkpyrouCCSyRw7sMlPHn1nbk0rJ9UJ1NdxOIkGy9z5E6ZfSLtEVP5P1Wt2RCTwF+5mR7BEbIZ4fX0Er/zpH/tor8/8mrBa2Uf0DNFbo"
    "Nym252m8nELcJtdpEsjZmf5eqctTiZCrJGfjPyaw2ui41nfvq8JqXj+CL+xZX1F8bhlPTs9PhhJAeCxoicQpJhRkjizHz4plu21PM9i9YSZtop"
    "YJAQ1WYw7bGcpPfM5RicCBtTPVluY9OCEqsRi6rE1HVd13VF/JlaOKU7PE3XM2hUWJRy4E3RNKXQee986i2IyNnpCQjL5fLo+OjF7/rk008/oT"
    "HMYiwTQSoVYNtby1dffvW3fv13wjA+fuPxcRxv3b4lTkIISb8m7LquE+dSB9KnkG2mruucCJGGGGMIMYZhGM7WZ8M4hnF48omn/9pf+7/u7u2N"
    "47hcrgB+5533nnzq8d3d7Rjj9GlbmhJYoSHyv/ylX/nylz7fdV0cw+nZyXKxck7EeZG0JyLd41Sj5N5Y/qpzXWNvTjUHl4mIncwm3VMdpOWena"
    "3zMTlroNOWCrbgWEaTwQZO8Jupy63QfBmRXDOmuX9WO9T4C8QYlTjAjyHu8b2reOVJ9+1nl++o+Afhcr5Lp/KHLc5y6nPncyRH9tYgGK6ktrJ+"
    "HD/zs0WazVyEHHYf5HPka7bLoPET5x/FfUj/5e76Gychz5HHEDWO+bESEXOMms40oIjkc8NLi1p1sswWDFltPVc2OKtqtbXXDhLybUw44TGpTS"
    "Ip/ZCyAlWVhTvfed+lDXyCqoHZ1fhnIM2XJz2pGB06TWOJXICmeYZ3fr0+A7BcrmLU3b39z372M6qh9nkmd0T60fvOr09PX/72q5cuX15ubfuu"
    "/+DzL7z//nsPHt5P84YEd2Ei8b6Ew3jvnfh0JCBr13UYhs04hDGIMGL8K3/l//zZ7/sMM+3u7S9X212/CFHv33/w9NNP+M6hpEhbcT6RKk7Ohp"
    "/7n//RrfdvOufGcTg9PV2ttvquTwb90kZRwDmXRrpMHGPUjHLTtI3mbk0yhU0SSuc9lJyTPHSoblI456qYmhlWwDJd7GUqc0rjqna6RNL616pn"
    "4IkX1eaRTZVjWt7pAaajJ8IdnxwPIfS8eXrxNjl3NzzOsw4h2XgkbkiHZcnRufVDRAzHz/65dshg0gbquLa2SqvGZepDmpZMWgrLS3jl6eFXz4"
    "JAR82TUw1hdOKQFQ+pqZibMJPZq4jFC42Fc88zHWhJ3ASzgNKHE85RaSDmzEoQllmPKP/0uYASSzdQEDPEeeecMKdFLVLTnuo+V3IlRNLvS8Px"
    "An4TkagxrdtpugXn3DiO4zj0i0XfL05PTz/3uR/Y2tqKmltE6TtKSA3HLII3XnvL++7ipf0LFy8sFivfLZ948uk33nj95ORYxKenKlPJkGCnaQ"
    "/1zjFR1ECEYRzO1uvNZi1ONuv1X/yL/5t/+8/82RjD9s7OYrHwfe+8851/+ODw9Gz95BPXUSlJJYaMkirv7bdv/9w/+DmNIxGt12eb9dlqte1E"
    "nIjzrpTj09eqUs+raXwKKhyJ9PSM7y7tyCqTjDkNDyfxhZmgmwrJjFhbwRdzbRymzlweDmcRRVG5JHJQ1Z8wZQV/0hxQCnJVEj49PQUI5IdIT/"
    "S3T2j/MD7GHLmqwxsWKD+i7uQZr3BaUhCqTdf2X+UammmzW4yqfjIvV2Bf+sHjjfjlEEOMIcagqjFGFo6KMSowjRzyriuUEElJvZEeEKegLALy"
    "0K/OZKZwOYUmjTVXO15B9GWRNKoZgEFqpgeFdZvvDN6R876S0BguIwNJoSXDdfIBcwa6VKstmF26Amlt2xRkoaR3OsYQQ3BObr779le+9FXfLV"
    "STjY6UUOCDfS+3bt4eNmFvf3dvd+fKlYvXH792+eqVx5989mf++J/Z2toVEXESwpjEA6qRSJNcRoSJoRqhGkPYbDYhjkw8bIYf/MEf+Uv/0X8i"
    "TlZbW/2iWyz7nZ3V5SsXr11/7Olnn7p758HLL7/hfNc4YoGpQHAvvfTK/Xt3vfdTzrUNm+OiQ50WD026P5qkpbXmTm82EyuiIio0ZWGWP8ix9F"
    "3vvEsdVMohi1mXy+KKJBBEihzbq9MyS+FgKKen8NQQys2e6mUpuXFZAokJ6Bg0N94BokSsE2Egxqinm+GD/Ls9HwEGJtsmDE55b2jid6kRF5V/"
    "JGSvd/YX26lhTbNu1CkNVo0JxH5Ft7fHtzYh4aoZQIiRSGLUmNKLph3OqMMmZxbnyDm1zt6p5Mj9/umjZGkIl/SmfB9PrwEbrQ+m6OkyXCOmGF"
    "NPaDLvaeTpkp4yoZMhxzmfjHVJDEqTTYBNMqDmbDCZjl9mEWJW1Ew7EaeKqBrCMI7jb/3Wb8UwFmaJTAGEi96fnpwen5zs7u8tl8t+sVgsl/t7"
    "+489dvXK1csvfuJ7/sK/9x+uVlvCQixjGMM4JnEnTyeOE2ERBcYwjuOYWl7Xr934q3/1/7S3t7Pou+Wy99577xZ9t721unz5wuOPX3vmuafefu"
    "vm+zdvee8rdCcdBaox0le+9JUQNpPZNlK+l5J4SQUNT87kmugBcuLE+ekugMI4yS9B/lbTfC9NF9LfiTTKTVfLHM8hnqbRT7KYTWPw2g9JrQYA"
    "KYWAin41lyzJhZL/HaN0LBpSnoAfXCO5WZIxLh2nwhiVV7j/pLxCcHl/MfHOPK12rqcjzuVd2PYN+6LnnFvdDW7Z5OVQa+udTFK12pXd+C7Fkz"
    "Gy5MiU/C867+j/19a7/NqWXed9Y4w519r7nHPvraJo0rIsA06kQJLjGEkcxO8kSGAYacS9JG4YgYM0giR/TnpppBV3EjiNBDFgkJREqsiiiiyx"
    "yCLrRRarimQ9bl3e9z1n773mHGOkMR5z7lOUDVkWq+49Z++15hyP7/t9U2WfYr+CqKAl7SGIKjKdVYAILoxCa5yIHLoKqUoBJLX5Hrjg0/5wyR"
    "zs4HxhJF1QsWELBbOwzOtym9DY1qHUYtNcRB/i2UMgIl6bqzj+KfyTqlJKHVRftNZfQLW1tqy7N99885OPP/ntv/I7W2+FEBU7S63EvT95/OzO"
    "nbvLUr3zBLy4LIhYSymF/vE/+S8bt//9f/tf9/vd8Xgk6qtzksiqeRYRZh0PpKrI//Q//y9/82/9zadPn+72q6lqABQK1KUgoshyebkrhG+/9f"
    "7F5cW9l+703u2bYhZC+vzXj9966639fjU0gX2ORCUNwTYDU9M/qA6QjDm/TB4Uez+3HYpmwmJYpU0/j8JuZxnuS2F3OIgi2XwlQNqUBVDsbFXM"
    "bma37CyIpkihI3OZWr+DtlxRu5+9V1Fh950Bc/cgV0JVKISs9Nv00w/xbwiU+MZvXYOB4RaFL4TA3IqeIMBz2duZeFvVRxG3Vh+pHwhE8RRMeg"
    "UPuLPtG3IcLCqFqgUnwMCl2MyT5oZWZHicY0Y63MMYcy0cafRKRKKSfSS4gSizHTWWB4RIEi6d0LkBEVnzZv+PtZTnaCn0tHfQZIbNLlii6jd8"
    "5vTCEKzaT1VrNWeD/TgPHnz+6ne/V9dF3LioRFgJnj55uizr/uKi1mVd13VZa63rslxc7K7uXHzpSy+31v+rf/pf/+f/+J+201Zr7a2dTlvnri"
    "LMUsiaQ1TV1puCbqft7/6dv/vf/rP/5nBzs1vWqY2035uo0LLU/W79nb/y1ZdffunNH73bWrO9v802l3X9yU/evv/Zx8uymnfZuR5EoNAjmzHv"
    "BH/m3VbtwQAR1CpeZHj9Ge5PUzKBggqIAzK8LgllKaBh3YCFxb9fmfLSUEKXY9+3q3ZM/BatR6wtx1WECAIibkjCzOFwmK0qAPTe3QsSshuGeg"
    "mP7sBj0IK3ZKM4eAQuFZl0TnNKNI6XcFq33VIVT7wJnZh3ihNs7JzaRAB95QcsqMIKilTMbWcWBFUROCNRWqOFiGKrUJFznuMUjgOAWEyr6YQm"
    "X9cqAtqEAMJ3Fy8wZuySud3tW7WvRkWJirXqJiKFJAZGq6DxFhHiFJ/qccfF5i6+6AyhTypU40MyrBiAzb27GGOf4Jvf/NbxcIjnQ3drffH8BR"
    "HduXO5LGVZ67LUutKyllJp2S37/W633y/Lsq71n/+L//Hf+nf+XeFukDJh6dxBvcZu3Nu2qWoptN/v/7v//l9cXF4qKBZrAogIgbyP94UH0bqr"
    "f/3f/qvLur7z1geISAFJ7V1fe+114UbF7LrQ2jY2rjki88vQHNVnaTwu/VMf0UFUuT4+0RGmZ/9SSIUjZRMRAWqpSy2WwxqgRFB3lNowHJh5RO"
    "slqAaEiqX0aETxoYp9YyzCQccTD9WQuD9jVdf7Fj+hmCBcVCts9+BzgIIgMCUU6a0YtpkWM9T5Oku46ZYcPFdEOpDo49XUKUjzDIwZLqqih4Wf"
    "shlzJdZEAOpr3JR0Ckawo40Hy6RyMD27d+r2biT82Vd8voKx/6bDCqu5jTAj6QwnBoDet87dhM7ZwURfLoFmlxioxFzYdrs52jMuWA57YiQQOH"
    "9VEXYdgsteTfHoHl+V3tuyLu+98/Y7b79X6sKd16Ueb65ba3fvXtWlrKtRb+y4wGKGqFKXdblz985+v371q1/9Z//8f1h3lzZatPWGiFCJXChC"
    "ALy5ufmDP/qjf/AP/9Hh5qbWajdfKGxd5GGfBRUshfa79fd/76+dtv7hB5/WUkUUoXxy//GbP/wREQkzERqlnErpvZuuOlc/RLPWGE2uaYMVC6"
    "vo3E/HG381YCCUZ+OYWCCVS9U0BjlqzYjYykjFQrlB81GRGKuqCFNQCYiQfA4HiAVc4+FjbnucLMXVZCP+GCAgQCFCAKLCnZk5HgZU1da6ql7i"
    "o1m4eQ7v1JzMzVPdM9l3wH4Izr1Iek5gnOBww/+YfrhM5g72KS1wXfiF5wKgq/5sJYqEpRTuzS5pibxJ0PFfBlB3BkX8BqI5TRVRAcn87gBzR+"
    "toOAmNVQFGcxj7YrFViQpHAxm9M5I9xPFXufYl7c5E/v0REhCGTt/VNtF+i90vUWRZrlwq5WlZFuHu9RWWrW1/+qff5M61YN+2x4+f3b13l0qh"
    "Qvb6+QmYIuJCy7IsS71zuV8X/Xt/7+//rf/gPz4cbgzdH7ErjEiFyMQGp9PpH/yDf/TSS/ds+n8mGNLJIOATEkTCi8v97/3eX3vy9MXHHz8gQs"
    "Xyox+988nHvwjfM5qVsRBpuvhMzOBNl3uvcteHwSpSVZFWCgl3cKCkgEKNht++TV+j27EYLQkRsUDjrqDFAz/8Xop5ZxEWRCzFnZwRxmpDe8/H"
    "ABFCArWnTwkpJ1CmcMwaKpVSzL23rS6LDoITWWr2BV1n1LCepeTN/WHKZ78YF+HXJM17DIUzW170gHgOCdUzsrnBIuL1XPW6yCbT3WLXh6lGkE"
    "qOT3BShkcBCeTlUkkvqYrgJCZAJDGVQ5DXkk8VT6JObCTIFUYxngJi6633FomWVhHFHQ02sIurdlbzqaT/ltCL7WiaNKX+ts+ILcUUa6UqCoWK"
    "FUGgwMKllL94/fXnz56tS3329Nmdu3dKqfa4OFdqWj3ZLHmttdayLOXqYvfyS3f+k//sv0hEVUHa7XaFSKR37r0zgO73F3/4R39ogsx4B1OCP+"
    "we2QeZE+PycvfX//rvPHz07MnjF6XUN9/4QW9H49Ywiy3xbDfvIII4wMXn/BoZObGkIUel1bIg4NZOvW/ZgDEz5JDZf/2scU16lPqbvL2tSPGl"
    "n8x/mcbeRF0ChWFWgXAe2gNmo4qYrEmkZI9QQftFmFlUSqlDK6xCCKK4gxcI2xCB3vLY317gT/QenVMpsI6XFc5RGSP7HW/FDeMtGTcOvMxOXw"
    "ifLD5GfGaIAoqCzIRI9rpZBW+jRA0hWPTZNDG5xk0ZlQDE6YhBUrRtp4a7inyhZyp1NDIXiApRERJE3LZjqcXdf4JKqpO7W0e2TwyLBVSFStWJ"
    "4OaIFMw82aKgCHwW5OrfMzvEttRt20Sk905Uai2/+uVHP3//Z7/z23+nrsvFxb53NnS1c9r0LOCyINZKq9a6lJWrYv2tL3/VJofb6cDcOp9A5O"
    "Zws7WGCrXWq4vLu3fu4CAwDEL8jDfS9AQQoKKqXl3uf/d3//KDB4+Q6vs/e3e/32WWVu+99X6HCgIYq8Jk3BT+SCrEvduRLqpAORSRUiqo7qls"
    "bTu2w7rbQeDMFcePOYN3PdwQFAiFfVxJAIpKUIJyYdVK6LEIncWrU+i3ilUVEppbky/bnyamER9YzoG+V+nCkg5pc66JKAvty6HC1uAyhQjTEC"
    "O5phlTnSPcW4nISmcT0XzwE5c9R+cNEJCdE+cpLgCAcAFPQFhFhLst/jQ19CpmBjetsN2nZvF0s9kQD2t6Vp0WUQq6ZjemmyFcMenwQAy7LG7E"
    "KyORaXREFInqsmxtExZA8O22TnzhSaqdOo9Q+p7z/8dV6IEGmHM6IjezxYOgUQuIsjBbz4hIwv07r3xbVPf7HceqePTsc5QdkALUQkulQlBrYe"
    "6/9/t/8O//R/+pCNRlt99drOv+8vLub/3WV7/08pfv3XtJFf72f/i3/+E//Ps31zeI5XwDDNGsDi3XaF4IVWW3X+699NIPf/zuL3/xYS219R6i"
    "yg5mYnSCjpePqTUR5qQ6+s7B1xLFipJS6m53UZelt1NrJzZcELNNB3KG50CsUgiDTwlKNAKW7R6LnW1c7d7pWTNohkQBEJNM2GzCM0HtXpzkx7"
    "lCG5hAhd5765vfsGKqSgUFVtiV054OCU49JymOWUwI0EQVfpOjIrRdOSPxLymRqdlfeilgyzoJ4d00IlUAwCt9JMGTU1HmDqpUSGOnR4TqL6HG"
    "wOY8DTugFZ5AlKpY46mFcROjj8+w1hGCZXqoSc7v3yjaqmBl5s4dFdiqSn80KF7AEptGU00QTLnWuV+xgVtInCKuPQ4LiOldwMUgMRAuBVNRld"
    "1u9xc/+MGTx0+cFR+0lMwMsYcKCYk0lKmhEgD90pde2u32hHSx219d3dmt+2XdXV5c3r26e3Xn7le/+tWff/DhK698e7fbeTYBAtKZd3eoiPQc"
    "fg9wOjWg5Qdv/Oj582dBZ0NVMIxdAv9D2OuPjy1yzdFhH63Y9hBJRQybbwuGWqrxY06nw9ZOPhRB384TVfvRWAzNIX5MYwlZvf3UcVq6RxHH2h"
    "xHAR65L+oVrKI6fhbiTYRS6hRFGpnliNwbIBqWitLkgKgCJO2SXoDSRAKbGjedQwVwjp/QxNAoAChNqfYYyW86IettjZlKNYhpFo44Zc90pYrH"
    "XX/QBedgLwUQ5tjvYa1VuINLVRzHGgIFVeOYT/G1rqEZFBxUBKQy1kQxdB4mI5zwMJm846UsAFKpC/fmhsLgtInk+FgAsZSKQxoFlPlNBgUDtF"
    "RDPPvZ4v2PWU/6ekyatdQlKgsyonxd64PPP//e916vde1dJIRKeg6+s9hPP4v874Wrq4tHD++/8forzNv1zYvr6+eO2UYoS6ml9N4/+fTjr33t"
    "a8tafFk2Q/v8jpoy2cKzZr1d7/rk2c1P3vxRLRSCI+y999aIsJTKzJ48LoJUEImw4EAgQgjZbcrfAZRjImAyYKRCVPf7K0LaTkfDQ/bOIsZo89"
    "2S7bwtsCkpsi71DvNp6pqDXqU0BTZjEtBURbqNYezyJDTrJuW7HTnojsM7baeCJbQcMf9WK9z4Dj6GL8S44Lztw0REpKlVB18dU2Zxxq3BGVR+"
    "7qNI72impM3NJl3q49IedzGFwdhwmIjOWjjC0nsDUP+tvNIQsCtRhxXdPmJyfJ6XHOx9v4qy/zLi7uuh7VScpiYOC0kEOREtde29YYiYIO1RuY"
    "cweL7toEXt58/1iXvP5qlyLJQH9n8k3wUDjqiUSkSde8IDpIsIf+Mb3+idxVW1Gl6YEONPGqdRQYiu6/LrBw+491KKtTeRKWiVP7FIKeWjDz+6"
    "uTkuSx1G7ZmBEoYbOEMagKgi0ce/+uWHP39vXVdzOBphrbVGWFyzPoXmmgZRRCzNMd4iUGHu3SstCVxQdqSIgLgsu91u39vG0u0HLIVsyTQ2tO"
    "I3ra0cyN+9qBOV8wu0woxtahy7k4y0kYGYFtFu/wsFfw5CoiFRomtrjQrZTjhcsGzFlILeq0/st4/mdiirz9IfvWQYn/7AlKnSSJ8804ziLADX"
    "L9Btpko143roSj6Vdp38PQ9QN0yy1eaiVCpGMEvO6siHbDFlyo35VJ/a3M1HMjZ+jPrHL0xVVqVCgI6LjzAFdAxMnD7ruooqCxMSpHx0nDeYN2"
    "PCIK2tH1qqEGp5qU5EiLkYtXkvhuvdtKz2w9dSuXPnbu95515rffuttz/88KO6LMzx1s9RB3Nu5SCiaev8ja/9m0IGYuVSqm3waq2EpW0bqO73"
    "+7feeuvVV797cbFy42TsDUIBwFT8jqGdiIDiD9944+b6xbKuvbFd0yIs3Oqy9N6jA0O7fk1lPU16wBUwcKbsjO82CvjQj5ZSdruLtm2iOvaz6I"
    "LbaNcIY2nsYiwvKwzpNcMaKRJ3otNyBVneimoyP1fAeqIduk8twJnWA5e6xHtBiUIW4dPGd+ApUVegCICZoGy3Uqw9Unt29vjHRKMGPqNkn/2P"
    "MHWLM2MXz3YbfKf9vHObXYdWlg2gJgIh1VpF2PmwIRo0JkIhd8zHTMl9H14CUhwiPkwv6TfUKNUdeDBXLC7nt8YBC2GtSyn1dDrEgZewsChH/S"
    "mxps51cmHTLGbEVhEf5JpyH/HWr60q3FsM7xRUza9sjyIzCwuzUCmPHz9+5duvlrL0LnoWfjAm68nUtYeqLst77/7s1W9/6+rqyv8BsZoXttNp"
    "205m1yy1UCn/1//5r07HDnPSOY6zdkZZwyTtenF9/PGPfgiqImxNbIRChXbPBWcAqiFzO1vneF1NOEAOoS8C0VsuXut0191uXZbR4+S2gK0N9V"
    "VeqgMJSYV9WIiu2bN6ZQZvaaxM8q4z+WBxGa21RSUcOS4lBRXuXYSXWm2NwdJZGAJs2xlXfXIBB9AysUh1YkKd00bP8inGhI9urTPgi3RwgNtv"
    "Jsw7fHs7qer1rn3KgY3OzSRB0qaHUkndB0QcGrQZMjsRFvzWNstozjDDvaeRmZPi0iTAYvYkMPk7s6Nd6tq5Z/cet4/m8Hpq4v0Xd7k2yK2GDY"
    "0gpkITvsD8L2McZmKxUoyl7VG6KgDYua+79c9f/c6z589N1HfmJTofx1r6ik10vvH1b3x+/75dw4VcWFMsCMDGzqrMfHmx/+6r3/32K9+9unPh"
    "fTiiyhz7gjrw5w5QBCwffPjLD37+/rrbc2fnIIiN5plKMQrzhAUGojE4sgWdBYa6btFn4QVjkx77QIoMAhzJgCqGfjWZEQCw+zbIREQ+gh4Rnw"
    "P47URwf+skXVSQPHE3GYNh19T1JKrKiFDcsIoiyqKtNxG14AOr5shTcCy5DYGfX8CvAUr4i2IB7vMOPetcRizD2X/RmVlvtsrfDqCwc3AsDCeU"
    "iCrVvT4q/bHPaDD8Pu7zYlUGlxRiLQsLm/wl5gWeOJVzc8xpHp6jbCbQYdL6iMoQtziMUs+W0HnAkJkVtdZFRTo3qytcf+zsA9PugILfXfYv26"
    "mZdFDHfuVBpCIDhK2m58qRh6oyN1GxvzflziIszMuyvv/++2+99VZdFrfj6axUmvy1oixaSn38+MmffOPf7HZLay39mQaziE/ez6DW+ul0/D/+"
    "5b9UFd96x9xM53zSPE7RKvzyozfffPb0iYnj06LZe2NmAEpZu70BdrlhWGmzq3e/BaL3VCJh/UxZs73zaYxAIBI4fxRjLzLmPYDC7JIab7Y9IX"
    "nkapGV2Wi6nFk5LWllgtvt1xjlqBJSaw0Ra11AZQxU/V9XBZC+XcGDEeiSk9gQc6nCb9CgxZ3m27bb76WOuN1cTswv3pyaN81Z6Yo/IT4IDNVY"
    "5hmNdbAqIpRSLUkCp3N0DqKJWj+lneEPjaXFPMbNtjXuvYEUSEZszs+sCUTAUmup6+l49EArf4F95k5hPo0f0nWtaWDLGY8CiFgoXwknGQBASS"
    "aNu7PNMwWlVCRQ7aNjVVCVw+Hm1e+8okb9mMDiZx5PVVXgzutu/9qfv/bRBz/b7XYsbezphFkECWLWh6DQtu3y6vL177/+/e/94PJy55EPHnWX"
    "JwiMLD1VVXj6/Pj6918rZQ5bVjC9EUut1ezD0QOR+AOqZMRhEbc4h5La2zo7zvw+zMN++MbFkKD+TaEIo2Mm7DUmsH8xEJWIKNxlSkYgcvtvkI"
    "xGrRRiHgySoi/CVDhIGQm98B98a6dY0w8jberz7SC41M8M43Y7zu83kOjHngGHw2AKM/kCim169uPt0LMsYtSxipC78okAEgBhsTN++Eact1sQ"
    "C6hP2y3CGqygV03dpr35xe4fNR0tYqQ1Wc+YSFeb6ITo3qS64fOkIGZmvxgnjN2Wu93OLCquoEXyKaG19c79ljwY4vJVr6lAlXkiFhgKa0J2oD"
    "szzMBot9CyrO4Wl0DRqoLC5cXlD99449NP7yNEMZy7rlQYxo6vNf7X//pft3YC5ZCVI2QzbX93LYSFCgkoImzb6f/+V//KW7o4ylVvkcESrFY+"
    "+OCj93/67m5dNZmfNoJsvZjzi9BKw1gbWOFgADW1b4x8XmWFdyhJcz5AxQ6mTFARcQ3gHKXngO1S1A03MLPH7Xq3bN8EQFEMAAoFrMtsMjYnc4"
    "hoQu3GlPLcnauIyuwhXAl0U9TQwQqAssBdfFzoqBH7BqGJhrN2Am+rYqa/ib4QBR9dLkwF3u2EwTltBgGowosLvt/ZJ/hssWRhJkFyn2VYsInC"
    "l5l7eLt/aIbHJOEVaEwMPWQXJQDu0aaHaDqE2oNeGm3DgF0BAEClBRF6b4Wix5xspEgFIisb8h1O3KMDQ3EaQti+hzT3+5Dvo9pgG1AQzLUA7F"
    "gKsMC9Usr9Tz954wc/ACqWMTwV/GMXwizLuvv5Bx+9/v3Xdvtd693zajBckUTcRUR6b6pS61KoCPP+Yv+dV7/7/k8/2O12ftzN1u7sOd3MhT/8"
    "4Q9vXjxb1iV6JPtQuXNbFrMFO3PecL0xRHbVcC7Z4Nz6DlPUNw2aAmVkbkTlFXdSgQUBdINHmpQniXsjGk00aehJapuOTsppviO90UfiCGSvgE"
    "2ezi8ybdvWTifrQ8MjbfNdoyhYqVEu8HoH12r3mU5sdIz/e+uDPsd220uIt6KhprCZvIBHA5GAER3pvnQJT1Z5KppunbgTrF0Otmi8QkRUI1Qo"
    "YWsYewlNSoKnZI1lEcZkRWm4LIxV7qKgWI1Y32XieZgMncEgBUSiuuxaO0Zbb7ecuwtdLpMzYD8yCnr9KXE1YQKCIC0F426Xqf3QEK9h2iZF2S"
    "QAnVmEv/vtPzseN/1CqlXqlnrnuizfffXVhw8+N+mzXfKUmoRwdJtpyThoCkpY7t///Gtf+9p+X01NllntmRoQGcT67NnNG3/xg3VdZqaWvTLM"
    "vZSKYBmdthC3PtA6erGPwra4lFVV7F0hpU9hWSKks2kh2revZplHJPHrrvgi3hWXsYgI9LGfkiLRMKmjZhCN8Baz7qDU+uvKiDD8oJbSHXht2w"
    "TWWlPvnhen0zpFRHGBw116CEpxwQW2NzJTbpWlt8KfXLoB56QqTKbMnH6I84uqKZh2pAV+inzw5G297SNWTaWWDUWxVtJsxFVDGaaRAU451k/p"
    "1hRTnGz8sRTyWj/8YCn8tm4wnb7gPBh3Ye52e2ExHak/AhIVtrXYI8ecDIQxnl1n3RaffSIKuPUGxkzJL16JH8bUXhoW3xg5CjOv6/6999756I"
    "MP12WdSAZjZiICLPri+uaVb32rWMytDzLKsqxE1JnFrDoBCwxlGajKUuuf/Mk3nz27sewKZyhPxltxN3x5//0P33/vrXVd09pl32nnzsJIZczG"
    "CAG01AJE5m2PQjeCCjXuOh/rpzIvvXkS4xl3TpsWYrDbuIdXE9Obd2tVjSG3Soimqoqy3TDMrMLWr8bAB+0csYZwcFAdHaau2bfP00QkoNxZ57"
    "YvkJcg/Uv48fRGCMJsbxrA0XnlPhc7dPamhqoCI71p8rWnZka/0Ebyvf5hHznxU+wekbrvS+My8YPNZjO+qYu0xhyX5QxbI11gxAnoXKpOutUY"
    "g2MiTAOPk/+sqNgYmohAtJaFSjlth/hLJe3hZjYLEEMWl5LJHmHsjl2Zg08dFQyJPIWRe2bj/2VZwCekkxtfGIkONzff+/73AKvqmJVkXDaLAt"
    "L7P/v5Wz/50bJW24+Rb880BrbpMC5UcinCvff9fv/22++++eOfXOz33gvM3ysakku64Ouvf//Fi2elLDG+93KFO2fKLxIa6dCXbJomtyne2Jcf"
    "aPt3BejcM4BANSleimlAQDSdt8aKmY1kN8nPojzBKWBFKc7dFEwbCwuSyQBngDJPv8NRPaoqFhJhialYa42Ilrr4emI0jnZe+4HdBV7Ch4CsQD"
    "EwtCgrW8/jWOZlhINOOCONsu5WnuFQb+KtyEE9m7KqipY9PL3qn2yMME9+p4PcOAL+HIuASi3FOv3RfNpsjSZCDAw9hOqZFTW6dQwKTq71yS69"
    "2fUfWWU0tJHqgw4FrXVp25a2+ojgkmHXx5LmDAxuqgMBgAzs7VICKm47zsmsS8AkhWBGlEoyr6tp0C1XpdBfvP69x0+euZRHxtFnWAUWeO217z"
    "15/HBZqmdpkKH1bLE2phqEIKI5kLDO/Hg6vPqd75hzZdZm2E9qaIynT1+8/v3XAvISEkoVUWltsyQze8WD8E+sLCo2kNNJUj+GOvFmut3WNWWs"
    "YdsVr9sl1F6KoIhWpUuUf4BQEIvmhtOHEoRk71vOcpXF8AbMtg0enQtNNk+JBFmIgxUiudxJuVQKlWK8XIqBcISEuwGyK92hJ3t8oUk+jENUR8"
    "QnaDoD480aw7fbxPxkBI7omRxU6SCU5qYQ60twn7anEhbmuMZ0bsdy5DPiv9WxKM5KU0nkq88NVGjsN2cBRIkBmWPzAyepqdP1ffTZKEpdd6YO"
    "U7M3ZVl2LCLczWqNOkBNUavJgBep5nQ2dBW58raxjSsMp5R5U2eIxpy+ljp5Xs+U8nWp7//svTd++KOlrr5YhxyOA7O+uD7++M0f+GycHZbj+X"
    "zDcG1YJGeEkw/3gHtbavnea689f35t78+ceKeqLApQ3nnnvZ//7KfruvNBmpgGk0XENtc+d1YR7vat2cFrYimvKHESK+mgkvuP5PgltJ3VaDyR"
    "MrhJFFjU9sn5VPtuOXfiGSAy5pk+tCjmXxMxI5XTtRB1wGBDTBO3/bSdU1CwPDlzwAZEz6shYYndMiBAZyW5vouPAcrgOiZUH74wZRsJmQlzwX"
    "OYfUyuNF15OkWs6S19DQLiS/2DrZ3sYAkPXizBPODEtL2Z0CIASoW4NwU5N2VMneqtoe5YkYqK0hjCeJCpyQKmlZpk7TFoC0Rg3MTwOpVaa61b"
    "O5WyaJaOGJr6jNQdq1KxRztlcUgkLmG185sQyLcyMdHJaDeLPShEojz0HKqq2jszy/F4eOVbf9Z1Viakw5Q+/fTTj95/d92tHu1gi5/hrxAiNA"
    "o9xvLGnvylFiJa1+UXH330i1/8qi4rc3w+YR3uTU9Nv/Pt7xyPNya4Eds/iAAgi/TWnJCQBj4bZjCfj77IubdggapqQNFk4/sWcyBMRrZ5Gl80"
    "dGNnKZRWHvhg1CE0GZzmpxGReQMSGh/VGQ5Mnp7lY9MccITOJu69b62t6+pC4qS0qXvc7LliS2+X7eXy+QDXw9l4MwMScbJU5MgqaWu3QrlBz4"
    "vTLwTR57yBVnxxtX1wainOM3tREuAhHJnBwFLb5JRaFg4ds05EplAznjs1R367xVCDzKTHMKQObGtYcpNmmZOAqE4LBIZkWfen41G4U/qGfXSe"
    "ubk+3p69LHms+ZVl8I60Oxg3MUqdpGwQIpVKpRrZUaI5tA+Pmfe73Zs/fOOTjz8zCG/gHpEFWOBXH334+PFDT187gwtZ8JAlfHi7Yjtu+5tFlY"
    "iWZT0ebt55+13VauLSGO6gqLLIg18//NEbr+93Oz+hi79wSf4rpdpdVkoBT5D30sdkzzqCS4bSPcldiITF2IQpUJMooAaUxOg8otJZgk3l2RWU"
    "mU3qI0oYOgrFnAfFWl0i2mS0rArFY0anwbI5YMfAElvbDGo+di+OyrfTLWLbEEG0id6DTwH7DF37DepP1dug7tSODm/5iCOdMl4UFX6T5E0BoN"
    "7VT3B7yECBmZ0cMa4nnB232e/ZRWTAXI6WYJB5VGM+I6k+9Nma/fmmFYh2RtxoZ/DWc6msj7mzTA7uGzjIWWtdEOl4PGSwqzfNVrvqVAmE2srz"
    "RDJYUEOkE6WV+k5pBNVGaB+WUvwJBi971IwCqgpaluX+55++9ud/XuuSIeF2GB+3/t67bzO3Wj2hvZZSTJMe/nUA3bZNRVV8lCfMptYznq+qvv"
    "3Wj1sPNFS8w8LCQm/++K1PPv5VqYt4nIZvQxGBmY2Lk0YWW68nWpZTPuf2CnQJiy9yYnIu02gEUQE5ewNfSxMC5Y4nkXNEJXykOqMWJ06YoqI7"
    "znOEA8b5tYckx3thsw4R30i5jP0bS4/A4JKdrXXyXdi+a2s4AaUxXMKjFW/UdaBjKRXDKlSgqIl1etlgYJvPqMA6YgZwntBMyn6JS/VO+yn3Ew"
    "4xTahmp+UETH1hjgGWUlOnm9twX9YNUW+yMRLm6wsxjvZu8Mb96WcyIhqN8TtNiR+zBx8iWavUup2OZrwS5cHOiD1y7GVk1qOqzL92/Hl2KISg"
    "Q4Mdp5O3cFlWtwWpk3P9fVNVFkL81re+dXNzyoEcAorozfXNhx++b5kWvnIsZB1FeLJsQ4idzRgrJufzIGnywOr33nv3+vo6NiWYcQvPr9t3X3"
    "2VuUU4lyN97efuvTvviBzGlTz7KDfC1qwiyp4RFmTo6Jkh3bj5/pBRqAB1FFOCRNz7SDuchIThgZjZZThOWfFnH9P5o8nQG50ZeihNwkmCDeOH"
    "pprLzLYQkvN6u3Id3WYKIbskSuXnd/AJQM313lks/fAfw0j2i9+A8uc7XxVG2h7CrX9h8hcj6c3F6ZesNN/kw1AfEQWJbPA9sk2fC9mxMvRkRk"
    "dXUJGBgZpSMXxuCWMpTzlFCdxGyJ8FR+JPBFdEeZw6s+yel3XH3MUONnBYbwx7YhhFBETe/0QPMie2hlDG8SqiQ9Yg/ueJDepqLenWUhdzq3A3"
    "ffbFxcU777z31lvvLMuOk/4I+OTp04e//qyWGtgNckco5OtXgjGfcDUqpaTykIVrrQ8+v//o8WOnpKHfTo31F7/45K0f/2C/38W17QWkqy6Ymd"
    "myuENf4vIGnfKtMk1AVKlQhASB+ODN5uQK5rccWsChDXRSMIuCmspv6tkguj6Pr4hH3N9eFh5gTiIbI2UeqEaGsc2HFJRVwH/B2Jj40IF7a7XW"
    "Gd0tKTdP3Gd4qXsX1O0efOI7v1simQHv8yJOz/d8lDMRTF4UjJh3TbTa/FZ7LVV2/HDHD1nHMHrIgdG77SAuR3drKWSghUotBlYxO4lMs83Y2P"
    "qibsBL84KJE1FEE29GMNBaQ9M29AY6pQkZitdGG0hLXajWtm1EBXwaBkFS5ERuo2vE0NeeUYbmXGRSY8U2ElHHiBxtVrksq7PbbPcQpAxL/BTV"
    "m5vrP/7jP41QAP/ZHz189OTxI6Ii44pVKihO0fRKUMMoYCo2kwIkihwAnj598tmnn5Zap0pFT5v+4C++//jhg2VdwfW66H8XAIi0trl/V28v7K"
    "YezBnYw3kVUAn/3kVsmeZrsJnzh/6eqCiVClPNOQiUXo7IvCTOTmm80TDEVTLwWVP/GTeuh0WY5UUVzDOpwMy9Ncvftk5ecgmO6MRc5/2hhD7q"
    "rtwH5VtWv2kpOnNbzq1M6SObpDRzqNhQVdwCAgPWe3C/yA2LOgUuAqjtNB48FZvsO1LN9S5WsHPvsTXILbkL0SiVmZ7zCGfaTsQ0k/nquSC6fS"
    "FNIL58n6NwDBND3syl04x26+XheBDuARmWTEQcCj53xklS4UYKoq8i42VDmLqXwSrGGL2WsDzpBCm3vRYzI8Ir337l008+XXeL8zO7Pvj1g+vr"
    "F66zJSSiatfc4PSc8Z87swEp0o1FiOuyno6HTz7+FSBJWCMV8MnTF69/77uUyFUBZwVYfynSeiMqpVQ4w8YmPT4VkdEJJXlCJ4bYhPSzxnnyyU"
    "DaP1V0205ZOSZoMvhB+VaPobqLQVzoS5pGt/GuTiGuxUO+8exG0swMam1LlhdOkHZVVmUEMltTQs8AtCte4cOKB3WD7/kQBROcMS8nMG4PV9lE"
    "x3Ju5bWSXn+jIxj0bv8lx0UxB4hGru2cKQ8AghMKXkWQiqjYxmKyw8oUmFyMxBOA7yn8QEdWVBqXJAaS8aeEqMK35L5aQNfvjjWkAtS6gMLWNm"
    "vEp6CFmZprdzVhZqepYOKMQpQ3iz5tQ92F1fkrLMJISLXkTC/Cw9W5CgDLsnzy8a9eeeWVq8u9ncGty2effda2E5VSShnB3I6NdPkB92C0kGnW"
    "kKgUKpY5EJ8DfvrpJ8wqw0uB77//wa9++cHFxd4IFj4QEzXkNgu31rxZiD7KCtEpOggH4w9GH+j0a18hlBRdqWsqyph+Dz6Ssop9IxF/okZFzS"
    "R1mOOW/YKaxCIyxoHDhh92nKz5fF/p+PDwnYv03kWEqGTKJI0xgjsXs8e3F7QxXOKLS3wKUAeQJP9+ndjno/L2qDM/MybJ9BdnqIPXNJFNscL1"
    "JX/KMhik49rEAfzXofAcJEZ7aGqtGLNmUU5dmtza9McjNpuJx0sR1r7h34Mpuu5cNWYGqyhxwLA0HhJIWJe1M/tAAnKCncSlOfE3d/rn/ubQHg"
    "SBH2J8bzI3tF1lobJYHGcslnIhbqsnc1r92bdeubk+2EDlcGwPH3ye3QhiqXXRAVxRAGT2zxDDHm6zh967/b1+SCHe/+xXRli09ci26Wuvfe90"
    "vC5YTQMkwvZR9N4VoLUmwtUME95ge1wjUQntnbrnHZCQZLpkrKPJqjs2ujh0IB6o5sdx730p1bqRgRyNEzZCDUa+EniBQRnPG+WcbxQm+GVSM+"
    "wndxbGZNBTAGjtBIjrumApASz1BVP+svY2x58rAIhyugOPIPUVt6ghcK7i0vEP0LS7yAtG5nW8I/gn1C8qgJadPl7kWZdpBWefAiFOhS0VnM3C"
    "6S4ziZAJSew5D9WpjhfXUw0iTVRHBK9HOEZPMsUAnLk8MuU0tiyZ2+zkUnLEMxLibrdvbWPuJogbrtBpuTJEDQhEhUrBaXWah66CWNKlOggMnb"
    "YQnPhS12QiJ/GNYzLEostS3/zxWz9448113XWWw+H05Mmj4hgrk7kpUTGest2rg+XjWe2ujIuryQeehPjg8/un0wmpmJL2s88f/eiHr6/rInZG"
    "lGJiy0hl6tw7s9RlXZYlmlLh3uLn96dLIsHbBjmDau7AxpRrJ/SEcIReYvpXWRj8nggrtn++omfh77EEt6nr1BvE6ad+uoU7jCa5sWa4avyBRv"
    "0Rka21WhdyaZ7CFBYWsknyhPZJZ8LCd+E+uPhuskrDOUYsCrtcCNIXADQ40ndBJyf9VIaiApQr/kT7CzCYgodd21gC7Rf3OcHk2R1U/pD5I5ph"
    "BMIKCMlq0KG4PZvEBPpusn4NVrj/zwjJMJnmVJRjpbRiDNuoAtRlIcTWTsVHfDxJxBWTJO2uCVIQGB42mWI+xu1ox7ApuaafhGqpxpM1rpn7h8"
    "e9qgBwc33z//4//599IzeHw5PHjwyaKMzqpk1BM6orluKhZ8zce0+pkP1drpNWEGYAePjo0YvrF6UWACh1/fGP3/74lx+UUkTsyuKcbViZ0HpD"
    "gFqW818vdM8GzrGNnKgmX+92m6rjTJ8OOUhBNprsk4WZArXtkHs0CokOZ8Kkg3YvqKbkZdBVh4sYpkjpibWPAy7h+6bWW299XXcZJZwBuZ5ZIk"
    "GNC2uAnbCscFc/Bzgp/KZACpjLxMHRsLQGzM0hRvU6dZaI01urQxKtF/IJSw+vo2uR7eDR6Kdh3jBMm1UvR0sxw65f7wIpQQIXM3L+G7GtGavV"
    "gXYfCg2a3C6jLB6ZyQMTPJeP4ypfd3sfZiSPeDgawM2G5lF2qrfXXSYAGOD8icqa02+dVkgRgSKOfMfBULTPszPv9rs/e+U7H3344cXlxc3hcP"
    "3iea2FO1tqgA9MjJ2FSFQqFbfeEHG3AQ+mhknVrQO11OfPnz179sx0sFuXV155RaTbMDB3Ypa2aZjN1jYkspxTK8vCJKGqQm54SLoM+bU0hq/Z"
    "6GjmoJOP0CS4Ni4P6G2ToGOMRsKFL1E4xKyXMnE7xpVxOkc2yTjI8yakaTs//kvCFG5QKeuVcKgPXN8PRCMgBXKVjaDSWPf4dI8H1aJxRkyg7X"
    "Exnp0hrhmP8nGKXTq71vM/pNzP6emSH4hoRhXP45KJdgozHeIWpg2p1KUytwD8ilf8Ia1LQlPq3VP/5Vb9gD3P6rrc68549tBe4fQW5zQAs3RY"
    "6iosrbWB1UNjfgGe5YTCYBDm5yQ6DJTnoQQQBb0G3qS6sdCtLhhPrap0ywESLaU8efLkj//4Ty4vdtc3N8+ePQnnhfi6rBAisAgV6r33sefUJP"
    "1EPLCkUgwLtdPh6ZMnCkBY3v/5L//i+39+eXFh/Fw/hdXzIaKV7g4v9iBgjKk7ICK7qDoQVggy9NbD2eRVTu6ObJI5zdvsod62rdY65+naP2mV"
    "q5nCNYIoUqoR3LQwJ2gOYYlKCWQpygiWx6AsQ8qMbR6+bZuC08MgpH+pGdfIrzd4H/g1aP9VdnS4wucAZSjEcqCZNA3LQELNvB+aSRhTWup4fW"
    "CmDUXiyYrPl/ZrlvEO2Ad0tt2AmfgMZ9gKnymBwbzcZavJto+jL5nfE6kf5x9Qdbr9IN7XuT8/j4+KNiB2kiOnw4fFhUqtx+MhXA+ag3eHduSn"
    "4JMqy2YidR0wzsnC6eOStE2Ob7NYRJG3KUR2ADvVE0wVxfv97mtf/9P7nz9tbTsebiSc+pMq1ni4WqsZOMwgksCVsPBb7YqkYUR+/OQxsyrUb3"
    "7zlQef31+WNYfqKmyqS9t/qwKL1mU3cEVYkIrmClvPgibxbHuYidkgEdbFLBJ9B9oAPEZZvTcMBXaaOzwLAEuY6SlvI1/k6owUDydxHpESkPQE"
    "OuYMMQ5rdMGtKkDrjZBqjvows7NdcE9Yzp6rEcmGyqcrfGCjlgEdOxNzT+DxmIdSknNw0vPqcMS6gSmrWAQFqBfya2hP5YxCNaYj0eMhx3hTRl"
    "U3UuDsRBef6TlEPYXZU5YehpsrnnI9h8eNBHNfVc6LR52i00PoKJMBfwiC7Iesy7q1zeYZ9giqDmC1TTBN3uxSVJ0S0KfndFQUowfyNSmrIECt"
    "VXNVGNxrCbWOlXbLsv70vZ9+61vfUZbnz5+5IAYplB9YbPAlIizuHvBW0J9GZrk9F1E9Ho/3799f6vLsxeFb3/zT3W4JpID66ihs9sxseUy11E"
    "ywSkArksm4p/GGignHfSdUTHpDDv7wFCQdvpnIMzbbEfdelwUASvGdZIr+RQWAqJBEuuhwWqkYuMn4dwn1cneddvDxrO1fZMjf1axSw9HhkaC1"
    "gvNLcYQphTHdvswoE/z3tyKgd7mjnwGIcbdi2jTI53qm7XYdOelZfC+O3duwIk/PsBezdIc/RmATUYecJ0BTGv5aSIVYMOJwvBgBrcBg0Y0LcM"
    "rjBphsDfkvmg0sIj5iN64Yg7JBbRqE/zh5RDhSBiDfMAkmje3oSqmn04kw6O4ggMk/zwmu5oML07RpzlGd0x0cvG0Np7XjTm3LfhWzzrTqlpmZ"
    "u4L88R9//fMHD7fTUQG6qVtd2KnMvTObyox71xReKxAWM92b8MAb2lKZZTudPvn4092+vvWTt3/203cvLy9FxBwqACjsRz6LsKjFYtfIQjOOi8"
    "cccurzjN9VJmIdpDM4dvHmI4OpzVZDGFqF0vsmIkblcJwZwhxkECDw4SvLuiOUYDpdOhiycv+sAl8syTIeBYtt4oVVxcwi+dVkmTrhXky0SWkI"
    "9b0L98ZwoQ8ITp7gBGd77BwHxHbRX76KmRoZo6NMaRqOQk3jpO9RLvgzW+oQQBe9Rer2GUMmWs9t8PmkpJZqqAvPf81MMjsyQUGB3F2II4jR+E"
    "XihncVgUi31iT945DHRyymMZEVQ8qsEXBNnkzqeo3dbn+8ud7tL1yDJoo021/1tkdFYQw//VfGPG5FhQS0kEmYRaVQJaKYjmo1UCeRJcKzHS4u"
    "x+CLi4t33/rJqUlYkyNWAZMTThKJNG4mBOWI+0ulb299zJcKPXz06xc3/PWvf53bRpcX9hCavnRUVwCgurUTKFApk8ATYyGJKlNOPaY/2wmIPn"
    "8IS3j6VEQFRBBIQTDk1JsnYxHiWQaJKSPItwWaCR+es6tu2h/CmuDvjlEpTstABXEJ9pTb4xYc18qYOyVIPBLWj1z6UYZDIRIYUcQENEA7ebrH"
    "Zzfw5SmlMHcSEN40PSdw/+aA32kAbMCMUbeVAjcrP2Ilk8yGVjtHgoopth4XafAwIy3AbiTzivW++ZR1TNBGJowm08VQpRDcX1S4DceRgOaSyg"
    "jYxDN7JIZnCiLEa0yA7NNZ6qKgbTtZpGl0mqObTRm3UV58CSNiYr1IVLdzkM7mwioEHmhTqJC9mUOb7j/PNPlAADgeb37yw9cAlajYojzxZxCr"
    "uczfiGRUrz7Te4/kA1gbpfbWfvrTD779yp/tL/Zm1AdA7t03+MJ2j7N0bq2UmnJtW0cluCjuZO86vBdy+oNEE6tJVYr0OMxICesbuLMw17ok49"
    "OuLCffeaKjdQROiwzGhLj9xbH2kf0UOwRTF1o1VtKU7EoGmVf+ANC2piJ1WTJVQkPQTFjy3nZsfnJKJCduWPSwh8eeJjyt6jHVfNMr6Fz4kHGS"
    "z4vO7Yhp8NBBeSp7eUz9aRcanpQRQ+OXSdwGgmfJn5L+Hvt2DBdtwihJCOK8GRhzRVefOe9TpzihtN5aFayDW0yx6J9CKVSEh1iBjH/oDkD7PI"
    "loWZbtdAh5ZO42XHOXEq0cuubglsJLEQiQAKtIxDojiIqlyBYs49E5+9hFY1ckURMWIuYucVp5c4iUMqDJzyWBH053uRs4qZBpc05te+WVVz77"
    "9ONaSujvHQwzV0+Wz0ml6AhLxBkDm24jb7NxustgyFwsHHYaocU+B0AVSq22C7FQrZH0hja8Gf2/DNqF8ybGNFHTepkZGJDuPZ2uIEz1BY4nzr"
    "6I03a0o8H8zMoCEpP4TJQLXZ4jUeKXtQ8WUS/x8US7GniX2FfoFMaqEZc95KXTaOE3iUkRAKCs/GuUgyhEAH2ubUYcdwydFDJXNur1hPTYQ1nr"
    "YvKLFDT7ttLqTBh4nBTghs1PRrenI91a3Op+FhDoGg5JxJNAgPRml4wtnERkt9uziNmrz3T6iff3hDZMAF50EeJyxPHeagaDDZmlFYa14JQsE2"
    "MmSZwuqDJL7323u3jp5S9bZm2yGyPmVDO9KKJ4hrYup1B+F8R/9Ktf/uKbf/L1i4t9cDl03iTZP+mJ3yK11FqshK6VCqbbPLz/lkUOLkgc6K2U"
    "Xw7qa6SI4shvUOF+Oh7c4CeQhydlTnHMKim6Mo8xQ/J2JptDmH8AP3tCG2ffiyQ8BYckTgC0ta1tJ9PKuIcrQb4e+xMbVMf+YyAGDayKBUlVL/"
    "XxLHDBM4yawBcSX2j+/+MU8JUN5JQU7f/knj9D4RgrjwylSZrjueeRmy3JooswLSihma+l2jjeMpF18D9COk25wB+iAYzNZKSK5txF4r9PQtMU"
    "K3hIHYrAbAwbsxxyiXCtSyl0Ot0MbPc4XAOjNFAOUYYNTKfMQmMV63xQplkbAK7LWgqCsBPaffM5JEI2KrCB4Z27L42OVBUhbtT5zgkIFLOkDY"
    "6Gh9JppVTKLz782Se/+nC/v1AVJGNDjcQVTU9Hb6BgNsIS7oPM8fM+PFXTwpS72WBbibBKF5WzXVfuqakYXCMVS0MhGMwYmDG2rmYjk2gDgGVd"
    "po5lgspAAHIg3iLNMVJyQ7Jedu2u6m63ozhDw1U7BvsmIBn5imME4Q5RUbiEx4hN40jMAc9MTBrbewXyoaLOCmSrUXHIsXOtCQjaLvm+TAK9KZ"
    "BYJx9UiDtDSi9B4wbQmISgB9wSmZyKgxqUBKSUqefqL0IkYUjK3CElSTGNY3LePYCIAzCzI5+TwT2Fayw5cV3327ZNfDGF/MHSUTY3OTDtYAxK"
    "PwwYJPFdhYceCLHWGmopyey+pOImiBFUQKXW5eLisvcNcdqq5qOHKBwMxRioW7xm7Dyl98ade+/bdip1xQhXsc91Rq9ZRczcW29IVOti2bxBUU"
    "MqlPANlSE2jgA7NWOheqQcBuZnhqgo2B4SQFgcJzmRulILlUn0oiACZoXxCOX4iOezY8rBhQTYTvk8Iw07wGMeO9LaBgC1rsk0PYMczxZHI1lF"
    "mrzJK9TIT0A7eLbqMZyTKf0PPvfs/1EAVLoN8h3qtKn1Hog2KnCzl4cCKMLqbm4dXjuc1Sho0+CwpqPktAQUCDNUoJYqKiNhTHLfYZV65mcoDM"
    "FhJkjTJJED69ID1yf52piBFka9O0pyHKaLSJpFUtV12SHg8XjtdMP0N4Mieg6RL8R8j5yfF9py/CxSNZRCiCCijoigakHW0bSMWBEZqbbebKjo"
    "nTv37MyRWBwPPBl4LxdXFlmMRCCMYi5AeDqdbg7Hy6u7kfbuMMh5wABGExXufSu1llqtn7RUYNFuwaAxrHd//vSUu1ogk3ZgyrdyPurwNuBpO8"
    "VP6w8g+R8YL9V4yCCvazibWI4I3BAty2glNIWVZ4Fik5ZAFLS1zQhA50Y2Ci68g07sKSxEOgm1pzkrFj2s8BwGEjBVdGck7Zwb0G+yHZytG85f"
    "zrKTR5WfipZRmmCJK29O1NA5Kj47QA/ujXh3mzuW4kBuzHhdyImzcfIsuEODpxpwJZxRbI42hOw/R1Mr+XcNPkyK4yLWONl72f7tLy6308n2cm"
    "P2MulIvTv19xyCxj1JC0f+nw7mr2pkiUGtlcYpS2dmxBTvRz7zsu72+4tIco9PlchgyjrF2FomhHsabPgZuqitbaWuu3UHICb2SywqlYqTNlLE"
    "lmZLJEw5PDSlHqrAJlHydjrDWHxxwO5vQpjeT3DesQ2QuLUNAWxZhSMODgGRmW3Rn2kcOgX3ZezMVOHDBGVCCfAlYgkaiH+6XsIEnQ1Ae2/bdi"
    "qljhxViODESIgOO6NpdjxZ0/4EYXfFiQLp8QKfxMs1vU5DXA44pVHQjN8+T/wDVTzPdFJA2ssDlC39dgmxzR4uJDdphSfzkc28fdNIDDBpKSJs"
    "rnZMtuP089uuBoaoGmQoyDx6DW4bcHFAvIBKeI7s5Il7Wx2DGfQ0SRoQIgCsy05B23aEUShS7EJCLWzrspQyRHKloWw1WKzj1Ek4BSi65EUz5z"
    "mEVv7CG2d2Ckuky6u723aScHUAYCEKArf/eBbpbDZo9SkxtbYxM7fW2nZ1dUdFaKrQ3ItkmHp78Vm2tqlCXRZTqaKnC+LE6XH9dCQIuNAvicaI"
    "I1k57YETSwYQSbiXQoYL8K8nHpJE2E5WNc91sccsqIfnXkF3gitaBnYM/KxUJhpY8NQw2Vq/915LQfQtYdgOUYUhSIIC2Uyy+ket/oI5rwtB+F"
    "Ifn2dM6PkNN7+PbmWahVkpeZlw+RMp/0oegLIlxVmIAqhYQRQzkgBtxtI17zyI0I9gjSXOkRCRuUO8emZ7s0g8sKRIkBFYGsg6AzrZ0DxvABuv"
    "hLbdb0ixOCuiyT/mTsiYnMtkoPc+E0vZ7S4Oh5spGMvo0/bdh+pPdUqG9HLfD5oReh2ioiHUQEIstUoIO1R1JJ0OjtjAT/feduvFsuyYu6UUiW"
    "rnrgOkIjaachabO+q0C9sLdzweVGFd9xbfPRz6U66YsGyNI+8aSqmJeqEAZg/vwcir0BDNsRl/NYJ34jvNU+xMVKki6WyAEdLgviGQdHWGZRNg"
    "RkNNs36YglXHFssZPDm68TwpslBVHXd+YxGkUorjZ5m5902kAwJLJxxJ0hPgw/nZkYrrBt8Lvu/XBp7Z4yf79zAD0CRmxXM06ZmJMFrFvrSHLD"
    "DrsXIRYccDhWQEhzlxQHztvTVbTULpbVIajCDfy2tiCHxTL9NaXuP0sb+EQpUWkySXxXnsVqpvklaAXuhDBAzqPNNLsyIC1GVtbeutJTNZQE2+"
    "HMvxhEP7TCXC8fKDVmcsJ4ghhxNmc3CuURRG/tyowzgAHUyKPmi5vLqbExgiVw7mxj6i2hiTbhnlIws/f/F8Wfe1LrnAdzRoRM57KHfvrMLMRr"
    "KBKVtoPMthk7fr0QV+ImMRDJrny/kI3n4YQcTWG5VSSi2ZT+9b0HjacfDULfLY1/Hny9XIHWDxyy2S6EWH/tKWnKE4c6iI9N6bacdLoWrzKtfY"
    "2/yfDCcQQn+fKNliBhB7dA1OkVURoD08IWhqMxecxvSDk49ppKPMXbvdDk41QPSMRHpY+NcsNlKCSbkce4URDYlJTJmzDDMqBBJvggUJbdsTyB"
    "lTIY1AuXNkkLoIW9OWNe0/wgBmNX3soCnS1XVW0PnMwC/tAXiFNFsgLnVZlt3xeDPgmUDo+s/hC/HTV8K4MEZv/hFRDIrzVwmcPNW6srBPTUdY"
    "QMQ25pxbvMe+urq7LqsvsUTsg3b/xOghNfWDlt0tIofDjQJcXd0rRBSHpgmbh+rIVQ8gzMy91mU8RDqN+nAksjBzbl7CfYdqU+XQ19jAbPobTM"
    "UCKlxqjTw/cImmqkimX9nHWdBz5BEBOS02IRgdB5D325r7rUiPDGRzqMzPzITCrbValqXWZAcPsdo0GxGX7wAzG3axlJL7WVUQFlZc9UXVI9xK"
    "wtbzvIdMnRg3+Wx+HGXsYPkA0gLPd3hgwfhEIbnWMvYKYFBHGCluabMezVHG6KmqYamZjeh7FlBjcpM5NdG3bRHyFHQfnNraULhDik4YE6w/SD"
    "WqsX5NpdXIEg+7BgLudhetbZ1bzCsl1A0jaTnSSbwsgdBnu/FsCFnEjjy7wWyqYpa/zEiwJ9ua2Gi6bMEYmEOkqzsvM7MYx3rEVsbbKyriq1fP"
    "CCFS1ePxeHF5ry6r68si0TYucFdOB2ujiUit1av4iBTxZYzh1VURbEyasmGK0a5MmMV0yQkOpyZwbzpbfsD/zDTIazghI7cDJxlWbIwwzMVRUR"
    "sfLJoCUZHEQ1ndazo7G+xrKO5666VWq8vmSlKDWJZ6t/QHByXAvnRy9iyIKhY9rvrCqXpnzeDEuojqmubM0Gy2cDxe88VJKz9COVgHTWepGppu"
    "+IFlkzHqgDlF1wiTqAHPAASspbJayDNr0CLRp1ia0hZ/WMOgwNytA1FgAAan16Q9H1whU8h1xElhDf0xTsYOQGMGUyrcBURUlmUlKsfjUaf543"
    "DlQ3LvsinQMYhPHYe7k1J4OcyNtqR2DlJGsovM4kFbIKRLcL+/qnVnRbJGFEyIAbypbq2NIWdrrZ1YZb+/tIdSVVi7L29T9TJyy9QawloXKkRG"
    "p4p2Iv8tP39zs+E/DMPYCcokm/Uco+QUnY7HUAmrnqWDB+6ayJLkprD02/+0Rz67BlhwKvoGEE0MUFTtSsx4bVUPFW5tE+V13Vn6twNybUoo0b"
    "x4FakZjOixI7ZuVYkHUkS1IO/xibV7iF9cANrgKAifkUiR+fQjZG0eD9hUa+WHKtu8Dh7gPTu7MrNMdXKcYIjc7ZeCjIiJawRLKSCu+jMumYQu"
    "CnGK9YW58LGXk27hK5HKJPkJFPfZtFdD/44ZeB9+E7KvIKpDAEQqtO73x8ON5rPm0gGJhBCESTo0JRdE7U2o0/sEoOGetyCXUuviqhocpigEOL"
    "dDyvAAI11cXNkSgnwIUwBnXhcWT1NUQ+bcHA67/dW67mwGSURENVjimNgRYVZhexOJsNZKWIy3H9Mpjs2xDGv5oOtO6FHwtCphBvT5fq67e+9d"
    "mEqJEkxur6zDuBfvJuqwZeZXkdsgGF6N5Dx4U+7fb6RfZJxMD1kNH49HRM8DNXv6+NhtOhgTOxy7AFdkinP7MTdUIoKoF/oQhrwJ581gLnjcEh"
    "R/kdcnM45IM1I+1B57eZT+1CEI99GwhM8dcv4xdqO32bT+Tw6dO1FdFhdGKhvfytR6HrLt4i8rMNKspImWyPDZ8KBBwvzlTNQyXChT6Ru0TFRV"
    "tn8xmaIgulv3pZTWNrU72gY/pdopa4WKwkx2V0AopUTU4owbyKCKVJNg8TQvHy1i2LIG+BQQiJyfpcrS9xcXRKVt20CgRIVjoiNA6GygtN65M8"
    "PV5T0ipFJsVulk7snS2jliZESaE8eqfX1+ZVix5SJtsuhsTASeqQU9mRSHscupM95T2EXaWiOqpdSUewfaHScaUHj159JjfoyZ4/E4C6ial4Z2"
    "3ICJlQKKAe7KF+ult7bZ/AmDkx8r9hi6DBlDyjgz9EaDSaVzNtmFPgFgmDwOiOcRGtlzwchY/KLJYhSbqgDKCz/xcn+QqnTkh8Ht+nSaXcW/4E"
    "EAmB9Wzs6ISjcAtk68AJ2A4Ahnk+rJZKhTOxS0JBmRI0R5joKk9198Ehmfo2lHbFRL6RZX2/WWuuyOh5uQkDpQRsGnZBr85ixRfOw+015DHxGV"
    "mt/qhcYezv6HM49WNHX25jMbVEQQ6fLqqnPrvbFwFxbR3puV9My9926lRu/9+YsX6+5iWdZhZXYuo28AfFhvuqsAy5Bno5Eq1EKFils3zmCQTj"
    "+YzNYwch1VJyHeSBWxDdtSq4IiTGkWIt5ypxM1+Hkj70Bn82ou0IHM1mw9pMTY+uwF9zwp75oDf962Jiy11kCM+grK9fpAZ9nAGTMYF7Kl8bCd"
    "nkTRB9EenoH2WTOaioN59KKBPJwBZDgJXXL5Y639aekPe2xcYIqVdydZRJnFiTmq8kkC7w+8z5xhbHhrKbaucWSbu0t6mp90ZERNkVacPvo0bm"
    "p8iRIlrITOwhyxfI59C6W6T+yLhzCP+Fhg1d3+UlV7axE3nQNiSXwwTtqG6Rmd30CyBYgO1apSMdSnIs5yo+HJnnxGw7zAvV1e3CllYW6epc08"
    "4Y0seQ6JcNu2trXLyzsUUNSJRKqg4kgjj7KwN7gxcykLIJmMxg5OQFDlUeZoZlKA1SmhsxyiKUi3V2glS6HW2kjpmbaGtk/HcdvYYtmi7CmJRZ"
    "OFDkEsyJADVu9qpChZdBTSEcwatGWfXW1tE+kW/2L3sKfrKEiCqRCG9cmnuGN9CyCRTefz9i66wnXRQ2zhJ3U2Kk46XcTkjk7ozjmJd5iToCz6"
    "bNWbzpDexInZqNNtPIqu0RuFKSkgrhL4A43CBkstNu/OCI1pI5wL5TnrO0fUfgVRLfZVYeRh6KTSo1CZ+TJDnAZ6qwQ4Z6RnkYS11rqsh8M1Jr"
    "JtjP50koCgya8o8B2JPPdR7ezND/q5yaNT253BPBG4gIHKl5Fsg4BUru68jFBg+Bg1pV1G6WXmw/Gw218syyqqhMbsC5R9vDSpUbD/03snpKWu"
    "hFgQCwHFHy4BLx3zkwwpQDTtb8yEUcLPnheACUuY+7quoyMwY0oZsrgYxUVgoV1BhOcahkl/pcruR3XcL3PbtmPvmw8aVEPTIyqWAeBnfW9boV"
    "rrzs5fmxgPzWTaylyVmqHampG9k/9Poh7DqjcVblTHliKaQvRA+7jmaYzj09KWiYUZcQ8IWFd5hnLw4Gn/RFJAZAUBzMlrgUnHAU8LF0VGck1G"
    "D7SdbW9NDMYlnKTk2EZkxs3AFDs/m8I+KNOeN8O1c/sXKQUhMfW1jxP5EpeS1LMxbgUA3e8vmaW1DRKAOcT4PsiPUyGGWRlAG9Wy/7GDgp5+Ux"
    "RlgeGs9ohNogEIm+eTCiK83+0LFWH27BpL3VEAwM4dVFvvInp1dS9zAayqoCCdqsaXEh+WiHBvBp6jUizARgHF13dARDmMEcARwTrW2Sq9O5I0"
    "5BbBWc4Q7xKvsSdPJnQvH6WgE2B45c9i+iLCI7I7e+u9tc5ba6ftKCKlEBH1vnlVL5piibxOW2+t9XXdl1JKpVorlWIuEItmQI0YdncXSLZjI7"
    "NWkyuYXh8k2Fa9HlKRSe6TB5j3hEMIndmmY4aDk1+XVnkMbmONVAC/26Pc0mkJhzMhNYIRwyPjIxygKZzG7RStN3GiRJIRRnhOzt8yHAIDaxW7"
    "sgQ8pIIxnjQIF6pw1oT+xKhImh4ci0Bpwcxxy7Isy7oe8zLEOQIykta+oE7KqPBbjPZYrbsWXBGkM4gjej0lEiyjco7BcaynxOu+v7zq3GfEBh"
    "KKkwr0eDysu4tl3TtM2mxylOtNHIgKySyMzsK1rhYDOINMU4MGFKpRyylItdjtRym+lCFLU0v81cnc7aYQ+6UwH3iJ3FSGwYlxLbsnJvbee2+9"
    "994z4GRd1/3+Yrfb1VJ73xQUycAWELptjdwP4c6gYI7N2ecB0S2bgdDT6RNU4ypXCtZKinshrbOV+AKfuigNwkEXuJN5UEkjLHvITGnYl6aTZ8"
    "ePckaD4M/KbZ04xHxfzxwYYyA47W/n7tri0ExIaezd2Cg6q869+ThZLMe4WW10l4nd4WMaKH49SxNIZinmPP1sl6OBNrFCffhjZbe/6HZMqKJa"
    "g+SPRXJQYYoOMGNHkmqtTpSIDTUbsz09hSqLuLsiGceYMyPI4kfyKFEB0P26X9edNTkerxfe6Oub623bLq/uEVq970Lx2BWJuTcl5lwmCOi9C/"
    "O6roUyWdX5PSPNXKeSJAOV0RwBKixobiQ0VqVTMJCo98a9QyqLPZsZz8CUrlAdY04NsjsLC/feeWsbiyhgKXWpy7IstVZD1LizL/aAvhbKrEJF"
    "iSkUIJy2IxCWUvJ2tVm9YXUGNRTJHTxe7Whk/mVPhzFCM2cYq8hen8600YEAGoY/w1voXLLieQQaTv8Rr/rMS5c5eGP6s63YiAUghAfkLO03th"
    "1460/IhZsPAIfWllXE8WEyActl7FZw4oyLDgtpappNwq+juB2/WQrZXC8uDqeKbFBSFfdZu5ClAtLpeEAqHAa8wRc/C0IIZ57PZiXVk6rJG4Bg"
    "T1ho9KC2IGSosQddBMdbrKnOCxyJdrtLG/DYY2SDyN766XTYX9ypdR1pP7kzj27VAweSUwwqwki01J19J2aco7BLYMSkTiK5HLjOuiWxAp0iLD"
    "miDnsOnwNPRUlw9KYxPKQivbfW+9bZ/oduxJ1CVJdlXddlqYaoy1R6E/eROe1C5Ijm8IyU7YyS6L231iyrYyB6EMDt2n5NhdqqIJaMvTBEgaad"
    "aCgiwaO1gC7hKYDnooIO87zO/FHH4A/l2plh4ozhpKeVn3ig2yiPxuQVYWwdwLPDw3BNUeS7f1QTrhPum8jWQSRE+6yd/JFx5BNGeFj08hehsy"
    "Fk3m5DSSmSqx4r8ifqz+ywh5FAmqZsHMupQrTfXxxPRzV+6bndwduY9IwGe9PV7b4eEQAQdq5mOtRqrRQIlakj9Un5NGSmaeHr7v7d/gKRmHtS"
    "kkSltU2F7t79EvmwfsJmhH5FcqLNEpAkad4QFlMcxudrzjpKOGzq16b8PJsv+M+Y/4lpsw0tx8xUSnLT7RXCiM2xa673trWtbZs5IY0UU+tSa0"
    "EqpRSrLEQ6Sx8KCncz4HQsaDHQVlicMIfatnPrnXsvtfjeDMljmSZAsMk/1MWurs4nqiM9OFLcCdEgGTbzYsUdPAflc9naCOLMfFP6Qhz9VGPG"
    "ZahKRQ9VnnWZBJNzCrWklCRqSK8DYToqAec07uClB+aVYjWNpv2BkNja40OEQyae8h93/8rIQM200JzTUMFkyQ04eKKxJY72ibrvwotkYCkOm5"
    "0uy4KIx9MRYimUtQVzh3n9PdIOFQAJiMoIBUt2gzVmpVSi0noPOXXmHGPIYjxNOvKMhG3KJwIAl1f3TtuJuxW4IMKH4+Hi6m6pNUNNI+c57bEu"
    "A7T/jRWxrtumsC9ZD2kz/diCzFE/pqW1T6b15h2gRU1xcivJvnZmJiAKQ1uMNGXbTq23zmwMATITbi11WS2FBsJbpCIW4zV0EelKMGSwm0sdNh"
    "n+PskFWmicUURO7aSWDhYEIInovoCdoyJ6fqvDIy3SMwWMOomNowgEsVt4h4cCTW9VjvY4wVjfEUzurrOmc6RQIGAp/ILkqEowV3GDLDJAXumO"
    "CCKMTzd7b1kFyTSOg3BF2BVQSk270/DjuWPNR2k20LR+LJLHh20nSF6+8FGeTbFjYxvYvIzI0ECJK9vjxoMLHggmY+yWZdmdjgfnt9uPoZnhpV"
    "EuCgyeTeQ3a8TLpKvErAzqShPhHucS4yTdjoICplBbEAVmERXufVnWdd333owaejodEYsNRSe5HAR3UIM6SDETFvaZTBdmpGIHC4WLzrfSrpih"
    "hB7a4CyLSa+RcVIeO6dQhflwuFaVrZ2Ox4OZjEVYEepSq3EliAjRGg9CdLhcnpsBei22blUygEn4A4wO4g4m5m5AflVIiFNKRo2yuW2bk8WnVZ"
    "WKJd65gHSYWhzbw7aKyM1wuC0yRdB+ed46VzgsepNKUAwOtsJZqizB7Lx141CyKiPhBmjR5wTbUF+lzgand1XP3mAwipQ5xXXuJH1LGVejteYw"
    "wdLRJlE2y5o32KPenbDfcQ9r5mZMsBTQeQ8dq7ZRYnrjIMPnpXnVJ4QiJEHRh+72e1Ftpw0GaR8Hiyev2gkolECahOH63SSsAMbbrHWxcM8pF0"
    "Bmtq8JQyhn1+LxCvYkXF7cYcN3ipyOp8uru+u6G7GYoMJivzlF6KwtwZwJr6qqnZuYbtvzjCDW9KiTO4toGNMMQOxrWP+HKL8rtzMBnLYjANa6"
    "LMtaylJKKYVqqZ6IGDD1wBQnY9IJeKqZ5kr5RZiaGuYggqEI7MyS+GcJ4Wsur1rv3LnWhbCUYBzaQFgzdF2z7yezU3l7rwPvbykYMbkdGYYARN"
    "B28MK2FDGBTNfWgF3ShOmOtRUOc1NE2+Aij9HFg2O2AXM+7pgr0sDqxyNMZDubXINmpOxYA46QQaTWtmSx4IimjlR3TX6pT/mYOeUFCnhGXD2j"
    "omJGIMDICff/yPgiGelqGoJ5xxmCNCpUl3V3fbh2y1+QCqNnwBw/RfxTPJdRfgzQnjCG0q2UigC9d68j/LyALF9DooUz8dYGYIiwrKsRaLbTEa"
    "ncubo7iew9zMQaBtHRcI7O0BQkvS91Wde9S++I0u0TGkDXxNiLh0ToGz8IBQyiJtXCYQm2fry4vCq1Gj0i9Pf+D9gfYkMsAg9joYSpZZ9jmt5E"
    "aU7J6r41HJJOhCQIp/kmOn0ROZ2OqmIT4DP2CAwtiv16GM+tjGGH46t9UeFjf9Mn+A8uqspt1aegOFG/dPLfYOJWYPZ9zVyraZUhO32WoxRP5I"
    "0MpMmmBCOMGoP+EBqAYGSfsWAgdGEZcmikd+Zu5BxE7L0Js0h3ZNMA3IuOrjWFNeEb1dF8BXpkyLYjtRuGJmfIANCM/zh5mjDYAL6uAN3v9iBq"
    "0QWhZcGRSTLQTaQjqHiI50fmXQgR7aEvcRMOYvOw/IG3hiYMgFSoVYh0x4uLS+58Op3u3nt5Wap/LtwHxlDUKS4IQbbViHDzReG6rkToxPhpJx"
    "ZDGgxbTECC42OxNCjbEiFCKTTFvLVSChnSm9nt2uCTTDMQSaY1mtQpdX0x9giJHEJO9XKQQBkvhAMr6nlJbAevIyFNMsN9O52cXUAWxQOhwdAh"
    "Qo8kshzqJLDUZua20xmejmQoRHzvTp+lYmMKhIkry/K9ZobODONOPQcggPLKTy34It1aQw2Hcxow6khZQz1n3Ew4RTGzXmYqxJMsjk4Rab1BBG"
    "2IyqS58cj0GMp4NrqNnrM5HGReGhRgytHipMucms9olMlUYCNjPTdhSd8stSzrerh5YWHRYYCEQOe4zFUGsCpENIgsI+Y2IHEIgFQKlRo0/rSn"
    "aAhBvfwT0CmLAlzdYeMoLKJQ6u7q6m5on6KnlRFS4YiTyGbN/87cFXRZ1hHjHuL90BV4HBXFM0lEKjwjZ1JHZe2B/dHMrdRqhVkEHhcDIMTDDQ"
    "QZvT5XKGC6q0IOWLCTgWJQ4vC3wKqlcVMBDF4+PFF+VJkkqPfefT3otAe1YsROuQiBHI50X4EMmhkM8hgktE6sofB0A5ALfJbxB+c6bXXnn63P"
    "4hY7U1DS6B0RQBZ5bkX7tANDS0uMwCiHjs15uSF4GYJHGIOQlDkqktWQGORjAoTWm4ZOyv+XASkLnvzAt0epSaF/h0EelvznxqMCosHoS+1ihr"
    "FAZlaiJdOGoVID2mtH5H6/671tx4P9G5n+N8mL4QwyBxPhNLP8zOgajkiy20Nh0B/TwEbFVMUhevMsACfPiwpza5sCfvkv/WWTpKgKSMY5gSvF"
    "5IzOEkYSJ24TlaDfkr9sOFjxYXUNG3vqQFxK4ZMbGR27EgH3kxmddbLkTuHNaOdPBAb4qTTMEIaEASjFtoI6rmVz8UsGE2NouP0ZISSFAfkHgM"
    "5dRVpvqmLVps5KacQM4Y48JjnzIXgB4UO52EvjHNcVa29QxR1eA/D02Guu8rJAriNNcc5/Dy+Kf2K6VXnBGqnPgYAWI5TBuHni1Yhil1K3PGtN"
    "I+A2XJcTxtZFxcVS13snIuNlEKZRURDQ7Glm3geQEMHa9BJjZCo6PdApvM4ohTnjHt09rPnMeYTqSPAAKiTMsSSlUmBZ1uPpsCy7YC5QpAYgEk"
    "pA1GJtc7aHVIsucKwfewvmN1dHNXKJ2syGmW1ubCenfcbFGzYrhkVUDjfXu3W32+/tOwlAuFpxAeap9x9ADH1kKjhDSjN3z5yw26qWWMCSm4zi"
    "u7WIZVDo1rs6qyi+CQ+sB1QBRLacIxgftgn8xLnjatUEYGgJg2uG4Z1PH4nNYjAFANH7e/6dK2E5R3ZUa76QMGkLtm1zw5mtQ9DPW6OzgShhUW"
    "+kUZM/ZG81AnnW3gxN9xLMWGfB0dAFbgibYgn9WZLaNWepNEXYj4svUYr2u1a9LnrNAp49NolR02uJMffOdQKOuD5IAamv4EP4M4cbZ/6RpQeJ"
    "sEmQw2s74wNTRugHkLCYQck4f2OgGu7BMOljoAYRgXyREA76hCZIJFf7pekHirvpYtuAiLjuL5i59c0EGQEGpMyEiI225k8yGTuG5kscy2Qg30"
    "XDPmcvmOkzOdiMGe+esR/GDDuejp37vZdfBi/ssXhpF0y+0AxPHGL3X/bO9isUH9njBPpyT4lZIEP4KgmVAyRfyomAqKe9I7CyKtiMDbHGZUVT"
    "gCqM/HoZkkb/S3DcKW4mTtF0rP0yKFrVTTn+NSOqcgTKC2SfEoyC1rYSP6fv6YmUWdiN/JJ46AgvGJRdBRN1RcehM+MnFrD291HVQ9FupzN+oT"
    "nD4SfMM2eMFKZ8UcUi10UOOtks4icfZGWZ3u4RJuH1WyjawvgQdN2sKxSJQo1m5zMqaO9dp5wd8UACb79SWZiuCCQyjZsqD/qbnoGZM6EtvnYK"
    "qfcY7aTzKDXGcuv1AWRmRFyXlaicjseBlKMko1Lc7FY3ZgqNVxM5K9aIkROFQqXWKsKiHGcBSGbXSArmbTiSZlnt3E6Hm3VZHaAYcudKxVMTsV"
    "iJFam3CDbBC+ewhc+YubGUUmiQuEzNZzK0HAQHWGnE9QQfJwXoGGz5mias2LZZFQ1j/RDL2zRIjJwzY9jJMGpMeVvDXN47995VRayI6gyDkIyT"
    "KEy3bevMhSg90z5VpwIT0TtW8LalYBWvlYJXojo7IkzwwBzDGwDQzrriaaGT+m03sdQGEs/Kp9tW0pS6+GO1yDOUzpp1J2pS6yavRDSEimc0Vh"
    "gxMaARWTlWJ26gHKpWTDM2c5t5RzDPo4NFOecBhuIUcm+Mg+oEOD6FM9VoYql0qF/wXFDrLrhJlWdOPSaii4tL2zsnqiwiKAlhaNZyVg5TuHtK"
    "vSTaGhs1hG9fTeohmeseajwfHU1aw+10VJU7d+5aUgkOcMH0m+a4JRye7g5gj/IGxFoWoxyYymzEQYeVLD/KQLnOeSwasQIxQzIQMFVHMHvbR3"
    "NUlk1hR2yGKhAgUUbGBsEYOvfUphONzBY7dFha2069t8A6EVLCbCFufVHVrR29WaIM4tYsCiLxJaIsPeMDg73t6dlBBxcbaOelFelp3ooX2FZ9"
    "AVDwC4K0HJLWWzy22Rkennpc5ZlKVy2YS9JIyUhpTzzXcia38TFuDKN9O4SIxMyIJMoYslOX3UT/SkjdYGqwqAgWnPSiZeQ0iYxwdAAwbZcOh9"
    "HYEuaqyJpJiC1WSIxhsnhMT5LhMc0bBaEaEfIUISllWdbd6XhY1zVeQkIAYU48fhSWipafM42QMSQ+9sWTKfpjj8IeO0M59FV7SHVqgUR659O2"
    "3bv3UilFlSuV3pt47HMFKCb1sFUjs4S/zrNZw6nTSql1WUxA5wsKGx17rE9QJ6LNnx4+FBVyVp0mAaH3VusK6AwIJ5p6rh6GQwVFBQhIUEEQC8"
    "zh6IgYgg37iEpsQUwnzcLCgqhItKyrjfcMNOCcUl9QEYt7AE6nkz0Kbnl1FZfUGpIgK30MohOnuSorkI56z59yUbZhlTXehBRjTVIg0LbTp+AB"
    "WDDLbNNkWgHmJ/CsFM1ffOFHIqDCoxqdIl48EJXZmZM6sT1SGobzWwEyQo6czRrrrElFgCgsnbmy1KWICAWODgxuFU1hGWmMGb2FSaiz89UPTo"
    "tnmFmuoCj+n0YIPCZi2Ocy6e4KUXXYqD0oZl131y+ebdu2rvvc2usYZI8bAyGJBKm/tXE/iiopGM+GiLh3rAhEKAg2C4g3QBSKVZKqLMLcD4eb"
    "ZVktYQIQem+Hww0RwVKhuLsmhSO+EXG9jbqFULj3frG/Ck4sqLBSvjnhRfTpv6O3fDMkAMpUiksCPRQNW99Cb+OzfashRtIns6rajD3ITCXqUv"
    "GTS7XY+l1YQIW1qZiQyGqDQqWWioBUyMSnrTW/Kl3IJmO5BdBaa61ZPkzSQNSFGUEDy3MmqogUztkn7DNV9V9Kop9KT5NV4FQKgFzA0zk8ajJd"
    "+Hil+mBWUXECasYeVBVBedWn5nVHHeHAFoSAmieN/XzJjvY3rwBJ1EWm1eHJ95IgegkkYTYG9sf23nS3tzGAqWecswxJ0XbLgeVwjd/mLIlKh/"
    "kyd0862aNhwk+ET9Kx3GqaP4xrbZgGSinWW9VaEfF4eLFbd4NfThaQBpkLQEQqQkiSWjw3gQ64ky0SCtFpO1EpFexRAyJ77Mz5g2PfIrK1rbXt"
    "L335K7XU3tvx5uZwuKZSai3Mfd0pkfkzSJIqMBQf7gwzqkipC2TeMJKIUCFUyqA7VilUzE2RuqUZfmkvLRGqcmvHZVkBCSx6HsTzvUOOF62+uG"
    "1ErMgskYjhEdZt5EkqYCEqlZY4BWjkezaOtJBQcrpmO7hMCMKytU2YTaEOFI2fCFDxPSSiYq6FSI20jRJjWIjzyFsxIkKbtEsov1Ez4YtZVngC"
    "ZTyLqMNYZp9gHa7a9ESMYbptELZVnsoo8vxGcZGEG+00EQCTFhwDGx4NKLpt1JtRHa1ScMgYp5IVAc2XENoxitmw/X9NuUP+NpKKH6s6XC2xVL"
    "H0g7Nf3mV8U2CczJQPypoDx7pv6OViYuZki/3+4vr6WWutGkedcMAvxRlSqqg2Y58iyXGqLpKjUupCvamokOmtAg9JMZ2RyFJQPR4O67rudhci"
    "en1z3dvxL//2V1Slta6KrTURqLWaGtkG/BKzFhVgvwsbYal1MaFMptOoiEfQq42G6Uy6GGLZweCIkca2nQCIqHotN3ywA7ZEKYGLFAdj5yZnxO"
    "pYKoWwlEKlVnWnh10KgihxrTj5DlwonYs7tNyL3I6eTkf1FBBEyPEpTJNnjUClmM/ZxAEGONO+dhUhgqyeMqotn2Hh3hl25TkMGk0sKaK5QLsJ"
    "x9038eRDZV2KPq9y3SVbxukqnVRCkEJ3mKXVMuYcKr4tsS4jtAHTwGBS/8S13Tsz91JWQpLcjhhSgcoATiKK2qMgMXwN5EJ48NUdSRmYoURJap"
    "xo4paEbqFC0UsGG+6cgmVLJBUAXZYd0XI4vLi6c88UmaWUacrmBD43sXgOkYawWhQBS/E3X9U1qyCqBSx1DEaFEbJyBdW2ba21l176ChEeDte9"
    "n373d//q6Xhall2tu+fPnwl3QO3dbxHEQlCCSpx0SGFm8jBQA65WUSGoLs4SG1kTq9joP7ngFHD05McAQGfZWtut+9z/Wqho1jiAIMpsxlGny/"
    "rpXKjaW2cSPZvBOlanc5it/ZwUEeVIKY3lfsYyTIkHtnbCzm3bThmf7NTpYW8YaQ0Ga7JyVyNu1reIthayna11lJr8NVf2RCgdKeBOXxBuAoSh"
    "+hwjVQQAqOMwjpcedFKfIVU5khw882B01Tp04aGSnATd4b4IKCoiovrlmQetU2VjjQhxuw5pD6IoM3eROhlFbb5Grvm2DbGNSdMjNCW8JUUGMX"
    "W0lBhzD7X0mhNy+RGS3+ifI1MFJy+IjWogCEa7/f76+vn+4rKUqmqnzCj/h4IEURAVJLk4SggCSiVaD3LVpYiSeMiBzyc94N159czbdljWuq77"
    "zv3Fi+d/9Xe+sm2n+w/uv3TvHvd+587d5y+ej30qK4Ao+ZAy9Wwinbnvl6uCxbkymlYaKVSEEmUipkgwRw+O9OJ414Qth3TKDEUiNJqwSGcRZT"
    "FiQqGylDpHizt62GR4NvGfRFhWdg5ZSG4OzXEkOm+h4sszaCsyA3Nv29Z7JxyyYg12BlARkYJZPaUeQFwWY4qfWJy4r3foTXXikea9Kp1pKc8X"
    "vTnhPQ870vDFx6tYf4PrF+aHppDcAB8mcvYs7FNRxlBNJoJtGBxHdJxjVQmLwU2oULrLB6ibCBiDuYyIYPA6TyyMfEcaZaXtbJx9iiBU6vjLxy"
    "+aqSHxVlil4ddA1Ic+uaHcqhgwAvzxj2iicVsbucfs2Gxm38Ph+s6dl935RhFSDTDh8fyT8RWW+NnIIqAmUrFkxNpaM2OHTTKBIEQzaMYR5r5t"
    "x/3+AgG4NeZW6yKNRXhZVzuP77300vFwPB5b8aAvn9ojlaC2+Qy2lGpeDXO8qC+7sLNJQIB797WqBGDXu2jNCFYkApXeTqVW7s116ITM4jIxIn"
    "B2BhXflApMW1yb6yRYIA4gYGEPY8GcwyMCUCEn5U1ebjibDBrBgFTU5u1UF5y2NQVroWJqmJD3+qhMRAj94XEXjSkr/d4eqKtAK4CIEmFyvhlx"
    "kdOiz0/w8jCKp0bN94Qhbj2rS6d9xioGEp7pFTpFo4AbzH38lRvV0BdHxTi0Q6mzxXRaxXY2OV2Y6eX2wckM7DzzgaREM6Qq3hPH9k9GPzBB5O"
    "IPt5tNQwAcidg2iiiBh8pf3WEGmCAmV5D78PPy8upwuOm9YRoVdcQpjigsHIiXQdLyTsi9tsuy2Fvq0DHPQUjrLINKb83UnugQ+/78+Yt1XZa6"
    "tG17cX395b/0ld/7vd9fluXx44fXL64PN9etbUMMpO5SFWbEUusSeHSMCSpNDD9r2qPQyE5JYg6pKsKFaGun1jfhzioAWOtay7rf7S8urvb7y6"
    "Wuxrwx7BpLoPB8RsROlJ7EHsMmbi+Jrc7jGrTlY2bjuXJadfBTYo4qKm07QQjrotDBTBoMPvdZZPUZ5ySEakBT0osDjTBX5XimzleCvuqzeR3v"
    "BreMHHR89Uy5mB1PAFWepkI2MdL+uIOJv3R+VMcgC85NFDri0XOslovUfL5H3x+PKPcW8r8w6o2gIo7Vmed128sobjgYmb4KAi4pHK/aGDBM8j"
    "+bEmW7D+ksQ5+YZ5aT68gwPIaI67pflvV4eAEILJGsFDGDMX9xzeJMsXabNjq1rZRSypJS5vNYRRFLjbVyR4QCnSyiT5483q27e/de/vJvfeXq"
    "8upv/I1/T0RPp+3q6s5uv6NSELG1bWutMwdJi3vvS11qXbLGTvEnd84XP3J2c1eCApJRcPaFnLZja+3q6t5ud7nW3VJXI/r6xp05dFFscdOqHA"
    "EskgE8GQKTXi8RRsdd+3SkYDAB/X/QW7rwoGOpGH1AhHvfWguDq3sv8Najj57GB+iiiMCAOQ/ffYORNCqqU/C9t9mWnUxYAlAiO7yBOXZ+Gtwj"
    "QB1Y9Pm/Jlts1UPKTcaIRWeLZAhp4oqj1GZT1JYmPozfUydFzqRT0glmj+nxDXSC1S1lxJfGlMLPurR9eaVEqnhOngK0/iBRFrmCiE8wBUCi4n"
    "t5tjOSJmEtDj+Ixpft/GLcX1xeP3+63xoVUqCpiLXq1CMBydTVIk6+IpMNxAwppDYaigKqLjpzWqxl/dnyuviXfe+llx8++PTJ00d/8Id/dHV1"
    "7/f/4I/efeedjz/+xYvrm3W3L8V2wrjQggCdmTWpdlKXJYp9SMW5uRzzibCP2nswQ8YD4QBIufmTnL2thRDQQhpobJ7j+7atMgwZTj5P/jZSAL"
    "wFkn6tOGIcJTd4Khwlv8SOT0UAyyxf1t5ba5tLDtLpMIZvvpP0dV8+PNNXPWW30gRkNH2PII44zWE0UWHhlZ9AfJlRxg0Ke413BCbzxPTGou7k"
    "me3ffUMLOIVRxvwwK715bRiXczgYJMNDktKhkDBwHxZbpaA8TMXs0RR1TPZFYJQTPqa0cQKEAMomKWP/lip3ncZSscofa/SJoONDck+PUEzVEq"
    "DljXFkGBnH3o6S3bo7lnI6HS4v7wyMDToEDBO9CGrqURsJ2sNkMipTPJmEkzuDJa76m2p/QFRNKkNkhHD36u52PH700YfH42l/cdW2rbXtdGxL"
    "XQtVs56J2deCoWxTCSSsdYlLJMDs8dGRR2RDxEwgue1oyBGFuRRqbWvtuNtfReMNhKh0ZokgRHUFq8tQRdViUolIpGOMONwcF1DJKbU2Us+AVE"
    "an7RVQRghMEyy7dVtrOddBcNd8uOdGfig68lhCq+TCLx8hIgEAWyy8mYYzD8zEUBqCGYTEs+7giU9lzqcv9lvVeKsoJ79uJXYOQ6/6QlLiOt7V"
    "PAUHMSz1aTqRWrLHDcplwIgnUMlwBCZKADSpRk46jGEJTlNHNEx5AM5wlCVkTbJr9qNJj9rp3I48YWjz6UniGQooCGGB4bEQP/nCqZUcWBEloI"
    "uLq+vr58u6s3A/TKGJzql6UadjetR1shx6BErvTVWpFFEtqBgEVEKY/JEpUcIvf/kr19f7Rw8fIT5ZlqUu+93+yrMcIGdF2j1vxzeEhWot1Yuu"
    "iEUYQSWTDXSC+qTUU5I1tp1Oy7q3YrvW6tV7pj2n1BE92oJGMmQITVO1OKI+PfczgYs0gfZybRN6ybQxhHkyKkgR2dqWO2RIbJQXppi1j+tYaM"
    "ioyMjINpXxyMcYzofXXvNXnMi+PgBXXOUGtE0kKB1PLJaKKOAr8NuRKIBQsCkfWERB7C+0kW0Qi4NuoBKhv+MnkByljqDPdE94/WeQCvsCJiiA"
    "IkISmWxGbBCUUskUlkl8mGgRGl7vEhk6aC+M/1RxWU3kKDvaBr83CR1E7l7BiI0HJLZQzkIms5s/hLH0RGPV0LYdluVewEowB7b2/hUiDmoLhk"
    "1aVFDQ/nyIoZwIU+QcEqKNJEXnzAEKRosWKpeXd/f7y85cqOZ2W4fUEOZtu5lu1mXJBE/VZNtp0hEwLOWZUu7+Tju7LbOtdxFZsVoejunyuXfR"
    "/H7jwIl7QERrKUmXzcWWTj+oPTkpxjJcPiFJ7rUA2eLQ1e8uH/IwRzvt4eHmsk8Jpc2BRaQsVc1aMDC5KShlM5FNeZgpYPcaO1bKHrk5NJ2eFE"
    "SiWmArcGJY0KE6GbUBiPr/A9gwxd8KrtW+AAAAAElFTkSuQmCC"
)


if __name__ == "__main__":
    _crash = open(CRASHLOG, "a")  # if Python itself ever crashes, the reason is written here
    faulthandler.enable(_crash)
    App().mainloop()
