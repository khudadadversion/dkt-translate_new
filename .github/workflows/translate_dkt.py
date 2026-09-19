#!/usr/bin/env python3
"""
NSW DKT question bank -> clean multilingual JSON (Gemini API)

Commands:
  python translate_dkt.py prepare     # merge car+rider, clean, add correct answer, shuffle options
  python translate_dkt.py explain     # English explanation for each question
  python translate_dkt.py translate   # translate question + options + explanation
  python translate_dkt.py translate fa,pa     # only some languages
  python translate_dkt.py validate    # quality check -> validation_report.json
  python translate_dkt.py status      # show progress (no API calls)
  python translate_dkt.py all         # prepare -> explain -> translate -> validate

Everything is resumable. Progress is written to dkt_nsw_multilang.json after EVERY batch,
plus a human-readable checkpoint.json. If the internet drops, the API times out, the power
goes off or you press Ctrl+C, just run the same command again - it continues where it stopped.

Put your API key in get_client() below.
"""

import json
import os
import random
import re
import signal
import sys
import time
from pathlib import Path

# =====================================================================
# CONFIG
# =====================================================================

CAR_FILE = "dkt-car.json"
RIDER_FILE = "dkt-rider.json"
OUTPUT_FILE = "dkt_nsw_multilang.json"
CHECKPOINT_FILE = "checkpoint.json"
REPORT_FILE = "validation_report.json"
ERROR_LOG = "errors.log"

MODEL_NAME = "gemini-2.5-flash"      # check with models.py which models your key can use
REQUEST_TIMEOUT_S = 180              # a single call never hangs longer than this

EXPLAIN_BATCH = 15
TRANSLATE_BATCH = 10
DELAY_BETWEEN_CALLS = 4              # seconds; raise to 12-15 on the free tier
BAD_RESPONSE_RETRIES = 3             # bad/invalid answers before the batch is split
MAX_WAIT = 600                       # longest single wait between retries (seconds)
TIME_BUDGET_MIN = float(os.environ.get("TIME_BUDGET_MIN", "0"))   # 0 = unlimited
SHUFFLE_SEED = 2026

LANGUAGES = {
    "zh-Hans": "Simplified Chinese, Mainland vocabulary, clear standard written Mandarin.",
    "zh-Hant": "Traditional Chinese using Hong Kong written vocabulary (readers are mostly Cantonese speakers), not Taiwan-specific terms.",
    "ar": "Modern Standard Arabic, simple and clear, understandable to Levantine, Iraqi and Egyptian readers.",
    "vi": "Natural Vietnamese as used by the Vietnamese community in Australia.",
    "pa": "Punjabi in Gurmukhi script (NOT Shahmukhi). Everyday Punjabi; common English loanwords like ਲਾਇਸੈਂਸ are fine.",
    "hi": "Hindi in Devanagari. Everyday spoken-style Hindi, not heavy Sanskritised Hindi; common English loanwords like लाइसेंस are fine.",
    "ne": "Nepali in Devanagari. Natural, simple Nepali.",
    "fa": "Persian (Farsi), simple formal Persian understandable to both Iranian and Afghan readers. Avoid Iran-only slang.",
}

GLOSSARY = {
    "fa": {
        "give way": "حق تقدم دادن / راه دادن",
        "roundabout": "میدان (دوربرگردان)",
        "learner licence": "گواهینامه آموزشی (L)",
        "provisional licence": "گواهینامه موقت (P1/P2)",
        "demerit points": "امتیاز منفی",
        "blood alcohol concentration (BAC)": "غلظت الکل خون (BAC)",
        "pedestrian crossing": "محل عبور عابر پیاده",
        "kerb": "جدول کنار خیابان",
        "freeway": "آزادراه",
        "dividing line": "خط وسط جاده",
        "overtake": "سبقت گرفتن",
        "school zone": "منطقه مدرسه",
    },
}

TEXT_FIXES = {
    "What does this sign means?": "What does this sign mean?",
    "carrying a disable person": "carrying a person with a disability",
}

MANUAL_REVIEW = {
    "CG062": "Mentions the RTA (now Transport for NSW) - update wording.",
    "ND040": "Street/drag racing penalty - verify against current law.",
}

ATTRIBUTION = ("Questions: © State of New South Wales (Transport for NSW), "
               "licensed under CC BY 4.0. Modified: reformatted, explanations added, translated.")

# =====================================================================
# SMALL HELPERS
# =====================================================================

class FatalError(Exception):
    """Something retrying will never fix: bad key, wrong model name, bad request."""

class BadResponse(Exception):
    """The model answered, but the answer was unusable."""


_STOP = {"now": False}

def _on_signal(signum, frame):        # Ctrl+C / kill -> finish the current batch, then exit
    _STOP["now"] = True
    print("\nStopping after the current batch... (progress is saved)")

for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, _on_signal)
    except (ValueError, AttributeError):
        pass


_START = time.time()

def should_stop():
    if _STOP["now"]:
        return True
    return TIME_BUDGET_MIN > 0 and (time.time() - _START) / 60 > TIME_BUDGET_MIN


def log_error(title, content):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} {title} ---\n{str(content)[:4000]}\n")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path, data):
    """Write to a temp file, then replace. A crash mid-write can never corrupt the file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint():
    if Path(CHECKPOINT_FILE).exists():
        try:
            return load_json(CHECKPOINT_FILE)
        except Exception:      # noqa: BLE001  - corrupted checkpoint is not important
            pass
    return {"phase": None, "done": {}, "failed": {}, "calls": 0, "updated": None}


def save_checkpoint(cp):
    cp["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json_atomic(CHECKPOINT_FILE, cp)


def clean_text(s):
    s = (s or "").strip()
    s = re.sub(r"\s*-\s*$", "", s)
    s = re.sub(r"\s{2,}", " ", s)
    for old, new in TEXT_FIXES.items():
        s = s.replace(old, new)
    return s


def clean_picture(p):
    return None if p in (None, "", "None") else p


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]

# =====================================================================
# STEP 1: PREPARE (no API)
# =====================================================================

def prepare():
    car = load_json(CAR_FILE)
    rider = load_json(RIDER_FILE)
    rng = random.Random(SHUFFLE_SEED)
    merged, skipped = {}, []

    def add(items, licence):
        for q in items:
            question = clean_text(q["question"])
            options_en = [clean_text(q[k]) for k in ("optionA", "optionB", "optionC")]
            if not question or any(not o for o in options_en):
                skipped.append({"title": q["title"], "licence": licence, "reason": "empty question/options"})
                continue

            correct_text = options_en[0]          # option A is the correct one in the official bank
            key = (q["title"], question, tuple(sorted(options_en)))
            pic = clean_picture(q.get("picture"))

            if key in merged:
                merged[key]["licences"].append(licence)
                merged[key]["images"][licence] = pic
                continue

            shuffled = options_en[:]
            rng.shuffle(shuffled)
            flags = []
            if pic:
                flags.append("needs_redrawn_image")
            if re.search(r"\d", question + " ".join(options_en)):
                flags.append("verify_numbers_with_current_handbook")
            if q["title"] in MANUAL_REVIEW:
                flags.append("manual_review: " + MANUAL_REVIEW[q["title"]])

            merged[key] = {
                "source_code": q["title"],
                "category": q["category"],
                "licences": [licence],
                "images": {licence: pic},
                "correct_index": shuffled.index(correct_text),
                "question": {"en": question},
                "options": [{"en": o} for o in shuffled],
                "explanation": {},
                "review_flags": flags,
            }

    add(car, "car")
    add(rider, "rider")

    counts = {}
    for item in merged.values():
        counts[item["source_code"]] = counts.get(item["source_code"], 0) + 1
    questions = []
    for item in merged.values():
        code = item["source_code"]
        qid = code if counts[code] == 1 else f"{code}-{item['licences'][0].upper()}"
        item["images"] = {k: v for k, v in item["images"].items() if v}
        questions.append({"id": qid, **item})

    # keep work already done
    if Path(OUTPUT_FILE).exists():
        old = {q["id"]: q for q in load_json(OUTPUT_FILE)["questions"]}
        kept = 0
        for q in questions:
            o = old.get(q["id"])
            if o and o["question"]["en"] == q["question"]["en"] and \
               [x["en"] for x in o["options"]] == [x["en"] for x in q["options"]]:
                q["question"], q["options"], q["explanation"] = o["question"], o["options"], o["explanation"]
                kept += 1
        print(f"Kept existing work for {kept} questions.")

    data = {
        "meta": {
            "state": "NSW",
            "source": "https://www.nsw.gov.au/sites/default/files/2021-08/driver-knowledge-test-questions-car.pdf",
            "attribution": ATTRIBUTION,
            "disclaimer": "Not affiliated with Transport for NSW or Service NSW.",
            "languages": ["en"] + list(LANGUAGES),
            "image_files_note": ("The 'images' field only stores the original file names. The official "
                                 "pictures are NOT covered by CC BY - redraw them before publishing."),
        },
        "skipped": skipped,
        "questions": questions,
    }
    save_json_atomic(OUTPUT_FILE, data)
    with_img = sum(1 for q in questions if q["images"])
    both = sum(1 for q in questions if len(q["licences"]) == 2)
    print(f"Prepared {len(questions)} questions ({both} shared by car+rider), "
          f"{with_img} with images, skipped {len(skipped)}.")

# =====================================================================
# GEMINI CLIENT + RETRY / CHECKPOINT LOGIC
# =====================================================================

_client = None
_schema_supported = True

def get_client():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types

        # On GitHub the key comes from the GEMINI_API_KEY secret.
        # On your laptop you can simply write it between the quotes below.
        key = os.environ.get("GEMINI_API_KEY") or "PUT_YOUR_API_KEY_HERE"

        if not key or key == "PUT_YOUR_API_KEY_HERE":
            sys.exit("No API key. Put it inside get_client(), or set GEMINI_API_KEY.")
        _client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000),   # milliseconds
        )
    return _client


FATAL_SIGNS = ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED",
               "is not found", "no longer available", "NOT_FOUND", "400 INVALID_ARGUMENT")
QUOTA_SIGNS = ("429", "RESOURCE_EXHAUSTED", "quota")
NETWORK_SIGNS = ("timeout", "timed out", "deadline", "connection", "connect",
                 "getaddrinfo", "dns", "ssl", "network", "unreachable", "reset by peer",
                 "503", "502", "504", "UNAVAILABLE", "INTERNAL", "500")


def classify_error(msg):
    low = msg.lower()
    if any(s.lower() in low for s in FATAL_SIGNS):
        return "fatal"
    if any(s.lower() in low for s in QUOTA_SIGNS):
        return "quota"
    if any(s in low for s in NETWORK_SIGNS):
        return "network"
    return "other"


def call_gemini(prompt, schema, temperature):
    """Network problems and quota limits are retried forever; only real errors stop the run."""
    global _schema_supported
    from google.genai import types

    attempt = 0
    bad_answers = 0
    while True:
        attempt += 1
        if should_stop():
            raise KeyboardInterrupt
        try:
            cfg = dict(temperature=temperature, response_mime_type="application/json")
            if schema and _schema_supported:
                cfg["response_schema"] = schema
            resp = get_client().models.generate_content(
                model=MODEL_NAME, contents=prompt,
                config=types.GenerateContentConfig(**cfg),
            )
            text = (resp.text or "").strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
            data = json.loads(text)
            time.sleep(DELAY_BETWEEN_CALLS)
            return data

        except KeyboardInterrupt:
            raise
        except json.JSONDecodeError as e:
            bad_answers += 1
            log_error("invalid json", f"{e}\n{text[:2000] if 'text' in dir() else ''}")
            if bad_answers >= BAD_RESPONSE_RETRIES:
                raise BadResponse("model kept returning invalid JSON")
            time.sleep(5)
        except Exception as e:                       # noqa: BLE001
            msg = str(e)
            kind = classify_error(msg)

            # old SDK without response_schema support -> switch it off and retry
            if schema and _schema_supported and ("response_schema" in msg or "unexpected keyword" in msg):
                _schema_supported = False
                print("  (this SDK version does not support response_schema - continuing without it)")
                continue

            log_error(f"{kind} error", msg)
            if kind == "fatal":
                raise FatalError(msg)
            if kind == "quota":
                wait = min(MAX_WAIT, 60 * attempt)
                print(f"  quota limit reached - waiting {int(wait)}s (progress is saved, "
                      f"you can also close the window and run again later)")
            elif kind == "network":
                wait = min(MAX_WAIT, 15 * 2 ** min(attempt, 6))
                print(f"  no connection / timeout - retrying in {int(wait)}s ({msg[:60]})")
            else:
                wait = min(MAX_WAIT, 20 * attempt)
                print(f"  error: {msg[:90]} - retrying in {int(wait)}s")
            time.sleep(wait + random.uniform(1, 5))


def run_batch_with_split(batch, worker, cp, phase_key):
    """worker(batch) -> dict id->result. A batch the model keeps failing is split in half,
    so one bad question can never block the rest."""
    try:
        return worker(batch)
    except BadResponse as e:
        if len(batch) == 1:
            qid = batch[0]["id"]
            cp["failed"].setdefault(phase_key, [])
            if qid not in cp["failed"][phase_key]:
                cp["failed"][phase_key].append(qid)
            log_error(f"giving up on {qid} ({phase_key})", str(e))
            print(f"  skipped {qid} for now (listed in checkpoint.json -> failed)")
            return {}
        mid = len(batch) // 2
        print(f"  batch failed, splitting ({len(batch)} -> {mid} + {len(batch) - mid})")
        out = run_batch_with_split(batch[:mid], worker, cp, phase_key)
        out.update(run_batch_with_split(batch[mid:], worker, cp, phase_key))
        return out

# =====================================================================
# STEP 2: ENGLISH EXPLANATIONS
# =====================================================================

EXPLAIN_PROMPT = """You are an experienced NSW (Australia) driving instructor helping new migrants
prepare for the Driver Knowledge Test.

For each question below, write a short explanation (1-3 sentences, plain simple English, B1 level)
of WHY the correct answer is right, based on NSW road rules and the NSW Road User Handbook.

Rules:
- Do NOT invent numbers, fines or rules that are not in the question or correct answer.
  If you are not sure a specific number is current, state the rule generally.
- If the question refers to a picture you cannot see, explain using the correct answer text only.
- Do not start with "The correct answer is". Explain the rule itself.
- confidence = "low" if you are unsure the rule is still current, or the question depends on the picture.

Return JSON: a list of {id, explanation, confidence} for every question, same ids.

Questions:
"""


def explain():
    from pydantic import BaseModel

    class Expl(BaseModel):
        id: str
        explanation: str
        confidence: str

    data = load_json(OUTPUT_FILE)
    cp = load_checkpoint()
    cp["phase"] = "explain"
    todo = [q for q in data["questions"] if not q["explanation"].get("en")]
    total = len(data["questions"])
    print(f"[explain] done {total - len(todo)}/{total}, remaining {len(todo)}")

    def worker(batch):
        payload = [{
            "id": q["id"],
            "category": q["category"],
            "has_picture": bool(q["images"]),
            "question": q["question"]["en"],
            "options": [o["en"] for o in q["options"]],
            "correct_answer": q["options"][q["correct_index"]]["en"],
        } for q in batch]
        result = call_gemini(EXPLAIN_PROMPT + json.dumps(payload, ensure_ascii=False, indent=1),
                             list[Expl], temperature=0.3)
        if not isinstance(result, list):
            raise BadResponse("not a list")
        out = {r["id"]: r for r in result if isinstance(r, dict) and r.get("id")
               and str(r.get("explanation", "")).strip()}
        if not out:
            raise BadResponse("no usable explanations")
        return out

    for batch in batches(todo, EXPLAIN_BATCH):
        if should_stop():
            print("Stopped (progress saved). Run the same command again to continue.")
            break
        try:
            got = run_batch_with_split(batch, worker, cp, "explain")
        except KeyboardInterrupt:
            break
        for q in batch:
            r = got.get(q["id"])
            if not r:
                continue
            q["explanation"]["en"] = str(r["explanation"]).strip()
            if str(r.get("confidence", "")).lower() == "low" and \
                    "low_confidence_explanation" not in q["review_flags"]:
                q["review_flags"].append("low_confidence_explanation")

        # ---- checkpoint after every batch ----
        save_json_atomic(OUTPUT_FILE, data)
        cp["calls"] = cp.get("calls", 0) + 1
        cp["done"]["explain"] = sum(1 for q in data["questions"] if q["explanation"].get("en"))
        save_checkpoint(cp)
        print(f"  explained {cp['done']['explain']}/{total}")

# =====================================================================
# STEP 3: TRANSLATION
# =====================================================================

def build_translate_prompt(lang, lang_desc, payload):
    glossary = GLOSSARY.get(lang)
    glossary_txt = ""
    if glossary:
        glossary_txt = "Use this terminology consistently:\n" + \
            "\n".join(f"- {k} = {v}" for k, v in glossary.items()) + "\n"

    return f"""You are a professional translator and a licensed driving instructor who is a native speaker
of the target language and has lived in Sydney for many years. You are translating the NSW (Australia)
Driver Knowledge Test for new migrants.

Target language: {lang} - {lang_desc}

Goal: the translation must read as if a native-speaking driving instructor wrote it.
It must NOT sound like machine translation. Readers may have limited education, so be clear and simple,
but keep the exact legal meaning.

Rules:
1. Translate meaning, not word-by-word. Rewrite sentences so they sound natural in {lang}.
2. Keep the exact meaning of every rule. Never add, remove or soften conditions
   (must / must not / may / only / always / unless).
3. Numbers: keep them exactly, written with Western digits 0-9 (e.g. 40 km/h, 0.05, 12 months).
4. Keep these in English exactly as written: L, P1, P2, BAC, km/h, NSW, and words painted on
   real signs (STOP, GIVE WAY, KEEP LEFT, NO STANDING, SCHOOL ZONE...). On first use add the
   translation in brackets, e.g. GIVE WAY (translation). Learners must recognise the real sign text.
5. Incomplete questions: if the English question is an unfinished sentence that the options complete,
   make the translated question and EACH translated option combine into a correct natural sentence.
   If the grammar of {lang} makes that awkward, turn it into a complete question and make each option
   a complete standalone answer.
6. The 3 options must stay in the same order and remain clearly different from each other.
   Do not make the correct answer longer or more detailed than the wrong ones.
7. "correct_index" is given only for context. Do not reveal it in the question or options.
8. Explanation: translate naturally and simply, like a teacher speaking to a student.
   If the explanation is empty, return an empty string for it.
9. Australian context: vehicles drive on the LEFT.
{glossary_txt}
Return JSON: a list with one object per input question: {{id, question, options (exactly 3), explanation}}.
Keep the same ids. No extra text.

Questions:
{json.dumps(payload, ensure_ascii=False, indent=1)}
"""


def translate(only_langs=None):
    from pydantic import BaseModel

    class Tr(BaseModel):
        id: str
        question: str
        options: list[str]
        explanation: str

    data = load_json(OUTPUT_FILE)
    cp = load_checkpoint()
    cp["phase"] = "translate"
    langs = {k: v for k, v in LANGUAGES.items() if not only_langs or k in only_langs}
    if only_langs:
        unknown = [l for l in only_langs if l not in LANGUAGES]
        if unknown:
            sys.exit(f"Unknown language code(s): {unknown}. Available: {list(LANGUAGES)}")
    total = len(data["questions"])

    for lang, desc in langs.items():
        # NOTE: translation does NOT wait for the explanations to be finished.
        todo = [q for q in data["questions"] if not q["question"].get(lang)]
        print(f"[{lang}] done {total - len(todo)}/{total}, remaining {len(todo)}")

        def worker(batch, lang=lang, desc=desc):
            payload = [{
                "id": q["id"],
                "category": q["category"],
                "question": q["question"]["en"],
                "options": [o["en"] for o in q["options"]],
                "correct_index": q["correct_index"],
                "explanation": q["explanation"].get("en", ""),
            } for q in batch]
            result = call_gemini(build_translate_prompt(lang, desc, payload), list[Tr], temperature=0.4)
            if not isinstance(result, list):
                raise BadResponse("not a list")
            out = {}
            for r in result:
                if not isinstance(r, dict) or not r.get("id"):
                    continue
                opts = r.get("options") or []
                if len(opts) != 3 or not str(r.get("question", "")).strip() \
                        or any(not str(o).strip() for o in opts):
                    log_error(f"bad item {lang}", json.dumps(r, ensure_ascii=False))
                    continue
                out[r["id"]] = r
            if not out:
                raise BadResponse("no usable translations in this batch")
            return out

        for batch in batches(todo, TRANSLATE_BATCH):
            if should_stop():
                print("Stopped (progress saved). Run the same command again to continue.")
                return
            try:
                got = run_batch_with_split(batch, worker, cp, f"translate:{lang}")
            except KeyboardInterrupt:
                return
            for q in batch:
                r = got.get(q["id"])
                if not r:
                    continue
                q["question"][lang] = str(r["question"]).strip()
                for opt, text in zip(q["options"], r["options"]):
                    opt[lang] = str(text).strip()
                expl = str(r.get("explanation", "")).strip()
                if expl and q["explanation"].get("en"):
                    q["explanation"][lang] = expl

            # ---- checkpoint after every batch ----
            save_json_atomic(OUTPUT_FILE, data)
            cp["calls"] = cp.get("calls", 0) + 1
            cp["done"][f"translate:{lang}"] = sum(1 for q in data["questions"] if q["question"].get(lang))
            save_checkpoint(cp)
            print(f"  [{lang}] {cp['done'][f'translate:{lang}']}/{total}")

# =====================================================================
# STATUS (no API)
# =====================================================================

def status():
    if not Path(OUTPUT_FILE).exists():
        print("No output file yet - run: python translate_dkt.py prepare")
        return
    data = load_json(OUTPUT_FILE)
    qs = data["questions"]
    total = len(qs)
    print(f"Questions: {total}   with images: {sum(1 for q in qs if q['images'])}")
    print(f"English explanations: {sum(1 for q in qs if q['explanation'].get('en'))}/{total}")
    for lang in LANGUAGES:
        q_done = sum(1 for q in qs if q["question"].get(lang))
        e_done = sum(1 for q in qs if q["explanation"].get(lang))
        print(f"  {lang:8} questions {q_done}/{total}   explanations {e_done}/{total}")
    cp = load_checkpoint()
    if cp.get("updated"):
        print(f"Last run: {cp['updated']}   API calls: {cp.get('calls', 0)}")
    for key, ids in (cp.get("failed") or {}).items():
        if ids:
            print(f"  failed in {key}: {len(ids)} -> {', '.join(ids[:10])}")
    print("Nothing is lost - running the same command again continues from here.")

# =====================================================================
# VALIDATE (no API)
# =====================================================================

SCRIPTS = {
    "zh-Hans": r"[\u4e00-\u9fff]", "zh-Hant": r"[\u4e00-\u9fff]",
    "ar": r"[\u0600-\u06ff]", "fa": r"[\u0600-\u06ff]",
    "hi": r"[\u0900-\u097f]", "ne": r"[\u0900-\u097f]",
    "pa": r"[\u0a00-\u0a7f]",
    "vi": r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữự]",
}
WRONG = {
    "fa": r"[ةيك]",
    "pa": r"[\u0600-\u06ff]",
    "zh-Hans": r"[們這個車輛駕駛線過]",
    "zh-Hant": r"[们这个车辆驾驶线过]",
}


def numbers(s):
    return sorted(re.findall(r"\d+(?:\.\d+)?", s))


def validate():
    data = load_json(OUTPUT_FILE)
    report = {"summary": {}, "problems": []}

    for lang in LANGUAGES:
        done = 0
        for q in data["questions"]:
            if not q["question"].get(lang):
                continue
            done += 1
            pairs = [(q["question"]["en"], q["question"][lang])] + \
                    [(o["en"], o.get(lang, "")) for o in q["options"]]
            issues = []
            for en, tr in pairs:
                if numbers(en) != numbers(tr):
                    issues.append(f"numbers differ: {numbers(en)} vs {numbers(tr)}")
                if not re.search(SCRIPTS[lang], tr, re.I):
                    issues.append("text not in expected script")
                if lang in WRONG and re.search(WRONG[lang], tr):
                    issues.append("wrong script/variant characters")
                if tr.strip() == en.strip() and len(en) > 15:
                    issues.append("left untranslated")
            if len({o.get(lang) for o in q["options"]}) < 3:
                issues.append("duplicate options")
            if issues:
                report["problems"].append({"id": q["id"], "lang": lang, "issues": sorted(set(issues))})
        report["summary"][lang] = f"{done}/{len(data['questions'])} translated"

    report["summary"]["missing_explanations"] = sum(1 for q in data["questions"]
                                                    if not q["explanation"].get("en"))
    report["summary"]["problems"] = len(report["problems"])
    save_json_atomic(REPORT_FILE, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Details: {REPORT_FILE}")

# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    langs = sys.argv[2].split(",") if len(sys.argv) > 2 else None

    try:
        if cmd == "prepare":
            prepare()
        elif cmd == "explain":
            explain()
        elif cmd == "translate":
            translate(langs)
        elif cmd == "validate":
            validate()
        elif cmd == "status":
            status()
        elif cmd == "all":
            if not Path(OUTPUT_FILE).exists():
                prepare()
            explain()
            translate(langs)
            validate()
        else:
            sys.exit(__doc__)
    except FatalError as e:
        print("\nSTOPPED - this will not be fixed by retrying:")
        print(f"  {e}")
        print("Check your API key and MODEL_NAME at the top of the script. Progress is saved.")
    except KeyboardInterrupt:
        print("\nStopped by user. Progress is saved - run the same command again to continue.")

    if os.name == "nt":
        input("\nFinished. Press Enter to close...")
