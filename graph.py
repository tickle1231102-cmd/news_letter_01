import operator
import os
import re
import requests
import feedparser
import trafilatura
from datetime import datetime, timedelta, timezone
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field
from openai import OpenAI
# google.colab import moved inside try (Colab only)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# --- 1. Client & Constants Setup ---
try:
    from google.colab import userdata
    api_key = userdata.get('OPENAI_API_KEY')
except Exception:
    api_key = os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(
    api_key=api_key,
    default_headers={"Accept-Encoding": "gzip, deflate"}
)

UA = {"User-Agent": "Mozilla/5.0 (newsletter-agent-course)"}
SOURCES = [
    ("OpenAI",     "https://openai.com/blog/rss.xml"),
    ("DeepMind",   "https://deepmind.google/blog/rss.xml"),
    ("TechCrunch", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("The Verge",  "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ("AI타임스",    "https://www.aitimes.com/rss/allArticle.xml"),
]
BATCH, TARGET = 40, 5
CRITERIA = ("독자는 AI를 실제 제품에 붙이는 국내 개발팀입니다.\n"
            "- 이번 주 일하는 방식이 바뀔 만한가\n"
            "- 지금 쓰는 도구·API의 가격·한도·정책이 실제로 변했나\n"
            "버릴 것: 발표 예정·로드맵만 있는 것, MOU·투자유치·수상 같은 홍보성 소식")

# --- 2. Schemas & State ---
class Draft(BaseModel):
    headline: str = Field(description="20자 내외의 한국어 헤드라인")
    summary:  str = Field(description="세 문장 요약. ~합니다체, 과장 없이 건조하게")
    why:      str = Field(description="국내 개발팀에게 왜 중요한지 한 문장")

class Pick(BaseModel):
    index: int = Field(description="후보 목록에서의 번호")
    reason: str = Field(description="왜 골랐는지 한 문장")
    event: str = Field(description="이 기사가 다루는 사건을 짧은 라벨로")

class Shortlist(BaseModel):
    picks: list[Pick]

class Verdict(BaseModel):
    ok:       bool      = Field(description="요약이 원문에 근거하면 true")
    problems: list[str] = Field(description="근거 없는 부분. 없으면 빈 목록")

class Brief(TypedDict):
    hours:     int
    collected: list
    picked:    list
    drafted:   Annotated[list, operator.add]
    verified:  list
    log:       Annotated[list, operator.add]

class ReportIn(TypedDict):
    item: dict

INIT = {"hours": 24, "collected": [], "picked": [], "drafted": [], "verified": [], "log": []}

# --- 3. Helper Functions ---
def strip_tags(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()

def published_at(entry):
    t = entry.get("published_parsed")
    return datetime(*t[:6], tzinfo=timezone.utc) if t else None

def ask_picks(items, n):
    listing = "\n".join(f"{i}. [{it['source']}] {it['title']}" for i, it in enumerate(items))
    sys = f"{CRITERIA}\n\n아래 목록에서 중요한 순서대로 {n}건을 고르세요. 같은 사건에는 같은 event 라벨을 붙이세요."
    out = client.chat.completions.parse(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": listing}],
        response_format=Shortlist).choices[0].message.parsed
    return [p for p in out.picks if 0 <= p.index < len(items)]

def draft(body):
    sys_prompt = ("당신은 국내 개발팀을 위한 AI 뉴스레터 기자입니다.\n"
                  "아래 기사 본문을 읽고 헤드라인·요약·왜 중요한지를 쓰세요.\n"
                  "'주목된다·기대를 모은다' 같은 기자체 표현은 쓰지 마세요.")
    return client.chat.completions.parse(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": body[:6000]}],
        response_format=Draft).choices[0].message.parsed

def check(d):
    sys_check = ("요약이 원문에서 뒷받침되는지 판정하세요.\n"
                 "헤드라인과 요약만 보고 판단하고, 번역이나 단위 환산은 문제가 아닙니다.")
    user = f"[원문]\n{d['body'][:5000]}\n\n[헤드라인]\n{d['headline']}\n\n[요약]\n{d['summary']}"
    return client.chat.completions.parse(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role": "system", "content": sys_check}, {"role": "user", "content": user}],
        response_format=Verdict).choices[0].message.parsed

# --- 4. Node Functions ---
def collect(s: dict) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=s["hours"])
    items, dead, seen = [], [], set()
    for name, url in SOURCES:
        try:
            feed = feedparser.parse(requests.get(url, headers=UA, timeout=20).content)
        except Exception:
            dead.append(name)
            continue
        for e in feed.entries:
            at = published_at(e)
            if not at or at < cutoff:
                continue
            key = e.link.split("?")[0].rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            items.append({"title": e.title, "url": e.link, "source": name, "at": at,
                          "summary": strip_tags(e.get("summary", ""))[:300]})
    return {"collected": items,
            "log": [f"① 수집   {s['hours']}시간 time window · {len(items)}건" + (f" · 응답 없음 {dead}" if dead else "")]}

def select(s: dict) -> dict:
    items = s["collected"]
    survivors = []
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        survivors += [chunk[p.index] for p in ask_picks(chunk, 8)]
    finals = ask_picks(survivors, TARGET)
    return {"picked": [survivors[p.index] for p in finals],
            "log": [f"② 선별   {len(items)} → 예선 {len(survivors)} → {len(finals)}건"]}

def fan_report(s: dict):
    return [Send("report", {"item": it}) for it in s["picked"]]

def report(s: ReportIn) -> dict:
    it = s["item"]
    try:
        res = requests.get(it["url"], headers=UA, timeout=15)
        body = trafilatura.extract(res.text) if res.status_code == 200 else None
    except Exception:
        body = None

    if not body or len(body) < 600:
        return {"drafted": [], "log": [f"   취재 제외 {it['source']} · 본문 {len(body or '')}자"]}
    d = draft(body)
    return {"drafted": [{**it, "body": body[:6000], **d.model_dump()}]}

def verify(s: dict) -> dict:
    kept, dropped = [], []
    for d in s["drafted"]:
        (kept if check(d).ok else dropped).append(d)
    return {"verified": kept,
            "log": [f"④ 검수   {len(s['drafted'])} → {len(kept)}건" + (f" · 불합격 {[x['source'] for x in dropped]}" if dropped else "")]}

# --- External Discord publish function ---
def send_discord_stub(run_id, lead, articles, webhook=None, dry_run=True):
    import json
    COLORS = {"모델·API": 0x0B6E77, "도구·프레임워크": 0x4C7C9C,
              "정책·규제": 0x8F5606, "사례·적용": 0x2E7D5B, "연구": 0x6A4A9C}
    DEFAULT = 0x5F7476
    TITLE_MAX, DESC_MAX, EMBED_MAX, TOTAL_MAX = 256, 4096, 10, 5800
    
    if not articles:
        embeds = [{"title": f"🗞️ {run_id}", "color": DEFAULT, "description": "오늘은 조용합니다."}]
    else:
        embeds = [{"title": f"🗞️ {run_id} · AI 브리핑", "description": lead, "color": DEFAULT}]
        for i, a in enumerate(articles, 1):
            desc = a["summary"]
            if a.get("why"):
                desc += f"\n\n💡 **{a['why']}**"
            embeds.append({
                "title":       f"{i}. {a['headline']}"[:TITLE_MAX],
                "description": desc[:DESC_MAX],
                "url":         a["url"],
                "color":       COLORS.get(a.get("topic", ""), DEFAULT),
                "footer":      {"text": f"{a['source']} · {a['when']}"},
            })
    
    total = lambda es: sum(len(e.get("title", "")) + len(e.get("description", "")) 
                           + len(e.get("footer", {}).get("text", "")) for e in es)
    while len(embeds) > EMBED_MAX or total(embeds) > TOTAL_MAX:
        embeds.pop()
        
    payload = {"username": "편집실", "embeds": embeds}
    if dry_run or not webhook:
        print(f"[dry-run] embed {len(payload['embeds'])}개 — 보내지 않음")
        return False
    
    r = requests.post(webhook, json=payload, timeout=20)
    return r.status_code in (200, 204)

def make_lead(arts):
    if not arts:
        return ""
    srcs = ", ".join(dict.fromkeys(a["source"] for a in arts))
    return f"오늘은 {len(arts)}건을 골랐습니다. ({srcs})"

def publish(s: dict) -> dict:
    arts = [{"headline": a["headline"], "summary": a["summary"], "why": a["why"],
             "url": a["url"], "source": a["source"], "topic": a.get("event", ""),
             "when": a["at"].strftime("%m-%d %H:%M")} for a in s["verified"]]
    today = datetime.now().strftime("%Y-%m-%d")
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    dry_run = os.environ.get('DRY_RUN', '0') == '1'
    sent = send_discord_stub(today, make_lead(arts), arts, webhook=webhook, dry_run=dry_run)
    label = f"{len(arts)}건" if arts else "조용합니다"
    return {"log": [f"⑤ 발행   {label} · {'보냄' if sent else 'dry-run'}"]}

# --- 5. Graph Builder ---
def build():
    g = StateGraph(Brief)
    g.add_node("collect", collect)
    g.add_node("select", select)
    g.add_node("report", report)
    g.add_node("verify", verify)
    g.add_node("publish", publish)
    
    g.add_edge(START, "collect")
    g.add_edge("collect", "select")
    g.add_conditional_edges("select", fan_report, ["report"])
    g.add_edge("report", "verify")
    g.add_edge("verify", "publish")
    g.add_edge("publish", END)
    return g

def run():
    return build().compile().invoke(INIT)
