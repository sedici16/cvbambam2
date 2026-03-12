import pandas as pd
from cv_extractor import extract_json, oracle_analyse
import docx
import fitz  # PyMuPDF
import tempfile
import re
import json
import os
import uuid
import sqlite3
import secrets
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
# LOCAL MODEL — re-enable on 2GB+ RAM server
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from huggingface_hub import InferenceClient as HFClient
from typing import List, Optional
from pydantic import BaseModel
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

MAX_DOCS = 5
FREE_CV_LIMIT = 10
DB_FILE = os.getenv("DB_PATH", "talent_pool.db")

# LOCAL MODEL — re-enable on 2GB+ RAM server
# _model = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SESSION_DAYS = 7

pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


# ── DB setup ───────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                plan TEXT DEFAULT 'free',
                cvs_processed INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                last_active TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                raw TEXT NOT NULL,
                text_for_match TEXT NOT NULL,
                saved_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON candidates(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON candidates(user_id)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS upgrade_requests (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                email TEXT NOT NULL,
                wise_reference TEXT NOT NULL,
                message TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS email_verifications (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('shizzer_mode', '0')")
        try:
            conn.execute("ALTER TABLE users ADD COLUMN verified INTEGER DEFAULT 0")
        except Exception:
            pass
        conn.execute("UPDATE users SET verified = 1 WHERE role = 'admin'")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS email_verifications (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)
        # Add verified column if it doesn't exist yet
        try:
            conn.execute("ALTER TABLE users ADD COLUMN verified INTEGER DEFAULT 0")
        except Exception:
            pass
        # Mark admin as verified
        conn.execute("UPDATE users SET verified = 1 WHERE role = 'admin'")

        # Create admin account from env or defaults
        admin_email = os.getenv("ADMIN_EMAIL", "admin@cvbambam.com")
        admin_password = os.getenv("ADMIN_PASSWORD", "admin1234")
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (admin_email,)).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, role, plan, cvs_processed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), admin_email, hash_password(admin_password), "admin", "pro", 0, datetime.now().isoformat())
            )
            print(f"Admin account created: {admin_email}")


init_db()


# ── Email helper ──────────────────────────────────────────────────────────────

def send_email(subject: str, body: str, to: str = None):
    gmail_user = os.getenv("GMAIL_USER")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")
    recipient = to or os.getenv("ADMIN_EMAIL", gmail_user)

    if not gmail_user or not gmail_password or "xxxx" in gmail_password:
        print(f"[EMAIL SKIPPED] to={recipient} | {subject}\n{body}")
        return

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = gmail_user
        msg["To"] = recipient
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, recipient, msg.as_string())
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")


# ── Auth helpers ───────────────────────────────────────────────────────────────

def create_session(user_id: str) -> str:
    session_id = secrets.token_hex(32)
    expires_at = (datetime.now() + timedelta(days=SESSION_DAYS)).isoformat()
    with get_db() as conn:
        conn.execute("INSERT INTO sessions (id, user_id, expires_at) VALUES (?, ?, ?)",
                     (session_id, user_id, expires_at))
    return session_id


def get_user_from_session(session_id: str) -> Optional[dict]:
    if not session_id:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT s.user_id, s.expires_at, u.email, u.role, u.plan, u.cvs_processed "
            "FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.id = ?",
            (session_id,)
        ).fetchone()
    if not row:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.now():
        return None
    return dict(row)


def get_current_user(request: Request) -> Optional[dict]:
    session_id = request.cookies.get("session_id")
    return get_user_from_session(session_id)


def require_user(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_admin(request: Request) -> dict:
    user = require_user(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user


def update_last_active(user_id: str):
    with get_db() as conn:
        conn.execute("UPDATE users SET last_active = ? WHERE id = ?",
                     (datetime.now().isoformat(), user_id))


# ── CV processing helpers ──────────────────────────────────────────────────────

def compute_similarity(ideal_profile, text_blocks):
    # LOCAL MODEL — re-enable on 2GB+ RAM server
    # embeddings = _model.encode([ideal_profile] + text_blocks)
    # scores = cosine_similarity([embeddings[0]], embeddings[1:])[0]
    # return scores.tolist()
    hf = HFClient(provider="hf-inference", api_key=os.getenv("HF_TOKEN"))
    all_texts = [ideal_profile] + text_blocks
    vecs = []
    for text in all_texts:
        result = hf.feature_extraction(text, model=EMBED_MODEL)
        arr = np.array(result, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr[0]
        vecs.append(arr)
    vecs = np.array(vecs)
    query = vecs[0:1]
    docs = vecs[1:]
    # cosine similarity manually
    scores = (docs @ query.T).flatten() / (
        np.linalg.norm(docs, axis=1) * np.linalg.norm(query) + 1e-9
    )
    return scores.tolist()


def compute_similarity_api(ideal_profile: str, text_blocks: list) -> list:
    from cv_extractor import client
    scores = []
    for text in text_blocks:
        prompt = f"""Score how well this candidate matches the job description.
Return ONLY a number between 0 and 1 (e.g. 0.87). Nothing else.

JOB DESCRIPTION:
{ideal_profile}

CANDIDATE SUMMARY:
{text[:1500]}

SCORE:"""
        try:
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3-0324",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            raw = response.choices[0].message.content.strip()
            score = float(re.search(r"[\d.]+", raw).group())
            scores.append(min(max(score, 0.0), 1.0))
        except Exception:
            scores.append(0.0)
    return scores


def flatten_value(cell):
    def format_dict(d):
        return ", ".join(f"{k.capitalize()}: {v}" for k, v in d.items())
    if isinstance(cell, list):
        return "\n".join(format_dict(item) if isinstance(item, dict) else str(item) for item in cell)
    return str(cell)


def flatten_lists_in_df(df):
    return df.map(flatten_value)


def read_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])


def read_pdf(path):
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)


def clean_and_parse_json(json_data):
    if not isinstance(json_data, str):
        return json_data
    cleaned = re.sub(r"^```(json)?", "", json_data.strip(), flags=re.IGNORECASE).strip("` \n")
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    cleaned = re.sub(r'(?<=:\s)(?=,|\})', 'null', cleaned)
    cleaned = '\n'.join([line for line in cleaned.splitlines() if line.strip()])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")


def row_to_text(row):
    return " | ".join(str(val) for val in row if isinstance(val, str) and val.strip())


def clean_cv_text(text: str) -> str:
    # Remove page numbers (lines that are just digits)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Remove separator lines
    text = re.sub(r'^[\-=_\*\.]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Collapse 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace per line
    text = '\n'.join(line.rstrip() for line in text.splitlines())
    return text.strip()


# ── Auth endpoints ─────────────────────────────────────────────────────────────

@app.post("/auth/signup")
async def signup(request: Request, email: str = Form(...), password: str = Form(...)):
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    with get_db() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email.lower(),)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered.")
        user_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, plan, cvs_processed, created_at, verified) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, email.lower(), hash_password(password), "user", "free", 0, datetime.now().isoformat(), 0)
        )
        token = secrets.token_hex(32)
        expires_at = (datetime.now() + timedelta(hours=24)).isoformat()
        conn.execute("INSERT INTO email_verifications (token, user_id, expires_at) VALUES (?, ?, ?)",
                     (token, user_id, expires_at))

    base_url = str(request.base_url).rstrip("/")
    verify_link = f"{base_url}/verify?token={token}"
    send_email(
        subject="Verify your Cv Bam Bam account",
        body=f"Hi,\n\nThanks for signing up! Click the link below to verify your email address:\n\n{verify_link}\n\nThis link expires in 24 hours.\n\nCv Bam Bam",
        to=email.lower()
    )
    return RedirectResponse(url="/signup?verify=1", status_code=303)


@app.post("/auth/login")
async def login(email: str = Form(...), password: str = Form(...)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
    if not row or not verify_password(password, row["password_hash"]):
        raise HTTPException(status_code=400, detail="Invalid email or password.")
    if not row["verified"]:
        raise HTTPException(status_code=403, detail="Please verify your email before logging in. Check your inbox.")
    session_id = create_session(row["id"])
    update_last_active(row["id"])
    response = RedirectResponse(url="/app", status_code=303)
    response.set_cookie("session_id", session_id, httponly=True, max_age=SESSION_DAYS * 86400)
    return response


@app.get("/verify")
async def verify_email(token: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM email_verifications WHERE token = ?", (token,)).fetchone()
        if not row:
            return RedirectResponse(url="/login?error=Invalid+or+expired+verification+link")
        if datetime.fromisoformat(row["expires_at"]) < datetime.now():
            conn.execute("DELETE FROM email_verifications WHERE token = ?", (token,))
            return RedirectResponse(url="/login?error=Verification+link+expired.+Please+sign+up+again.")
        conn.execute("UPDATE users SET verified = 1 WHERE id = ?", (row["user_id"],))
        conn.execute("DELETE FROM email_verifications WHERE token = ?", (token,))
    return RedirectResponse(url="/login?verified=1", status_code=303)


@app.post("/auth/logout")
async def logout(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id:
        with get_db() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_id")
    return response


@app.post("/auth/change-password")
async def change_password(request: Request, current_password: str = Form(...), new_password: str = Form(...)):
    user = require_user(request)
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    with get_db() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user["user_id"],)).fetchone()
    if not verify_password(current_password, row["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    with get_db() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                     (hash_password(new_password), user["user_id"]))
    return {"status": "updated"}


@app.get("/auth/me")
def get_me(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401)
    return {
        "email": user["email"],
        "plan": user["plan"],
        "role": user["role"],
        "cvs_processed": user["cvs_processed"],
        "cvs_remaining": max(0, FREE_CV_LIMIT - user["cvs_processed"]) if user["plan"] == "free" else None,
    }


# ── CV processing ──────────────────────────────────────────────────────────────

@app.post("/process")
async def process_files(request: Request, files: List[UploadFile] = File(...), ideal_profile: str = Form(...)):
    user = require_user(request)

    if user["plan"] == "free" and user["cvs_processed"] >= FREE_CV_LIMIT:
        raise HTTPException(status_code=403, detail=f"Free plan limit reached ({FREE_CV_LIMIT} CVs). Please upgrade to Pro.")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_DOCS:
        raise HTTPException(status_code=400, detail=f"Max {MAX_DOCS} files allowed.")

    # Check remaining quota
    if user["plan"] == "free":
        remaining = FREE_CV_LIMIT - user["cvs_processed"]
        files = files[:remaining]

    data = []
    raw_data = []
    cv_texts = []  # cleaned raw text per CV for Oracle mode
    for upload in files:
        suffix = os.path.splitext(upload.filename)[1].lower()
        if suffix not in (".pdf", ".docx"):
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await upload.read())
            tmp_path = tmp.name

        try:
            text = read_pdf(tmp_path) if suffix == ".pdf" else read_docx(tmp_path)
            cv_texts.append(clean_cv_text(text))
            json_data = extract_json(text)
            json_parsed = clean_and_parse_json(json_data)
            data.append(json_parsed)
            raw_data.append(json_parsed)
        except Exception as e:
            data.append({"error": str(e), "source": upload.filename})
            raw_data.append(None)
            cv_texts.append("")
        finally:
            os.unlink(tmp_path)

    if not data:
        raise HTTPException(status_code=400, detail="No valid files processed.")

    # Update usage count
    processed_count = len([r for r in raw_data if r is not None])
    with get_db() as conn:
        conn.execute("UPDATE users SET cvs_processed = cvs_processed + ? WHERE id = ?",
                     (processed_count, user["user_id"]))
    update_last_active(user["user_id"])

    df = pd.json_normalize(data)
    preview_df = flatten_lists_in_df(df.copy())
    preview_df["text_for_match"] = preview_df.apply(row_to_text, axis=1)

    similarities = compute_similarity(ideal_profile, preview_df["text_for_match"].tolist())
    preview_df["similarity"] = [round(s, 4) for s in similarities]

    try:
        preview_df.insert(preview_df.columns.get_loc("name") + 1, "similarity", preview_df.pop("similarity"))
    except Exception:
        pass

    preview_df = preview_df.sort_values(by="similarity", ascending=False)
    preview_df.drop(columns=["text_for_match"], inplace=True)

    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    preview_df.to_csv(tmp_csv.name, index=False)
    csv_filename = os.path.basename(tmp_csv.name)

    rows = preview_df.where(pd.notna(preview_df), other=None).to_dict(orient="records")

    saveable = []
    oracle_candidates = []
    for i, raw in enumerate(raw_data):
        if raw is not None:
            flat_row = flatten_lists_in_df(pd.json_normalize([raw])).iloc[0]
            name = raw.get("name", f"Candidate {i+1}")
            saveable.append({
                "name": name,
                "raw": raw,
                "text_for_match": row_to_text(flat_row),
            })
            # Find this candidate's similarity score from sorted df
            score_row = preview_df[preview_df.get("name", pd.Series()) == name]
            score = float(score_row["similarity"].values[0]) * 100 if not score_row.empty else 0
            oracle_candidates.append({
                "name": name,
                "score": round(score, 1),
                "cv_text": cv_texts[i],
            })

    # Sort oracle candidates by score, keep top 5
    oracle_candidates = sorted(oracle_candidates, key=lambda x: x["score"], reverse=True)[:5]

    return {
        "columns": list(preview_df.columns),
        "rows": rows,
        "csv_file": csv_filename,
        "saveable": saveable,
        "oracle_candidates": oracle_candidates,
    }


# ── Oracle mode ────────────────────────────────────────────────────────────────

class OracleRequest(BaseModel):
    candidates: list   # [{name, score, cv_text}]
    job_description: str

@app.post("/oracle")
async def oracle(req: OracleRequest, request: Request):
    require_user(request)
    if not req.candidates or not req.job_description:
        raise HTTPException(status_code=400, detail="Missing candidates or job description.")
    results = oracle_analyse(req.candidates[:5], req.job_description)
    return {"results": results}


# ── Talent pool ────────────────────────────────────────────────────────────────

class SaveCandidateRequest(BaseModel):
    name: str
    category: str
    raw: dict
    text_for_match: str


@app.post("/save-candidate")
def save_candidate(req: SaveCandidateRequest, request: Request):
    user = require_user(request)
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM candidates WHERE name = ? AND category = ? AND user_id = ?",
            (req.name, req.category, user["user_id"])
        ).fetchone()
        if existing:
            return {"status": "exists", "message": f"{req.name} already in pool under {req.category}."}
        conn.execute(
            "INSERT INTO candidates (id, user_id, name, category, raw, text_for_match, saved_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), user["user_id"], req.name, req.category, json.dumps(req.raw), req.text_for_match, datetime.now().isoformat())
        )
    return {"status": "saved"}


@app.get("/pool-candidates")
def get_pool_candidates(request: Request, category: str = None):
    user = require_user(request)
    with get_db() as conn:
        if category:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE user_id = ? AND LOWER(category) = LOWER(?)",
                (user["user_id"], category)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM candidates WHERE user_id = ?", (user["user_id"],)).fetchall()
    candidates = [{"id": r["id"], "name": r["name"], "category": r["category"],
                   "raw": json.loads(r["raw"]), "saved_at": r["saved_at"]} for r in rows]
    return {"candidates": candidates}


@app.get("/pool-categories")
def get_pool_categories(request: Request):
    user = require_user(request)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT category FROM candidates WHERE user_id = ? ORDER BY category",
            (user["user_id"],)
        ).fetchall()
    return {"categories": [r["category"] for r in rows]}


@app.get("/search-candidates")
def search_candidates(request: Request, name: str = None, skills: str = None, location: str = None, min_exp: int = None):
    user = require_user(request)
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM candidates WHERE user_id = ?", (user["user_id"],)).fetchall()

    results = []
    for r in rows:
        raw = json.loads(r["raw"])
        if name and name.lower() not in r["name"].lower():
            continue
        if location and location.lower() not in (raw.get("location") or "").lower():
            continue
        if min_exp is not None:
            try:
                exp = int(str(raw.get("years_of_experience", 0)).split()[0])
                if exp < min_exp:
                    continue
            except (ValueError, TypeError):
                continue
        if skills:
            candidate_skills = raw.get("skills", [])
            skills_text = " ".join(candidate_skills).lower() if isinstance(candidate_skills, list) else str(candidate_skills).lower()
            if skills.lower() not in skills_text:
                continue
        results.append({"id": r["id"], "name": r["name"], "category": r["category"],
                        "raw": raw, "saved_at": r["saved_at"]})
    return {"candidates": results}


@app.delete("/pool-candidates/{candidate_id}")
def delete_candidate(candidate_id: str, request: Request):
    user = require_user(request)
    with get_db() as conn:
        conn.execute("DELETE FROM candidates WHERE id = ? AND user_id = ?", (candidate_id, user["user_id"]))
    return {"status": "deleted"}


@app.post("/match-pool")
async def match_pool(request: Request, ideal_profile: str = Form(...), category: str = Form(...)):
    user = require_user(request)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM candidates WHERE user_id = ? AND LOWER(category) = LOWER(?)",
            (user["user_id"], category)
        ).fetchall()
    filtered = [{"id": r["id"], "name": r["name"], "category": r["category"],
                 "raw": json.loads(r["raw"]), "text_for_match": r["text_for_match"]} for r in rows]

    if not filtered:
        raise HTTPException(status_code=404, detail=f"No candidates in category '{category}'.")

    similarities = compute_similarity(ideal_profile, [c["text_for_match"] for c in filtered])

    results = []
    for candidate, score in zip(filtered, similarities):
        raw = candidate["raw"]
        results.append({
            "name": candidate["name"],
            "similarity": round(score, 4),
            "category": candidate["category"],
            "email": raw.get("email", ""),
            "job_title": raw.get("job_title", ""),
            "years_of_experience": raw.get("years_of_experience", ""),
            "location": raw.get("location", ""),
            "skills": ", ".join(raw.get("skills", [])) if isinstance(raw.get("skills"), list) else raw.get("skills", ""),
            "id": candidate["id"],
            "raw": raw,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {"results": results, "category": category}


# ── Upgrade requests ──────────────────────────────────────────────────────────

@app.post("/request-upgrade")
async def request_upgrade(request: Request, wise_reference: str = Form(...), message: str = Form("")):
    user = require_user(request)
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM upgrade_requests WHERE user_id = ? AND status = 'pending'",
            (user["user_id"],)
        ).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="You already have a pending upgrade request.")
        conn.execute(
            "INSERT INTO upgrade_requests (id, user_id, email, wise_reference, message, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), user["user_id"], user["email"], wise_reference, message, "pending", datetime.now().isoformat())
        )

    send_email(
        subject=f"Upgrade Request from {user['email']}",
        body=f"User: {user['email']}\nWise Reference: {wise_reference}\nMessage: {message}\n\nApprove at: http://localhost:8080/admin"
    )
    return {"status": "submitted"}


# ── Admin endpoints ────────────────────────────────────────────────────────────

@app.get("/admin/users")
def admin_get_users(request: Request):
    require_admin(request)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, email, role, plan, cvs_processed, created_at, last_active FROM users ORDER BY created_at DESC"
        ).fetchall()
    return {"users": [dict(r) for r in rows]}


@app.post("/admin/users/{user_id}/plan")
def admin_set_plan(user_id: str, request: Request, plan: str = Form(...)):
    require_admin(request)
    if plan not in ("free", "pro"):
        raise HTTPException(status_code=400, detail="Plan must be 'free' or 'pro'.")
    with get_db() as conn:
        conn.execute("UPDATE users SET plan = ? WHERE id = ?", (plan, user_id))
    return {"status": "updated"}


@app.get("/admin/upgrade-requests")
def admin_upgrade_requests(request: Request):
    require_admin(request)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM upgrade_requests ORDER BY created_at DESC"
        ).fetchall()
    return {"requests": [dict(r) for r in rows]}


@app.post("/admin/upgrade-requests/{req_id}/approve")
def admin_approve_upgrade(req_id: str, request: Request):
    require_admin(request)
    with get_db() as conn:
        row = conn.execute("SELECT * FROM upgrade_requests WHERE id = ?", (req_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Request not found.")
        conn.execute("UPDATE users SET plan = 'pro' WHERE id = ?", (row["user_id"],))
        conn.execute("UPDATE upgrade_requests SET status = 'approved' WHERE id = ?", (req_id,))

    send_email(
        subject="Your CV Bam Bam account has been upgraded to Pro!",
        body=f"Hi,\n\nYour account ({row['email']}) has been upgraded to Pro. You now have unlimited CV processing.\n\nThank you!\nCV Bam Bam"
    )
    return {"status": "approved"}


@app.post("/admin/upgrade-requests/{req_id}/reject")
def admin_reject_upgrade(req_id: str, request: Request):
    require_admin(request)
    with get_db() as conn:
        conn.execute("UPDATE upgrade_requests SET status = 'rejected' WHERE id = ?", (req_id,))
    return {"status": "rejected"}


@app.delete("/admin/users/{user_id}")
def admin_delete_user(user_id: str, request: Request):
    require_admin(request)
    with get_db() as conn:
        conn.execute("DELETE FROM candidates WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return {"status": "deleted"}


@app.get("/admin/stats")
def admin_stats(request: Request):
    require_admin(request)
    with get_db() as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users WHERE role != 'admin'").fetchone()[0]
        pro_users = conn.execute("SELECT COUNT(*) FROM users WHERE plan = 'pro' AND role != 'admin'").fetchone()[0]
        total_cvs = conn.execute("SELECT SUM(cvs_processed) FROM users WHERE role != 'admin'").fetchone()[0] or 0
        total_candidates = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    return {
        "total_users": total_users,
        "pro_users": pro_users,
        "free_users": total_users - pro_users,
        "total_cvs_processed": total_cvs,
        "total_candidates_saved": total_candidates,
    }


# ── File download ──────────────────────────────────────────────────────────────

@app.get("/download/{filename}")
def download_csv(filename: str, request: Request):
    require_user(request)
    if not re.match(r'^[\w\-]+\.csv$', filename):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="text/csv", filename="results.csv")


# ── Page routes ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("templates/index.html")


@app.get("/login", include_in_schema=False)
def login_page():
    return FileResponse("templates/login.html")


@app.get("/signup", include_in_schema=False)
def signup_page():
    return FileResponse("templates/signup.html")


@app.get("/app", include_in_schema=False)
def cv_app(request: Request):
    if not get_current_user(request):
        return RedirectResponse(url="/login")
    return FileResponse("templates/app.html")


@app.get("/pool", include_in_schema=False)
def pool_page(request: Request):
    if not get_current_user(request):
        return RedirectResponse(url="/login")
    return FileResponse("templates/pool.html")


@app.get("/upgrade", include_in_schema=False)
def upgrade_page(request: Request):
    if not get_current_user(request):
        return RedirectResponse(url="/login")
    return FileResponse("templates/upgrade.html")


@app.get("/settings", include_in_schema=False)
def settings_page(request: Request):
    if not get_current_user(request):
        return RedirectResponse(url="/login")
    return FileResponse("templates/settings.html")


@app.get("/contact", include_in_schema=False)
def contact_page():
    return FileResponse("templates/contact.html")


@app.get("/pricing", include_in_schema=False)
def pricing_page():
    return FileResponse("templates/pricing.html")


@app.post("/contact")
async def contact_submit(
    name: str = Form(...),
    email: str = Form(...),
    message: str = Form(...)
):
    try:
        send_email(
            subject=f"Contact form: {name} <{email}>",
            body=f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        )
        return JSONResponse({"ok": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin", include_in_schema=False)
def admin_page(request: Request):
    user = get_current_user(request)
    if not user or user["role"] != "admin":
        return RedirectResponse(url="/login")
    return FileResponse("templates/admin.html")


@app.get("/{filename}.html", include_in_schema=False)
def serve_html(filename: str):
    if not re.match(r'^[\w\-]+$', filename):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    file_path = f"templates/{filename}.html"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
