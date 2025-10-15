#!/usr/bin/env python3
import os, re, json, textwrap
from typing import List, Dict, Any
from dotenv import load_dotenv
import streamlit as st

# ---- Env / SDK ----
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    st.stop()  # stops rendering with a message
os.environ.setdefault("GEMINI_API_KEY", API_KEY)

try:
    from google import genai
except Exception as e:
    st.error("google-genai not installed. Run: pip install --upgrade google-genai")
    st.stop()

client = genai.Client()
MODEL_NAME = "gemini-2.5-flash"

# ---- Utils ----
def _extract_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = min([i for i in [text.find("{"), text.find("[")] if i != -1] or [-1])
    if start == -1:
        return text
    end = max(text.rfind("}"), text.rfind("]"))
    if end != -1 and end >= start:
        return text[start:end+1]
    return text

def request_mcqs(topic: str, count: int = 5, options_per_q: int = 4) -> List[Dict[str, Any]]:
    prompt = textwrap.dedent(f"""
    You are a quiz generator. Produce EXACT JSON (no commentary, no code fences) for {count} MCQs on "{topic}".
    Each question must have {options_per_q} options and may have one OR multiple correct answers.
    Use zero-based indices in "correct_indices". Schema:

    [
      {{
        "question": "Concise question?",
        "options": ["Option 1","Option 2","Option 3","Option 4"],
        "correct_indices": [0]   // or [0,2]
      }}
    ]

    Rules:
    - Return ONLY JSON.
    - Keep questions/options short and unambiguous.
    """).strip()

    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = getattr(resp, "text", None) or getattr(resp.candidates[0].content.parts[0], "text", str(resp))
    raw = _extract_json(text)
    data = json.loads(raw)

    mcqs = []
    for item in data[:count]:
        q = {
            "question": str(item.get("question","")).strip(),
            "options": [str(x).strip() for x in item.get("options", [])][:options_per_q],
            "correct_indices": sorted({int(i) for i in item.get("correct_indices", [])})
        }
        if not q["question"] or len(q["options"]) != options_per_q:
            continue
        q["correct_indices"] = [i for i in q["correct_indices"] if 0 <= i < options_per_q] or [0]
        mcqs.append(q)
    if not mcqs:
        raise RuntimeError("Empty/invalid MCQs from model.")
    return mcqs

def grade_locally(mcqs, user_ans):
    results, correct = [], 0
    for i, (q, picks) in enumerate(zip(mcqs, user_ans), start=1):
        cs, ps = set(q["correct_indices"]), set(picks)
        ok = (ps == cs)
        correct += int(ok)
        results.append({
            "num": i, "question": q["question"], "options": q["options"],
            "your": sorted(picks), "correct": sorted(q["correct_indices"]), "ok": ok
        })
    return {"results": results, "score": correct, "total": len(mcqs)}

def request_explanations(mcqs):
    payload = []
    for q in mcqs:
        payload.append({
            "question": q["question"],
            "correct_answers": [q["options"][i] for i in q["correct_indices"]],
            "all_options": q["options"]
        })
    prompt = textwrap.dedent(f"""
    Provide 1–2 sentence explanations for the following MCQs. Explain briefly
    why the listed correct answers are correct. Return a JSON array of strings
    in the same order as input.

    INPUT:
    {json.dumps(payload, ensure_ascii=False)}
    """).strip()
    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = getattr(resp, "text", None) or getattr(resp.candidates[0].content.parts[0], "text", str(resp))
    raw = _extract_json(text)
    try:
        arr = json.loads(raw)
    except Exception:
        arr = [line.strip() for line in text.splitlines() if line.strip()]
    if len(arr) < len(mcqs):
        arr += [""] * (len(mcqs) - len(arr))
    return [str(x) for x in arr[:len(mcqs)]]

# ---- UI Styling ----
st.set_page_config(page_title="QuizCraft", page_icon="🧠", layout="centered")
st.markdown("""
<style>
/* 🌈 General App Background */
.stApp {
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  color: #0f172a;
  font-family: 'Inter', sans-serif;
}

/* 🧠 Quiz Card */
.quiz-card {
  background: white;
  border-radius: 16px;
  padding: 20px 24px;
  margin: 16px 0;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
  border: 1px solid #e2e8f0;
  transition: all 0.3s ease-in-out;
}
.quiz-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(0,0,0,0.12);
}

/* 🏷️ Headings */
h1, h2, h3 {
  color: #1e293b !important;
  font-weight: 700;
}

/* ✨ Buttons */
.stButton > button {
  border-radius: 10px;
  padding: 10px 18px;
  background: linear-gradient(90deg, #6366f1, #3b82f6);
  color: white;
  border: none;
  font-weight: 600;
  transition: background 0.3s ease;
}
.stButton > button:hover {
  background: linear-gradient(90deg, #4f46e5, #2563eb);
}

/* 💬 Pills for correctness */
.pill {
  display:inline-block;
  padding:6px 12px;
  border-radius:999px;
  font-size:0.85rem;
  font-weight:600;
  margin-bottom:4px;
}
.ok {
  background:#dcfce7;
  color:#166534;
}
.bad {
  background:#fee2e2;
  color:#991b1b;
}

/* 🧭 Sidebar */
[data-testid="stSidebar"] {
  background: #1e293b !important;
  color: white;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
  color: white !important;
}
.stNumberInput label, .stTextInput label {
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ---- Sidebar Controls ----
with st.sidebar:
    st.title("🧪 QuizCraft")
    topic = st.text_input("Topic", placeholder="e.g., Binary Search, Photosynthesis", value="Binary Search")
    num_q = st.number_input("Number of questions", 1, 20, 5, 1)
    n_opts = st.number_input("Options per question", 2, 6, 4, 1)
    regen = st.button("✨ Generate MCQs")

# ---- Session state ----
if "mcqs" not in st.session_state:
    st.session_state.mcqs = []
if "answers" not in st.session_state:
    st.session_state.answers = []  # list[List[int]]

# ---- Generate ----
if regen:
    with st.spinner("Generating MCQs…"):
        try:
            mcqs = request_mcqs(topic, int(num_q), int(n_opts))
            st.session_state.mcqs = mcqs
            st.session_state.answers = [[] for _ in mcqs]
            st.success("MCQs ready! Scroll down 👇")
        except Exception as e:
            st.error(f"Generation failed: {e}")

st.title("🧠 QuizCraft")
st.caption("Generate multiple-choice quizzes, select your answers, and grade locally. (Gemini powers the questions.)")

# ---- Render Questions ----
if st.session_state.mcqs:
    for idx, q in enumerate(st.session_state.mcqs, start=1):
        with st.container():
            st.markdown(f"<div class='quiz-card'><h3>Q{idx}. {q['question']}</h3>", unsafe_allow_html=True)
            # Use multiselect for both single & multi-answer (simple)
            labels = [f"{i+1}) {opt}" for i, opt in enumerate(q["options"])]
            # Preselect stored answers (convert indices->labels)
            pre = [labels[i] for i in st.session_state.answers[idx-1]] if st.session_state.answers[idx-1] else []
            picked_labels = st.multiselect(
                f"Pick one or more option(s) for Q{idx}",
                options=labels,
                default=pre,
                key=f"q{idx}_ms"
            )
            # Map labels back to zero-based indices
            selected = sorted([labels.index(l) for l in picked_labels])
            st.session_state.answers[idx-1] = selected
            st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        do_grade = st.button("✅ Check answers")
    with col2:
        want_expl = st.button("💡 Get brief explanations")
    with col3:
        clear = st.button("🧹 Clear answers")

    if clear:
        st.session_state.answers = [[] for _ in st.session_state.mcqs]
        st.rerun()

    if do_grade:
        graded = grade_locally(st.session_state.mcqs, st.session_state.answers)
        st.subheader(f"Score: {graded['score']} / {graded['total']}")

        # Per-question feedback
        for r in graded["results"]:
            your = ", ".join(str(i+1) for i in r["your"]) or "—"
            corr = ", ".join(str(i+1) for i in r["correct"])
            pill = "<span class='pill ok'>Correct</span>" if r["ok"] else "<span class='pill bad'>Incorrect</span>"
            st.markdown(f"""
            <div class='quiz-card'>
              <strong>Q{r['num']}</strong> {pill}<br/>
              <em>Your choice(s):</em> {your}<br/>
              <em>Correct choice(s):</em> {corr}
            </div>
            """, unsafe_allow_html=True)

    if want_expl:
        with st.spinner("Asking Gemini for explanations…"):
            try:
                exps = request_explanations(st.session_state.mcqs)
                st.subheader("Explanations")
                for i, exp in enumerate(exps, start=1):
                    st.markdown(f"<div class='quiz-card'><strong>Q{i}.</strong> {exp}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not fetch explanations: {e}")
else:
    st.info("Use the sidebar to generate a quiz.")
