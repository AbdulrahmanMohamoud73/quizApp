#!/usr/bin/env python3
"""
local_gemini_mcq_quiz.py

Terminal MCQ quiz app that:
 - asks for a topic
 - asks Google Gemini to generate MCQs (JSON: question, options, correct_indices)
 - lets the user tick one or multiple options per question (e.g., "2" or "1,3")
 - grades locally and shows the score
 - (optional) asks Gemini for brief explanations after grading

Env:
  - Put your key in .env as: API_KEY=your_gemini_key
  - pip install --upgrade google-genai python-dotenv
"""

import os
import json
import re
import textwrap
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise SystemExit("API_KEY not set. Create a .env with API_KEY=...")

# Make sure the SDK sees the key
os.environ.setdefault("GEMINI_API_KEY", API_KEY)

try:
    from google import genai
except Exception as e:
    raise SystemExit(
        "The google-genai SDK is required. Install with:\n\n    pip install --upgrade google-genai\n\n"
        "Import error: " + str(e)
    )

# Initialize the client (SDK reads GEMINI_API_KEY)
client = genai.Client()

# Choose your model (adjust if needed)
MODEL_NAME = "gemini-2.5-flash"

# ===== Helpers =====

def _extract_json(text: str) -> str:
    """
    Try to extract the first JSON object/array from model text, even if surrounded by prose or code fences.
    """
    # Remove code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

    # Heuristic: find the first '{' or '[' and the last matching '}' or ']'
    start = min([i for i in [text.find("{"), text.find("[")] if i != -1] or [-1])
    if start == -1:
        return text  # hope it's already pure JSON
    # naive scan to the end; JSON from the model is usually small
    end = max(text.rfind("}"), text.rfind("]"))
    if end != -1 and end >= start:
        return text[start : end + 1]
    return text


def request_mcqs(topic: str, count: int = 5, options_per_q: int = 4) -> List[Dict[str, Any]]:
    """
    Ask Gemini for MCQs in strict JSON:
    [
      {
        "question": "string",
        "options": ["A", "B", "C", "D"],
        "correct_indices": [1]   # zero-based indices; may contain multiple indices
      },
      ...
    ]
    """
    prompt = textwrap.dedent(f"""
    You are a quiz generator. Produce EXACT JSON (no commentary, no code fences) for {count} MCQs on "{topic}".
    Each question must have {options_per_q} options and may have one OR multiple correct answers.
    Use this schema precisely (zero-based indices):

    [
      {{
        "question": "Concise, clear question?",
        "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
        "correct_indices": [0]  // or [0,2] for multi-correct
      }}
    ]

    Rules:
    - Return ONLY JSON, no extra text.
    - Keep questions and options short and unambiguous.
    - Avoid trick questions; ensure correctness is widely accepted.
    """).strip()

    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = getattr(resp, "text", None)
    if text is None:
        try:
            text = resp.candidates[0].content.parts[0].text
        except Exception:
            text = str(resp)

    raw = _extract_json(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse MCQ JSON: {e}\nRaw:\n{text}")

    # Basic validation + trimming
    mcqs = []
    for item in data[:count]:
        q = {
            "question": str(item.get("question", "")).strip(),
            "options": [str(x).strip() for x in item.get("options", [])][:options_per_q],
            "correct_indices": list(map(int, item.get("correct_indices", []))),
        }
        if not q["question"] or len(q["options"]) != options_per_q:
            continue
        # Deduplicate + sort indices, keep only in-range
        q["correct_indices"] = sorted({i for i in q["correct_indices"] if 0 <= i < options_per_q})
        if not q["correct_indices"]:
            # If model forgot, default to first option to avoid crashes (not ideal, but robust)
            q["correct_indices"] = [0]
        mcqs.append(q)

    if not mcqs:
        raise RuntimeError("Model returned empty/invalid MCQs.")

    return mcqs


def prompt_multi_indices(n_options: int) -> List[int]:
    """
    Read user input like "2" or "1,3" or "2 4" and return zero-based indices.
    """
    while True:
        raw = input("Your choice(s) (e.g., 2 or 1,3): ").strip()
        if not raw:
            print("Please enter at least one option number.")
            continue
        # Allow separators: comma/space/slash
        parts = re.split(r"[,\s/]+", raw)
        try:
            picks = sorted({int(p) for p in parts if p != ""})
        except ValueError:
            print("Only numbers please, e.g., 2 or 1,3")
            continue
        if any(p < 1 or p > n_options for p in picks):
            print(f"Pick numbers between 1 and {n_options}.")
            continue
        # Convert to zero-based
        return [p - 1 for p in picks]


def grade_locally(mcqs: List[Dict[str, Any]], user_answers: List[List[int]]) -> Dict[str, Any]:
    """
    Compare user selected indices vs correct_indices.
    Returns dict with per-question results and total score.
    """
    results = []
    correct_count = 0
    for i, (q, ans) in enumerate(zip(mcqs, user_answers), start=1):
        correct_set = set(q["correct_indices"])
        ans_set = set(ans)
        is_correct = ans_set == correct_set
        if is_correct:
            correct_count += 1
        results.append({
            "number": i,
            "question": q["question"],
            "options": q["options"],
            "your_indices": sorted(ans),
            "correct_indices": sorted(q["correct_indices"]),
            "is_correct": is_correct,
        })
    return {
        "results": results,
        "score": correct_count,
        "total": len(mcqs),
    }


def request_explanations(mcqs: List[Dict[str, Any]]) -> List[str]:
    """
    (Optional) Ask Gemini for a 1-2 sentence explanation per question/answer.
    Returns a list of explanations aligned to mcqs order.
    """
    payload = []
    for q in mcqs:
        correct_opts = [q["options"][i] for i in q["correct_indices"]]
        payload.append({
            "question": q["question"],
            "correct_answers": correct_opts,
            "all_options": q["options"],
        })

    prompt = textwrap.dedent(f"""
    Provide 1–2 sentence explanations for the following MCQs. For each item, explain briefly
    why the listed correct answers are correct (avoid revealing extra answers).
    Return a JSON array of strings, same order as input, one explanation per question.

    INPUT JSON:
    {json.dumps(payload, ensure_ascii=False)}
    """).strip()

    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = getattr(resp, "text", None)
    if text is None:
        try:
            text = resp.candidates[0].content.parts[0].text
        except Exception:
            text = str(resp)

    raw = _extract_json(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: split lines
        data = [line.strip() for line in text.splitlines() if line.strip()]

    # Ensure length matches
    if len(data) < len(mcqs):
        data += [""] * (len(mcqs) - len(data))
    return [str(x) for x in data[:len(mcqs)]]


# ===== Main =====

def main():
    print("=== Local Gemini MCQ Quiz (terminal) ===")
    topic = input("Enter a topic for the quiz: ").strip()
    if not topic:
        print("No topic entered. Exiting.")
        return

    # Number of questions
    while True:
        try:
            num_q = int(input("How many questions? (default 5): ").strip() or "5")
            if num_q > 0:
                break
            print("Please enter a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")

    # Options per question
    while True:
        try:
            n_opts = int(input("How many options per question? (default 4): ").strip() or "4")
            if n_opts >= 2:
                break
            print("Please enter an integer >= 2.")
        except ValueError:
            print("Please enter a valid integer.")

    print(f"\nGenerating {num_q} MCQ(s) about '{topic}' — please wait...\n")
    try:
        mcqs = request_mcqs(topic, num_q, n_opts)
    except Exception as e:
        print("Error generating MCQs:", e)
        return

    # Ask user the questions
    user_answers: List[List[int]] = []
    for idx, q in enumerate(mcqs, start=1):
        print(f"\nQ{idx}. {q['question']}")
        for i, opt in enumerate(q["options"], start=1):
            print(f"  {i}) {opt}")
        print("  (One or more may be correct)")
        picks = prompt_multi_indices(len(q["options"]))
        user_answers.append(picks)

    # Grade locally
    print("\nChecking answers...\n")
    graded = grade_locally(mcqs, user_answers)
    print(f"Score: {graded['score']} / {graded['total']}\n")

    # Show per-question feedback
    for r in graded["results"]:
        your = ", ".join(str(i+1) for i in r["your_indices"]) or "—"
        corr = ", ".join(str(i+1) for i in r["correct_indices"])
        status = "✅ Correct" if r["is_correct"] else "❌ Incorrect"
        print(f"Q{r['number']}: {status}")
        print(f"   Your choice(s): {your}")
        print(f"   Correct choice(s): {corr}")

    # Optional explanations
    want_expl = input("\nWould you like brief explanations? (y/N): ").strip().lower() == "y"
    if want_expl:
        try:
            exps = request_explanations(mcqs)
            print("\n=== Explanations ===")
            for i, exp in enumerate(exps, start=1):
                print(f"Q{i}: {exp}")
        except Exception as e:
            print("Could not fetch explanations:", e)

    print("\n=== End ===")


if __name__ == "__main__":
    main()
