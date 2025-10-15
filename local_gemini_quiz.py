#!/usr/bin/env python3
"""
local_gemini_quiz.py

Local terminal quiz app that:
 - asks for a topic
 - asks Google Gemini to generate quiz questions
 - lets the user answer the questions
 - asks Gemini to grade the answers and provide correct answers + explanations

WARNING: This example uses a placeholder API key hard-coded into the script.
Do NOT use a real production key in code you share. Replace the placeholder with your key locally.
"""

import os
import textwrap
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access your API key
api_key = os.getenv("API_KEY")
try:
    from google import genai
except Exception as e:
    raise SystemExit(
        "The google-genai SDK is required. Install with:\n\n    pip install --upgrade google-genai\n\n"
        "Import error: " + str(e)
    )

# === EDIT THIS: placeholder API key (per your request) ===
API_KEY = os.getenv("API_KEY")  # <-- replace with your key locally if running
# =======================================================

# You can either set the environment variable used by the SDK or set it programmatically:
os.environ.setdefault("GEMINI_API_KEY", API_KEY)

# Initialize the client (SDK will read GEMINI_API_KEY)
client = genai.Client()

# Choose a model available to you; gemini-2.5-flash is example — change if needed.
MODEL_NAME = "gemini-2.5-flash"  # adjust if you need a different model

def generate_questions(topic: str, count: int = 5) -> List[str]:
    """
    Ask Gemini to generate `count` numbered quiz questions about `topic`.
    Returns a list of question strings.
    """
    prompt = (
        f"Generate {count} concise, clear quiz-style questions about the topic: \"{topic}\".\n"
        "Respond ONLY with a numbered list of questions (1., 2., ...). Do not include answers or extra commentary."
    )
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    # Depending on SDK version, response text is available as resp.text
    text = getattr(resp, "text", None)
    if text is None:
        # Fallback attempt to inspect structure
        try:
            text = resp.candidates[0].content.parts[0].text
        except Exception:
            text = str(resp)

    # Parse numbered list into clean question strings
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    questions = []
    current = ""
    for line in lines:
        # If line starts with "1." or "1)" etc, treat as new question
        if line.lstrip().split()[0].rstrip(".").rstrip(")").isdigit():
            # finish previous
            if current:
                questions.append(current.strip())
            # remove the leading numbering
            parts = line.split(".", 1)
            if len(parts) == 2:
                current = parts[1].strip()
            else:
                # try parentheses style
                parts = line.split(")", 1)
                current = parts[1].strip() if len(parts) == 2 else line
        else:
            # continuation of the previous question (rare)
            current += " " + line
    if current:
        questions.append(current.strip())

    # If parsing fails, and we got fewer questions than requested, try a simpler split
    if len(questions) < max(1, min(3, count)):
        # split by blank lines and hope for the best
        alt = [p.strip() for p in text.split("\n\n") if p.strip()]
        questions = [q for q in alt][:count]

    return questions[:count]

def grade_answers(questions: List[str], user_answers: List[str]) -> str:
    """
    Send the questions + user answers to Gemini asking for grading.
    We'll request a clear, numbered output containing for each question:
      - "Correct" or "Incorrect"
      - The correct answer
      - A brief explanation (1-2 sentences)
    We request the response as a numbered list matching the question numbers.
    Returns the raw text response from Gemini.
    """
    assert len(questions) == len(user_answers)

    # Build the prompt
    pairs = []
    for i, (q, a) in enumerate(zip(questions, user_answers), start=1):
        pairs.append(f"{i}) Question: {q}\nYour answer: {a if a.strip() else '(no answer)'}")
    pairs_text = "\n\n".join(pairs)

    prompt = textwrap.dedent(f"""
    You are an assistant that grades short-answer quiz responses.
    For each of the following numbered questions and the student's answer, respond with:
      - The question number
      - 'Result: Correct' or 'Result: Incorrect'
      - 'Correct answer:' followed by the best short correct answer
      - 'Explanation:' one or two brief sentences explaining why the correct answer is correct (concise)
    Output MUST be a numbered list that directly corresponds to the input numbers (1., 2., 3., ...).
    Keep answers terse and unambiguous.

    Here are the questions and the student's answers:

    {pairs_text}
    """).strip()

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    # Attempt to extract text
    text = getattr(resp, "text", None)
    if text is None:
        try:
            text = resp.candidates[0].content.parts[0].text
        except Exception:
            text = str(resp)

    return text

def prompt_int(prompt_text: str, default: int = None) -> int:
    while True:
        val = input(prompt_text)
        if not val.strip() and default is not None:
            return default
        try:
            v = int(val)
            if v > 0:
                return v
            print("Please enter a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")

def main():
    print("=== Local Gemini Quiz (terminal) ===")
    topic = input("Enter a topic for the quiz: ").strip()
    if not topic:
        print("No topic entered. Exiting.")
        return

    num_q = prompt_int("How many questions would you like? (default 5): ", default=5)

    print(f"\nGenerating {num_q} question(s) about '{topic}' — please wait...\n")
    try:
        questions = generate_questions(topic, num_q)
    except Exception as e:
        print("Error while requesting questions from Gemini:", e)
        return

    if not questions:
        print("Gemini returned no questions. Try again or change the topic.")
        return

    print("Questions:")
    for i, q in enumerate(questions, start=1):
        print(f"{i}. {q}")

    print("\nType your answers. Press Enter when done with each answer.\n")
    user_answers = []
    for i, q in enumerate(questions, start=1):
        ans = input(f"Answer {i}: ").strip()
        user_answers.append(ans)

    print("\nGrading your answers (Gemini)...\n")
    try:
        grading = grade_answers(questions, user_answers)
    except Exception as e:
        print("Error while grading with Gemini:", e)
        return

    print("=== Grading Result ===\n")
    print(grading)
    print("\n=== End ===")

if __name__ == "__main__":
    main()
