import json
import os
import re
from threading import Lock
from uuid import uuid4

import openai
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI


load_dotenv()

app = Flask(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:0.5b")
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# MVPではDBを使わず、問題と提出結果をメモリに一時保存する。
# サーバーを終了すると内容は消える。
question_sets = {}
submissions = {}
store_lock = Lock()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


def problem(status, title, detail, invalid_params=None):
    body = {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": request.path,
    }
    if invalid_params:
        body["invalid_params"] = invalid_params

    response = jsonify(body)
    response.status_code = status
    response.content_type = "application/problem+json"
    return response


def get_llm_client():
    """環境変数に応じたLLMクライアントとモデル名を返す。"""
    if LLM_PROVIDER == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        client = OpenAI(
            api_key=api_key,
            base_url=GEMINI_BASE_URL,
            timeout=60.0,
            max_retries=0,
        )
        return client, GEMINI_MODEL

    if LLM_PROVIDER == "ollama":
        client = OpenAI(
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
            timeout=180.0,
            max_retries=0,
        )
        return client, OLLAMA_MODEL

    raise RuntimeError("LLM_PROVIDER must be gemini or ollama")


def extract_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def validate_generated_question_set(data):
    if not isinstance(data, dict):
        raise ValueError("生成結果がJSONオブジェクトではありません")

    passage = data.get("passage")
    questions = data.get("questions")
    if not isinstance(passage, str) or not passage.strip():
        raise ValueError("英文がありません")
    if not isinstance(questions, list) or len(questions) != 2:
        raise ValueError("設問数が2問ではありません")

    normalized_questions = []
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError("設問の形式が不正です")

        question = item.get("question")
        choices = item.get("choices")
        correct_choice = item.get("correct_choice")
        evidence = item.get("evidence")
        explanation = item.get("explanation")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("設問文がありません")
        if not isinstance(choices, list) or len(choices) != 4:
            raise ValueError("選択肢数が4つではありません")
        if any(not isinstance(choice, str) or not choice.strip() for choice in choices):
            raise ValueError("空の選択肢があります")
        if len(set(choices)) != 4:
            raise ValueError("選択肢が重複しています")
        if isinstance(correct_choice, bool) or not isinstance(correct_choice, int) or not 0 <= correct_choice <= 3:
            raise ValueError("正解番号が0から3の範囲ではありません")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("正解の根拠がありません")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("日本語解説がありません")

        normalized_questions.append(
            {
                "question_id": f"q{index}",
                "question": question.strip(),
                "choices": choices,
                "correct_choice": correct_choice,
                "evidence": evidence.strip(),
                "explanation": explanation.strip(),
            }
        )

    return {
        "passage": passage.strip(),
        "questions": normalized_questions,
    }


def create_prompt(level, document_format):
    return f"""
Create one original English reading comprehension exercise.

Requirements:
- Difficulty: {level}
- Format: business {document_format}
- Passage length: 120 to 160 English words
- Create exactly 2 questions
- Each question must have exactly 4 unique choices
- correct_choice must be an integer from 0 to 3
- evidence must be an exact quotation from the passage
- explanation must be written in Japanese
- The correct answer, evidence, and explanation must be consistent
- Do not reproduce an existing TOEIC question
- Return only valid JSON without Markdown

Required JSON structure:
{{
  "passage": "English passage",
  "questions": [
    {{
      "question": "English question",
      "choices": ["choice 1", "choice 2", "choice 3", "choice 4"],
      "correct_choice": 0,
      "evidence": "Exact quotation from the passage",
      "explanation": "Japanese explanation"
    }},
    {{
      "question": "English question",
      "choices": ["choice 1", "choice 2", "choice 3", "choice 4"],
      "correct_choice": 0,
      "evidence": "Exact quotation from the passage",
      "explanation": "Japanese explanation"
    }}
  ]
}}
"""


@app.route("/api/v1/question-sets", methods=["POST"])
def create_question_set():
    if not request.is_json:
        return problem(400, "Bad Request", "Content-Typeをapplication/jsonにしてください。")

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return problem(400, "Bad Request", "正しいJSONを送信してください。")
    if set(data) - {"level", "format"}:
        return problem(422, "Validation Error", "許可されていない項目が含まれています。")

    level = data.get("level")
    document_format = data.get("format")
    invalid_params = []

    if level not in {"beginner", "intermediate"}:
        invalid_params.append({"name": "level", "reason": "beginnerまたはintermediateを指定してください。"})
    if document_format != "email":
        invalid_params.append({"name": "format", "reason": "MVPではemailのみ指定できます。"})
    if invalid_params:
        return problem(422, "Validation Error", "入力値が仕様を満たしていません。", invalid_params)

    try:
        client, model = get_llm_client()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You create accurate English exercises and return only valid JSON.",
                },
                {"role": "user", "content": create_prompt(level, document_format)},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        generated = extract_json(completion.choices[0].message.content)
        validated = validate_generated_question_set(generated)

    except (json.JSONDecodeError, ValueError) as error:
        app.logger.warning("LLM output validation failed: %s", error)
        return problem(502, "LLM Failure", "AIが正しい形式の問題を生成できませんでした。もう一度お試しください。")
    except openai.APITimeoutError:
        app.logger.exception("LLM API timed out (provider=%s)", LLM_PROVIDER)
        return problem(504, "LLM Timeout", "問題の生成に時間がかかっています。もう一度お試しください。")
    except (openai.APIConnectionError, openai.APIStatusError) as error:
        app.logger.exception("LLM API failed (provider=%s): %s", LLM_PROVIDER, error)
        return problem(502, "LLM Failure", "AIサービスとの通信に失敗しました。もう一度お試しください。")
    except RuntimeError as error:
        app.logger.error("Configuration error: %s", error)
        return problem(500, "Internal Server Error", "サーバーの設定が完了していません。")
    except Exception:
        app.logger.exception("Unexpected question generation error")
        return problem(500, "Internal Server Error", "予期しないエラーが発生しました。")

    question_set_id = str(uuid4())
    stored_question_set = {
        "question_set_id": question_set_id,
        "level": level,
        "format": document_format,
        **validated,
    }

    with store_lock:
        question_sets[question_set_id] = stored_question_set

    public_question_set = {
        "question_set_id": question_set_id,
        "level": level,
        "format": document_format,
        "passage": validated["passage"],
        "questions": [
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "choices": item["choices"],
            }
            for item in validated["questions"]
        ],
    }

    response = jsonify(public_question_set)
    response.status_code = 201
    response.headers["Location"] = f"/api/v1/question-sets/{question_set_id}"
    return response


@app.route("/api/v1/question-sets/<question_set_id>/submissions", methods=["POST"])
def create_submission(question_set_id):
    if not request.is_json:
        return problem(400, "Bad Request", "Content-Typeをapplication/jsonにしてください。")

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return problem(400, "Bad Request", "正しいJSONを送信してください。")
    if set(data) != {"answers"}:
        return problem(422, "Validation Error", "answersだけを送信してください。")

    with store_lock:
        question_set = question_sets.get(question_set_id)
    if not question_set:
        return problem(404, "Not Found", "指定された問題セットが存在しません。")

    answers = data.get("answers")
    if not isinstance(answers, list) or len(answers) != 2:
        return problem(
            422,
            "Validation Error",
            "2問分の回答を送信してください。",
            [{"name": "answers", "reason": "回答数は2件必要です。"}],
        )

    answer_map = {}
    for index, answer in enumerate(answers):
        if not isinstance(answer, dict):
            return problem(422, "Validation Error", "回答の形式が不正です。")
        if set(answer) != {"question_id", "selected_choice"}:
            return problem(422, "Validation Error", "回答項目が仕様と一致しません。")
        question_id = answer.get("question_id")
        selected_choice = answer.get("selected_choice")
        if question_id in answer_map:
            return problem(422, "Validation Error", "同じ設問への回答が重複しています。")
        if isinstance(selected_choice, bool) or not isinstance(selected_choice, int) or not 0 <= selected_choice <= 3:
            return problem(422, "Validation Error", f"answers[{index}].selected_choiceは0から3で指定してください。")
        answer_map[question_id] = selected_choice

    expected_ids = {item["question_id"] for item in question_set["questions"]}
    if set(answer_map) != expected_ids:
        return problem(422, "Validation Error", "問題セットに含まれる2問へ回答してください。")

    results = []
    score = 0
    for item in question_set["questions"]:
        selected_choice = answer_map[item["question_id"]]
        correct = selected_choice == item["correct_choice"]
        if correct:
            score += 1
        results.append(
            {
                "question_id": item["question_id"],
                "selected_choice": selected_choice,
                "correct": correct,
                "correct_choice": item["correct_choice"],
                "evidence": item["evidence"],
                "explanation": item["explanation"],
            }
        )

    submission_id = str(uuid4())
    submission = {
        "submission_id": submission_id,
        "question_set_id": question_set_id,
        "score": score,
        "total": 2,
        "results": results,
    }
    with store_lock:
        submissions[submission_id] = submission

    return jsonify(submission), 201


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
