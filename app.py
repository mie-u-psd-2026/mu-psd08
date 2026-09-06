import json
import os
import re
import time
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
time_attacks = {}
detailed_explanations = {}
store_lock = Lock()

TIME_ATTACK_SET_COUNT = 5
TIME_ATTACK_TIME_LIMIT_SECONDS = 600
GENERATION_MAX_ATTEMPTS = 2


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
    passage_translation = data.get("passage_translation")
    questions = data.get("questions")
    if not isinstance(passage, str) or not passage.strip():
        raise ValueError("英文がありません")
    if not isinstance(passage_translation, str) or not passage_translation.strip():
        raise ValueError("英文の日本語訳がありません")
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
        if evidence.strip() not in passage:
            raise ValueError(f"設問{index}の根拠が本文に含まれていません")
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
        "passage_translation": passage_translation.strip(),
        "questions": normalized_questions,
    }

LEVEL_INSTRUCTIONS = {
    "beginner": (
        "Use common everyday and business vocabulary. "
        "Use short and simple sentences. "
        "Create direct questions whose answers are explicitly stated in the passage."
    ),
    "intermediate": (
        "Use moderately varied business vocabulary. "
        "Include some complex sentences. "
        "Create a mixture of detail and purpose questions."
    ),
    "advanced": (
        "Use advanced business vocabulary and complex sentence structures. "
        "Include implicit relationships. "
        "Create at least one inference question."
    ),
}

def create_prompt(level, document_format):
    difficulty_instruction = LEVEL_INSTRUCTIONS[level]

    return f"""
Create one original English reading comprehension exercise.

Requirements:
- Difficulty: {level}
- Difficulty guideline: {difficulty_instruction}
- Format: business {document_format}
- Passage length: 120 to 160 English words
- passage_translation must be a natural Japanese translation of the entire passage
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
  "passage_translation": "Japanese translation of the entire passage",
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


def generate_question_set_with_llm(level, document_format):
    """LLMで1つの英文と2問を生成し、形式不正時は再生成する。"""
    client, model = get_llm_client()
    last_error = None

    for attempt in range(1, GENERATION_MAX_ATTEMPTS + 1):
        try:
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
            return validate_generated_question_set(generated)
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            app.logger.warning(
                "LLM output validation failed on attempt %s/%s: %s",
                attempt,
                GENERATION_MAX_ATTEMPTS,
                error,
            )

    raise last_error


def public_question_set(question_set):
    """正解・根拠・解説を除いた、画面表示用の問題セットを返す。"""
    return {
        "question_set_id": question_set["question_set_id"],
        "level": question_set["level"],
        "format": question_set["format"],
        "passage": question_set["passage"],
        "questions": [
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "choices": item["choices"],
            }
            for item in question_set["questions"]
        ],
    }


def llm_error_response(error):
    """問題生成時の例外を共通のHTTPエラーへ変換する。"""
    if isinstance(error, (json.JSONDecodeError, ValueError)):
        app.logger.warning("LLM output validation failed: %s", error)
        return problem(502, "LLM Failure", "AIが正しい形式の問題を生成できませんでした。もう一度お試しください。")
    if isinstance(error, openai.APITimeoutError):
        app.logger.exception("LLM API timed out (provider=%s)", LLM_PROVIDER)
        return problem(504, "LLM Timeout", "問題の生成に時間がかかっています。もう一度お試しください。")
    if isinstance(error, (openai.APIConnectionError, openai.APIStatusError)):
        app.logger.exception("LLM API failed (provider=%s): %s", LLM_PROVIDER, error)
        return problem(502, "LLM Failure", "AIサービスとの通信に失敗しました。もう一度お試しください。")
    if isinstance(error, RuntimeError):
        app.logger.error("Configuration error: %s", error)
        return problem(500, "Internal Server Error", "サーバーの設定が完了していません。")

    app.logger.exception("Unexpected question generation error")
    return problem(500, "Internal Server Error", "予期しないエラーが発生しました。")


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

    if level not in {"beginner", "intermediate", "advanced"}:
        invalid_params.append({"name": "level", "reason": "beginner、intermediateまたはadvancedを指定してください。"})
    if document_format != "email":
        invalid_params.append({"name": "format", "reason": "MVPではemailのみ指定できます。"})
    if invalid_params:
        return problem(422, "Validation Error", "入力値が仕様を満たしていません。", invalid_params)

    try:
        validated = generate_question_set_with_llm(level, document_format)
    except Exception as error:
        return llm_error_response(error)

    question_set_id = str(uuid4())
    stored_question_set = {
        "question_set_id": question_set_id,
        "level": level,
        "format": document_format,
        **validated,
    }

    with store_lock:
        question_sets[question_set_id] = stored_question_set

    response = jsonify(public_question_set(stored_question_set))
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
        "passage_translation": question_set["passage_translation"],
        "results": results,
    }
    with store_lock:
        submissions[submission_id] = submission

    return jsonify(submission), 201


def find_explanation_target(question_set_id):
    """詳細説明対象と、1回制限に使うキー、採点済みかを返す。"""
    question_set = question_sets.get(question_set_id)
    if question_set:
        graded = any(
            item["question_set_id"] == question_set_id
            for item in submissions.values()
        )
        return question_set, f"question_set:{question_set_id}", graded

    for time_attack_id, time_attack in time_attacks.items():
        question_set = next(
            (
                item
                for item in time_attack["question_sets"]
                if item["question_set_id"] == question_set_id
            ),
            None,
        )
        if question_set:
            # タイムアタックは5セット全体で詳細説明を1回だけ利用できる。
            return question_set, f"time_attack:{time_attack_id}", time_attack["finished"]

    return None, None, False


def create_detailed_explanation_prompt(passage, selected_text):
    return f"""
You are an English teacher helping a Japanese learner.

Full passage:
{passage}

Selected sentence:
{selected_text}

Explain only the selected sentence in clear Japanese. Include:
1. A natural Japanese translation
2. The sentence's grammatical structure
3. Important words, phrases, and expressions

Keep the explanation concise and useful for learning. Return plain Japanese text only.
"""


@app.route("/api/v1/question-sets/<question_set_id>/explanations", methods=["POST"])
def create_detailed_explanation(question_set_id):
    """採点後、本文から選択した1文をAIが日本語で詳しく説明する。"""
    if not request.is_json:
        return problem(400, "Bad Request", "Content-Typeをapplication/jsonにしてください。")

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return problem(400, "Bad Request", "正しいJSONを送信してください。")
    if set(data) != {"selected_text"}:
        return problem(422, "Validation Error", "selected_textだけを送信してください。")

    selected_text = data.get("selected_text")
    if not isinstance(selected_text, str) or not selected_text.strip():
        return problem(422, "Validation Error", "説明する英文を選択してください。")
    selected_text = selected_text.strip()
    if len(selected_text) > 500:
        return problem(422, "Validation Error", "選択できる英文は500文字以内です。")

    with store_lock:
        question_set, usage_key, graded = find_explanation_target(question_set_id)
        if not question_set:
            return problem(404, "Not Found", "指定された問題セットが存在しません。")
        if not graded:
            return problem(409, "Conflict", "詳細説明は採点後に利用できます。")
        if selected_text not in question_set["passage"]:
            return problem(422, "Validation Error", "本文に含まれる英文を選択してください。")
        if usage_key in detailed_explanations:
            return problem(409, "Conflict", "詳細説明は1度だけ利用できます。")

        # 同時クリックによる複数回送信を防ぐため、LLM通信前に利用中として確保する。
        detailed_explanations[usage_key] = {"status": "processing"}

    try:
        client, model = get_llm_client()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You explain English grammar and phrases accurately in Japanese.",
                },
                {
                    "role": "user",
                    "content": create_detailed_explanation_prompt(
                        question_set["passage"], selected_text
                    ),
                },
            ],
            temperature=0.2,
        )
        explanation_text = (completion.choices[0].message.content or "").strip()
        if not explanation_text:
            raise ValueError("詳細説明が空です")
    except Exception as error:
        # 生成に失敗した場合は「1回」を消費せず、再試行を許可する。
        with store_lock:
            if detailed_explanations.get(usage_key, {}).get("status") == "processing":
                detailed_explanations.pop(usage_key, None)

        if isinstance(error, openai.APITimeoutError):
            app.logger.exception("Detailed explanation timed out")
            return problem(504, "LLM Timeout", "詳細説明の生成に時間がかかっています。もう一度お試しください。")
        if isinstance(error, (openai.APIConnectionError, openai.APIStatusError)):
            app.logger.exception("Detailed explanation API failed: %s", error)
            return problem(502, "LLM Failure", "AIサービスとの通信に失敗しました。もう一度お試しください。")
        if isinstance(error, RuntimeError):
            app.logger.error("Configuration error: %s", error)
            return problem(500, "Internal Server Error", "サーバーの設定が完了していません。")

        app.logger.exception("Detailed explanation generation failed")
        return problem(502, "LLM Failure", "詳細説明を生成できませんでした。もう一度お試しください。")

    result = {
        "explanation_id": str(uuid4()),
        "question_set_id": question_set_id,
        "selected_text": selected_text,
        "explanation": explanation_text,
        "remaining_uses": 0,
    }
    with store_lock:
        detailed_explanations[usage_key] = {"status": "completed", **result}

    return jsonify(result), 201


def validate_time_attack_answers(question_set, answers, allow_unanswered=False):
    """タイムアタックの現在セットに対する回答を検証する。"""
    if not isinstance(answers, list) or len(answers) > 2:
        raise ValueError("回答数は0件から2件で指定してください。")
    if not allow_unanswered and len(answers) != 2:
        raise ValueError("2問分の回答を送信してください。")

    answer_map = {}
    expected_ids = {item["question_id"] for item in question_set["questions"]}
    for answer in answers:
        if not isinstance(answer, dict) or set(answer) != {"question_id", "selected_choice"}:
            raise ValueError("回答項目が仕様と一致しません。")
        question_id = answer.get("question_id")
        selected_choice = answer.get("selected_choice")
        if question_id not in expected_ids or question_id in answer_map:
            raise ValueError("設問IDが不正または重複しています。")
        if selected_choice is not None and (
            isinstance(selected_choice, bool)
            or not isinstance(selected_choice, int)
            or not 0 <= selected_choice <= 3
        ):
            raise ValueError("selected_choiceは0から3またはnullで指定してください。")
        if selected_choice is None and not allow_unanswered:
            raise ValueError("次へ進むには2問すべてに回答してください。")
        answer_map[question_id] = selected_choice

    if not allow_unanswered and set(answer_map) != expected_ids:
        raise ValueError("2問すべてに回答してください。")
    return answer_map


def time_attack_result(time_attack, timed_out):
    """未回答を不正解として、タイムアタック全10問を採点する。"""
    results = []
    score = 0
    answer_maps = time_attack["answers"]

    for set_index, question_set in enumerate(time_attack["question_sets"]):
        answer_map = answer_maps.get(question_set["question_set_id"], {})
        for item in question_set["questions"]:
            selected_choice = answer_map.get(item["question_id"])
            correct = selected_choice == item["correct_choice"]
            if correct:
                score += 1
            results.append(
                {
                    "set_number": set_index + 1,
                    "question_set_id": question_set["question_set_id"],
                    "passage": question_set["passage"],
                    "passage_translation": question_set["passage_translation"],
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "choices": item["choices"],
                    "selected_choice": selected_choice,
                    "correct": correct,
                    "correct_choice": item["correct_choice"],
                    "evidence": item["evidence"],
                    "explanation": item["explanation"],
                }
            )

    result = {
        "time_attack_id": time_attack["time_attack_id"],
        "score": score,
        "total": TIME_ATTACK_SET_COUNT * 2,
        "timed_out": timed_out,
        "elapsed_seconds": min(
            int(time.monotonic() - time_attack["started_at"]),
            TIME_ATTACK_TIME_LIMIT_SECONDS,
        ),
        "results": results,
    }
    time_attack["finished"] = True
    time_attack["result"] = result
    return result


@app.route("/api/v1/time-attacks", methods=["POST"])
def create_time_attack():
    """中級2問の問題セットを5つ生成し、最初のセットだけを返す。"""
    if not request.is_json:
        return problem(400, "Bad Request", "Content-Typeをapplication/jsonにしてください。")

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return problem(400, "Bad Request", "正しいJSONを送信してください。")
    if set(data) - {"level", "format"}:
        return problem(422, "Validation Error", "許可されていない項目が含まれています。")
    if data.get("level") != "intermediate" or data.get("format") != "email":
        return problem(422, "Validation Error", "タイムアタックは中級・email形式のみ利用できます。")

    generated_sets = []
    try:
        for _ in range(TIME_ATTACK_SET_COUNT):
            validated = generate_question_set_with_llm("intermediate", "email")
            question_set_id = str(uuid4())
            generated_sets.append(
                {
                    "question_set_id": question_set_id,
                    "level": "intermediate",
                    "format": "email",
                    **validated,
                }
            )
    except Exception as error:
        return llm_error_response(error)

    time_attack_id = str(uuid4())
    time_attack = {
        "time_attack_id": time_attack_id,
        "question_sets": generated_sets,
        "current_set_index": 0,
        "answers": {},
        "started_at": time.monotonic(),
        "finished": False,
        "result": None,
    }
    with store_lock:
        time_attacks[time_attack_id] = time_attack

    return jsonify(
        {
            "time_attack_id": time_attack_id,
            "current_set": 1,
            "total_sets": TIME_ATTACK_SET_COUNT,
            "total_questions": TIME_ATTACK_SET_COUNT * 2,
            "time_limit_seconds": TIME_ATTACK_TIME_LIMIT_SECONDS,
            "question_set": public_question_set(generated_sets[0]),
        }
    ), 201


@app.route("/api/v1/time-attacks/<time_attack_id>/answers", methods=["POST"])
def submit_time_attack_answers(time_attack_id):
    """現在セットの回答を保存し、次のセットまたは最終結果を返す。"""
    if not request.is_json:
        return problem(400, "Bad Request", "Content-Typeをapplication/jsonにしてください。")
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or set(data) != {"question_set_id", "answers"}:
        return problem(422, "Validation Error", "question_set_idとanswersを送信してください。")

    with store_lock:
        time_attack = time_attacks.get(time_attack_id)
    if not time_attack:
        return problem(404, "Not Found", "指定されたタイムアタックが存在しません。")
    if time_attack["finished"]:
        return problem(409, "Conflict", "このタイムアタックはすでに終了しています。")

    if time.monotonic() - time_attack["started_at"] >= TIME_ATTACK_TIME_LIMIT_SECONDS:
        with store_lock:
            result = time_attack_result(time_attack, timed_out=True)
        return jsonify({"finished": True, "result": result}), 200

    current_index = time_attack["current_set_index"]
    question_set_id = data.get("question_set_id")
    submitted_index = next(
        (
            index
            for index, item in enumerate(time_attack["question_sets"])
            if item["question_set_id"] == question_set_id
        ),
        None,
    )
    if submitted_index is None or submitted_index > current_index:
        return problem(409, "Conflict", "まだ表示されていない問題セットには回答できません。")
    current_question_set = time_attack["question_sets"][submitted_index]

    try:
        answer_map = validate_time_attack_answers(current_question_set, data.get("answers"))
    except ValueError as error:
        return problem(422, "Validation Error", str(error))

    with store_lock:
        time_attack["answers"][current_question_set["question_set_id"]] = answer_map
        if submitted_index < current_index:
            return jsonify({"finished": False, "updated": True}), 200
        time_attack["current_set_index"] += 1
        if time_attack["current_set_index"] == TIME_ATTACK_SET_COUNT:
            result = time_attack_result(time_attack, timed_out=False)
            return jsonify({"finished": True, "result": result}), 200
        next_index = time_attack["current_set_index"]
        next_question_set = time_attack["question_sets"][next_index]

    return jsonify(
        {
            "finished": False,
            "current_set": next_index + 1,
            "total_sets": TIME_ATTACK_SET_COUNT,
            "question_set": public_question_set(next_question_set),
        }
    ), 200


@app.route("/api/v1/time-attacks/<time_attack_id>/finish", methods=["POST"])
def finish_time_attack(time_attack_id):
    """時間切れ時の回答途中データを保存し、全10問を採点する。"""
    if not request.is_json:
        return problem(400, "Bad Request", "Content-Typeをapplication/jsonにしてください。")
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or set(data) != {"answer_sets"}:
        return problem(422, "Validation Error", "回答データの形式が不正です。")

    with store_lock:
        time_attack = time_attacks.get(time_attack_id)
    if not time_attack:
        return problem(404, "Not Found", "指定されたタイムアタックが存在しません。")
    if time_attack["finished"]:
        return jsonify(time_attack["result"]), 200

    answer_sets = data.get("answer_sets")
    if not isinstance(answer_sets, list):
        return problem(422, "Validation Error", "answer_setsは配列で指定してください。")

    validated_answer_sets = {}
    revealed_sets = time_attack["question_sets"][: time_attack["current_set_index"] + 1]
    revealed_map = {item["question_set_id"]: item for item in revealed_sets}
    try:
        for answer_set in answer_sets:
            if not isinstance(answer_set, dict) or set(answer_set) != {"question_set_id", "answers"}:
                raise ValueError("各回答セットにはquestion_set_idとanswersが必要です。")
            question_set_id = answer_set.get("question_set_id")
            if question_set_id not in revealed_map or question_set_id in validated_answer_sets:
                raise ValueError("問題セットIDが不正または重複しています。")
            validated_answer_sets[question_set_id] = validate_time_attack_answers(
                revealed_map[question_set_id],
                answer_set.get("answers", []),
                allow_unanswered=True,
            )
    except ValueError as error:
        return problem(422, "Validation Error", str(error))

    with store_lock:
        time_attack["answers"].update(validated_answer_sets)
        result = time_attack_result(time_attack, timed_out=True)
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
