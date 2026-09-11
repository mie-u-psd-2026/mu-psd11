from flask import Flask, request, jsonify, send_from_directory
from ollama import Client
import json

app = Flask(__name__)

# ========================================
# Ollama設定
# ========================================

client = Client(host="http://localhost:11434")

OLLAMA_MODEL = "qwen3.5:0.8b"


# ========================================
# 開発時のキャッシュ無効化
# ========================================

@app.after_request
def add_header(response):

    if app.debug:
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response


# ========================================
# トップページ
# ========================================

@app.route("/")
def index():

    return send_from_directory(
        app.static_folder,
        "index.html"
    )


# ========================================
# 用途ごとの指示
# ========================================

PURPOSE_PROMPTS = {

    "report": """
大学レポートとして適切な文章にする。
・口語的な表現を、自然な文章表現に修正する
・文体を統一する
・簡潔で読みやすくする
・過度に難しい表現や専門的すぎる表現は使わない
・「ちゃんと」「かなり」「いろんな」「マスターする」
  などの口語表現は、文脈に合った自然な表現へ修正する
・「実験」「研究」「調査」など、内容を表す重要な語は
  原文の意味を維持し、理由なく変更しない
・筆者の感想や推測を、客観的事実として書き換えない
""",

    "email": """
メールとして自然な文章にする。
相手に失礼のない表現にする。
硬すぎる表現を避け、簡潔で読みやすくする。
""",

    "es": """
就職活動のESとして適切な文章にする。
簡潔で具体的な表現にする。
同じ表現の繰り返しを減らす。
経験や実績を勝手に追加しない。
誇張しない。
""",

    "normal": """
自然で読みやすい日本語にする。
誤字脱字や不自然な表現を修正する。
冗長な表現を簡潔にする。
"""
}


# ========================================
# 推敲レベル
# ========================================

LEVEL_PROMPTS = {

    "light": """
修正は最小限にする。
誤字脱字や明らかに不自然な表現だけを修正する。
原文をできるだけ残す。
""",

    "normal": """
原文を活かしながら、
自然で分かりやすい文章に修正する。
""",

    "strong": """
文章全体を積極的に改善する。
必要であれば構成や語順も変更する。
ただし意味や事実は変更しない。
"""
}


# ========================================
# AIのJSONを整形
# ========================================

def format_result(result, original_text):

    # -----------------------------
    # 推敲後文章
    # -----------------------------

    revised_text = str(
        result.get("revised_text", "")
    ).strip()

    if not revised_text:
        revised_text = "推敲結果を取得できませんでした."


    # -----------------------------
    # 修正点
    # -----------------------------

    changes = result.get("changes", [])

    if not isinstance(changes, list):
        changes = []

    valid_changes = []

    for change in changes:

        if not isinstance(change, dict):
            continue

        before = str(
            change.get("before", "")
        ).strip()

        after = str(
            change.get("after", "")
        ).strip()

        reason = str(
            change.get("reason", "")
        ).strip()

        # 空データを除外
        if not before or not after:
            continue

        # 修正前の文章が原文に存在するか確認
        if before not in original_text:
            continue

        # 修正後の文章が推敲結果に存在するか確認
        if after not in revised_text:
            continue

        valid_changes.append({
            "before": before,
            "after": after,
            "reason": reason
        })

        # 最大5件
        if len(valid_changes) >= 5:
            break

    changes = valid_changes


    # -----------------------------
    # 評価
    # -----------------------------

    scores = result.get("scores", {})

    if not isinstance(scores, dict):
        scores = {}


    def normalize_score(value):

        try:
            score = int(value)

            # 0～100に制限
            return max(0, min(100, score))

        except (TypeError, ValueError):
            return 0


    readability = normalize_score(
        scores.get("readability")
    )

    conciseness = normalize_score(
        scores.get("conciseness")
    )

    naturalness = normalize_score(
        scores.get("naturalness")
    )


    # -----------------------------
    # 総評
    # -----------------------------

    summary = str(
        result.get("summary", "")
    ).strip()

    if not summary:
        summary = "総評を取得できませんでした。"


    # -----------------------------
    # 表示用テキスト作成
    # -----------------------------

    lines = []

    lines.append("【推敲後の文章】")
    lines.append(revised_text)

    lines.append("")
    lines.append("【主な修正点】")


    if changes:

        for i, change in enumerate(changes, start=1):

            before = change["before"]
            after = change["after"]
            reason = change["reason"]

            lines.append(
                f'{i}. 「{before}」'
            )

            lines.append(
                f'   → 「{after}」'
            )

            lines.append(
                f'理由：{reason}'
            )

            lines.append("")

    else:

        lines.append(
            "大きな修正はありません。"
        )


    lines.append("【文章評価】")

    lines.append(
        f"読みやすさ：{readability}点／100点"
    )

    lines.append(
        f"簡潔さ：{conciseness}点／100点"
    )

    lines.append(
        f"自然さ：{naturalness}点／100点"
    )

    lines.append("")

    lines.append("【総評】")
    lines.append(summary)


    return "\n".join(lines)

# ========================================
# 推敲API
# ========================================

@app.route("/send_api", methods=["POST"])
def send_api():

    data = request.get_json(silent=True)


    # -----------------------------
    # 入力チェック
    # -----------------------------

    if not data or "text" not in data:

        return jsonify({
            "error": "文章が送信されていません。"
        }), 400


    if not isinstance(data["text"], str):

        return jsonify({
            "error": "文章の形式が正しくありません。"
        }), 400


    received_text = data["text"].strip()


    if not received_text:

        return jsonify({
            "error": "文章を入力してください。"
        }), 400


    # -----------------------------
    # purpose / level
    # -----------------------------

    purpose = data.get(
        "purpose",
        "normal"
    )

    level = data.get(
        "level",
        "normal"
    )


    if purpose not in PURPOSE_PROMPTS:
        purpose = "normal"

    if level not in LEVEL_PROMPTS:
        level = "normal"


    purpose_prompt = PURPOSE_PROMPTS[purpose]

    level_prompt = LEVEL_PROMPTS[level]


    # ========================================
    # システムプロンプト
    # ========================================

    system_prompt = f"""
あなたは日本語文章の推敲アシスタントです。

次の条件に従ってユーザーの文章を推敲してください。

用途：
{purpose_prompt}

推敲レベル：
{level_prompt}

原文の意味や事実を維持しながら、
用途に適した自然で読みやすい文章に改善してください。

【必ず行うこと】
・不自然な表現を自然な日本語に修正する
・用途に合わない口語表現を修正する
・冗長な表現を必要に応じて簡潔にする
・選択された推敲レベルに応じて文章を改善する

【禁止事項】
・原文にない事実を追加しない
・原文の意味や主張を変更しない
・頻度や程度を勝手に変更しない
・「実験」「研究」「調査」などの重要な語を理由なく変更しない
・不自然に難しい言葉へ置き換えない

【修正点】
・実際に変更した箇所だけを記録する
・beforeとafterが同じ修正は記録しない
・beforeは原文に実際に存在する表現にする
・afterは推敲後の文章に実際に存在する表現にする
・reasonはその変更だけについて簡潔に説明する
・最大5件

【文章評価】
推敲後の文章を0～100の整数で評価する。
読みやすさ、簡潔さ、自然さをそれぞれ評価する。
安易に100点にせず、改善の余地も考慮する。

【総評】
推敲後の文章について1～2文で簡潔に評価する。
実際に行っていない修正について説明しない。

前置きや挨拶は出力しない。
"""


    try:

        # ========================================
        # Ollamaへ送信
        # ========================================

        chat_response = client.chat(

            model=OLLAMA_MODEL,

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": received_text
                }

            ],

            think=False,

            # JSON出力を要求
            format={
                "type": "object",

                "properties": {

                    "revised_text": {
                        "type": "string"
                    },

                    "changes": {

                        "type": "array",

                        "maxItems": 5,

                        "items": {

                            "type": "object",

                            "properties": {

                                "before": {
                                    "type": "string"
                                },

                                "after": {
                                    "type": "string"
                                },

                                "reason": {
                                    "type": "string"
                                }

                            },

                            "required": [
                                "before",
                                "after",
                                "reason"
                            ]

                        }

                    },

                    "scores": {

                        "type": "object",

                        "properties": {

                            "readability": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100
                            },

                            "conciseness": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100
                            },

                            "naturalness": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100
                            }

                        },

                        "required": [
                            "readability",
                            "conciseness",
                            "naturalness"
                        ]

                    },

                    "summary": {
                        "type": "string"
                    }

                },

                "required": [
                    "revised_text",
                    "changes",
                    "scores",
                    "summary"
                ]
            },

            options={

                "num_ctx": 8192,

                # 推敲なのでランダム性を低めにする
                "temperature": 0.1

            }
        )


        # ========================================
        # Ollamaレスポンス取得
        # ========================================

        message = chat_response.get(
            "message",
            {}
        )

        content = (
            message.get("content") or ""
        ).strip()


        app.logger.info(
            f"Ollama JSON response: {content}"
        )


        if not content:

            return jsonify({
                "error":
                "AIから有効な応答を取得できませんでした。"
            }), 502


        # ========================================
        # JSON解析
        # ========================================

        try:

            result = json.loads(content)

        except json.JSONDecodeError:

            app.logger.error(
                f"Invalid JSON from Ollama: {content}"
            )

            return jsonify({
                "error":
                "AIの応答形式が正しくありませんでした。"
            }), 502


        # ========================================
        # Python側で表示形式を作る
        # ========================================

        processed_text = format_result(
            result,
             received_text
        )


        return jsonify({

            "message":
                "文章を推敲しました。",

            "processed_text":
                processed_text

        })


    except Exception:

        app.logger.exception(
            "Ollama API call failed."
        )

        return jsonify({
            "error":
            "AIサービスとの通信中にエラーが発生しました。"
        }), 500


# ========================================
# 起動
# ========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )