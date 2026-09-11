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
大学レポートに適した文章へ推敲する。

・口語的な表現を自然な文章表現へ修正する
・文体をできるだけ統一する
・簡潔で読みやすい文章にする
・過度に難しい表現や専門的すぎる表現は使わない
・「ちゃんと」「かなり」「いろんな」「でも」などの
  口語表現は、文脈に合った自然な表現へ修正する
・「実験」「研究」「調査」など内容を表す重要な語は
  原文の意味を維持し、理由なく変更しない
・筆者の感想や推測を客観的事実として書き換えない
""",

    "email": """
メールとして自然で適切な文章へ推敲する。

・相手に失礼のない表現にする
・必要に応じて丁寧な表現へ修正する
・硬すぎる表現は避ける
・簡潔で読みやすい文章にする
・原文にない事情や情報を追加しない
""",

    "es": """
就職活動のESとして適切な文章へ推敲する。

・簡潔で具体的な表現にする
・同じ表現の繰り返しを減らす
・読み手に伝わりやすい文章にする
・経験や実績を勝手に追加しない
・内容を誇張しない
""",

    "normal": """
自然で読みやすい日本語へ推敲する。

・誤字脱字を修正する
・不自然な表現を修正する
・冗長な表現を簡潔にする
・原文の意味や内容を維持する
"""
}


# ========================================
# 推敲レベル
# ========================================

LEVEL_PROMPTS = {

    "light": """
修正は最小限にする。

・誤字脱字を修正する
・明らかに不自然な表現のみ修正する
・原文の表現をできるだけ残す
""",

    "normal": """
原文を活かしながら、
自然で分かりやすい文章に修正する。

不自然な表現や用途に合わない表現は、
意味を変えない範囲で積極的に改善する。
""",

    "strong": """
文章全体を積極的に改善する。

必要であれば文章構成や語順も変更する。

ただし、
原文の意味・事実・主張は変更しない。
"""
}


# ========================================
# 修正点比較用
# ========================================

def normalize_for_match(text):
    """
    AIが「生成AI」と「生成 AI」のように
    空白を入れる場合があるため、
    比較するときだけ空白・改行・タブを除去する。
    """

    return "".join(str(text).split())


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
        revised_text = "推敲結果を取得できませんでした。"


    # -----------------------------
    # 修正点
    # -----------------------------

    changes = result.get("changes", [])

    if not isinstance(changes, list):
        changes = []

    valid_changes = []

    # 比較用
    normalized_original = normalize_for_match(
        original_text
    )

    normalized_revised = normalize_for_match(
        revised_text
    )


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


        # -----------------------------
        # 空データを除外
        # -----------------------------

        if not before or not after:
            continue


        # 比較用に空白を除去
        normalized_before = normalize_for_match(
            before
        )

        normalized_after = normalize_for_match(
            after
        )


        # -----------------------------
        # beforeとafterが同じなら除外
        # -----------------------------

        if normalized_before == normalized_after:
            continue


        # -----------------------------
        # beforeが原文に存在するか
        # -----------------------------

        if normalized_before not in normalized_original:
            continue


        # -----------------------------
        # afterが推敲後文章に存在するか
        # -----------------------------

        if normalized_after not in normalized_revised:
            continue


        # -----------------------------
        # 理由が空なら補完
        # -----------------------------

        if not reason:

            reason = (
                "文章をより自然で読みやすい"
                "表現に修正しました。"
            )


        valid_changes.append({

            "before": before,

            "after": after,

            "reason": reason

        })


        # 最大5件
        if len(valid_changes) >= 5:
            break


    changes = valid_changes


    # ========================================
    # 文章評価
    # ========================================

    scores = result.get("scores", {})

    if not isinstance(scores, dict):
        scores = {}


    def normalize_score(value):

        try:

            score = int(value)

            # 0～100点に制限
            return max(
                0,
                min(100, score)
            )

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


    # ========================================
    # 総評
    # ========================================

    summary = str(
        result.get("summary", "")
    ).strip()

    if not summary:

        summary = (
            "総評を取得できませんでした。"
        )


    # ========================================
    # 表示用テキスト
    # ========================================

    lines = []


    # -----------------------------
    # 推敲後文章
    # -----------------------------

    lines.append(
        "【推敲後の文章】"
    )

    lines.append(
        revised_text
    )

    lines.append("")


    # -----------------------------
    # 主な修正点
    # -----------------------------

    lines.append(
        "【主な修正点】"
    )


    if changes:

        for i, change in enumerate(
            changes,
            start=1
        ):

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

        # 原文と推敲後が違うのに
        # changesだけ取得できなかった場合
        if (
            normalized_original
            != normalized_revised
        ):

            lines.append(
                "文章全体の表現を調整しました。"
            )

        else:

            lines.append(
                "大きな修正はありません。"
            )


    # -----------------------------
    # 文章評価
    # -----------------------------

    lines.append(
        "【文章評価】"
    )

    lines.append(
        f"読みやすさ："
        f"{readability}点／100点"
    )

    lines.append(
        f"簡潔さ："
        f"{conciseness}点／100点"
    )

    lines.append(
        f"自然さ："
        f"{naturalness}点／100点"
    )

    lines.append("")


    # -----------------------------
    # 総評
    # -----------------------------

    lines.append(
        "【総評】"
    )

    lines.append(
        summary
    )


    return "\n".join(lines)


# ========================================
# 推敲API
# ========================================

@app.route(
    "/send_api",
    methods=["POST"]
)
def send_api():

    # -----------------------------
    # JSON取得
    # -----------------------------

    data = request.get_json(
        silent=True
    )


    if not data:

        return jsonify({
            "error":
            "JSONデータを取得できませんでした。"
        }), 400


    # -----------------------------
    # 入力文章
    # -----------------------------

    received_text = str(
        data.get("text", "")
    ).strip()


    if not received_text:

        return jsonify({
            "error":
            "文章を入力してください。"
        }), 400


    # -----------------------------
    # 用途
    # -----------------------------

    purpose = data.get(
        "purpose",
        "normal"
    )


    if purpose not in PURPOSE_PROMPTS:

        purpose = "normal"


    # -----------------------------
    # 推敲レベル
    # -----------------------------

    level = data.get(
        "level",
        "normal"
    )


    if level not in LEVEL_PROMPTS:

        level = "normal"


    purpose_prompt = (
        PURPOSE_PROMPTS[purpose]
    )

    level_prompt = (
        LEVEL_PROMPTS[level]
    )


    # ========================================
    # システムプロンプト
    # ========================================

    system_prompt = f"""
あなたは日本語文章の校正・推敲を行います。

【用途】
{purpose_prompt}

【推敲レベル】
{level_prompt}

原文の意味や事実を維持しながら、
用途に適した自然で読みやすい文章に改善してください。


【必ず行うこと】

・不自然な表現を自然な日本語に修正する

・用途に合わない口語表現を修正する

・冗長な表現を必要に応じて簡潔にする

・選択された推敲レベルに応じて文章を改善する


【重要】

文章に改善できる箇所が存在する場合は、
原文をそのまま返さず、
実際に文章を改善してください。


【禁止事項】

・原文にない事実を追加しない

・原文の意味や主張を変更しない

・筆者の感想を事実として変更しない

・頻度や程度を勝手に変更しない

・因果関係を勝手に追加しない

・「実験」「研究」「調査」などの
  重要な語を理由なく変更しない

・不自然に難しい言葉へ置き換えない


【修正点】

changesには、
実際に変更した箇所だけを記録してください。

beforeは、
原文に実際に存在する文字列を
そのまま使用してください。

afterは、
推敲後の文章に実際に存在する文字列を
そのまま使用してください。

beforeとafterを
同じ文章にしてはいけません。

修正理由は、
その変更だけについて
簡潔に説明してください。

主な修正点は最大5件です。


【文章評価】

推敲後の文章を
0～100の整数で評価してください。

・読みやすさ
・簡潔さ
・自然さ

をそれぞれ評価してください。

安易に100点にせず、
改善の余地も考慮してください。


【総評】

推敲後の文章について
1～2文で簡潔に評価してください。

実際に行っていない修正について
説明してはいけません。


前置きや挨拶は不要です。
"""


    # ========================================
    # JSON Schema
    # ========================================

    output_schema = {

        "type": "object",

        "properties": {

            "revised_text": {
                "type": "string"
            },

            "changes": {

                "type": "array",

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
                        "type": "integer"
                    },

                    "conciseness": {
                        "type": "integer"
                    },

                    "naturalness": {
                        "type": "integer"
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

    }


    # ========================================
    # Ollamaへ送信
    # ========================================

    try:

        response = client.chat(

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

            format=output_schema,

            think=False,

            options={

                "num_ctx": 8192,

                "temperature": 0.1

            }

        )


    except Exception as e:

        print(
            "Ollama error:",
            e
        )

        return jsonify({

            "error":
            "AIとの通信に失敗しました。"
            "Ollamaが起動しているか確認してください。"

        }), 500


    # ========================================
    # AI回答取得
    # ========================================

    try:

        content = response["message"]["content"]

    except Exception:

        return jsonify({

            "error":
            "AIから正常な回答を取得できませんでした。"

        }), 502


    if not content:

        return jsonify({

            "error":
            "AIから空の回答が返されました。"

        }), 502


    # ========================================
    # JSON解析
    # ========================================

    try:

        result = json.loads(
            content
        )

    except json.JSONDecodeError:

        print(
            "Invalid JSON:",
            content
        )

        return jsonify({

            "error":
            "AIの回答形式が正しくありませんでした。"

        }), 502


    if not isinstance(result, dict):

        return jsonify({

            "error":
            "AIの回答形式が正しくありませんでした。"

        }), 502


    # ========================================
    # 表示形式に変換
    # ========================================

    processed_text = format_result(
        result,
        received_text
    )


    # ========================================
    # フロントへ返す
    # ========================================

    return jsonify({

        "processed_text":
        processed_text

    })


# ========================================
# Flask起動
# ========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )