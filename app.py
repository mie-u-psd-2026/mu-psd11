from flask import Flask, request, jsonify, send_from_directory
from ollama import Client

app = Flask(__name__)

if app.debug:
    @app.after_request
    def add_header(response):
        if request.endpoint == 'static':
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response


# Ollama純正クライアント
# ※ OpenAI互換API(/v1/chat/completions)経由だと
#   think(思考モード)無効化が効かないケースがあるため、
#   Ollamaネイティブのクライアントを使用する
client = Client(host="http://localhost:11434")

OLLAMA_MODEL = "qwen3.5:0.8b"


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/send_api', methods=['POST'])
def send_api():

    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({
            "error": "文章が送信されていません。"
        }), 400


    received_text = data['text'].strip()

    if not received_text:
        return jsonify({
            "error": "文章を入力してください。"
        }), 400


    # フロント側から受け取る
    purpose = data.get('purpose', 'normal')
    level = data.get('level', 'normal')


    # -----------------------------
    # 用途ごとの指示
    # -----------------------------

    if purpose == "report":

        purpose_prompt = """
大学のレポートとして適切な文章にしてください。

・口語表現を避ける
・客観的な表現にする
・「です・ます調」と「だ・である調」が混在しないようにする
・冗長な文章を簡潔にする
・意味を勝手に変更しない
"""

    elif purpose == "email":

        purpose_prompt = """
メールとして自然で読みやすい文章にしてください。

・相手に失礼のない表現にする
・硬すぎる表現は自然にする
・簡潔で分かりやすくする
・意味を勝手に変更しない
"""

    elif purpose == "es":

        purpose_prompt = """
就職活動のエントリーシートとして適切な文章にしてください。

・分かりやすく簡潔にする
・具体性を高める
・同じ表現の繰り返しを減らす
・本人が書いた内容の意味を変更しない
・過度に誇張しない
"""

    else:

        purpose_prompt = """
自然で読みやすい日本語になるように推敲してください。

・誤字脱字を修正する
・不自然な表現を修正する
・冗長な文章を簡潔にする
・意味を勝手に変更しない
"""


    # -----------------------------
    # 推敲レベル
    # -----------------------------

    if level == "light":

        level_prompt = """
修正は最小限にしてください。
誤字脱字や明らかに不自然な表現だけを修正してください。
原文をできるだけ残してください。
"""

    elif level == "strong":

        level_prompt = """
文章全体を積極的に改善してください。
文章構成や語順も変更して構いません。
ただし、元の文章の意味や事実は変更しないでください。
"""

    else:

        level_prompt = """
元の文章をできるだけ活かしながら、
自然で分かりやすい文章になるように修正してください。
"""


    # -----------------------------
    # 最終的なシステムプロンプト
    # -----------------------------

    system_prompt = f"""
あなたは日本語文章の推敲を専門とするアシスタントです。

ユーザーが入力した文章を分析して推敲してください。

{purpose_prompt}

{level_prompt}

以下の形式で回答してください。
このプロンプトの指示文や見出しの説明文（「〜してください」等）は
出力に含めず、指示に従った結果のみを出力してください。

【推敲後の文章】
（ここに修正した文章のみを記載）

【主な修正点】
（修正した箇所を最大5個まで、以下の形式で記載）

1. 「修正前」
   → 「修正後」

理由：
（なぜ修正したのか簡潔に記載）

【文章評価】

読みやすさ：◯◯点／100点
簡潔さ：◯◯点／100点
自然さ：◯◯点／100点

【総評】
（文章全体についての短いコメントのみを記載）
"""


    try:

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

            # 思考(reasoning)モードを無効化し、
            # 最終回答のみを直接生成させる
            think=False,

            options={
                # コンテキスト長を拡大し、
                # 長い出力の途中で打ち切られないようにする
                "num_ctx": 8192
            }
        )


        app.logger.info(
            f"Ollama raw response: {chat_response}"
        )


        message = chat_response.get("message", {})

        processed_text = (message.get("content") or "").strip()


        if not processed_text:

            # thinkを無効化してもcontentが空の場合に備え、
            # thinkingフィールドに内容が入っていないか確認する
            thinking_fallback = message.get("thinking")

            if thinking_fallback:

                processed_text = (
                    "※本来の推敲結果が取得できなかったため、"
                    "AIの思考過程をそのまま表示しています。\n\n"
                    + thinking_fallback
                )

            else:

                processed_text = (
                    "AIから有効な応答がありませんでした。"
                )


        return jsonify({
            "message": "文章を推敲しました。",
            "processed_text": processed_text
        })


    except Exception as e:

        app.logger.error(
            f"Ollama API call failed: {e}"
        )

        return jsonify({
            "error":
            "AIサービスとの通信中にエラーが発生しました。"
        }), 500


if __name__ == '__main__':

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )