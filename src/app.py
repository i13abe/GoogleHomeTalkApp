import torch
import transformers
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer

# Flask app生成
app = Flask(__name__)

# 生成AI準備
MODEL_PATH = "models/llama2_7b_elyza_fast_inst"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

# pipline作成
pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)

# eosなどの設定
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

# prompt準備
messages = [
    {"role": "system", "content": "あなたは有能なAIで、我々の質問に対し適切な回答を日本語でします。その際分からないことはわからないと答えて構いません。"},
]


@app.route("/", methods=["GET"])
def test():
    return "This is test get method."


@app.route("/talk", methods=["POST"])
def talk():
    # POSTで受け取るメッセージ抽出
    message = request.json.get("result").get("parameters").get("message")

    # メッセージの保存
    messages.append({"role": "user", "content": message})

    # promptの生成
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 生成AIの回答取得
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # 返答の保存
    messages.append({"role": "assistant", "content": outputs[0]["generated_text"].lstrip(prompt)})

    response = {"response": outputs[0]["generated_text"].lstrip(prompt)}
    return jsonify(response)


if __name__ == "__main__":
    app.run(port=8000)
