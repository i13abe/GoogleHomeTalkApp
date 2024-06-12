import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# modelのパス
# model_id = "models/llama3_8b"
# model_id = "models/llama2_7b_hf"
model_id = "models/llama2_7b_elyza_fast_inst"

# model読み込み
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

# streamer設定
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# pipline作成
pipeline = transformers.pipeline(
    "text-generation", model=model, tokenizer=tokenizer, streamer=streamer
)

# eosなどの設定
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

# prompt
ask = "あなたの自己紹介をしてください。"
messages = [
    {"role": "system", "content": "あなたは有能なAIで、我々の質問に対し適切な回答を日本語でします。"},
    {"role": "user", "content": ask},
]
prompt = pipeline.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# 推論
outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

# print(outputs[0]["generated_text"].lstrip(prompt))
