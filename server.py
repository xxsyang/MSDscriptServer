from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests

app = Flask(__name__)

# Load GPT-2 model
model_path = "./gpt2-autofill-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    max_length = data.get('max_length', 100)

    if not prompt:
        return jsonify({'error': 'Empty prompt received'}), 400

    # Explicitly include attention_mask
    encoded_input = tokenizer(prompt, return_tensors='pt', padding=True)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id
    )
    completion = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({'completion': completion})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8964)
    
