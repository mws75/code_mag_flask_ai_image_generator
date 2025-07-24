from flask import Flask, render_template, jsonify, url_for, request
from flask_executor import Executor
from PIL import Image, ImageDraw
import safetensors
import transformers
from diffusers import StableDiffusionPipeline
import torch
from huggingface_hub import login
import os
import certifi
from dotenv import load_dotenv

# load .env file 
load_dotenv()

# Fix SSL issues if needed
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token is None: 
    raise ValueError("HF_TOKEN not found in .env file")

# Login to Hugging Face (if using private/gated models)
login(hf_token)

app = Flask(__name__)
executor = Executor(app)

def generate_image(generation_prompt):
    model_path = "models\stable-diffusion-v1-4"
    print("loading pipeline")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        safety_checker=None,
        torch_dtype=torch.float32
    )
    print("pipeline loaded")
    pipe.to("cpu")

    print("Generating image")
    image = pipe(
        prompt=generation_prompt,
        negative_prompt="",
        num_inference_steps=5,
        height=64,
        width=64,
        guidance_scale=3.0
    ).images[0]
    print("image generated")
    image.save("static/image.png")
    print("image saved")
    return 'static/image.png'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_generation', methods=['POST'])
def start_generation():
    if os.path.exists('static/image.png'): os.remove('static/image.png')
    generation_prompt = request.json['generation-prompt']
    executor.sumbit(generate_image(generation_prompt))
    return jsonify({'message': 'generating image'})

@app.route('/check_generation')
def check_generation():
    # Check if the image file exists
    if os.path.exists('static/image.png'):
        return jsonify({'message': 'generation complete', 
                        'image': 'static/image.png'})
    else:
        return jsonify({'message': 'generation running'})
    
@app.route('/image')
def serve_image(): 
    return url_for('static', filename='image.png')

if __name__ == '__main__': app.run(debug=True)
