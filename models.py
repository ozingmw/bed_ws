import os
import base64
import cv2
import re
import openai
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

import prompts
from openai import AzureOpenAI


# load azure model
# you can use another model api
class Azure():
    def __init__(self):
        load_dotenv()
        self.deployment_name = 'gpt-4o'
        self.azure_client = self.create_azure_client()
    
    def create_azure_client(self):
        return AzureOpenAI(
            api_key=os.getenv("api_key"),
            api_version=os.getenv("api_version"),
            azure_endpoint=os.getenv("azure_endpoint"),
        )
    
    def encode_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_base64
    
    def encode_image_path(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate(self, image):
        base64_image = self.encode_image(image)

        prompt = prompts.GPT_GENERATE_PROMPT

        ex_image_1 = self.encode_image_path('./prompt_image_정면.png')
        ex_image_2 = self.encode_image_path('./prompt_image_측면.png')

        try:
            response = self.azure_client.chat.completions.create(
                model = self.deployment_name,
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{prompt.split('{EX_IMAGE_1}')[0]}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex_image_1}", "detail": "high"}},
                            {"type": "text", "text": f"{prompt.split('{EX_IMAGE_1}')[1].split('{EX_IMAGE_2}')[0]}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex_image_2}", "detail": "high"}},
                            {"type": "text", "text": f"{prompt.split('{EX_IMAGE_2}')[1]}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}},
                        ]
                    }
                ],
                max_tokens=4096,
            ).to_dict()

            answer = response['choices'][0]['message']['content']
            answer = re.sub(r'[^\uAC00-\uD7A3]', '', answer) # 한글만

            if answer not in ['정면', '측면']:
                answer = '에러'

        except openai.BadRequestError as e:
            response = e.body
            answer = '에러'

        return answer