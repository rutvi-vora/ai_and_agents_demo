from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate a caption for this image in about 50 words."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://images.pexels.com/photos/879109/pexels-photo-879109.jpeg?_gl=1*1sea6a7*_ga*MjM2MzIzMjEuMTc2NTQzNDk3MA..*_ga_8JE65Q40S6*czE3NjU0MzQ5NzAkbzEkZzEkdDE3NjU0MzUyMDIkajYwJGwwJGgw"
                    }
                }
            ]
        }
    ])

print(response.choices[0].message.content)