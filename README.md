# 🔍 Open-Vocabulary Object Search (CLIP + SAM)

This project is a computer vision system that allows users to find objects in images using natural language queries such as "a dog", "a cat", or "a sofa". It combines segmentation and vision-language models to localize objects in images.

It uses:
- Segment Anything Model (SAM)
- CLIP (vision-language model from Hugging Face)

---

# 🚀 Features
- Upload an image
- Enter a text query
- Detect and highlight matching object
- Show confidence score
- Handles "no match found" cases

---

# 🤗 Hugging Face Setup (IMPORTANT)

This project uses models from Hugging Face, including gated models.

## 1. Accept model access (required)

Go to:
https://huggingface.co/facebook/sam3

- Click **“Request Access”** or **“Accept Terms”**
- Wait for approval (usually instant for students)

---

## 2. Login via terminal

Run:

hf auth login

- Paste your Hugging Face Access Token
- Get it from:
  https://huggingface.co/settings/tokens
- Create a **Read token**
- Paste it (it will not show while typing — this is normal)

---

## 3. Test download (recommended)

Run:

hf download facebook/sam3 config.json

If no **401 Unauthorized** error appears, setup is correct.

---

## 🛠️ Fix token permissions (if errors occur)

Go to:
https://huggingface.co/settings/tokens

Then:
- Edit your token
- Set **Permissions → Read access**
- Enable **Gated Repositories access**
- Save changes

---

# 🧱 Setup Instructions

## 1. Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git  
cd YOUR_REPO  

---

## 2. Create virtual environment (recommended)
python -m venv venv  
source venv/bin/activate      # Mac/Linux  
venv\Scripts\activate         # Windows  

---

## 3. Install dependencies
pip install -r requirements.txt  

If requirements file is missing:
pip install gradio matplotlib opencv-python torch transformers segment-anything

---

## 4. Download SAM checkpoint

Download from:
https://github.com/facebookresearch/segment-anything

Place file here:

models/sam_vit_h_4b8939.pth  

Final structure:
project/
│── models/
│     └── sam_vit_h_4b8939.pth
│── app.py
│── main.py
│── utils.py

---

## 5. Run the app
python app.py  

---

## 6. Open in browser
http://127.0.0.1:7860  

---

# 🧪 How to use
1. Upload an image
2. Enter a text query (e.g., "a dog", "a cat", "a sofa")
3. Click submit
4. View highlighted object with confidence score
