# CVBambam

AI-powered CV screening for small businesses. Upload CVs, get a ranked shortlist in seconds.

**Live at [cvbambam.com](https://cvbambam.com)**

## What It Does

1. Define your ideal candidate profile
2. Upload up to 5 CVs (PDF or DOCX)
3. The app extracts key info (skills, experience, education) using an LLM
4. Candidates are ranked by similarity to your ideal profile
5. Download the results as a structured comparison table

## Tech Stack

- **Backend**: FastAPI + Gradio
- **CV Parsing**: PyMuPDF (PDF), python-docx (DOCX)
- **AI**: Hugging Face Inference API for text extraction and analysis
- **Ranking**: Sentence similarity scoring via a remote embedding service
- **Frontend**: Gradio UI with custom static assets

## Setup

```bash
git clone https://github.com/sedici16/cvbambam2.git
cd cvbambam2
pip install -r requirements.txt
python app.py
```

## Environment Variables

```
HF_TOKEN=your-huggingface-token
```

## Project Structure

```
cvbambam2/
├── app.py              # FastAPI + Gradio app, handles uploads and ranking
├── cv_extractor.py     # LLM-based CV parsing and JSON extraction
├── requirements.txt
├── static/             # CSS, JS, images
└── templates/          # HTML templates
```
