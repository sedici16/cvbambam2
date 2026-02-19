import gradio as gr
import pandas as pd
from cv_extractor import extract_json
import docx
import fitz  # PyMuPDF
import tempfile
import re
import json
import os
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr
app = FastAPI()
import threading
from fastapi.responses import FileResponse
from gradio.routes import mount_gradio_app
from fastapi.responses import RedirectResponse
import requests
from fastapi.staticfiles import StaticFiles


app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/", StaticFiles(directory="templates", html=True), name="static")



def compute_similarity_remote(ideal_profile, text_blocks):
    response = requests.post(
        "https://theoracle-sim-fun.hf.space/similarity",
        json={"ideal_profile": ideal_profile, "text_blocks": text_blocks},
        timeout=30
    )
    return response.json()


MAX_DOCS = 5  # ❗ change this to limit uploads

def flatten_lists_in_df(df):
    def flatten_cell(cell):
        if isinstance(cell, list):
            return "\n".join(
                format_dict(item) if isinstance(item, dict) else str(item)
                for item in cell
            )
        return str(cell)

    def format_dict(d):
        # Customize this format string if needed
        return ", ".join(f"{k.capitalize()}: {v}" for k, v in d.items())

    return df.applymap(flatten_cell)




def read_docx(file_obj):
    doc = docx.Document(file_obj.name)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file_obj):
    doc = fitz.open(file_obj.name)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def clean_and_parse_json(json_data):
    if not isinstance(json_data, str):
        return json_data

    # Remove triple backticks and optional 'json'
    cleaned = re.sub(r"^```(json)?", "", json_data.strip(), flags=re.IGNORECASE).strip("` \n")

    # Remove trailing commas from objects and arrays
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

    # Replace missing/null values (e.g. "email": , → "email": null)
    cleaned = re.sub(r'(?<=:\s)(?=,|\})', 'null', cleaned)

    # Remove extra blank lines or non-JSON whitespace
    cleaned = '\n'.join([line for line in cleaned.splitlines() if line.strip()])

    # Optional: log cleaned string for debugging
    # print("🧹 Cleaned JSON string:\n", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")


# Combine all non-null, string-like fields into one text block per CV
def row_to_text(row):
    return " | ".join(str(val) for val in row if isinstance(val, str) and val.strip())



def process_files(files, ideal_profile):
    if not files:
        return "No files uploaded.", None, pd.DataFrame()

    if len(files) > MAX_DOCS:
        return f"❌ You can only upload up to {MAX_DOCS} documents.", None, pd.DataFrame()

    data = []
    for file in files:
        if file.name.endswith(".docx"):
            text = read_docx(file)
        elif file.name.endswith(".pdf"):
            text = read_pdf(file)
        else:
            continue

        json_data = extract_json(text)
        print ("extraction: ", json_data)

        try:
            json_parsed = clean_and_parse_json(json_data)
            data.append(json_parsed)
        except Exception as e:
            data.append({"error": f"Failed to parse JSON: {str(e)}", "source": file.name})

    df = pd.json_normalize(data)

    # Create a shallow copy for preview
    preview_df = df.copy()

    preview_df = flatten_lists_in_df(preview_df)


 


    preview_df["text_for_match"] = preview_df.apply(row_to_text, axis=1)

    
    similarities = compute_similarity_remote(ideal_profile, preview_df["text_for_match"].tolist())



    preview_df["similarity"] = similarities

    # Move 'similarity' column right after 'name' if 'name' exists

    try:
        preview_df.insert(preview_df.columns.get_loc("name") + 1, "similarity", preview_df.pop("similarity"))
    except Exception:
        pass  # or log a warning if needed




    preview_df = preview_df.sort_values(by="similarity", ascending=False)

    preview_df.drop(columns=["text_for_match"], inplace=True)


    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    preview_df.to_csv(tmp_file.name, index=False)


    return (
      "✅ Extraction complete. Download your CSV below.",
      tmp_file.name,
      preview_df.head(10),
      
      )
    



with gr.Blocks(title="Custom CV Extractor") as demo:

    gr.Markdown("### 📄 Upload Your CVs (PDF or DOCX)")
    gr.Markdown("📝 **Tip:** Write a short profile (1–2 sentences) focusing only on key skills and experience. Long inputs may reduce matching accuracy.")

    with gr.Row():
        file_input = gr.File(file_types=[".pdf", ".docx"], file_count="multiple", label="Upload CVs")

        # ✅ Add this outside the row, or in a new row if needed
        profile_input = gr.Textbox(
            label="Ideal Candidate Profile",
            placeholder="e.g. 5+ years experience in Python, NLP, and FastAPI",
            lines=3,
            value="5+ years experience in Python and NLP"
        )

    extract_btn = gr.Button("🧠 Extract Data")

    status_output = gr.Textbox(label="Status")
    csv_output = gr.File(label="📁 Download CSV")

    preview_output = gr.Dataframe(label="Preview (first 5 rows)")

    

    extract_btn.click(
    fn=process_files,
    inputs=[file_input, profile_input],  # ✅ Corrected input list
    outputs=[status_output, csv_output, preview_output]
)




#@app.get("/", include_in_schema=False)
#def root():
#    return FileResponse("templates/index.html")



mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", 10000))  # use PORT env var or default to 10000
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
