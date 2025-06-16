# cv_extractor.py

from huggingface_hub import InferenceClient

HF_TOKEN = "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"

client = InferenceClient(
    provider="novita",
    api_key=HF_TOKEN
)

def extract_json(input_text):
    prompt = f"""
    Please extract the following fields in JSON format from the CV text, if available:
    Your task is to extract structured data from the following CV/resume text. Return a single, valid JSON object only. Do not include Markdown formatting or text outside the JSON.

    Field definitions (all optional):
    - name
    - summary (concise bio)
    - job_title (current role)
    - location (current city, country)
    - years_of_experience
    - education (list, with degree and institution)
    - work_experience (companies with roles, in order)
    - skills (technical and managerial)
    - certifications
    - languages (with fluency level)
    - email
    - phone
    - portfolio_website
    - github_profile
    - speaking_engagements
    - community_involvement
    - writing_portfolio
    - interests

    Return only the JSON object. 

    EXAMPLE INPUT:
    Hi, I'm Alice Bianchi, a marketing manager from Florence with 6 years of experience. I graduated from the University of Bologna and worked at Acme Corp and BrightMedia. My skills include SEO, branding, and Google Analytics. I hold an AWS Certified Marketing Specialist certificate. I speak Italian and English. Contact me at alice.b@example.com or +39 389 0011223. You can view my profile at linkedin.com/in/alicebianchi and my portfolio at alicebianchi.design.

    EXAMPLE JSON OUTPUT:
    {{
        "name": "Alice Bianchi",
        "summary": "Alice Bianchi is a marketing manager based in Florence with 6 years of experience. She specializes in SEO, branding, and analytics, and has worked with major Italian agencies. She holds an AWS Marketing certificate and speaks English and Italian.",
        "job_title": "marketing manager",
        "location": "Florence",
        "years_of_experience": 6,
        "education": ["University of Bologna"],
        "work_experience": ["Acme Corp", "BrightMedia"],
        "skills": ["SEO", "branding", "Google Analytics"],
        "certifications": ["AWS Certified Marketing Specialist"],
        "languages": ["Italian", "English"],
        "email": "alice.b@example.com",
        "phone": "+39 389 0011223",
        "linkedin_profile": "linkedin.com/in/alicebianchi",
        "portfolio_website": "alicebianchi.design"
    }}

    USER INPUT:
    {input_text}

    Return a valid JSON object with no comments, no trailing commas, and no markdown syntax (no backticks).
    
    

    JSON OUTPUT:
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5000,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"
