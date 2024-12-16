import toml
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import warnings
import pandas as pd
import os
import PyPDF2
import requests
import json
import google.generativeai as genai

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure Gemini API

api_key_1 = "AIzaSyBL8-yYHEOP8KVMylVxSaRwCvHy3wq9W34"
api_key_2 = "AIzaSyBOpUUPOSk7afxINUUX40ZhiBfsrfRr66c"
api_key_3 = "AIzaSyDhZlco7vv-ptc3N-yN3a5pBeVXXWUWaSg"
api_key_4 = "AIzaSyCVkPQgNHqkq-lT3f3JoblD7zc0SkTubCQ"

genai.configure(api_key="AIzaSyBL8-yYHEOP8KVMylVxSaRwCvHy3wq9W34")

def get_gemini_response_direct(prompt):
    """Get the Gemini model response."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text[7:-4]

def generate_prompt_and_save_to_json(resume_str, jd_str, json_template, jd_hard_skills_list=None, jd_soft_skills_list=None, job_expected_experience=None, job_expected_education_list=None, candidate_id=None):
    """Generate prompt for Gemini API and save the response to a JSON file."""
    prompt = f"""
    Given the following resume text: {resume_str},
    and the following job description: {jd_str},
    perform an advanced extraction and analysis of the resume. Evaluate its alignment with the job description and provide ratings based on its impact in a general crowd and among its competitors.

    Provide a properly formatted JSON output using the following structure:

    {json_template}

    Additionally:
    - Analyze the alignment of the resume's skills with the hard skills: {jd_hard_skills_list} and soft skills: {jd_soft_skills_list} required by the job.
    - Assess the resume's experience against the expected experience for the job: {job_expected_experience}.
    - Compare the resume's education against the expected qualifications: {job_expected_education_list}.
    - Assign scores for each section, including education, skills, experience, projects, certifications, and an overall impact score, as floats without '/10'.
    - Compulsorily include exactly 3 entries for "missing_skills", "recommended_training_programs", and "career_advancement_recommendations".

    Ensure to return only the json file.
    """

    # Fetch the response using Gemini API
    response = get_gemini_response_direct(prompt)

    # Define the raw path for the output folder
    output_folder = r"C:\Users\PETDEVI1957\Desktop\SEM5\ATS_JR\SEM5-STREAMLIT\output_jsons"
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Save response to a JSON file in the specified folder
    output_filename = os.path.join(output_folder, f"candidate_{candidate_id}.json")
    with open(output_filename, 'w') as json_file:
        json.dump(json.loads(response), json_file, indent=4)

    return response

def process_uploaded_file(uploaded_file):
    """Process the uploaded file and extract its content."""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        pdf_text = "".join(page.extract_text() for page in pdf_reader.pages)
        return pdf_text
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode('utf-8')
    elif uploaded_file.type in ["image/jpeg", "image/png"]:
        st.sidebar.error("Image file processing is not implemented yet.")
    else:
        st.sidebar.error("Unsupported file type.")
        return None

# Streamlit UI Setup
st.set_page_config(
    initial_sidebar_state="expanded",
    page_title="ATS and Job Recommender",
    menu_items={
        'Get Help': 'https://drive.google.com/drive/folders/1gosDbNFWAlPriVNjC8_PnytQv7YimI1V?usp=drive_link',
        'Report a bug': "mailto:agronexumlive@gmail.com",
        'About': "### A web application built using NLP frameworks, ML techniques, and GEN-AI APIs."
    },
    page_icon="analysis.png",
    layout="wide"
)

st.markdown('''# **Extract and Parse Resumes with Advanced Analysis**''')
add_vertical_space(2)

# Sidebar
st.sidebar.header("ATS and Job Recommender Tool")
st.sidebar.markdown("**Upload resumes and job details for analysis.**")
st.sidebar.markdown("\n")

# Collect relevant job information
job_description = st.text_area("Enter Job Description:")
hard_skills = st.text_area("Enter Hard Skills (comma-separated):").split(',')
soft_skills = st.text_area("Enter Soft Skills (comma-separated):").split(',')
expected_experience = st.text_input("Enter Expected Experience (e.g., '5 years'):")
expected_education = st.text_area("Enter Expected Education (comma-separated):").split(',')

# File uploader
uploaded_files = st.sidebar.file_uploader("Upload resumes (multiple files allowed):", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files and st.sidebar.button("Start Analysis"):
    if not job_description:
        st.error("Please provide a job description.")
    else:
        st.markdown("#### Processing resumes...")
        candidate_id = 1

        for uploaded_file in uploaded_files:
            resume_content = process_uploaded_file(uploaded_file)
            if resume_content:
                st.text(f"Processing file: {uploaded_file.name}")

                # JSON template
                json_template = {
                    "personal_information": {
                        "full_name": "",
                        "contact_info": {
                            "email": "",
                            "phone_number": "",
                            "linkedin_url": ""
                        }
                    },
                    "skills": [],
                    "education": [],
                    "experience": [],
                    "missing_skills": [],
                    "recommended_training_programs": [],
                    "career_advancement_recommendations": []
                }

                # Call Gemini API and save JSON
                response = generate_prompt_and_save_to_json(
                    resume_str=resume_content,
                    jd_str=job_description,
                    json_template=json.dumps(json_template),
                    jd_hard_skills_list=hard_skills,
                    jd_soft_skills_list=soft_skills,
                    job_expected_experience=expected_experience,
                    job_expected_education_list=expected_education,
                    candidate_id=candidate_id
                )

                st.text(f"Processed candidate {candidate_id}: {uploaded_file.name}")
                candidate_id += 1

        st.success("All resumes processed successfully!")
