import streamlit as st
import requests
import fitz  # PyMuPDF for PDF processing

API_URL = "http://127.0.0.1:8000/query/"  # Adjust if backend runs on a different port

# Centered subtitle
st.markdown("<h1 style='text-align: center;'>Bounce Insights</h1>", unsafe_allow_html=True)

# File uploaders for two PDF reports
st.subheader("Upload Reports")
pdf1 = st.file_uploader("Upload First PDF", type=["pdf"])
pdf2 = st.file_uploader("Upload Second PDF", type=["pdf"])

# Text input for query
st.subheader("Enter Your Query")
query = st.text_area("Type your query here...")

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Submit button
if st.button("Submit"):
    if pdf1 and pdf2 and query:
        # Extract text from PDFs
        text1 = extract_text_from_pdf(pdf1)
        text2 = extract_text_from_pdf(pdf2)

        # Send data to FastAPI backend
        payload = {"query": query, "pdf1_text": text1, "pdf2_text": text2}
        response = requests.post(API_URL, json=payload)

        # Display response
        if response.status_code == 200:
            st.subheader("Response from RAG Model")
            st.write(response.json()["response"])
        else:
            st.error("Error processing request. Please try again.")
    else:
        st.warning("Please upload both PDFs and enter a query.")