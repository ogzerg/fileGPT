import os
import pytesseract
import streamlit as st
from tempfile import TemporaryDirectory
from program import Program
os.environ["OPENAI_API_KEY"] = "API KEY HERE"
pytesseract.pytesseract.tesseract_cmd = (
    "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)


def main():
    st.set_page_config(
        page_title="Ask From File. Acceptable Exceptions:\n\
        .doc, .docx, .csv, .pdf, All Image Files"
    )  # Get File From User
    temp_dir = TemporaryDirectory()
    file = st.file_uploader("Upload A File")
    if file is not None:
        file_extension = file.name.split(".")[-1].lower()  # Obtaining the File Extension
        uploadedFile = temp_dir.name + "/" + file.name
        with open(uploadedFile, "wb") as f:  # Save the file to a temporary path
            f.write(file.read())
        question = st.text_input(label="Ask A Question")  # Get Question From User
        Program(uploadedFile=uploadedFile,question=question,file_extension=file_extension,st=st)

if __name__ == "__main__":
    main()
