import pypdf
import sys

def read_pdf(file_path):
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        with open('pdf_content.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Successfully extracted text to pdf_content.txt. Num pages: {len(reader.pages)}")
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    read_pdf("HeartCheck IA - TCC (4).pdf")
