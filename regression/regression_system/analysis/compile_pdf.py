import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "regression_system")
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

def compile_pdf():
    md_path = ANALYSIS_DIR / "report.md"
    html_path = ANALYSIS_DIR / "report.html"
    pdf_path = ANALYSIS_DIR / "report.pdf"
    
    if not os.path.exists(md_path):
        print("Hata: report.md bulunamadı!")
        return

    print("1. Markdown dosyası HTML'e dönüştürülüyor...")
    # Convert markdown to html using pandoc
    try:
        pandoc_path = shutil.which("pandoc")
        if not pandoc_path:
            print("Pandoc bulunamadı. Pandoc'u PATH'e ekleyin.")
            return
        subprocess.run([
            pandoc_path,
            md_path, 
            "-o", html_path
        ], check=True)
    except Exception as e:
        print("Pandoc hatası:", e)
        return

    # Add beautiful CSS styling to the HTML file
    print("2. HTML dosyasına özel CSS stilleri ekleniyor...")
    with open(html_path, "r") as f:
        html_content = f.read()

    css_style = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    @page {
        size: A4;
        margin: 20mm;
    }
    
    body {
        font-family: 'Outfit', 'Helvetica Neue', Arial, sans-serif;
        color: #2d3748;
        line-height: 1.5;
        margin: 0;
        padding: 0;
        background-color: #ffffff;
    }
    h1 {
        font-size: 24px;
        color: #1a365d;
        text-align: center;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px double #e2e8f0;
    }
    h2 {
        font-size: 18px;
        color: #2b6cb0;
        margin-top: 30px;
        padding-bottom: 6px;
        border-bottom: 2px solid #e2e8f0;
        page-break-before: always; /* Her ana bölüm yeni sayfada başlasın */
    }
    h1 + h2 {
        page-break-before: avoid; /* Başlıktan hemen sonraki ilk h2 sayfa kesmesi yapmasın */
    }
    h3 {
        font-size: 14px;
        color: #4a5568;
        margin-top: 15px;
    }
    p, li {
        font-size: 12px;
        color: #4a5568;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 11px;
        page-break-inside: avoid;
    }
    th, td {
        border: 1px solid #cbd5e0;
        padding: 8px 10px;
        text-align: left;
    }
    th {
        background-color: #ebf8ff;
        color: #2b6cb0;
        font-weight: 600;
    }
    tr:nth-child(even) {
        background-color: #f7fafc;
    }
    img {
        width: 450px; /* LibreOffice scaling için sabit genişlik */
        height: auto;
        display: block;
        margin: 15px auto;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-radius: 6px;
        page-break-inside: avoid; /* Grafiklerin ortadan bölünmesini engeller */
    }
    hr {
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 25px 0;
    }
</style>
"""
    # Insert style tag right before the body or at the beginning
    # Also add explicit width attribute to all img tags in HTML for bulletproof LibreOffice scaling
    html_content = html_content.replace("<img ", '<img width="450" ')
    styled_html = css_style + html_content

    with open(html_path, "w") as f:
        f.write(styled_html)

    print("3. HTML dosyası LibreOffice ile PDF'e dönüştürülüyor...")
    try:
        libreoffice_path = shutil.which("libreoffice")
        if not libreoffice_path:
            print("LibreOffice bulunamadı. PDF dönüştürme adımı atlandı.")
            return
        subprocess.run([
            libreoffice_path,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(ANALYSIS_DIR),
            html_path
        ], check=True)
        print(f"[✓] PDF başarıyla oluşturuldu: {pdf_path}")
    except Exception as e:
        print("LibreOffice dönüştürme hatası:", e)

if __name__ == "__main__":
    compile_pdf()
