import pypandoc
import os

md_file = r"C:\Users\aashi\.gemini\antigravity\scratch\ema_angle_backtest\Interview_Prep.md"
docx_file = r"C:\Users\aashi\.gemini\antigravity\scratch\ema_angle_backtest\Interview_Prep_Questions.docx"

try:
    pypandoc.get_pandoc_version()
except OSError:
    print("Downloading pandoc...")
    pypandoc.download_pandoc()

print("Converting...")
output = pypandoc.convert_file(
    md_file,
    'docx',
    outputfile=docx_file
)
print("Converted successfully to:", docx_file)
