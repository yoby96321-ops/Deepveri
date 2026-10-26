import os
from pathlib import Path

project_dir = Path(__file__).resolve().parent

os.chdir(project_dir)

print("=" * 60)
print("正在啟動 真假圖 AI 網站")
print("網址：http://localhost:8501")
print("=" * 60)

os.system("streamlit run app.py")