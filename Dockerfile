# 使用輕量版的 Python 基礎映像檔
FROM python:3.10-slim

# 設定容器內的工作目錄
WORKDIR /app

# 安裝系統依賴（Streamlit 與某些套件可能需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 複製並安裝 Python 依賴套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案所有內容到容器中
COPY . .

# 告知 Streamlit 綁定到 Cloud Run 指定的 Port (通常是 8080)
EXPOSE 8080

# 運行 Streamlit 的指令
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]