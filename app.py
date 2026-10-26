from pathlib import Path
from PIL import Image
import base64
import requests
import os
import time
import threading

import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

# =====================================================
# 基本設定
# =====================================================
MODEL_PATH = Path("models") / "test_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# 自動關閉 Streamlit
# =====================================================
HEARTBEAT_FILE = "browser_alive.txt"
AUTO_CLOSE_SECONDS = 60

def update_heartbeat():
    with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
        f.write(str(time.time()))

def auto_shutdown_checker():
    while True:
        time.sleep(2)

        if not os.path.exists(HEARTBEAT_FILE):
            continue

        try:
            with open(HEARTBEAT_FILE, "r", encoding="utf-8") as f:
                last_time = float(f.read().strip())
        except:
            continue

        if time.time() - last_time > AUTO_CLOSE_SECONDS:
            print("瀏覽器已關閉，自動停止 Streamlit...")
            os._exit(0)

# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config(
    page_title="DeepVeri",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "shutdown_thread_started" not in st.session_state:
    st.session_state.shutdown_thread_started = True
    update_heartbeat()
    threading.Thread(target=auto_shutdown_checker, daemon=True).start()

st.title("DeepVeri🧠")
st.caption("功能：真假圖判斷、AI解釋、Google Lens 找網圖來源")

# =====================================================
# API KEY
# =====================================================
DEFAULT_IMGBB_API_KEY = "0c1b161c3cb6f2d6f8ba4272048b0d95"
DEFAULT_SERPAPI_KEY = "bfb0bb83a42a8502e046f21e44cc222ac74baff2cce7c4e75e1503d00a357500"

with st.sidebar:
    st.header("API 設定")

    user_imgbb = st.text_input(
        "ImgBB API Key（已預設，可填自己的）",
        type="password"
    )

    user_serpapi = st.text_input(
        "SerpApi API Key（已預設，可填自己的）",
        type="password"
    )

    IMGBB_API_KEY = user_imgbb.strip() if user_imgbb.strip() else DEFAULT_IMGBB_API_KEY
    SERPAPI_KEY = user_serpapi.strip() if user_serpapi.strip() else DEFAULT_SERPAPI_KEY

# =====================================================
# 模型檢查
# =====================================================
if not MODEL_PATH.exists():
    st.error("缺少模型：models/best_model_better_diffusion.pth")
    st.stop()

# =====================================================
# 載入模型
# =====================================================
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        class_to_idx = checkpoint.get("class_to_idx", {"fake": 0, "real": 1})
    else:
        model.load_state_dict(checkpoint)
        class_to_idx = {"fake": 0, "real": 1}

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model.to(device)
    model.eval()

    return model, idx_to_class

model, idx_to_class = load_model()

# =====================================================
# 圖片前處理
# =====================================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# AI 解釋
# =====================================================
def make_explanation(pred_class, confidence, fake_prob, real_prob):

    if pred_class == "fake":
        return f"""
### 模型判斷數據
- 假圖機率：{fake_prob*100:.2f}%
- 真圖機率：{real_prob*100:.2f}%
- 信心值：{confidence:.2f}%

### AI 評價
模型認為這張圖片可能存在 AI 生成特徵，例如紋理不自然、光影不一致或細節平滑化。
"""

    elif pred_class == "real":
        return f"""
### 模型判斷數據
- 真圖機率：{real_prob*100:.2f}%
- 假圖機率：{fake_prob*100:.2f}%
- 信心值：{confidence:.2f}%

### AI 評價
模型認為這張圖片較接近真實拍攝圖片，紋理與光影分布較自然。
"""

    else:
        return f"""
### 模型判斷數據
- 假圖機率：{fake_prob*100:.2f}%
- 真圖機率：{real_prob*100:.2f}%
- 信心值：{confidence:.2f}%

### AI 評價
模型目前無法明確判定，真假特徵接近。
"""

# =====================================================
# ImgBB 上傳
# =====================================================
def upload_to_imgbb(api_key, uploaded_file):
    image_bytes = uploaded_file.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    url = "https://api.imgbb.com/1/upload"

    data = {
        "key": api_key,
        "image": image_base64
    }

    r = requests.post(url, data=data, timeout=30)
    result = r.json()

    image_url = result["data"]["url"]

    return image_url

# =====================================================
# Google Lens 搜尋
# =====================================================
def google_lens_search(serpapi_key, image_url):

    url = "https://serpapi.com/search.json"

    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": serpapi_key
    }

    r = requests.get(url, params=params, timeout=40)
    data = r.json()

    results = []

    visual_matches = data.get("visual_matches", [])

    for item in visual_matches[:10]:
        results.append({
            "title": item.get("title", "無標題"),
            "link": item.get("link", ""),
            "source": item.get("source", ""),
            "thumbnail": item.get("thumbnail", "")
        })

    return results

# =====================================================
# 上傳圖片
# =====================================================
uploaded = st.file_uploader(
    "上傳圖片",
    type=["jpg", "jpeg", "png", "jfif", "webp"]
)

if uploaded:

    img = Image.open(uploaded).convert("RGB")

    st.image(
        img,
        caption="上傳圖片",
        use_container_width=True
    )

    # =================================================
    # AI 判斷
    # =================================================
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        prob = torch.softmax(output, dim=1)[0]

    fake_idx = None
    real_idx = None

    for idx, name in idx_to_class.items():

        if name.lower() == "fake":
            fake_idx = idx

        elif name.lower() == "real":
            real_idx = idx

    fake_prob = prob[fake_idx].item()
    real_prob = prob[real_idx].item()

    fake_percent = fake_prob * 100
    real_percent = real_prob * 100

    # =================================================
    # 判定區間
    # =================================================
    if fake_prob >= 0.60:
        pred_class = "fake"

    elif real_prob >= 0.60:
        pred_class = "real"

    else:
        pred_class = "uncertain"
    # =================================================
    # 顯示結果
    # =================================================
    st.subheader("1. 真假圖片判斷")

    if pred_class == "fake":
        st.error("判定：假圖")

    elif pred_class == "real":
        st.success("判定：真圖")

    else:
        st.warning("判定：不確定")

    st.write(f"假圖 ：{fake_percent:.2f}%")
    st.write(f"真圖 ：{real_percent:.2f}%")

    # =================================================
    # AI 評價
    # =================================================
    st.subheader("2. AI 評價")

    confidence = max(fake_percent, real_percent)

    st.markdown(
        make_explanation(
            pred_class,
            confidence,
            fake_prob,
            real_prob
        )
    )

    # =================================================
    # Google Lens
    # =================================================
    st.subheader("3. Google Lens 以圖搜圖")

    if st.button("開始 Google Lens 以圖搜圖"):

        try:
            with st.spinner("正在上傳圖片..."):
                image_url = upload_to_imgbb(IMGBB_API_KEY, uploaded)

            with st.spinner("正在搜尋 Google Lens..."):
                results = google_lens_search(
                    SERPAPI_KEY,
                    image_url
                )

            if not results:
                st.warning("沒有找到相關結果")

            else:

                for i, item in enumerate(results, 1):

                    st.markdown(f"### 結果 {i}")

                    if item["thumbnail"]:
                        st.image(item["thumbnail"], width=220)

                    st.write("標題：", item["title"])
                    st.write("來源：", item["source"])

                    if item["link"]:
                        st.markdown(
                            f"[開啟相關網址]({item['link']})"
                        )

                    st.divider()

        except Exception as e:
            st.error(f"搜尋失敗：{e}")

# =====================================================
# 瀏覽器心跳
# =====================================================
components.html(
    """
    <script>
    setInterval(function(){
        window.parent.postMessage({type: 'streamlit:rerun'}, '*');
    }, 2000);
    </script>
    """,
    height=0
)

update_heartbeat()