import streamlit as st
import cv2
import numpy as np
import os
import shutil
import urllib.request
import sys
import json
import scipy.stats
import scipy.integrate

# ==========================================
# 0. 環境設定とパッチ (最優先)
# ==========================================

# --- A. Scipyバージョン対策 ---
if not hasattr(scipy.stats, 'binom_test'):
    scipy.stats.binom_test = scipy.stats.binomtest
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

# --- B. 保存先ディレクトリの準備 ---
writable_dir = os.path.join(os.getcwd(), 'model_weights')
os.makedirs(writable_dir, exist_ok=True)

# --- C. モデルファイルの強制手動ダウンロード ---
MODEL_URLS = {
    "mobilenet0.25_Final.pth": "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth",
    "ResMaskNet_Z_resmasking_dropout1_rot30.pth": "https://huggingface.co/py-feat/resmasknet/resolve/main/ResMaskNet_Z_resmasking_dropout1_rot30.pth",
    "pfld_model_best.pth.tar": "https://huggingface.co/py-feat/pfld/resolve/main/pfld_model_best.pth.tar",
}

def download_if_missing(filename, url):
    file_path = os.path.join(writable_dir, filename)
    # ファイルが存在し、サイズが0でないならスキップ
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with st.spinner(f"準備中: {filename} をダウンロードしています..."):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            except Exception as e:
                st.error(f"ダウンロード失敗: {filename}\nエラー: {e}")
                st.stop()

# --- D. 設定ファイル(model_list.json)を強制的に作成 ---
# ライブラリのバージョン差異によるエラーを防ぐため、
# 現在手元にあるファイルだけを定義した正しいJSONをここで作ります。
def create_custom_model_list():
    model_list_data = {
        "face_model": {
            "retinaface": {
                "urls": [MODEL_URLS["mobilenet0.25_Final.pth"]],
                "file": "mobilenet0.25_Final.pth",
                "sha256": "skip" # ハッシュチェックをスキップさせる意図
            }
        },
        "landmark_model": {
            "pfld": {
                "urls": [MODEL_URLS["pfld_model_best.pth.tar"]],
                "file": "pfld_model_best.pth.tar",
                "sha256": "skip"
            }
        },
        "emotion_model": {
            "resmasknet": {
                "urls": [MODEL_URLS["ResMaskNet_Z_resmasking_dropout1_rot30.pth"]],
                "file": "ResMaskNet_Z_resmasking_dropout1_rot30.pth",
                "sha256": "skip"
            }
        },
        "au_model": {},       # 不要なので空定義
        "facepose_model": {}  # 不要なので空定義
    }
    
    json_path = os.path.join(writable_dir, 'model_list.json')
    with open(json_path, 'w') as f:
        json.dump(model_list_data, f)

# ==========================================
# E. ライブラリ読み込みとパスの強制書き換え
# ==========================================
import feat
from feat import Detector
from PIL import Image

# パス設定関数を書き換え
def patched_get_resource_path():
    return writable_dir

# ありとあらゆるモジュールの get_resource_path を書き換える
if hasattr(feat.utils, 'get_resource_path'):
    feat.utils.get_resource_path = patched_get_resource_path
if hasattr(feat.pretrained, 'get_resource_path'):
    feat.pretrained.get_resource_path = patched_get_resource_path

for module_name, module in list(sys.modules.items()):
    if module_name.startswith('feat'):
        if hasattr(module, 'get_resource_path'):
            module.get_resource_path = patched_get_resource_path

# ==========================================
# アプリ本体
# ==========================================

@st.cache_resource
def load_detector():
    st.info("システムの準備中...")
    
    # 1. モデルファイルのダウンロード
    for fname, url in MODEL_URLS.items():
        download_if_missing(fname, url)
        
    # 2. 正しい設定ファイル(JSON)を生成・保存
    create_custom_model_list()

    # 3. Detectorの初期化
    # ここで指定する名前(retinafaceなど)は、上で作ったJSONのキーと完全一致するため、
    # ValueErrorは理論上発生しません。
    return Detector(
        face_model="retinaface",
        landmark_model="pfld",
        emotion_model="resmasknet",
        au_model=None,
        facepose_model=None
    )

def annotate_image(img_array, results):
    img = img_array.copy()
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    for index, row in results.iterrows():
        x, y, w, h = 0, 0, 0, 0
        
        # カラム名の違いを吸収
        if 'FaceRectX' in row:
            x, y, w, h = int(row['FaceRectX']), int(row['FaceRectY']), int(row['FaceRectWidth']), int(row['FaceRectHeight'])
        elif 'face_x' in row:
            x, y, w, h = int(row['face_x']), int(row['face_y']), int(row['face_width']), int(row['face_height'])
        
        if w > 0:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            emotion_texts = []
            for emo in emotion_cols:
                if emo in row and row[emo] > 0.05:
                    emotion_texts.append(f"{emo}: {row[emo]:.2f}")

            if emotion_texts:
                text = ", ".join(emotion_texts[:2])
            else:
                text = "Neutral"
            
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img

st.title("表情分析アプリ")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # RGBAなどの場合はRGBに変換
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    detector = load_detector()

    st.write("分析中...")
    try:
        result = detector.detect_image(img_bgr)
    except Exception as e:
        st.error(f"解析エラー: {e}")
        st.stop()

    if result.empty:
        st.warning("顔が検出されませんでした。")
    else:
        st.write("解析結果データ:")
        st.dataframe(result)
        
        annotated_img_bgr = annotate_image(img_bgr, result)
        annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_img_rgb, caption='解析結果', use_column_width=True)