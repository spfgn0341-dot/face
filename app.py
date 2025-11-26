import streamlit as st
import cv2
import numpy as np
import os
import shutil
import urllib.request
import sys
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
# 必要最小限のモデル（顔検出、ランドマーク、感情）のみダウンロード
MODEL_URLS = {
    # 顔検出 (RetinaFace)
    "mobilenet0.25_Final.pth": "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth",
    # 感情認識 (ResMaskNet)
    "ResMaskNet_Z_resmasking_dropout1_rot30.pth": "https://huggingface.co/py-feat/resmasknet/resolve/main/ResMaskNet_Z_resmasking_dropout1_rot30.pth",
    # ランドマーク検出 (PFLD - 感情分析の前処理で必要)
    "pfld_model_best.pth.tar": "https://huggingface.co/py-feat/pfld/resolve/main/pfld_model_best.pth.tar",
}

def download_if_missing(filename, url):
    file_path = os.path.join(writable_dir, filename)
    if not os.path.exists(file_path):
        with st.spinner(f"モデルをダウンロード中: {filename} ..."):
            try:
                # User-Agent偽装（403エラー回避）
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            except Exception as e:
                st.error(f"ダウンロード失敗: {filename}\nエラー: {e}")
                st.stop()

# ==========================================
# D. ライブラリ読み込みと「総書き換え」
# ==========================================
import feat
from feat import Detector
from PIL import Image

# 読み込まれている全てのモジュールをチェックし、パス設定関数を書き換える
def patched_get_resource_path():
    return writable_dir

if hasattr(feat.utils, 'get_resource_path'):
    feat.utils.get_resource_path = patched_get_resource_path
if hasattr(feat.pretrained, 'get_resource_path'):
    feat.pretrained.get_resource_path = patched_get_resource_path

for module_name, module in list(sys.modules.items()):
    if module_name.startswith('feat'):
        if hasattr(module, 'get_resource_path'):
            module.get_resource_path = patched_get_resource_path

# model_list.json の救出
original_feat_dir = os.path.dirname(feat.__file__)
possible_paths = [
    os.path.join(original_feat_dir, 'resources', 'model_list.json'),
    os.path.join(original_feat_dir, '..', 'resources', 'model_list.json')
]
for p in possible_paths:
    if os.path.exists(p):
        shutil.copy(p, os.path.join(writable_dir, 'model_list.json'))
        break

# ==========================================
# アプリ本体
# ==========================================

@st.cache_resource
def load_detector():
    st.info("モデルの準備とロード中...")
    
    # 1. 必要なファイルだけダウンロード
    for fname, url in MODEL_URLS.items():
        download_if_missing(fname, url)

    # 2. Detectorの初期化
    # 【重要】ファイルを持っていないモデル（AU, FacePose）は None にして読み込ませない
    return Detector(
        face_model="retinaface",
        landmark_model="pfld",
        emotion_model="resmasknet",
        au_model=None,       # これを読み込むとエラーになるので無効化
        facepose_model=None  # これも無効化
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
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    detector = load_detector()

    st.write("分析中...")
    try:
        # detect_image実行時も不要な検出をスキップするよう念のため指定
        result = detector.detect_image(img_bgr)
    except Exception as e:
        st.error(f"解析エラー: {e}")
        st.stop()

    if result.empty:
        st.warning("顔が検出されませんでした。")
    else:
        st.write("解析結果データ:")
        # 必要なカラムだけ表示（AUなどのカラムが無い場合のエラー回避）
        st.dataframe(result)
        annotated_img_bgr = annotate_image(img_bgr, result)
        annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_img_rgb, caption='解析結果', use_column_width=True)