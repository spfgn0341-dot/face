import streamlit as st
import cv2
import numpy as np
import os
import shutil
import urllib.request
import scipy.stats
import scipy.integrate

# ==========================================
# 0. 初期設定とパッチ処理 (最優先実行)
# ==========================================

# --- A. Scipyエラー回避パッチ ---
if not hasattr(scipy.stats, 'binom_test'):
    scipy.stats.binom_test = scipy.stats.binomtest
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

# --- B. 保存先の確保とパス書き換え ---
import feat
import feat.utils
import feat.pretrained

# 書き込み可能なローカルディレクトリを作成
writable_dir = os.path.join(os.getcwd(), 'model_weights')
os.makedirs(writable_dir, exist_ok=True)

# py-feat が参照するパスをローカルに向けるパッチ
def patched_get_resource_path():
    return writable_dir

feat.utils.get_resource_path = patched_get_resource_path
feat.pretrained.get_resource_path = patched_get_resource_path

# --- C. モデルファイルの強制手動ダウンロード (Hugging Face対応) ---
# 最新の正しいURLリスト
MODEL_URLS = {
    # 顔検出 (RetinaFace)
    "mobilenet0.25_Final.pth": "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth",
    # 感情認識 (ResMaskNet)
    "ResMaskNet_Z_resmasking_dropout1_rot30.pth": "https://huggingface.co/py-feat/resmasknet/resolve/main/ResMaskNet_Z_resmasking_dropout1_rot30.pth",
    # ランドマーク検出 (PFLD)
    "pfld_model_best.pth.tar": "https://huggingface.co/py-feat/pfld/resolve/main/pfld_model_best.pth.tar",
}

def download_if_missing(filename, url):
    file_path = os.path.join(writable_dir, filename)
    if not os.path.exists(file_path):
        with st.spinner(f"モデルをダウンロード中: {filename} ..."):
            try:
                # User-Agentを設定しないと拒否される場合があるため設定
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            except Exception as e:
                # ダウンロード失敗時はここで明確に停止させる
                st.error(f"モデルのダウンロードに失敗しました。\nファイル名: {filename}\nエラー: {e}")
                st.stop()

# model_list.json のコピー処理 (設定ファイル)
original_feat_dir = os.path.dirname(feat.__file__)
possible_resource_dirs = [
    os.path.join(original_feat_dir, 'resources'),
    os.path.join(original_feat_dir, '..', 'resources'),
]
src_json_path = None
for p in possible_resource_dirs:
    if os.path.exists(os.path.join(p, 'model_list.json')):
        src_json_path = os.path.join(p, 'model_list.json')
        break

if src_json_path:
    dst_json_path = os.path.join(writable_dir, 'model_list.json')
    if not os.path.exists(dst_json_path):
        shutil.copy(src_json_path, dst_json_path)

# ==========================================
# アプリ本体
# ==========================================

from feat import Detector
from PIL import Image

@st.cache_resource
def load_detector():
    st.info("モデルファイルをチェックしています...")
    
    # 必要なモデルを順番にチェック＆ダウンロード
    download_if_missing("mobilenet0.25_Final.pth", MODEL_URLS["mobilenet0.25_Final.pth"])
    download_if_missing("ResMaskNet_Z_resmasking_dropout1_rot30.pth", MODEL_URLS["ResMaskNet_Z_resmasking_dropout1_rot30.pth"])
    download_if_missing("pfld_model_best.pth.tar", MODEL_URLS["pfld_model_best.pth.tar"])

    # 全て揃った状態でDetectorを初期化
    return Detector()

def annotate_image(img_array, results):
    img = img_array.copy()
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    for index, row in results.iterrows():
        x, y, w, h = 0, 0, 0, 0
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

    # Detectorロード（初回はここからダウンロード処理が走ります）
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