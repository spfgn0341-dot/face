import streamlit as st
import cv2
import numpy as np
import os
import shutil
import scipy.stats
import scipy.integrate  # 追加: 積分モジュールのインポート

# ==========================================
# 0. 初期設定とパッチ処理 (最優先実行)
# ==========================================

# ------------------------------------------
# A. Scipyエラー回避 (binom_test & simps)
# ------------------------------------------

# 1. binom_test のパッチ (scipy 1.12+ 対策)
if not hasattr(scipy.stats, 'binom_test'):
    scipy.stats.binom_test = scipy.stats.binomtest

# 2. simps のパッチ (scipy 1.13+ 対策: 今回のエラー原因)
#    py-feat が古い simps を呼ぼうとするので、新しい simpson に置き換える
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

# ------------------------------------------
# B. モデル保存先と設定ファイルの修復
# ------------------------------------------
import feat
import feat.utils
import feat.pretrained

# 1. 書き込み可能なローカルディレクトリを作成
writable_dir = os.path.join(os.getcwd(), 'model_weights')
os.makedirs(writable_dir, exist_ok=True)

# 2. model_list.json をローカルにコピーする
original_feat_dir = os.path.dirname(feat.__file__)
# リソースフォルダの場所を探す
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
    # 常に上書きコピー（古い情報が残らないように）
    shutil.copy(src_json_path, dst_json_path)

# 3. パッチ適用: 保存先をローカルに向ける
def patched_get_resource_path():
    return writable_dir

feat.utils.get_resource_path = patched_get_resource_path
feat.pretrained.get_resource_path = patched_get_resource_path

# ==========================================
# アプリ本体
# ==========================================

from feat import Detector
from PIL import Image

@st.cache_resource
def load_detector():
    return Detector()

def annotate_image(img_array, results):
    img = img_array.copy()
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    for index, row in results.iterrows():
        # カラム名の揺らぎ吸収
        x, y, w, h = 0, 0, 0, 0
        if 'FaceRectX' in row:
            x, y, w, h = int(row['FaceRectX']), int(row['FaceRectY']), int(row['FaceRectWidth']), int(row['FaceRectHeight'])
        elif 'face_x' in row: 
            x, y, w, h = int(row['face_x']), int(row['face_y']), int(row['face_width']), int(row['face_height'])
        
        if w > 0:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            emotion_texts = []
            for emo in emotion_cols:
                if emo in row:
                    if row[emo] > 0.05:
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

    st.info("モデルをロード・ダウンロード中...（数分かかる場合があります）")
    
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