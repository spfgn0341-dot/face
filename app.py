import streamlit as st
import cv2
import numpy as np
import os
import shutil
import scipy.stats

# ==========================================
# 0. 初期設定とパッチ処理（最優先実行）
# ==========================================

# ------------------------------------------
# A. Scipyエラー回避
# ------------------------------------------
# scipy 1.12以降で削除された binom_test を binomtest に置き換えます
if not hasattr(scipy.stats, 'binom_test'):
    scipy.stats.binom_test = scipy.stats.binomtest

# ------------------------------------------
# B. パスと設定ファイルの修復 (FileNotFoundError & PermissionError対策)
# ------------------------------------------
import feat
import feat.utils
import feat.pretrained

# 1. 書き込み可能なローカルディレクトリを作成
writable_dir = os.path.join(os.getcwd(), 'model_weights')
os.makedirs(writable_dir, exist_ok=True)

# 2. オリジナルのインストール先から model_list.json を探し出し、ローカルにコピーする
#    (これをしないと Detector() が設定ファイルを見つけられず落ちる)
original_feat_dir = os.path.dirname(feat.__file__) # site-packages/feat
original_resources_dir = os.path.join(original_feat_dir, 'resources')
json_filename = 'model_list.json'

src_json_path = os.path.join(original_resources_dir, json_filename)
dst_json_path = os.path.join(writable_dir, json_filename)

# コピー実行（まだローカルになければ）
if os.path.exists(src_json_path) and not os.path.exists(dst_json_path):
    shutil.copy(src_json_path, dst_json_path)

# 3. パッチ適用: py-feat が参照するパスを、準備したローカルフォルダに向ける
def patched_get_resource_path():
    return writable_dir

feat.utils.get_resource_path = patched_get_resource_path
feat.pretrained.get_resource_path = patched_get_resource_path

# ==========================================
# アプリ本体
# ==========================================

from feat import Detector
from PIL import Image

# モデルのロードをキャッシュ
@st.cache_resource
def load_detector():
    # 引数は空（デフォルト設定）
    # パッチのおかげで、ローカルフォルダの設定ファイルを読み、ローカルにモデルをDLします
    return Detector()

def annotate_image(img_array, results):
    img = img_array.copy()
    emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    for index, row in results.iterrows():
        x = int(row['FaceRectX'])
        y = int(row['FaceRectY'])
        w = int(row['FaceRectWidth'])
        h = int(row['FaceRectHeight'])

        # 矩形描画
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

    # RGB → BGR 変換
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.info("モデルを準備中...（初回のみダウンロードに数分かかります）")
    
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