import streamlit as st
import os
import sys
import shutil
import urllib.request
import json
import scipy.stats
import scipy.integrate
import numpy as np
import cv2
from PIL import Image

# ==========================================
# 1. Scipy 互換性パッチ
# ==========================================
if not hasattr(scipy.stats, 'binom_test'):
    scipy.stats.binom_test = scipy.stats.binomtest
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

# ==========================================
# 2. ファイルシステムとモデルの準備
# ==========================================
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, 'model_weights')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URLS = {
    "mobilenet0.25_Final.pth": "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth",
    "ResMaskNet_Z_resmasking_dropout1_rot30.pth": "https://huggingface.co/py-feat/resmasknet/resolve/main/ResMaskNet_Z_resmasking_dropout1_rot30.pth",
    "pfld_model_best.pth.tar": "https://huggingface.co/py-feat/pfld/resolve/main/pfld_model_best.pth.tar",
}

def download_file(filename, url):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

# 設定ファイル生成 (AUとFacePoseを偽装設定する)
def create_model_config():
    config_path = os.path.join(MODEL_DIR, 'model_list.json')
    # 既存のファイルを指すようにしてファイル存在チェックをパスさせる
    dummy_file = "mobilenet0.25_Final.pth" 
    
    config_data = {
        "face_detectors": {
            "retinaface": {
                "file": "mobilenet0.25_Final.pth",
                "urls": [MODEL_URLS["mobilenet0.25_Final.pth"]],
                "sha256": "skip"
            }
        },
        "landmark_detectors": {
            "pfld": {
                "file": "pfld_model_best.pth.tar",
                "urls": [MODEL_URLS["pfld_model_best.pth.tar"]],
                "sha256": "skip"
            }
        },
        "emotion_detectors": {
            "resmasknet": {
                "file": "ResMaskNet_Z_resmasking_dropout1_rot30.pth",
                "urls": [MODEL_URLS["ResMaskNet_Z_resmasking_dropout1_rot30.pth"]],
                "sha256": "skip"
            }
        },
        "au_detectors": {
            # "xgb" を指定されたら、とりあえず存在するファイルを指しておく
            "xgb": {
                "file": dummy_file, 
                "urls": [],
                "sha256": "skip"
            }
        },
        "facepose_detectors": {
            # "img2pose" も同様
            "img2pose": {
                "file": dummy_file,
                "urls": [],
                "sha256": "skip"
            }
        }
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f)

# ==========================================
# 3. Detector ローダー (Import Delay & Class Mocking)
# ==========================================

@st.cache_resource
def load_detector_safe():
    st.info("モデル環境を構築中...")
    
    # A. ダウンロード & 設定ファイル生成
    for fname, url in MODEL_URLS.items():
        download_file(fname, url)
    create_model_config()
    
    # B. パッチ関数定義
    def patched_get_resource_path():
        return MODEL_DIR
    
    # C. インポート
    import feat
    import feat.utils
    import feat.pretrained
    import feat.detector # ここで Detector クラスなどが読み込まれる
    
    # D. パス書き換え (get_resource_path)
    feat.utils.get_resource_path = patched_get_resource_path
    feat.pretrained.get_resource_path = patched_get_resource_path
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith('feat') and hasattr(module, 'get_resource_path'):
            module.get_resource_path = patched_get_resource_path

    # E. クラスの無効化 (Mocking)
    # AUとFacePoseは、クラスの初期化時にファイルをロードしようとするので、
    # クラスそのものを「何もしないダミークラス」に差し替えます。
    
    class MockDetectorPart:
        def __init__(self, *args, **kwargs):
            # 初期化時は何もしない（ファイルのロードを回避）
            pass
        def detect(self, *args, **kwargs):
            # 検出時はNoneを返すか、空の結果を返す
            return None
            
    # ライブラリ内のクラス定義を上書き
    feat.detector.AUDetector = MockDetectorPart
    feat.detector.FacePoseDetector = MockDetectorPart
    
    # F. 初期化
    from feat import Detector
    
    # バリデーションを通過させるために、有効な名前("xgb", "img2pose")を指定する。
    # ただし、上のMock化により、実際にロードは行われません。
    detector = Detector(
        face_model="retinaface",
        landmark_model="pfld",
        emotion_model="resmasknet",
        au_model="xgb",         # Noneだとエラーになるので有効な名前を指定
        facepose_model="img2pose" # 同上
    )
    return detector

# ==========================================
# 4. アプリケーション UI
# ==========================================

st.title("表情分析アプリ")

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

            text = ", ".join(emotion_texts[:2]) if emotion_texts else "Neutral"
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img

uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    try:
        detector = load_detector_safe()
    except Exception as e:
        st.error(f"モデルのロードに失敗しました。\n詳細: {e}")
        st.stop()

    st.write("分析中...")
    try:
        result = detector.detect_image(img_bgr)
    except Exception as e:
        st.error(f"画像解析中にエラーが発生しました: {e}")
        st.stop()

    if result.empty:
        st.warning("顔が検出されませんでした。")
    else:
        st.write("解析結果データ:")
        st.dataframe(result)
        
        annotated_img_bgr = annotate_image(img_bgr, result)
        annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_img_rgb, caption='解析結果', use_column_width=True)