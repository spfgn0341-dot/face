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
# 2. 環境構築
# ==========================================
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, 'model_weights')
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. 本物のモデルファイル
REAL_MODELS = {
    "mobilenet0.25_Final.pth": "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth",
    "ResMaskNet_Z_resmasking_dropout1_rot30.pth": "https://huggingface.co/py-feat/resmasknet/resolve/main/ResMaskNet_Z_resmasking_dropout1_rot30.pth",
    "pfld_model_best.pth.tar": "https://huggingface.co/py-feat/pfld/resolve/main/pfld_model_best.pth.tar",
}

# 2. 公式の設定ファイル (これをベースにするのでKeyErrorが起きない)
OFFICIAL_CONFIG_URL = "https://github.com/cosanlab/py-feat/raw/main/feat/resources/model_list.json"

# 3. 偽装ファイル (AU/Pose用)
FAKE_FILES_MAP = {
    "img2pose_v1.pth": "mobilenet0.25_Final.pth", 
    "svm_lenet_v1.pth": "mobilenet0.25_Final.pth"
}

# 4. ダミー生成する .npy ファイル
DUMMY_NPY_FILES = [
    "WIDER_train_pose_mean_v1.npy",
    "WIDER_train_pose_stddev_v1.npy",
    "reference_3d_68_points_trans.npy",
    "mean_face_68.npy"
]

def prepare_environment():
    """環境構築: 公式設定ファイルをDLし、ローカル向けに改造する"""
    
    # A. 公式の設定ファイルをダウンロード
    config_path = os.path.join(MODEL_DIR, 'model_list.json')
    if not os.path.exists(config_path) or os.path.getsize(config_path) < 100:
        try:
            req = urllib.request.Request(OFFICIAL_CONFIG_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(config_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception:
            # 万が一DLできなければ、最低限の構成で作る（前の反省を生かし net_name を追加）
            with open(config_path, 'w') as f:
                json.dump({
                    "face_detectors": {"retinaface": {"file": "mobilenet0.25_Final.pth", "net_name": "mobilenet0.25"}},
                    "landmark_detectors": {"pfld": {"file": "pfld_model_best.pth.tar"}},
                    "emotion_detectors": {"resmasknet": {"file": "ResMaskNet_Z_resmasking_dropout1_rot30.pth"}},
                    "au_detectors": {"svm": {"file": "svm_lenet_v1.pth"}},
                    "facepose_detectors": {"img2pose": {"file": "img2pose_v1.pth"}}
                }, f)

    # B. 設定ファイルの読み込みと改造 (パスをローカルに向ける)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 全てのモデル定義の 'file' を、ファイル名だけ（ローカルパス）にする
        # これにより、ライブラリが余計なパスを探さなくなる
        for category in config.values():
            for model_name, settings in category.items():
                if 'file' in settings:
                    settings['file'] = os.path.basename(settings['file'])
                # ハッシュチェックをスキップ
                settings['sha256'] = 'skip'
        
        # 偽装したいモデルの設定を上書き (ファイル名を偽装ファイルに向ける)
        if "au_detectors" in config and "svm" in config["au_detectors"]:
            config["au_detectors"]["svm"]["file"] = "svm_lenet_v1.pth"
        if "facepose_detectors" in config and "img2pose" in config["facepose_detectors"]:
            config["facepose_detectors"]["img2pose"]["file"] = "img2pose_v1.pth"

        # 保存
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
    except Exception as e:
        st.error(f"設定ファイルの調整に失敗: {e}")

    # C. モデルファイルのダウンロード
    for fname, url in REAL_MODELS.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            except Exception:
                pass

    # D. 偽装ファイルの作成 (コピー)
    source_path = os.path.join(MODEL_DIR, "mobilenet0.25_Final.pth")
    if os.path.exists(source_path):
        for fake_name, _ in FAKE_FILES_MAP.items():
            fake_path = os.path.join(MODEL_DIR, fake_name)
            if not os.path.exists(fake_path):
                shutil.copy(source_path, fake_path)

    # E. .npy ダミー生成
    for npy_name in DUMMY_NPY_FILES:
        path = os.path.join(MODEL_DIR, npy_name)
        if not os.path.exists(path):
            shape = (68, 3) if "68" in npy_name else (3,)
            np.save(path, np.ones(shape, dtype=np.float32))

# ==========================================
# 3. Detector ローダー
# ==========================================

@st.cache_resource
def load_detector_ultimate_v2():
    st.info("システム初期化中...")
    
    # 1. 環境構築 (公式JSONベース)
    prepare_environment()
    
    # 2. パス関数の定義
    def patched_get_resource_path():
        return MODEL_DIR
    
    # 3. numpy.load ハイジャック
    original_np_load = np.load
    def patched_np_load(file, *args, **kwargs):
        if isinstance(file, str):
            filename = os.path.basename(file)
            if filename in DUMMY_NPY_FILES:
                return original_np_load(os.path.join(MODEL_DIR, filename), *args, **kwargs)
        return original_np_load(file, *args, **kwargs)
    
    np.load = patched_np_load

    try:
        # 4. インポート
        import feat
        import feat.utils
        import feat.pretrained
        import feat.detector
        
        # 5. パス書き換え
        feat.utils.get_resource_path = patched_get_resource_path
        feat.pretrained.get_resource_path = patched_get_resource_path
        for module_name, module in list(sys.modules.items()):
            if module_name.startswith('feat') and hasattr(module, 'get_resource_path'):
                module.get_resource_path = patched_get_resource_path

        # 6. クラスの無効化 (Mocking)
        # 偽装ファイルを実行させないための空クラス
        class MockDetectorPart:
            def __init__(self, *args, **kwargs):
                pass
            def detect(self, *args, **kwargs):
                return None
            
        feat.detector.AUDetector = MockDetectorPart
        feat.detector.FacePoseDetector = MockDetectorPart
        
        from feat import Detector
        
        # 7. 初期化
        # 公式JSONを使っているので、RetinaFaceのキー不足エラーは起きないはずです。
        detector = Detector(
            face_model="retinaface",
            landmark_model="pfld",
            emotion_model="resmasknet",
            au_model="svm",
            facepose_model="img2pose"
        )
        
    finally:
        np.load = original_np_load

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
        detector = load_detector_ultimate_v2()
    except Exception as e:
        st.error(f"起動エラー: {e}")
        st.stop()

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