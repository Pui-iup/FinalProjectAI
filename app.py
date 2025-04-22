import sys

import openai
sys.path.append("D:/App/Demo/model/GroundingDINO/")  # Thêm thư mục gốc vào sys.path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import torch
from openai import OpenAI
from groundingdino.util.inference import load_model, load_image, predict as grounding_predict  # tránh trùng tên
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
import requests
import os

# Cấu hình Grounding DINO
CONFIG_PATH = "model/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "model/weights/groundingdino_swint_ogc.pth"
TEXT_PROMPT = "person , animal , car , tree , flower , fruit , leaf"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
KEYWORDS_TREE = ["tree", "flower", "fruit", "leaf"]  # dùng để quyết định đi tiếp B2

# API của PlantNet
API_KEY = "2b10tYCp6YTdKLkLKSb7dE4LO"
API_URL = f"https://my-api.plantnet.org/v2/identify/all?api-key={API_KEY}"

# client = OpenAI


# Tải mô hình DINO
try:
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    model.eval()
    model = model.to("cpu")
    print("✅ Grounding DINO model loaded successfully on CPU.")
except Exception as e:
    print(f"❌ Error loading Grounding DINO model: {e}")
    model = None

# Tải mô hình YOLO một lần khi khởi động ứng dụng
try:
    model_yolo = YOLO("model/best.pt")
    print("✅ YOLO model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model_yolo = None

# Flask app
app = Flask(__name__)
CORS(app)
# B1: Nhận diện bằng DINO
def detect_objects_dino(image_path):
    if model is None:
        print("⚠️ Grounding DINO model is not loaded.")
        return []

    try:
        image_source = Image.open(image_path).convert("RGB")
        _, image = load_image(image_path)

        boxes, logits, phrases = grounding_predict(
            model,
            image,
            TEXT_PROMPT,
            BOX_THRESHOLD,
            TEXT_THRESHOLD,
        )

        processed_phrases_with_duplicates = []
        for phrase in phrases:
            words = phrase.lower().split()
            unique_words_in_phrase = sorted(list(set(words)))
            processed_phrase = " ".join(unique_words_in_phrase)
            processed_phrases_with_duplicates.append(processed_phrase)

        # Loại bỏ các phrase trùng lặp trong danh sách
        processed_phrases = sorted(list(set(processed_phrases_with_duplicates)))

        print("🔎 DINO đã nhận diện (tránh lặp từ và phrase):", processed_phrases)

        found_tree_keywords = False
        for phrase in processed_phrases:
            for keyword in KEYWORDS_TREE:
                if keyword in phrase:
                    found_tree_keywords = True
                    break
            if found_tree_keywords:
                break

        if found_tree_keywords:
            print("👉 Có phát hiện từ khóa liên quan đến cây/cỏ. Tiếp tục sang B2.")
            return processed_phrases # Trả về danh sách các phrase đã xử lý (duy nhất)
        else:
            print("⛔ Không phát hiện từ khóa liên quan đến cây/cỏ. Dừng tại B1.")
            return []

    except Exception as e:
        print(f"❌ Error during DINO object detection: {e}")
        return []



# B2: Gửi ảnh cho PlantNet API (không gửi thông tin về organ)
def identify_plantnet(image_path, detected_keywords):
    try:
        with open(image_path, 'rb') as img_file:
            files_data = {'images': img_file}

            print("📤 Gửi ảnh tới PlantNet mà không có thông tin về organ.")

            response = requests.post(API_URL, files=files_data)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ PlantNet API request failed with status code: {response.status_code}, response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error during PlantNet API call: {e}")
        return None

# B3: Dự đoán tình trạng lá sầu riêng (sử dụng mô hình đã tải)
def predict_leaf_condition(image_path):
    if model_yolo is None:
        print("⚠️ YOLO model is not loaded.")
        return {"error": "YOLO model is not loaded."}
    try:
        # Tiến hành dự đoán
        results = model_yolo(image_path)
        predictions = []
        for *xyxy, conf, cls in results[0].boxes.data.tolist():
            label = model_yolo.names[int(cls)]
            predictions.append({
                'xmin': int(xyxy[0]),
                'ymin': int(xyxy[1]),
                'xmax': int(xyxy[2]),
                'ymax': int(xyxy[3]),
                'confidence': float(conf),
                'class': label,
            })
        return {"predictions": predictions, "original_image_path": image_path} # Trả về cả đường dẫn ảnh gốc
    except Exception as e:
        print(f"❌ Error during YOLO leaf condition prediction: {e}")
        return {"error": str(e)}

def draw_bbox_on_image(image_path, predictions):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Chọn một font và kích thước. Bạn có thể cần điều chỉnh đường dẫn font.
        try:
            font = ImageFont.truetype("arial.ttf", 20)  # Sử dụng font Arial, kích thước 20
        except IOError:
            font = ImageFont.load_default() # Nếu không tìm thấy font, dùng font mặc định

        for pred in predictions:
            xmin, ymin, xmax, ymax = int(pred['xmin']), int(pred['ymin']), int(pred['xmax']), int(pred['ymax'])
            label = pred['class']
            confidence = pred['confidence']

            if label == 'Leaf_Healthy':
                outline_color = "blue"
            elif label == 'Leaf_Blight':
                outline_color = "red"
            else:
                outline_color = "black"  # Màu mặc định cho các nhãn khác

            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=outline_color, width=2)

            # Vẽ chữ với font và màu đã chọn
            text = f"{label}: {confidence:.2f}"
            draw.text((xmin, ymin - 25), text, fill=outline_color, font=font) # Dịch chữ lên trên bbox

        # Lưu ảnh đã vẽ vào bộ nhớ
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    except Exception as e:
        print(f"❌ Error drawing bounding boxes: {e}")
        return None
# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')
# Định nghĩa route cho header
@app.route("/header")
def header():
    return render_template("header.html")

# Định nghĩa route cho footer
@app.route("/footer")
def footer():
    return render_template("footer.html")

@app.route('/upload')
def upload_page():
    return render_template('upload.html')
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        leaf_condition_data = data.get("leaf_condition_data", [])
        dino_detections = data.get("dino_detections", [])
        plantnet_results = data.get("plantnet_results")
        not_durian = data.get("not_durian", False)
        is_durian = data.get("is_durian")
        best_match = data.get("best_match", "")
        prompt = ""

        if leaf_condition_data:
            detected_classes = [condition['class'] for condition in leaf_condition_data]
            class_list = ', '.join(detected_classes)
            if "Leaf_Blight" in detected_classes or "Leaf Blight" in detected_classes:
                prompt = (
                    f"The image contains these detected leaf conditions: {class_list}. "
                    "Please provide advice on durian leaf blight, including causes, treatment, and suitable pesticides. "
                    "If healthy leaves are also present, advise monitoring and treating infected leaves immediately."
                )
            elif "Leaf_Healthy" in detected_classes:
                prompt = (
                    f"The image contains these detected leaf conditions: {class_list}. "
                    "The durian leaves appear healthy. Please advise the user to continue monitoring for any future issues."
                )
            else:
                prompt = (
                    f"The image contains these detected leaf conditions: {class_list}. "
                    "Please give general advice on the condition of the durian plant based on these findings."
                )

        elif dino_detections and not plantnet_results:
            object_list = ', '.join(dino_detections)
            prompt = (
                f"The image includes the following objects detected by the model: {object_list}. "
                "Based on this, give feedback. These may not be relevant to the durian diagnosis model. "
                "Recommend uploading a clear durian leaf image for better analysis."
            )

        elif dino_detections and plantnet_results and not_durian:
            object_list = ', '.join(dino_detections)
            prompt = (
                f"Objects detected: {object_list}. PlantNet suggests the plant is likely: {best_match}. "
                "This is not a durian leaf. Suggest the user upload a durian-related image if possible."
            )

        elif dino_detections and plantnet_results and is_durian and not leaf_condition_data:
            object_list = ', '.join(dino_detections)
            prompt = (
                f"DINO detected: {object_list}. PlantNet match: {best_match} (durian). "
                "However, our model could not detect the leaf condition. The image may be unclear or not contain diseased durian leaves. "
                "Apologize and suggest the user upload a clearer image showing durian leaves."
            )

        elif user_message:
            prompt = user_message
        else:
            return jsonify({"error": "No valid data to process."}), 400

        if prompt:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Your name are GreenMind"
                            "You are a friendly and knowledgeable agricultural expert. "
                            "Always answer in a clean, readable format with proper line breaks and bullet points. "
                            "Help farmers by giving clear, simple advice about durian leaf health, especially leaf blight. "
                            "Use clear sections: Causes, Prevention, Treatment, and Recommended Pesticides. "
                            "Keep responses under 300 tokens and always answer in English. "
                            "Format should be easy to read on mobile."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7,
            )
            reply = response.choices[0].message.content.strip()
            return jsonify({"reply": reply})
        else:
            return jsonify({"reply": "No prompt to process."}), 200

    except Exception as e:
        return jsonify({"reply": "Error communicating with AI: " + str(e)}), 500


# @app.route("/chat", methods=["POST"])
# def chat():
#     try:
#         data = request.get_json()
#         user_message = data.get("message", "").strip()

#         if not user_message:
#             return jsonify({"error": "Message is empty."}), 400

#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are an agricultural expert who only answers questions related to plant diseases, "
#                         "especially leaf blight on durian trees. "
#                         "If the user asks unrelated questions, politely respond that you do not have information in that field. "
#                         "Always respond in English."
#                     )
#                 },
#                 {"role": "user", "content": user_message}
#             ],
#             max_tokens=300,
#             temperature=0.7,
#         )

#         reply = response.choices[0].message.content
#         return jsonify({"reply": reply})

#     except Exception as e:
#         return jsonify({"reply": "Lỗi khi kết nối với AI: " + str(e)}), 500

# Hàm main_pipeline tích hợp B1 → B2 → B3
def main_pipeline(image_path):
    response_data = {"status": "Bắt đầu phân tích...", "steps": []}
    is_durian = False # Khởi tạo biến is_durian

    # B1: DINO nhận diện các đối tượng
    response_data["status"] = "Đang xử lý bước 1: Nhận diện đối tượng (DINO)..."
    dino_detected = detect_objects_dino(image_path)
    response_data["steps"].append({"step": "DINO", "result": dino_detected})

    plantnet_results_data = None
    leaf_condition_data = None
    message = ""
    output_image_base64 = None

    # B2: PlantNet nhận diện thực vật
    response_data["status"] = "Đang xử lý bước 2: Nhận diện loài (PlantNet)..."
    plantnet_results = identify_plantnet(image_path, dino_detected)
    plantnet_results_data = plantnet_results
    response_data["steps"].append({"step": "PlantNet", "result": plantnet_results})

    if plantnet_results:
        # print("📤 PlantNet trả về:", plantnet_results)
        for result in plantnet_results.get('results', []):
            common_names = result.get('species', {}).get('commonNames', [])
            if any("durian" in name.lower() for name in common_names):
                is_durian = True
                print("✅ PlantNet xác nhận có khả năng là sầu riêng.")
                break

    # B3: Dự đoán tình trạng lá sầu riêng (sử dụng mô hình đã tải)
    if "leaf" in dino_detected and is_durian:
        response_data["status"] = "Đang xử lý bước 3: Phân tích tình trạng lá (YOLO)..."
        message = "Phát hiện lá sầu riêng → kiểm tra tình trạng"
        leaf_condition_result = predict_leaf_condition(image_path)
        leaf_condition_data = leaf_condition_result.get("predictions")
        response_data["steps"].append({"step": "YOLO", "result": leaf_condition_data})
        message = "Lá sầu riêng được nhận diện và kiểm tra tình trạng"
        if leaf_condition_data:
            original_image_path = leaf_condition_result.get("original_image_path")
            output_image_base64 = draw_bbox_on_image(original_image_path, leaf_condition_data)
            response_data["output_image"] = f"data:image/jpeg;base64,{output_image_base64}"
    elif "leaf" in dino_detected and not is_durian:
        message = "Phát hiện lá nhưng PlantNet không nhận diện là sầu riêng."
    elif not "leaf" in dino_detected and is_durian:
        message = "PlantNet nhận diện là sầu riêng nhưng không phát hiện lá."
    else:
        message = "Không phát hiện lá hoặc không nhận diện được sầu riêng."

    response_data["dino_detections"] = dino_detected
    response_data["plantnet_results"] = plantnet_results_data
    response_data["leaf_condition"] = leaf_condition_data
    response_data["message"] = message
    response_data["status"] = "Hoàn tất phân tích!"
    response_data["is_durian"] = is_durian # Thêm is_durian vào response
    return response_data


# Route cho dự đoán
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided", "status": "Lỗi: Không có file ảnh"}), 400

    image_file = request.files['image']
    upload_folder = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    image_path = os.path.join(upload_folder, image_file.filename)
    image_file.save(image_path)

    response_data = main_pipeline(image_path)

    return jsonify(response_data)

# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

