# Phát Triển Hệ Thống Nhận Diện Bệnh Cháy Lá trên Lá Sầu Riêng

## Mô tả

Dự án này phát triển một hệ thống nhận diện bệnh cháy lá trên cây sầu riêng sử dụng ba mô hình học sâu (deep learning models):

- **Grounding DINO**: Mô hình nhận diện đối tượng giúp xác định các đối tượng quan trọng trên lá sầu riêng.
- **Pl@ntNet**: Một mô hình nhận dạng cây cối giúp nhận diện các loài cây và tình trạng của chúng.
- **YOLO (You Only Look Once)**: Mô hình phân tích hình ảnh để nhận diện các đối tượng trong thời gian thực.

## Thư Viện Cần Cài Đặt

Dưới đây là các thư viện cần thiết cho dự án:

```python
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


```

Cài Đặt Mô Hình
Để cài đặt mô hình, bạn cần tải các file mô hình từ Google Drive. Sau khi tải về, bạn cần lưu các mô hình vào thư mục model/ trong dự án của bạn.

Tải các file mô hình từ Google Drive.

Giải nén các file mô hình vào thư mục model/ trong dự án của bạn.

link cài model : https://drive.google.com/drive/folders/1RHGFJnzVF0kTmoFlo1ig_DyQKQfyNjrH?usp=drive_link

Cài Đặt Các Thư Viện Phụ Thuộc trong requirements.txt
