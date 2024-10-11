from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# Định nghĩa thư mục upload
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Hàm kiểm tra định dạng ảnh hợp lệ
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Xử lý ảnh denoising
def denoise_image(image_path):
    img = cv2.imread(image_path)
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    denoised_image_path = os.path.join('uploads', f"denoising_{os.path.basename(image_path)}")
    denoised_image_path = denoised_image_path.replace(os.sep, '/')  # Thay đổi phân cách thư mục thành "/"
    cv2.imwrite(os.path.join('static', denoised_image_path), denoised_img)
    return denoised_image_path

# Xử lý ảnh sharpening
def sharpen_image(image_path):
    img = cv2.imread(image_path)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  # Kernel sharpen
    sharpened_img = cv2.filter2D(img, -1, kernel)
    sharpened_image_path = os.path.join('uploads', f"sharpening_{os.path.basename(image_path)}")
    sharpened_image_path = sharpened_image_path.replace(os.sep, '/')  # Thay đổi phân cách thư mục thành "/"
    cv2.imwrite(os.path.join('static', sharpened_image_path), sharpened_img)
    return sharpened_image_path

# Xử lý ảnh Edge Detection (Sobel, Prewitt, Canny)
def apply_edge_detection(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Sobel Edge Detection
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    sobel_edge_path = os.path.join('uploads', f"sobel_{os.path.basename(image_path)}")
    sobel_edge_path = sobel_edge_path.replace(os.sep, '/')  # Thay đổi phân cách thư mục thành "/"
    cv2.imwrite(os.path.join('static', sobel_edge_path), sobel_edges)

    # Prewitt Edge Detection (Sử dụng Sobel)
    prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_x = cv2.filter2D(img.astype(np.float32), -1, prewitt_kernel_x)
    prewitt_y = cv2.filter2D(img.astype(np.float32), -1, prewitt_kernel_y)
    prewitt_edges = cv2.sqrt(prewitt_x**2 + prewitt_y**2)
    prewitt_edge_path = os.path.join('uploads', f"prewitt_{os.path.basename(image_path)}")
    prewitt_edge_path = prewitt_edge_path.replace(os.sep, '/')  # Thay đổi phân cách thư mục thành "/"
    cv2.imwrite(os.path.join('static', prewitt_edge_path), prewitt_edges)

    # Canny Edge Detection
    canny_edges = cv2.Canny(img, 100, 200)
    canny_edge_path = os.path.join('uploads', f"canny_{os.path.basename(image_path)}")
    canny_edge_path = canny_edge_path.replace(os.sep, '/')  # Thay đổi phân cách thư mục thành "/"
    cv2.imwrite(os.path.join('static', canny_edge_path), canny_edges)

    return sobel_edge_path, prewitt_edge_path, canny_edge_path

# Trang chủ để upload ảnh
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Giả sử bạn có các hàm xử lý ảnh sau
            denoised_image = denoise_image(filepath)
            sharpened_image = sharpen_image(filepath)
            sobel_image, prewitt_image, canny_image = apply_edge_detection(filepath)

            return render_template('show_image.html', 
                                   filename=filename, 
                                   denoised_path=denoised_image, 
                                   sharpened_path=sharpened_image, 
                                   sobel_path=sobel_image,
                                   prewitt_path=prewitt_image,
                                   canny_path=canny_image)
    return render_template('index.html')

# Chạy Flask app
if __name__ == '__main__':
    app.run(debug=True)
