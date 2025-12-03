from flask import Flask, render_template, jsonify, redirect, url_for
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, 'static', 'images')

def mat_to_base64(mat):
    """Convert image matrix to base64."""
    _, buffer = cv2.imencode('.jpg', mat)
    return base64.b64encode(buffer).decode('utf-8')

def resize_image_fixed_width(img, target_width=800):
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = target_width / w
    return cv2.resize(img, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)

def crop_black_borders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return img[y:y+h, x:x+w]
    return img

def stitch_two_manually(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        return img1

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return img1

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    corners2_transformed = cv2.perspectiveTransform(corners2, H)

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    all_corners = np.concatenate((corners1, corners2_transformed), axis=0)

    xmin, ymin = np.int32(all_corners.min(axis=0).flatten() - 0.5)
    xmax, ymax = np.int32(all_corners.max(axis=0).flatten() + 0.5)

    if (xmax - xmin) > 6000 or (ymax - ymin) > 6000:
        return img1

    translation = [-xmin, -ymin]
    H_translate = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]], dtype=np.float32)

    result = cv2.warpPerspective(img2, H_translate.dot(H), (xmax - xmin, ymax - ymin))
    img1_shifted = cv2.warpPerspective(img1, H_translate, (xmax - xmin, ymax - ymin))

    mask = img1_shifted[:, :, 0] > 0
    result[mask] = img1_shifted[mask]

    return crop_black_borders(result)

def process_stitching():
    filenames = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
    images = []

    for fname in filenames:
        path = os.path.join(IMAGE_FOLDER, fname)
        img = cv2.imread(path)
        if img is None:
            return None, f"Missing {fname}"
        images.append(resize_image_fixed_width(img, 600))

    try:
        stitcher = cv2.Stitcher_create(mode=1)
        status, pano = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            return pano, None
    except:
        pass

    result = images[0]
    for i in range(1, len(images)):
        result = stitch_two_manually(result, images[i])

    return result, None

def compute_sift_logic():
    path = os.path.join(IMAGE_FOLDER, '2.jpg')
    img = cv2.imread(path)
    if img is None:
        return None, None, None, None, "Missing 2.jpg"

    img = resize_image_fixed_width(img, 600)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    g1 = cv2.GaussianBlur(gray, (3, 3), 1.3)
    g2 = cv2.GaussianBlur(gray, (5, 5), 2.6)
    dog = cv2.absdiff(g1, g2)

    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    _, th = cv2.threshold(dog_norm, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    scratch_kp = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 2:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                scratch_kp.append(cv2.KeyPoint(float(x), float(y), 5))

    img_scratch = cv2.drawKeypoints(img, scratch_kp, None, color=(0, 255, 0))

    sift = cv2.SIFT_create()
    kp_cv, _ = sift.detectAndCompute(gray, None)
    img_cv = cv2.drawKeypoints(img, kp_cv, None, color=(0, 0, 255))

    return img_scratch, img_cv, len(scratch_kp), len(kp_cv), None

@app.route('/')
def home():
    return redirect(url_for('module4'))

@app.route('/module4')
def module4():
    return render_template('module4.html')

@app.route('/module4/run_stitch', methods=['POST'])
def run_stitch():
    pano, error = process_stitching()
    if error:
        return jsonify({'success': False, 'error': error})

    return jsonify({'success': True, 'pano_image': mat_to_base64(pano)})

@app.route('/module4/run_sift', methods=['POST'])
def run_sift():
    scratch, cv_ver, c1, c2, error = compute_sift_logic()
    if error:
        return jsonify({'success': False, 'error': error})

    return jsonify({
        'success': True,
        'scratch_img': mat_to_base64(scratch),
        'cv_img': mat_to_base64(cv_ver),
        'scratch_count': c1,
        'cv_count': c2
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
