import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# ------------------ SETTINGS ------------------
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = r"C:\Users\taral\OneDrive\Desktop\CoE 2025\Models\sam_vit_b_01ec64.pth"
IMAGE_FOLDER = r"C:\Users\taral\OneDrive\Desktop\CoE 2025\Images\Sample"

MIN_AREA = 50
MAX_AREA = 5000

# ------------------ LOAD SAM ------------------
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
predictor = SamPredictor(sam)
print("SAM loaded")

# ------------------ HELPER FUNCTIONS ------------------
def detect_blobs_otsu(gray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    merged = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
        if MIN_AREA < area < MAX_AREA and 0.5 < circularity < 1.5:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                points.append((cx, cy))
    return points

def detect_blobs_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 50, 50])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                points.append((cx, cy))
    return points

def get_fallback_points(image):
    h, w = image.shape[:2]
    return [(w//2, h//2), (w//4, h//4), (3*w//4, h//4), (w//4, 3*h//4), (3*w//4, 3*h//4)]

# ------------------ MAIN LOOP ------------------
segmented_images = {}  # store masks in-memory

for img_name in sorted(os.listdir(IMAGE_FOLDER)):
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Skipping {img_name}, could not read")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Multi-strategy blob detection
    points_otsu = detect_blobs_otsu(gray)
    points_hsv = detect_blobs_hsv(image)
    candidate_points = points_otsu + points_hsv

    # Fallback if no points found
    if len(candidate_points) == 0:
        candidate_points = get_fallback_points(image)

    input_points = np.array(candidate_points)
    input_labels = np.ones(len(input_points))

    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )

    if masks is not None and len(masks) > 0:
        mask = masks[0].astype(np.uint8)
        segmented_images[img_name] = mask
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        segmented_images[img_name] = mask

    # ------------------ CREATE OVERLAY ------------------
    overlay = image.copy()
    overlay[mask > 0] = [255, 0, 0]  # highlight mask in red
    alpha = 0.5
    blended = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)

    # ------------------ SHOW SIDE BY SIDE ------------------
    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].imshow(image)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(blended)
    axs[1].set_title("Segmented Overlay")
    axs[1].axis('off')

    plt.suptitle(f"{img_name}")
    plt.show()

print(f"Segmentation done for {len(segmented_images)} images. Ready for scaling-law pipeline.")