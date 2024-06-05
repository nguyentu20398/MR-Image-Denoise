import glob
import os.path

import cv2
import numpy as np
from loguru import logger
if "script" in os.getcwd():
    os.chdir("..")

from script.demo_strategy4_functions import Inference

NOISE_MAP_SIGMAS = [i * 5 for i in range(1, 15)]
cv2.namedWindow('Image Origin', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Cropped', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Results', cv2.WINDOW_AUTOSIZE)

show_cropped_images = None
show_concatenated_images = None


def callBack(*arg):
    pass


def cvt_gray2_bgr(gray_image):
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)


def resize_image_to_target_width(input_image, target_width):
    original_height, original_width = input_image.shape[:2]
    ratio = target_width / original_width
    target_height = int(original_height * ratio)

    resized_img = cv2.resize(input_image, (target_width, target_height))
    return resized_img


def create_trackbar():
    cv2.namedWindow('PARAM', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('sigma', 'PARAM', 0, 4, callBack)
    cv2.createTrackbar('image_path', 'PARAM', 0, 59, callBack)
    cv2.createTrackbar('box_size', 'PARAM', 50, 100, callBack)


def write_key_text(image, key, font_scale=0.6, color=(255, 255, 255), thickness=2):
    # Create a sub-image for writing the key text
    key_text = str(key) if isinstance(key, int) else key
    key_image = np.zeros((40, image.shape[1], 3), dtype=np.uint8)
    text_size = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = 25
    cv2.putText(key_image, key_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return key_image


def write_metrics_text(image, psnr, ssim):
    # Create a sub-image for writing PSNR and SSIM metrics
    psnr_text = f"PSNR: {psnr:.2f}"
    ssim_text = f"SSIM: {ssim:.4f}"
    metrics_text = f"{psnr_text}\n{ssim_text}"
    metrics_image = np.zeros((40, image.shape[1], 3), dtype=np.uint8)
    text_size = cv2.getTextSize(metrics_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = ((image.shape[1] - text_size[0]) // 2) + 20
    text_y = 15
    cv2.putText(metrics_image, psnr_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1)
    cv2.putText(metrics_image, ssim_text, (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
    return metrics_image


def concatenate_images_with_text(images, keys, psnrs, ssims, sigma):
    concatenated_images = []
    for key, image in zip(keys, images):
        if isinstance(key, int):
            key_image = write_key_text(image, key=f"NoiseMap:{key}")
        else:
            key_image = write_key_text(image, key)
        metrics_image = np.zeros((40, image.shape[1], 3), dtype=np.uint8)
        if isinstance(key, int):
            metrics_image = write_metrics_text(image, psnrs[key], ssims[key])
        elif key == "Noise":
            metrics_image = write_key_text(image, key=f"Sigma={sigma}", color=(0,100,200))

        concatenated_image = np.vstack((key_image, image, metrics_image))
        concatenated_images.append(concatenated_image)
    return concatenated_images


def display_images(images):
    num_rows = len(images) // 8
    num_rows += 1 if len(images) % 8 != 0 else 0
    canvas_height = images[0].shape[0] * num_rows
    canvas_width = images[0].shape[1] * min(8, len(images))
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    for i, image in enumerate(images):
        row = i // 8
        col = i % 8
        start_row = row * images[0].shape[0]
        start_col = col * images[0].shape[1]
        canvas[start_row:start_row + image.shape[0], start_col:start_col + image.shape[1]] = image
    return canvas


def pad_image(image, pad_size):
    # Create a border around the image with the specified pad size
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)
    return padded_image


def show_cropped_window(event, x, y, flags, param):
    global sigma, images, keys, psnrs, ssims, cropped_images, concatenated_images, show_cropped_images, show_concatenated_images
    box_size = cv2.getTrackbarPos('box_size', 'PARAM')
    cropped_images = [np.zeros_like(original_image) for _ in range(len(keys))]

    if event == cv2.EVENT_MOUSEMOVE:
        for i, image in enumerate(images):
            img = images[i]
            # Crop area from the current image
            x1, y1 = max(0, x - box_size // 2), max(0, y - box_size // 2)
            x2, y2 = min(img.shape[1], x + box_size // 2), min(img.shape[0], y + box_size // 2)
            cropped_image = pad_image(img[y1:y2, x1:x2], 3)
            cropped_images[i] = cv2.resize(cropped_image, (181, 217))

            images[i] = cv2.rectangle(images[i], (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 150, 0), 1)
        logger.success(f"{x1}, {y1}, {x2}, {y2}")
        cropped_images = concatenate_images_with_text(cropped_images, keys, psnrs, ssims, sigma)
        show_cropped_images = display_images(cropped_images)
        cv2.imshow('Cropped', show_cropped_images)

        # Update the Results window with highlighted areas
        concatenated_images = concatenate_images_with_text(images, keys, psnrs, ssims, sigma)
        show_concatenated_images = display_images(concatenated_images)
        cv2.imshow('Results', show_concatenated_images)


def save_images(folder, sigma, image_path):
    global show_cropped_images, show_concatenated_images
    if show_cropped_images is not None and show_concatenated_images is not None:
        cv2.imwrite(f'{folder}/sigma{sigma}_cropped_{image_path}'.replace(".npy", ".png"), show_cropped_images)
        cv2.imwrite(f'{folder}/sigma{sigma}_concatenated_{image_path}'.replace(".npy", ".png"),
                    show_concatenated_images)

        cv2.imwrite('concatenated_image.jpg', show_concatenated_images)


if __name__ == "__main__":
    create_trackbar()
    demo_ui = Inference()
    data_paths = glob.glob(f"data/test/origin/*.npy")
    sigmas = [10, 25, 35, 45, 55]
    cv2.setMouseCallback('Image Origin', show_cropped_window)
    saved_folder = "report/LV_ThS_TuPhuongNguyen/paper2"
    while True:
        numer_path = cv2.getTrackbarPos('image_path', 'PARAM')
        sigma = sigmas[cv2.getTrackbarPos('sigma', 'PARAM')]
        logger.success(os.path.basename(data_paths[numer_path]))
        img_results, psnrs, ssims = demo_ui.inference(data_paths[numer_path], sigma, NOISE_MAP_SIGMAS)
        original_image = img_results["Original"]
        images = [cvt_gray2_bgr(original_image)]
        images.append(cvt_gray2_bgr(img_results["Noise"]))
        for noise_map_sigma in NOISE_MAP_SIGMAS:
            images.append(cvt_gray2_bgr(img_results[noise_map_sigma]))

        keys = ["Original", "Noise"]
        keys.extend(NOISE_MAP_SIGMAS)
        cv2.imshow(f"Image Origin", original_image)

        # Handle key press events
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_images(saved_folder, sigma, os.path.basename(data_paths[numer_path]))

    # Close all windows
    cv2.destroyAllWindows()
