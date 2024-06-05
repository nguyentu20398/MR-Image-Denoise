import glob
import os.path

import cv2
import numpy as np
if "script" in os.getcwd():
    os.chdir("..")

from script.demo_inference_functions import DemoUI

cv2.namedWindow('Stacked Images', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Image Origin', cv2.WINDOW_AUTOSIZE)


def callBack(*arg):
    pass


def create_trackbar():
    cv2.namedWindow('param', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('sigma', 'param', 0, 4, callBack)
    cv2.createTrackbar('run_flag', 'param', 0, 1, callBack)

    cv2.createTrackbar('image_path', 'param', 0, 59, callBack)
    cv2.createTrackbar('box_size', 'param', 50, 100, callBack)


def cvt_gray2_bgr(gray_image):
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)


def mouse_callback(event, x, y, flags, param):
    global images, cropped_images, show_images

    if event == cv2.EVENT_MOUSEMOVE:
        # Define bounding box size
        box_size = cv2.getTrackbarPos('box_size', 'param')

        # Iterate through each image
        for i in range(len(images)):
            img = images[i]

            # Crop area from the current image
            x1, y1 = max(0, x - box_size // 2), max(0, y - box_size // 2)
            x2, y2 = min(img.shape[1], x + box_size // 2), min(img.shape[0], y + box_size // 2)
            cropped_img = img[y1:y2, x1:x2]
            images[i] = cv2.rectangle(images[i], (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 150, 0), 1)

            # Update the global list of cropped images
            cropped_images[i] = cropped_img

        # Display the stacked images and cropped areas
        display_images()


def resize_image_to_target_width(input_image, target_width):
    original_height, original_width = input_image.shape[:2]
    ratio = target_width / original_width
    target_height = int(original_height * ratio)

    resized_img = cv2.resize(input_image, (target_width, target_height))
    return resized_img


def display_images():
    global stacked_image, psnr, ssim
    # Create a blank image to display the stacked images and cropped areas
    height, width = images[0].shape[:2]
    stacked_image = np.zeros((int(2.25 * height) + 90, 7 * width + 60, 3), dtype=np.uint8)

    # Create a text matrix to write text
    text_matrix = np.zeros((30, 7 * width + 60, 3), dtype=np.uint8)

    # Populate the stacked image with original images, cropped areas, PSNR, and SSIM
    for i in range(len(images)):
        stacked_image[30:height + 30, i * width + (i * 10):i * width + (i * 10) + width, :] = images[i]
        stacked_image[height + 60:2 * height + 60, i * width + (i * 10):i * width + (i * 10) + width, :] = \
            cv2.resize(cropped_images[i], (images[i].shape[1], images[i].shape[0]))

    # Display the text on the text matrix
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 1

    # Define image names
    image_names = ['Original', 'Noise', 'Strategy1', 'Strategy2', 'Strategy3', 'Strategy4', 'Strategy5']

    for i, name in enumerate(image_names):
        text_size = cv2.getTextSize(name, font, font_scale, font_thickness)[0]
        text_x = i * width + (i * 10) + (width - text_size[0]) // 2
        text_y = 20
        cv2.putText(text_matrix, name, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Display PSNR and SSIM under "Strategy*" images
        if name.startswith("Strategy"):
            psnr_text = f"PSNR: {psnr[name]:.2f}"
            ssim_text = f"SSIM: {ssim[name]:.4f}"

            psnr_size = cv2.getTextSize(psnr_text, font, font_scale, font_thickness)[0]
            ssim_size = cv2.getTextSize(ssim_text, font, font_scale, font_thickness)[0]

            psnr_x = i * width + (i * 10) + (width - psnr_size[0]) // 2
            ssim_x = i * width + (i * 10) + (width - ssim_size[0]) // 2

            psnr_y = 2 * height + 90
            ssim_y = psnr_y + 20

            cv2.putText(stacked_image, psnr_text, (psnr_x, psnr_y), font, font_scale, (255, 255, 255), font_thickness,
                        cv2.LINE_AA)
            cv2.putText(stacked_image, ssim_text, (ssim_x, ssim_y), font, font_scale, (255, 255, 255), font_thickness,
                        cv2.LINE_AA)

    # Vertically stack the text matrix with the stacked image
    stacked_image = np.vstack((text_matrix, stacked_image))

    # Display the stacked image
    cv2.imshow("Stacked Images", resize_image_to_target_width(stacked_image, 1910))

if __name__ == "__main__":
    create_trackbar()
    demo_ui = DemoUI()
    # cv2.namedWindow(f"Image Origin")
    cv2.setMouseCallback("Image Origin", mouse_callback)
    data_paths = glob.glob(f"data/test/origin/*.npy")
    sigmas = [10, 25, 35, 45, 55]


    while True:
        numer_path = cv2.getTrackbarPos('image_path', 'param')
        sigma = sigmas[cv2.getTrackbarPos('sigma', 'param')]
        run_flag = cv2.getTrackbarPos('run_flag', 'param')

        number_data_path = os.path.basename(data_paths[numer_path])
        demo_ui.run(number_data_path=number_data_path, sigma=sigma, run_flag=run_flag)

        result = demo_ui.get_result()

        original_image = result["image"]["Original"]

        images = [cvt_gray2_bgr(original_image)]
        images.append(cvt_gray2_bgr(result["image"]["Noise"]))
        images.append(cvt_gray2_bgr(result["image"]["Strategy1"]))
        images.append(cvt_gray2_bgr(result["image"]["Strategy2"]))
        images.append(cvt_gray2_bgr(result["image"]["Strategy3"]))
        images.append(cvt_gray2_bgr(result["image"]["Strategy4"]))
        images.append(cvt_gray2_bgr(result["image"]["Strategy5"]))
        psnr = result["psnr"]
        ssim = result["ssim"]

        # Display the original images
        cv2.imshow(f"Image Origin", images[0])
        cropped_images = [np.zeros_like(original_image) for _ in range(len(images))]

        # Break the loop when the 'Esc' key is pressed
        waitkey = cv2.waitKey(1)
        if waitkey == ord('q'):
            break
        if waitkey == ord('s'):
            saved_path = f"sigma{sigma}_{numer_path}.png"
            cv2.imwrite(saved_path, stacked_image)

    # Close all windows
    cv2.destroyAllWindows()
