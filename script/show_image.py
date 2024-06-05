import cv2
import numpy as np

cv2.namedWindow('Stacked Images', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Image Origin', cv2.WINDOW_AUTOSIZE)

save_full_images = {
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
    6: None,
    7: None,
}
save_crop_images = {
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
    6: None,
    7: None,
}


def callBack(*arg):
    pass

def create_trackbar():
    cv2.namedWindow('param', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('sigma', 'param', 0, 4, callBack)
    cv2.createTrackbar('image_path', 'param', 0, 59, callBack)
    cv2.createTrackbar('box_size', 'param', 50, 100, callBack)

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
    global stacked_image
    # Create a blank image to display the stacked images and cropped areas
    height, width = images[0].shape[:2]
    stacked_image = np.zeros((2 * height + 60, 7 * width + 60, 3), dtype=np.uint8)

    # Create a text matrix to write text
    text_matrix = np.zeros((30, 7 * width + 60, 3), dtype=np.uint8)

    # Add space at the top of the stacked image
    # stacked_image[30:height + 30, :] = 255

    # Populate the stacked image with original images and cropped areas
    for i in range(len(images)):
        stacked_image[30:height+30, i * width + (i * 10):i * width + (i * 10) + width, :] = images[i]
        stacked_image[height + 60:2 * height + 60, i * width + (i * 10):i * width + (i * 10) + width, :] = \
        cv2.resize(cropped_images[i], (images[i].shape[1], images[i].shape[0]))
        save_full_images[i] = stacked_image[30:height+30, i * width + (i * 10):i * width + (i * 10) + width, :]
        save_crop_images[i] = stacked_image[height + 60:2 * height + 60, i * width + (i * 10):i * width + (i * 10) + width, :]


    # Display the text on the text matrix
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Define image names
    image_names = ['Original', 'Noise', 'Strategy1', 'Strategy2', 'Strategy3', 'Strategy4', 'Strategy5']

    for i, name in enumerate(image_names):
        text_size = cv2.getTextSize(name, font, font_scale, font_thickness)[0]
        text_x = i * width + (i * 10) + (width - text_size[0]) // 2
        text_y = 20
        cv2.putText(text_matrix, name, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Vertically stack the text matrix with the stacked image
    stacked_image = np.vstack((text_matrix, stacked_image))

    # Display the stacked image

    cv2.imshow("Stacked Images", stacked_image)


if __name__ == "__main__":
    create_trackbar()

    # cv2.namedWindow(f"Image Origin")
    cv2.setMouseCallback("Image Origin", mouse_callback)

    sigmas = [10, 25, 35, 45, 55]
    while True:
        numer_path = cv2.getTrackbarPos('image_path', 'param')
        sigma = sigmas[cv2.getTrackbarPos('sigma', 'param')]

        # Path to the original image
        path_image1 = f"report/LV_ThS_TuPhuongNguyen/LuanVan/origin/{numer_path}.png"

        # Load the original image
        img1 = cv2.imread(path_image1)

        # Load other images based on sigma values
        images = [img1]
        show_images = [img1]

        noise_img = cv2.imread(f"report/LV_ThS_TuPhuongNguyen/LuanVan/noise/sigma{sigma}/{numer_path}.png")
        images.append(noise_img)
        show_images.append(noise_img)

        for i in range(5):
            img_path = f"report/LV_ThS_TuPhuongNguyen/LuanVan/strategy{i + 1}/sigma{sigma}/{numer_path}.png"
            images.append(cv2.imread(img_path))
            show_images.append(cv2.imread(img_path))

        # Initialize the list of cropped images
        cropped_images = [np.zeros_like(img1) for _ in range(len(images))]
        # Display the original images
        cv2.imshow(f"Image Origin", images[0])

        # Break the loop when the 'Esc' key is pressed
        waitkey = cv2.waitKey(1)
        if waitkey == ord('q'):
            break
        if waitkey == ord('s'):
            saved_folder = "report/LV_ThS_TuPhuongNguyen/LuanVan/report"
            saved_path = f"{saved_folder}/sigma{sigma}_{numer_path}.png"
            for i, name in enumerate(['Original', 'Noise', 'Strategy1', 'Strategy2', 'Strategy3', 'Strategy4', 'Strategy5']):
                cv2.imwrite(f"{saved_folder}/sigma{sigma}_{numer_path}_{name}_full.png", save_full_images[i])
                cv2.imwrite(f"{saved_folder}/sigma{sigma}_{numer_path}_{name}_cropped.png", save_crop_images[i])

            cv2.imwrite(saved_path, stacked_image)

    # Close all windows
    cv2.destroyAllWindows()
