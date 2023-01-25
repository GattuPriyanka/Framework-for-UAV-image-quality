import cv2
import os

def computePixelRange(folderPath):
    avg_overExposed = 0
    avg_UnderExposed = 0
    img_count = 0

    for f in os.listdir(folderPath):
        if os.path.isfile(os.path.join(folderPath, f)):
            full_path = os.path.join(folderPath,f)
            img_count += 1
            image = cv2.imread(full_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            underExposed = 0
            overExposed = 0
            height, width = gray_image.shape

            for row in range(height):
                for col in range(width):
                    if gray_image[row][col] <= 5:
                        underExposed += 1
                    elif gray_image[row][col] >= 250:
                        overExposed += 1
            if(underExposed > 0):
                avg_UnderExposed = (underExposed/ (width * height))
            if(overExposed > 0):
                avg_overExposed = (overExposed / (width * height))

    print("Under Exposed: ", (avg_UnderExposed/img_count))
    print("Over Exposed: ", (avg_overExposed/img_count))
    return (avg_UnderExposed/img_count),(avg_overExposed/img_count)

