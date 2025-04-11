from PIL import Image
import numpy as np

def floyd_steinberg_dithering(image):
    """Applies Floyd-Steinberg dithering to a grayscale image."""
    img_array = np.array(image, dtype=np.float32)
    height, width = img_array.shape
    
    for y in range(height - 1):
        for x in range(1, width - 1):
            old_pixel = img_array[y, x]
            new_pixel = 255 * (old_pixel > 127)
            img_array[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            img_array[y, x + 1] += quant_error * 7 / 16
            img_array[y + 1, x - 1] += quant_error * 3 / 16
            img_array[y + 1, x] += quant_error * 5 / 16
            img_array[y + 1, x + 1] += quant_error * 1 / 16
            
    return np.clip(img_array, 0, 255).astype(np.uint8)


def process_image(input_path, output_path):
    """Loads an image, rescales it to 128 width, applies dithering, and saves as binary text."""
    image = Image.open(input_path).convert('L')  # Convert to grayscale
    aspect_ratio = image.height / image.width
    new_size = (128, int(128 * aspect_ratio))
    resized_image = image.resize(new_size, Image.LANCZOS)
    dithered_array = floyd_steinberg_dithering(resized_image)
    
    with open(output_path, 'w') as f:
        for row in dithered_array:
            f.write(''.join('1' if pixel > 127 else '0' for pixel in row) + '\n')


if __name__ == "__main__":
    input_image_path = "obrazek2.jpg"  # Ścieżka do obrazu wejściowego
    output_text_path = "obraz.txt"  # Ścieżka do pliku wynikowego
    
    process_image(input_image_path, output_text_path)
    print("Proces zakończony. Zapisano wynik do:", output_text_path)
