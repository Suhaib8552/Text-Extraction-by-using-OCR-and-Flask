from flask import Flask, render_template, request, jsonify
import pytesseract
from PIL import Image
import PyPDF2
import os
import cv2
import numpy as np
from spellchecker import SpellChecker
import re
from scipy.ndimage import rotate

selected_roi = None
app = Flask(__name__)


@app.route('/')
def index():
    image_path = "/static/pics/logo.jpeg"
    return render_template('index.html', image_path=image_path)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/developersdetails')
def developersdetails():
    return render_template('developersdetails.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')    

@app.route('/softwareguide')
def softwareguide():
    return render_template('softwareguide.html')    


@app.route('/extract', methods=['POST'])
def extract():
    global selected_roi

    # Get the uploaded file from the request
    file = request.files['file']
    file_ext = file.filename.rsplit('.', 1)[1].lower()

    if file_ext == 'pdf':
        extracted_text = extract_from_pdf(file)
    else:
        is_captcha = request.form.get('captcha')  # Check if it's a CAPTCHA image
        if is_captcha:
            extracted_text = extract_text_from_captcha(file)
        elif selected_roi is not None:
            extracted_text = extract_from_selected_roi(file, selected_roi)
            selected_roi = None
        else:
            extracted_text = extract_from_image(file)

    return jsonify({'text': extracted_text})

def extract_from_selected_roi(file, roi):
    # Load and preprocess the image
    image = Image.open(file)
    grayscale_image = image.convert('L')
    processed_image = preprocess_image(image)
    thresholded_image = grayscale_image.point(lambda x: 0 if x < 127 else 255, '1')

    # Extract text from the selected ROI
    selected_region = thresholded_image.crop(roi)
    extracted_text = pytesseract.image_to_string(selected_region, lang='hin+eng', config='--psm 6 --oem 1')

    # Apply spellchecker and autocorrection to English words
    spell = SpellChecker()
    words = extracted_text.split()
    corrected_words = []
    for word in words:
        if word.isalpha() and not is_vertical_word(word):
            corrected_word = spell.correction(word.lower())
            if corrected_word != word.lower():
                corrected_words.append('<span class="autocorrect">' + corrected_word + '</span>')
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)

    # Reconstruct the corrected text
    corrected_text = ' '.join(corrected_words)

    # Remove HTML tags for non-highlighted words

    return corrected_text

def extract_from_pdf_snip(file, roi):
    # Load the PDF file
    pdf = PdfFileReader(file)

    # Get the total number of pages in the PDF
    total_pages = pdf.getNumPages()

    # Prompt the user to enter the page number for snipping
    page_number = int(input('Enter the page number for snipping (1 to {}): '.format(total_pages)))

    # Read the specified page from the PDF
    page = pdf.getPage(page_number - 1)

    # Extract the page as an image
    page_image = page.extract_images()[0][1]

    # Convert the page image to PIL format
    image = Image.fromarray(page_image)

    # Preprocess the image and extract text from the selected ROI
    grayscale_image = image.convert('L')
    processed_image = preprocess_image(image)
    thresholded_image = grayscale_image.point(lambda x: 0 if x < 127 else 255, '1')
    selected_region = thresholded_image.crop(roi)
    extracted_text = pytesseract.image_to_string(selected_region, lang='hin+eng', config='--psm 6 --oem 1')

    # Apply spellchecker and autocorrection to English words
    spell = SpellChecker()
    words = extracted_text.split()
    corrected_words = []
    for word in words:
        if word.isalpha() and not is_vertical_word(word):
            corrected_word = spell.correction(word.lower())
            if corrected_word != word.lower():
                corrected_words.append('<span class="autocorrect">' + corrected_word + '</span>')
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)

    # Reconstruct the corrected text
    extracted_text = ' '.join(corrected_words)

    # Remove HTML tags for non-highlighted words

    return extracted_text

def extract_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    extracted_text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        lines = page_text.strip().split("\n")
        corrected_lines = []

        # Apply spellchecker and autocorrection to English words
        spell = SpellChecker()
        for line in lines:
            words = line.split()
            corrected_words = []
            for word in words:
                if word.isalpha() and not is_vertical_word(word):
                    corrected_word = spell.correction(word.lower())
                    if corrected_word is not None and corrected_word != word.lower():
                        corrected_words.append('<span class="autocorrect">' + corrected_word + '</span>')
                    else:
                        corrected_words.append(word)
                else:
                    corrected_words.append(word)
            corrected_line = ' '.join(corrected_words)
            corrected_lines.append(corrected_line)

        # Join the corrected lines with newlines
        corrected_page_text = '\n'.join(corrected_lines)
        extracted_text += corrected_page_text + "\n\n"

    return extracted_text

def extract_from_image(file):
    # Load and preprocess the image
    image = Image.open(file)
    grayscale_image = image.convert('L')
    processed_image = preprocess_image(image)
    thresholded_image = grayscale_image.point(lambda x: 0 if x < 127 else 255, '1')

    # Extract text from the image
    extracted_text = pytesseract.image_to_string(thresholded_image, lang='eng+hin', config='--psm 6 --oem 1')

    # Apply spellchecker and autocorrection to English words
    spell = SpellChecker()
    words = extracted_text.split()
    corrected_words = []
    for word in words:
        if word.isalpha() and not is_vertical_word(word):
            corrected_word = spell.correction(word.lower())
            if corrected_word != word.lower():
                corrected_words.append('<span class="autocorrect">' + corrected_word + '</span>')
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)

    # Reconstruct the corrected text
    corrected_text = ' '.join(corrected_words)

    # Remove HTML tags for non-highlighted words

    return corrected_text

def is_vertical_word(word):
    vertical_characters = ['१', '२', '३', '४', '५', '६', '७', '८', '९', '०']  # Add more vertical characters if needed
    return any(char in word for char in vertical_characters)
    

def preprocess_image(image):
    # Convert PIL Image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    vertical_lines = detect_vertical_lines(opencv_image)

    rotated_image = rotate_image(opencv_image,vertical_lines)

    # Apply image preprocessing techniques (e.g., resizing, rotation, thresholding, etc.)
    resized_image = resize_image(rotated_image, max_height=1200)
   
    thresholded_image = threshold_image(resized_image)

    # Convert back to PIL Image format
    processed_image = Image.fromarray(cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2RGB))

    return processed_image

def detect_vertical_lines(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply morphological operations to enhance vertical lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    return vertical_lines

def rotate_image(image, vertical_lines):
    # Perform line detection using Hough transform
    lines = cv2.HoughLinesP(vertical_lines, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Calculate the average angle of the lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
    avg_angle = np.mean(angles)

    # Rotate the image by the average angle
    rotated_image = rotate(image, avg_angle, reshape=False)

    return rotated_image


def resize_image(image, max_height):
    height, width = image.shape[:2]
    if height > max_height:
        ratio = max_height / height
        new_width = int(width * ratio)
        new_height = max_height
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image
    return image

def threshold_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding
    thresholded = cv2.adaptiveThreshold(gray, 127, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    return thresholded

def extract_text_from_captcha(file):
    image = Image.open(file)

    # Preprocess the CAPTCHA image
    processed_image = preprocess_captcha_image(image)

    # Perform CAPTCHA solving
    text = solve_captcha(processed_image)

    return text

def preprocess_captcha_image(image):
    # Convert the image to grayscale
    grayscale_image = image.convert('L')

    # Apply thresholding to convert the image to black and white
    thresholded_image = grayscale_image.point(lambda x: 0 if x < 128 else 255, '1')

    return thresholded_image

def solve_captcha(image):
    # Perform contour detection to identify individual characters
    contours, _ = cv2.findContours(np.array(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours from left to right based on their x-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    captcha_text = ""
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the character region from the original image
        character_region = image.crop((x, y, x + w, y + h))

        # Perform OCR on the character region
        character = pytesseract.image_to_string(character_region, config='--psm 10 --oem 3')

        # Append the character to the CAPTCHA text
        captcha_text += character

    return captcha_text

def is_hindi_word(word):
    # Check if the word contains any Devanagari characters (indicating Hindi)
    return any(char >= '\u0900' and char <= '\u097F' for char in word)    


@app.route('/snip', methods=['POST'])
def snip():
    global selected_roi

    # Get the uploaded file from the request
    file = request.files['file']
    file_ext = file.filename.rsplit('.', 1)[1].lower()

    if file_ext == 'pdf':
        selected_roi = snip_from_pdf(file)
    else:
        # Read the image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Display the image and let the user select the ROI
        roi = cv2.selectROI("Select ROI", image)
        cv2.destroyWindow("Select ROI")

        # Store the selected ROI coordinates
        selected_roi = roi

    return jsonify({'message': 'ROI selected successfully.'})

def snip_from_pdf(file):
    # Load the PDF file
    pdf = PdfFileReader(file)

    # Get the total number of pages in the PDF
    total_pages = pdf.getNumPages()

    # Prompt the user to enter the page number for snipping
    page_number = int(input('Enter the page number for snipping (1 to {}): '.format(total_pages)))

    # Read the specified page from the PDF
    page = pdf.getPage(page_number - 1)

    # Extract the page as an image
    page_image = page.extract_images()[0][1]

    # Display the page image and let the user select the ROI
    image = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)
    roi = cv2.selectROI("Select ROI", image)
    cv2.destroyWindow("Select ROI")

    # Return the selected ROI coordinates
    return roi







if __name__ == '__main__':
    app.run(debug=True)
