import requests

# Set the API endpoint URL
url = 'http://localhost:4000/'

# Set the path to the image file you want to upload
image_path = '/Users/maheshalyana/Documents/Documents - Mahesh का MacBook Air/college/cancer prediction/SegmentationAPI/testimg.jpg'

# Create a multipart/form-data request with the image file
files = {'file': open(image_path, 'rb')}

# Send the POST request to the API endpoint
response = requests.post(url, files=files)

# Check the response status code
if response.status_code == 200:
    print(response)
else:
    print('Error:', response.text)
