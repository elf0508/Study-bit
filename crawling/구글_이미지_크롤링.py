# 기존에 google_images_download 라이브러리를 설치 했다면, 지우고
# 업뎃 버전으로 재설치 한다.

# cmd --> pip uninstall google_images_download 

# --> (1) git clone https://github.com/Joeclinton1/google-images-download.git
# 또는 (2) pip install git+https://github.com/Joeclinton1/google-images-download.git


from google_images_download import google_images_download

response = google_images_download.googleimagesdownload() 

arguments = {"keywords":"Polar bears,baloons,Beaches","limit":20,"print_urls":True}  # creating list of arguments
paths = response.download(arguments)   # passing the arguments to the function
print(paths)   # printing absolute paths of the downloaded images