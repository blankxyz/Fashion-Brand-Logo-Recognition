from bs4 import BeautifulSoup
import urllib.request

def get(max_count = 1):
    base_url = "http://10000img.com/"
    url = "http://10000img.com/ran.php"
    
    count = 1
    while count <= max_count:
        print("+------[ %d번 째 이미지 ]------+" % count)
        
        html = urllib.request.urlopen(url)
        source = html.read()
        
        soup = BeautifulSoup(source, "html.parser")
        
        img = soup.find("img")
        img_src = img.get("src")
        img_url = base_url + img_src
        img_name = img_src.replace("/", "")
        
        urllib.request.urlretrieve(img_url, "./img/" + img_name)
        
        print("이미지 src:", img_src)
        print("이미지 url:", img_url)
        print("이미지 명:", img_name)
        print("\n")
        count += 1 
    else:
        print("크롤링 종료")
get(1)                