import matplotlib.pyplot as plt

from skimage import io

url = "https://www.google.com/search?biw=1365&bih=926&tbm=isch&sxsrf=ACYBGNRMkNK_mtFKKt4CStaK5wH7bT03mg%3A1569570937865&sa=1&ei=ecCNXfa_NMPYhwPigbSgBw&q=%EC%8A%88%ED%94%84%EB%A6%BC+%EB%A1%9C%EA%B3%A0&oq=%EC%8A%88%ED%94%84%EB%A6%BC+%EB%A1%9C%EA%B3%A0&gs_l=img.3..35i39.121638.122352..122674...0.0..1.290.554.0j2j1......0....1..gws-wiz-img.NxpPhirXhqg&ved=0ahUKEwi24M2xw_DkAhVD7GEKHeIADXQQ4dUDCAc&uact=5"

#UserAgent 에서 걸림 구글에서 컴퓨터라고 차단함
headers = {'User-Agent':'Mozilla/5.0', 'referer' : 'http://m.naver.com')


img = io.imread(url, headers=headers)

plt.imshow(img)

plt.show()

plt.savefig("이미지이름.png")

