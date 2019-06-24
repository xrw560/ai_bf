from bs4 import BeautifulSoup
import requests

url="https://www.woyaogexing.com/touxiang/nan/2018/663561.html"

r=requests.get(url) #请求该网页，得到网页源代码
soup=BeautifulSoup(r.content.decode('utf-8'),"html.parser") #把网页源代码解析为beautifulsoup节点形式

#得到所有的图片
# all_img=soup.find_all('img',attrs={"class":"lazy"}) #查找代码里，属性class值为lazy的img标签，返回列表
# num=0
# for once_img in all_img: #遍历列表，每次得到的是一个img标签
#     once_img_src="https:"+once_img.attrs['src'] #取到img标签里的src属性，并且拼接上https，得到图片链接
#     img_resource=requests.get(once_img_src,stream=True) #请求图片链接，得到图像资源
#     with open("./HeadImg/%d.jpg"%(num),'wb') as file: #创建文件，准备保存资源
#         for j in img_resource.iter_content(1024): #每次从得到的资源里读取一部分，遇到大文件不会卡顿
#             file.write(j) #把读取到的这一部分写入到文件里
#     num+=1

this_div=soup.find('div',attrs={'class':"hot-tags"}) #返回列表
all_a=this_div.find_all('a')
for once_a in all_a:
    print(once_a.string)

