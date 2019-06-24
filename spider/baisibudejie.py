import requests
from bs4 import BeautifulSoup
import re
import time
#取到百思不得姐段子  里的
# 用户名，时间，标题，图片(保存名为标题)

# 1 分析，各个数据的位置

def download(page=1):
    url="http://www.budejie.com/%d"%(page)
    headers={
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"
    }
    r=requests.get(url,headers=headers)
    soup=BeautifulSoup(r.text,"html.parser")
    total=soup.find('div',attrs={'class':'j-r-list'}).ul.contents
    while "\n" in total:
        total.remove("\n")

    duanzi_info=[]
    for once_li in total:
        author=once_li.find('a',attrs={'class':'u-user-name'}).string
        # print(author)
        addTime=once_li.find("span",attrs={'class':'u-time'}).string
        # print(addTime)
        title=once_li.find('div',attrs={'class':'j-r-list-c-desc'}).find('a').string
        # print((title))
        image=once_li.find('div',attrs={'class':'j-r-list-c-img'}).find('img').attrs['data-original']
        # print(image)
        #把图片下载下载
        image_resource=requests.get(image,headers=headers,stream=True)
        last_name=image[-3:] #后缀名
        save_title=re.sub("[\\/\|\*\?><\"':]|(\n)",'',title)[0:20]
        # print(save_title)
        # exit()
        with open("./Budejie/%s.%s"%(save_title,last_name),'wb') as file:
            for j in image_resource.iter_content(1024):
                file.write(j)

        #把段子的信息保存起来，保存成csv文件
        duanzi_info.append("%s,%s,%s,%s\n"%(author,addTime,save_title,"./Budejie/%s.%s"%(title,last_name)))

    with open("budejie.csv",'a',encoding='utf-8') as file1:
        for once_info in duanzi_info:
            file1.write(once_info)
    print('第%d页下载成功'%(page))

for i in range(5,11):
    download(i)
    time.sleep(2)

# list1=['1','\n','\n','\n']
# konggeIndex=[]
# for i in range(len(list1)):
#     if list1[i]=='\n':
#         konggeIndex.append(i)
#
# print(konggeIndex)
# for j in konggeIndex:
#     list1.remove('\n')
#
# print(list1)




# str1="./Budejie/精华整理 | 减肥期间应该怎么吃？.jpg"
# res=re.sub("[\\/\|\*\?><\"':]",'',str1)
# print(res)



