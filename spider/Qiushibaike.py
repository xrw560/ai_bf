# 要求：
'''
使用xpath 取到用户头像，用户名，用户性别，用户年龄
糗事内容

'''
import requests
from lxml.html import etree

url="https://www.qiushibaike.com/"
r=requests.get(url)
html=etree.HTML(r.content.decode())
all_qiushi=html.xpath("//div[@id='content-left']/div")
for one in all_qiushi:
    userImg=one.xpath("./div[1]/a[1]/img/@src")
    if userImg:
        userImg="http:"+userImg[0]
        username = one.xpath("./div[1]/a[2]/h2/text()")[0]
        userage = one.xpath("./div[1]/div/text()")[0]
        usersex = one.xpath("./div[1]/div/@class")[0]
        usersex = usersex[14:-4]
    else:
        userImg="https://static.qiushibaike.com/images/thumb/anony.png?v=b61e7f5162d14b7c0d5f419cd6649c87"
        username="匿名用户"
        userage="0"
        usersex='man'

    userQiushi = ''.join(one.xpath("./a[1]/div/span[1]/text()")).replace("\n",'')
    userQiushiImg=one.xpath("./div[@class='thumb']/a/img/@src")
    if userQiushiImg:
        userQiushiImg="https:"+userQiushiImg[0]
    else:
        userQiushiImg="暂无"
    # print(userQiushiImg)
    one_userinfo="%s,%s,%s,%s,%s,%s"%(username,userage,usersex,userImg,userQiushi,userQiushiImg)
    print(one_userinfo)