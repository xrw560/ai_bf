from lxml import etree
import requests
'''
解析 etree.HTML()
//任意位置搜索
//nodename 找nodename标签  
//div           === soup.find('div')
//div[@class='u-txt']   === soup.find('div',attrs={'class':'u-txt'})




'''


# url="http://www.budejie.com"
# headers={
#         "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"
#     }
# r=requests.get(url,headers=headers)
# html=etree.HTML(r.content.decode("utf-8"))
# html=etree.parse()
# print(html)

#parse 严格准守w3c规范
html=etree.parse(open('index.html','rb'))
# print(html.xpath("//div")) #查找所有的div标签
# print(html.xpath("//div[@class='content']")) # 查找网页中所有div class为content的
# print(html.xpath("//div[@class='content']/ul/li[1]")) #找到第一个li
# print(html.xpath("//div[@class='content']/ul/li[last()]/text()")) #最后一个li的文本
# print(html.xpath("//li[position()<3]/text()")) #查找所在位置小于3的
# print(html.xpath("//li[@cc]/text()")) #查找带有cc属性的li标签
# print(html.xpath("//*[@cc]/text()")) #查找带有cc属性的任意标签
# for i in html.xpath("//ul[li>22]"): #查找li标签值大于22 的ul
#     print('-'*22)
#     print(i.text)


# 常用方法
# text() 取文本
#  @class 取class属性  @* 取所有属性
# html.xpath("//div[@info='*']/ul/li/text()")
# print(html.xpath("//div[1]/@*"))





