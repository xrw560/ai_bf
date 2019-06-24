import requests
from bs4 import BeautifulSoup
import re

url = "https://www.woyaogexing.com/touxiang/nan/2018/663561.html"

# 使用正则匹配
# r = requests.get(url)
# res_text = r.content.decode('utf-8')
# print(res_text)
# regx = re.compile("<img class=\"lazy\" src=\"(.*?)\"")
# all_img = regx.findall(res_text)
# for i in all_img:
#     print("https:" + i)
# print(all_img)

# 使用bs4 解析 返回的文本
r = requests.get(url)
res_text = r.content.decode('utf-8')

# print(type(res_text))
# Beautifuls(txt,method)
# txt 要解析的文本字符串
# method  解析方式
#   html.parser 使用html的方式去解析字符串，解析成节点的形式
#   xml 使用xml的方式解析字符串，
soup = BeautifulSoup(res_text, "html.parser")

# Tag 直接取网页标签,只会返回找到的第一个标签，
# name 标签名 / attrs 返回标签的属性

# print(soup.title)
# print(soup.title.attrs)
# print(soup.img.attrs)
# 找head标签下的link标签
# print(soup.head.link)

# NavigableString 对象，取到标签内的文字
# string 取得标签内的文字，适用于双标签  <title>***</titile>   <img />单标签只有属性
# strings 递归去取该标签内所有的标签里的文字，返回可迭代对象
# text 递归去取该标签内所有的标签里的文字，返回文本
# print(soup.title.string) #付胤:小清新男生文艺范高清头像._QQ男生头像_我要个性网
# print(soup.head.string)#只能取该标签内的文字，标签内嵌套的标签文字取不到
# for i in soup.body.strings:  # strings 递归去取该标签内所有的标签里的文字
#     print(i, end='')

# BeautifulSoup对象 ,用不到
# print(soup.name)
# print(soup.attrs)

# comment 对象 ，等同于string，输出会去掉注释，用不到

# contents 取直接子节点 返回列表
# children 返回可迭代对象
# for i in soup.body.contents:
#     print(i)
#     print("-"*30)

# print(soup.body.contents[5])

# descendants 所有的子孙节点
# print(soup.body.descendants)

# parent 获取父节点
# parents 获取所有父节点
soup1=BeautifulSoup(open("index.html",'rb').read(),'html.parser')
print(soup1.div.ul.li.parent)

# ------------find_all-------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# soup1=BeautifulSoup(open("index.html",'rb').read(),'html.parser')
# find_all(name,attrs,text,**kwargs) 以列表形式返回搜索到的内容

# name 按照Tagname 标签名查找 ！！！！！！
# print(soup1.find_all('li')) #找到网页中所有的li
# print(soup1.find_all(['p','li']))# 找到p标签和li标签
# print(soup1.find_all(re.compile("h[1-6]"))) #可以通过正则匹配标签名

# attrs 通过属性筛选 ！！！！
#  attrs={"属性名1":"属性值1","属性名2":"属性值2"}
# print(soup1.find_all("ul",attrs={"class":"top",'info':re.compile('^\w{3}$')}))

# text 查找字符串内容
# print(soup1.find_all(text="音乐")) #找到标签里字符串内容为音乐的
# print(soup1.find_all(text=re.compile("^[\u4e00-\u9fa5]{2}$")))

# limit 限定查找个数
# print(soup1.find_all(text=re.compile("^[\u4e00-\u9fa5]{2}$"),limit=2))

# 如果查找的内容只有一个，可以不使用find_all   用find就ok
# find()   只返回一个，不是列表形式，而是对象形式
