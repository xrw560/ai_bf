import requests
import json

url = "https://www.woyaogexing.com/touxiang/nv/2018/662775.html"
'''
requests.get(url,**kwargs)
url 要请求的地址
data post方式请求参数
headers 设置请求消息头  一般设置User-Agent就ok
files 给服务器发送文件
cookies 请求的时候附带cookie
proxies 设置代理
verify  是否验证ssl
'''
'''
返回状态码
1** 正在请求
2** 请求成功
3** 重定向
4** 请求内容错误
5** 服务器错误
'''

# response = requests.get(url)
# print(type(response))
# # 查看状态码
# print(response.status_code)
# # 查看编码
# print(response.encoding)
# # 查看返回的文本 html代码 text方法会自动解码，有可能会解析错误
# print(response.text)
# # 查看网页的字节方式的响应 content 需要自定义解码形式
# print(response.content.decode("utf-8"))
# 假如返回的是json格式 通用数据交换格式 xml格式 json格式
# json()方法 把json字符串转换为字典或者列表
# url = "https://github.com/timeline.json"
# r = requests.get(url)
# result = r.json()
# print(r.text)
# print(result['message'])

# cookie
# url = "http://www.ibeifeng.com"
# r = requests.get(url)
# # 自己保存cookie
# wecookie = r.cookies
# # 第二次请求的时候带上cookie
# s = requests.get(url, cookies=wecookie)
# print(s.text)
# 
# 模拟登陆 cookies
# s = requests.post("http://www.antvv.com/login/dologin.php", data={'uname': "admin", 'upwd': '123456'})
# wecookie = s.cookies
#
# r = requests.get("http://www.antvv.com/login/index.php", cookies=wecookie)
# print(r.text)

# 使用requests.session()方法保存cookie
# s = requests.session()  # 模拟了一个浏览器
# s.post("http://www.antvv.com/login/dologin.php", data={'uname': "admin", 'upwd': '123456'})
# response = s.get("http://www.antvv.com/login/index.php")
# print(response.text)

# 下载文件
'''
    1.图片的地址
    2.请求图片，得到字节格式的图片资源
    3.保存文件
        3.1 本地新建一个空文件
        3.2 把图片资源写入到空文件里
        3.3 关闭文件
'''
# 小文件
# url = "https://img2.woyaogexing.com/2018/08/30/c9828fd2ccfc49caa8693e0e550b20a9!400x400.jpeg"
# r = requests.get(url)
# # print(r.content)
# with open("a.jpg", 'wb') as file:
#     file.write(r.content)

# 大文件的形式
# url = "https://img2.woyaogexing.com/2018/08/30/c9828fd2ccfc49caa8693e0e550b20a9!400x400.jpeg"
# r = requests.get(url, stream=True)
# # print(r.raw.read(1024))  #返回字节形式
# file = open('b.jpg', 'wb')
# for i in r.iter_content(1024):  # 推荐使用
#     file.write(i)
# file.close()

# # 发送文件和发送json 文件和json不能同时发送
# file = open('spider_note.txt', 'rb')
#
# url = "http://httpbin.org/post"
# header = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"
# }
# datadict = [{"a": "aaa"}]
# print(type(datadict))
# # r = requests.post(url, headers=header, data={"name": "zhangsan"}, files={'file': file})
# # r = requests.post(url, headers=header, data=json.dumps(datadict), files={'file': file})
# r = requests.post(url, headers=header, data=json.dumps(datadict))
# print(r.text)

# 设置代理
# url = "http://httpbin.org/post"
# header = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"
# }
# proxies = {
#     "http": "118.190.95.43:9001",
# }
# r = requests.post(url, headers=header, proxies=proxies)
# print(r.text)

# SSL验证 verify
# http和https

url = "https://www.12306.cn"
r = requests.get(url, verify=False)
print(r.text)
