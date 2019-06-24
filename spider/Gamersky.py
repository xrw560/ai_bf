from bs4 import BeautifulSoup
import requests
import json
page=1
while page<5:
    url="http://pic.gamersky.com/home/getimagesindex?sort=time_desc&pageIndex=%d&pageSize=50&nodeId=21089"%(page)
    r=requests.get(url)
    json_str=r.json()
    print(json_str)
    data=json.loads(json_str)['body']

    for oncedata in data:
        print(oncedata['originImg'])
    print('第%d页下载成功'%(page))
    page+=1