#在播放页面的network里面找到了音乐的资源地址，
#我们在点击每首歌的时候都可以获取到这个链接，但是，太麻烦了
#我们想要的是下载列表里所有的歌曲，需要找到音乐资源请求的通用位置
#对比音乐资源url
# http://isure.stream.qqmusic.qq.com/C4000049gKZo2S2J1N.m4a?vkey=161F7C9892556C8FE7E471774E0AE86C25DF46F145F5394EEFE0107C45FF2874EF8DF960EA72B9568F9C351111D3F3CFB718AD72D5B330A4&guid=674695650&uin=0&fromtag=66
# http://isure.stream.qqmusic.qq.com/C400004T877T2q0iCt.m4a?vkey=5B0134952DDAAD7ED27C9D90EFB248D559FB12AE897CA4E470702399A710B3858272A92BCB1FAC88508911CD6704C007EA4D96B72248B427&guid=674695650&uin=0&fromtag=66
#对比之后发现，音乐文件名称不同，vkey不同
#分析网址的参数，发现 vkey guid 是必须的   uin和fromtag 不是必须的
#去列表页找到了跟音乐资源很相似的
    # 资源url C400001WO6e403aibq.m4a
    # 列表页      001WO6e403aibq
    # C400+列表页里的音乐id+.m4a
    #结论，想要下载音乐列表里所有的音乐时，只需要拿到列表页所有音乐的url，拼接就能得到文件名

# 意外惊喜 找到了 列表页里所有的音乐信息，是以动态请求获取的
# https://c.y.qq.com/qzone/fcg-bin/fcg_ucc_getcdinfo_byids_cp.fcg?type=1&json=1&utf8=1&onlysong=0&disstid=4571471093&format=jsonp&g_tk=5381&jsonpCallback=playlistinfoCallback&loginUin=0&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0

#找到了vkey,请求该页面即可得到vkey
# https://c.y.qq.com/base/fcgi-bin/fcg_music_express_mobile3.fcg?g_tk=5381&jsonpCallback=MusicJsonCallback05026006874348621&loginUin=0&hostUin=0&format=json&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0&cid=205361747&callback=MusicJsonCallback05026006874348621&uin=0&songmid=002K8HEt2Ng0f4&filename=C400002K8HEt2Ng0f4.m4a&guid=674695650
#分析得到vkey网址的参数
# g_tk: 5381---g_tk: 5381
# jsonpCallback: MusicJsonCallback058423163768663855---jsonpCallback: MusicJsonCallback5703131309752927
# loginUin: 0---loginUin: 0
# hostUin: 0---hostUin: 0
# format: json---format: json
# inCharset: utf8---inCharset: utf8
# outCharset: utf-8---outCharset: utf-8
# notice: 0---notice: 0
# platform: yqq---platform: yqq
# needNewCode: 0---needNewCode: 0
# cid: 205361747---cid: 205361747
# callback: MusicJsonCallback058423163768663855---callback: MusicJsonCallback5703131309752927
# uin: 0---uin: 0
# songmid: 000fIVRG0lsVUH---songmid: 001L1lqm4UAdyo
# filename: C400003ZYkMo3g8c2U.m4a---filename: C400001L1lqm4UAdyo.m4a
# guid: 674695650---guid: 674695650
#去掉相同项，最后发现之后 jsonpcallback ，callback,songmid,filename 不同
# songmid和filename我们已经设置过，发现jsoncallback 后面的值 其实是一个随机数，就可以直接请求了


import requests
import re
import json
import random
import os

headers={
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"
}
#现在可以通过列表页歌曲id，获取到歌曲的实际播放资源url
def song_id_get_real_url(songid="003OUlho2HcRHC",songname="演员"):
    filename="C400"+songid+".m4a"
    callback="MusicJsonCallback"+str(random.randint(10000000,99999999))

    #获取vkey的url
    getvkeyUrl="https://c.y.qq.com/base/fcgi-bin/fcg_music_express_mobile3.fcg"
    #
    params={
        "g_tk": "5381",
        "jsonpCallback": callback,
        "loginUin": "0",
        "hostUin": "0",
        "format": "json",
        "inCharset": "utf8",
        "outCharset": "utf-8",
        "notice": "0",
        "platform": "yqq",
        "needNewCode": "0",
        "cid": "205361747",
        "callback": callback,
        "uin": "0",
        "songmid": songid,
        "filename": filename,
        "guid": "674695650",
    }


    #请求该url，准备获取vkey
    r=requests.get(getvkeyUrl,params=params)
    #得到值之后，发现有一部分干扰数据，需要使用正则取出
    regx=re.compile("MusicJsonCallback\d+\((.*?)\)")
    result=regx.search(r.text)
    #把得到的json字符串解析成字典或者列表
    result_dict=json.loads(result.group(1))
    #获取到字典里的vkey字段
    vkey=result_dict['data']['items'][0]['vkey']
    #把文件名和vkey拼接到url里，得到的就是真实的资源地址
    real_url="http://isure.stream.qqmusic.qq.com/"+filename+"?vkey="+vkey+"&guid=674695650&uin=0&fromtag=66"
    return {"real_url":real_url,"songname":songname}

def download_music(info,sonpath=''):
    basepath="./Music/"
    if sonpath=="":
        savepath=basepath
    else:
        sonpath=re.sub("[\|/\*\?\"><:\\\\]",'',sonpath)
        savepath=basepath+sonpath+'/'
        if os.path.exists(savepath)==False:
            os.mkdir(savepath)
    r=requests.get(info['real_url'],headers=headers,stream=True)
    with open(savepath+info['songname']+".mp3",'wb') as file:
        for j in r.iter_content(1024):
            file.write(j)
    print('%s下载成功'%info['songname'])

#通过歌单来下载歌曲
def list_get_songs(disstid="4276387932"):
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
        "referer":"https://y.qq.com/n/yqq/playsquare/4277269185.html",
    }
    url="https://c.y.qq.com/qzone/fcg-bin/fcg_ucc_getcdinfo_byids_cp.fcg?type=1&json=1&utf8=1&onlysong=0&disstid="+disstid+"&format=jsonp&g_tk=5381&jsonpCallback=playlistinfoCallback&loginUin=0&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0"
    r=requests.get(url,headers=headers)
    dissinfo=json.loads(r.text[21:-1])
    song_list=[]
    for once_song in dissinfo['cdlist'][0]['songlist']:
        songname=once_song['songname']
        songid=once_song['songmid']
        song_list.append({"songname":songname,'songid':songid})
    dissname=dissinfo['cdlist'][0]['dissname']
    return {"songlist":song_list,"dissname":dissname}

#通过歌手下载热门歌曲
def singer_hot_songs(singermid="0025NhlN2yWrP4"):
    url="https://c.y.qq.com/v8/fcg-bin/fcg_v8_singer_track_cp.fcg?g_tk=5381&jsonpCallback=MusicJsonCallbacksinger_track&loginUin=0&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0&singermid="+singermid+"&order=listen&begin=0&num=30&songstatus=1"
    #    https://c.y.qq.com/v8/fcg-bin/fcg_v8_singer_track_cp.fcg?g_tk=5381&jsonpCallback=MusicJsonCallbacksinger_track&loginUin=0&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0&singermid=0025NhlN2yWrP4&order=listen&begin=0&num=30&songstatus=1
    r=requests.get(url,headers=headers)
    songinfo=json.loads(r.text[31:-2])
    song_list=[]
    for once_song in songinfo['data']['list']:
        songname = once_song['musicData']['songname']
        songid = once_song['musicData']['songmid']
        song_list.append({"songname": songname, 'songid': songid})
    singername=songinfo['data']['singer_name']
    return {'songlist':song_list,"singername":singername}
#下载单首歌曲
# songinfo=song_id_get_real_url("004bRWFg3fej9y",'彩虹')
# download_music(songinfo)

#下载一个歌单里所有的歌曲
# song_list=list_get_songs("4242003865")
# for once_song in song_list['songlist']:
#     songinfo=song_id_get_real_url(once_song['songid'],once_song['songname'])
#     download_music(songinfo,song_list['dissname'])

#通过歌手下载热门歌曲
song_list=singer_hot_songs("001BLpXF2DyJe2")
for once_song in song_list['songlist']:
    songinfo=song_id_get_real_url(once_song['songid'],once_song['songname'])
    download_music(songinfo,song_list['singername'])