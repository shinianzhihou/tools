from bs4 import BeautifulSoup
from urllib.parse import quote
import requests
import re

def get_pages_from(url,data=None):
    #有多少页
    wb_data = requests.get(url)
    soup = BeautifulSoup(wb_data.text, 'lxml')
    try:
        pages = int(soup.select('#num > span > b > font:nth-of-type(2)')[0].get_text())
    except IndexError:
        pages = 0
    return pages

def get_titles_and_hrefs_from(url,data=None):
    #获得书籍的标题和链接，返回一个字典
    wb_data = requests.get(url)
    soup = BeautifulSoup(wb_data.text, 'lxml')
    titles = soup.select('h3')
    hrefs = soup.select('h3 > a')
    for title, href in zip(titles, hrefs):
        # print(title)
        data = {
            'title': title.get_text(),
            'href': href.get("href")
        }
    return data

def get_urls_from(url,data=None):
    #获得每本书的链接
    urls = []
    wb_data = requests.get(url)
    soup = BeautifulSoup(wb_data.text, 'lxml')
    hrefs = soup.select('h3 > a')
    for href in hrefs:
        urls.append('http://ico.bit.edu.cn/opac/'+href.get('href'))
    return urls

def get_info_from(url,data=None):
    urls = [url]
    wb_data = requests.get(url)
    soup = BeautifulSoup(wb_data.text, 'lxml')
    titles = soup.select('#item_detail > dl:nth-of-type(1) > dd > a')
    outers = soup.select('#item_detail > dl:nth-of-type(2)> dd')
    i = 2
    try:
        while re.search("出版",outers[0].get_text()) == None:
            i = i+1
            outers = soup.select('#item_detail > dl:nth-of-type({})> dd'.format(str(i)))
        for title,outer,url in zip(titles,outers,urls):
            data = {
                'title':title.get_text(),
                'outer':outer.get_text(),
                'url':str(url)
            }
    except IndexError:
            data = {
                'title':titles[0].get_text(),
                'outer':'It\'s hard to say',
                'url':str(url)
            }
    return data

def get_info():
    f = open('Info','w')
    title = quote(input('Please enter the keyword: '))# 将输入转化为URL编码
    url = 'http://ico.bit.edu.cn/opac/openlink.php?&title={}&page=1'.format(title)
    pages = get_pages_from(url)
    if pages == 0:
        print('No details!')
        return
    url_1 = ['http://ico.bit.edu.cn/opac/openlink.php?&title={}&page={}'.format(title,i) for i in range(1,pages)]
    list = []
    for url_1_1 in url_1:
        urls = get_urls_from(url_1_1)
        for url in urls:
            info = get_info_from(url)
            print(info)
            f.write('name:{0:15}outer:{1:20}url:{2}\n'.format(str(info['title']),str(info['outer']),str(info['url'])))
            list.append(info)
    print(list)
    f.close()

if __name__ == '__main__':
    get_info()
