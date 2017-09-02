from bs4 import BeautifulSoup
from urllib.parse import quote
import requests
import re
import pymysql

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
                'outer':re.sub(r'([\d,\s.-]+)','',outer.get_text()),
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
        f.write('{0:{3}^35}{1:{3}^20}{2:^60}\n'.format('书名','出版社','链接',chr(12288)))
        urls = get_urls_from(url_1_1)
        for url in urls:
            info = get_info_from(url)
            '''try:
            #数据库操作
                db = pymysql.connect(
                    host = 'localhost',
                    port = 3306,
                    user = 'root',
                    passwd = '',
                    db = 'Library',
                    charset = 'utf8mb4'
                )
                cursor = db.cursor()#获取游标
                cursor.execute('insert into getInfoForBit values("{}", "{}", "{}")'\
                               .format(str(info['title']).encode('utf-8'),str(info['outer']).encode('utf-8'),str(info['url']).encode('utf-8')))
                db.close()
            except Exception as e:
                db.rollback()
                print('gg了', e)
                db.close()
            '''
            print('{0:{3}^35}{1:{3}^20}{2:>60}'.format(info['title'],info['outer'],info['url'],chr(12288)))
            f.write('{0:{3}^35}{1:{3}^20}{2:>60}\n'.format(info['title'],info['outer'],info['url'],chr(12288)))
            list.append(info)
    print(list)
    f.close()

if __name__ == '__main__':
    get_info()
