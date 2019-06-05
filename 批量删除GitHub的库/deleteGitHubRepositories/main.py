#!usr/bin/python3  
# -*- coding:utf-8 _*-
""" 
@author:ShiNian 
@file: main.py 
@time: 2019/06/05 
"""
from selenium import webdriver
from configparser import ConfigParser

url = "https://github.com/login"
# 读取配置文件,配置文件无需加入‘’
cp = ConfigParser()
cp.read('config.cfg')
username = cp.get('usr_inf','username')
password = cp.get('usr_inf','password')
nickname = cp.get('usr_inf','nickname')
# 要删除的库
print(username,password)
reps = ["deep-image-prior",
        "pixel-cnn",
        "text-classification-cnn-rnn",
        "fast-style-transfer",
        "Semantic-UI-React",
        "fast-rcnn",
        "library",
        "react",
        "OpenCV-Python-Tutorial",
        # "My-world",
        "Wechat",
        "weui",
        "app",
        "facerec-python",
        "wecqupt",
        "BossSensor",
        "Library-1", ]
chromedriver = './driver/chromedriver'
wb = webdriver.Chrome(chromedriver)

## 登录
wb.get(url=url)
wb.find_element_by_id('login_field').clear()
wb.find_element_by_id('login_field').send_keys(username)
wb.find_element_by_id('password').clear()
wb.find_element_by_id('password').send_keys(password)
wb.find_element_by_css_selector('.btn-block').click()

## 删除
for rep in reps:
    try:
        wb.get(url="https://github.com/" + nickname + '/' + rep + '/' + 'settings')
        wb.find_element_by_css_selector(
            '#options_bucket > div.Box.Box--danger > ul > li:nth-child(4) > details > summary').click()
        wb.find_element_by_xpath(
            '//*[@id="options_bucket"]/div[9]/ul/li[4]/details/details-dialog/div[3]/form/p/input').send_keys(rep)
        wb.find_element_by_xpath(
            '//*[@id="options_bucket"]/div[9]/ul/li[4]/details/details-dialog/div[3]/form/button').click()
    except Exception as e:
        print(e)
