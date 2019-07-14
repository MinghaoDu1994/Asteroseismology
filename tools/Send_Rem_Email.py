#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:46:27 2018

@author: duminghao
"""
class Send:
    def __init__(self):
        import yagmail
        yag=yagmail.SMTP(user = 'wfduminghao@163.com', password = '8528269a', host = 'smtp.163.com') 
# 发送邮件
        yag.send(to = ['wfduminghao@hotmail.com'],
         subject = 'Mission Completed', contents = 
         ['Your Program has finished, check it！！'])
        
        
Send()




