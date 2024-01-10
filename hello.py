# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:35:36 2022

@author: Dell
"""

from flask import Flask

app = Flask(__name__)

@app.route("\")
def hello_world():
    return "<p>Hello, World!</p>"