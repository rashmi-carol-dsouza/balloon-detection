from flask import Flask,render_template
# import requests
# import json
# import pandas as pd
# import sys
# from model import geoloc

app = Flask(__name__)


@app.route("/",methods=["GET","POST"])
def home():
    return render_template("index.html")


# @app.route("/toilets",methods=["GET"])
# def toilet():
#     args = request.args
#     lat1 = args.get('lat')
#     lng1 = args.get('lng')
#     timing = args.get('open24')
#     cost = args.get('free')
#     if_accessible = args.get('accessible')
#     result = geoloc(lat1,lng1,timing,cost,if_accessible)
#     return result
        

if __name__ == "__main__":
    app.run(debug=True)