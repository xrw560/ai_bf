# -*- coding: utf-8 -*-

import pymysql
import pandas as pd

conn = pymysql.connect(host="localhost", port=3306, user='root', password='root', db='tlshop', charset='utf8')

sql = "select * from t_account"
df = pd.read_sql(sql, conn)
print(df)
