import sqlite3
import pandas as pd
con = sqlite3.connect(".pymon")
df = pd.read_sql_query("SELECT * from TEST_METRICS", con)
df.to_csv('metrics.csv')
con.close()
