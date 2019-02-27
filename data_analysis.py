"""Modul for analysing a dataset."""

from database import Database
import pandas as pd

my_db = Database()
my_db.open_db()  # open database

# read all task-sets into a dataframe
taskset_table = pd.read_sql("SELECT * FROM TaskSet limit 5", my_db.db_connection)

# read all tasks into a dataframe
task_table = pd.read_sql("SELECT * FROM Task limit 5", my_db.db_connection)

my_db.close_db()  # close database

print(taskset_table)
print(task_table)