import sqlite3
import pudb
import os
import sys
import struct
import numpy as np

def print_header(col_data):
    # get headers
    for col in col_data:
        print(f"{col['name'].center(16)} |",end='')
    print('')
    print('-'*18*len(col_data))


def print_table(col_data,data,no_blob=True):
    print_header(col_data)

    # print data
    keys = list(data.keys())
    n_rows = len(data[keys[0]])
    for row_idx in range(n_rows):
        for col in col_data:
            if no_blob and col['dtype'] == 'BLOB':
                print(f"BLOB".center(16),'|',end='')
            else:
                print(f"{data[col['name']][row_idx]}".center(16),'|',end='')
        print()


class DB_reader:
    def __init__(self,db_path):
        con = sqlite3.connect(db_path)
        self.cursor = con.cursor()

    def print_tables(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        data = self.cursor.fetchall()
        print('Available Tables:')
        for row in data:
            print(row)

    def get_image_fn(self,image_id):
        image_data = self.get_table_data('images')
        out_fn = None
        for id,fn in zip(image_data['image_id'],image_data['name']):
            if id == image_id: out_fn = fn
        return out_fn

    def get_image_id_from_fn(self,im_fn):
        image_data = self.get_table_data('images')
        out_id = None
        for id,fn in zip(image_data['image_id'],image_data['name']):
            if fn == im_fn: out_id = id
        return out_id

    def get_column_info(self,table_name):
        self.cursor.execute(f"PRAGMA table_info({table_name});")
        data = self.cursor.fetchall()
        out = []
        for row in data:
            out.append({'name':row[1],'dtype':row[2]})
        return out

    def get_table_data(self,table_name):
        column_names = self.get_column_info(table_name)
        out = {}
        for col in column_names:
            out[col['name']] = []
        self.cursor.execute(f"SELECT * FROM {table_name};")
        data = self.cursor.fetchall()
        for row in data:
            for col_idx,col in enumerate(row):
                out[column_names[col_idx]['name']].append(col)
        return out

    def print_table_by_name(self,table_name):
        print_table(self.get_column_info(table_name),self.get_table_data(table_name))

def extract_keypoints(table_data):
    n_images = len(table_data['image_id'])
    out = {}
    for image_id,n_rows,n_cols,data in zip(table_data['image_id'],table_data['rows'],table_data['cols'],table_data['data']):
        n_floats = n_rows*n_cols
        if n_floats > 0:
            float_array = struct.unpack(f'{n_floats}f',data)
            float_table = np.array(float_array).reshape(n_rows,n_cols)
            out[image_id] = float_table
        else:
            out[image_id] = np.array([[]])
    return out

def extract_matches(match_data):
    out = []
    for pair_id,n_rows,n_cols,data in zip(match_data['pair_id'],match_data['rows'],match_data['cols'],match_data['data']):
        im2_id = image_id2 = pair_id % 2147483647
        im1_id = (pair_id - im2_id) // 2147483647
        n_ids = n_rows*n_cols
        if n_ids > 0:
            int32_array = struct.unpack(f'{n_ids}i',data)
            int32_table = np.array(int32_array).reshape(n_rows,n_cols)
        else:
            int32_table = None
        out.append([[im1_id,im2_id],int32_table])
    return out
























