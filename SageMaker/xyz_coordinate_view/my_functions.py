import os
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as po
import plotly.graph_objs as go
import plotly
from mpl_toolkits.mplot3d import Axes3D
plotly.offline.init_notebook_mode(connected=True)

# 可視化結果画像を保存する為のディレクトリ作成関数
def make_result_dir(dir_path, file_path):
    make_dir_path = Path('./images/{}/{}'.format(dir_path, file_path.replace('.txt', '')))
    print(make_dir_path)
    if not make_dir_path.exists():
        make_dir_path.mkdir(parents=True)
    return make_dir_path

# データをdf化する関数
def make_df_return_file_name(pre_test_path, pre_test_file):
    with open(pre_test_path+'/'+pre_test_file, 'r') as f:
        l_strip1 = [s.strip().replace('\t', " ") for s in f.readlines()]
    columns = l_strip1[0].replace('TIME STAMP', 'TIME_STAMP').replace("STEP COUNT", 'STEP_COUNT').split()[0:-1]
    data = [i.split() for i in l_strip1[2:]]

    df = pd.DataFrame(data, columns=columns)
    return df, pre_test_file

# 1変数の勾配計算
def compute_and_view_grad(df, make_dir_path, file_name):
    x = np.array(df['X'], dtype=np.float32)
    y = np.array(df['Y'], dtype=np.float32)
    z = np.array(df['Z'], dtype=np.float32)
    
    x_x = np.arange(x.shape[0])
    dx = np.gradient(x)
    ddx = np.gradient(np.gradient(x))
    # xの勾配を計算
    # 結果を表示
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.plot(x_x, x, "r-o", label="x")
    plt.plot(x, dx, "g-o", label="dx")
    plt.plot(x, ddx, "b-o", label="ddx")
    plt.grid()
    plt.legend()
    plt.title("{} x coordinate and gradient".format(file_name))
#     plt.savefig('./'+str(make_dir_path)+'/{}_x_move_and_gradient.png'.format(file_name))
#     plt.show()
    
    x_y = np.arange(y.shape[0])
    dy = np.gradient(y)                 # yの勾配を計算
    ddy = np.gradient(np.gradient(y))
    # 結果を表示
    plt.subplot(1, 3, 2)
    plt.plot(x_y, y, "r-o", label="y")
    plt.plot(x_y, dy, "g-o", label="dy")
    plt.plot(x_y, ddy, "b-o", label="ddy")
    plt.grid()
    plt.legend()
    plt.title("{} y coordinate and gradient".format(file_name))
#     plt.savefig('./'+str(make_dir_path)+'/{}_y_move_and_gradient.png'.format(file_name))
#     plt.show()
    
    x_z = np.arange(z.shape[0])
    dz = np.gradient(z)                 # zの勾配を計算
    ddz = np.gradient(np.gradient(z))
    # 結果を表示
    plt.subplot(1, 3, 3)
    plt.plot(x_z, z, "r-o", label="z")
    plt.plot(x_z, dz, "g-o", label="dz")
    plt.plot(x_z, ddz, "b-o", label="ddz")
    plt.grid()
    plt.legend()
    plt.title("{} z coordinate and gradient".format(file_name))
    plt.savefig('./'+str(make_dir_path)+'/{}_xyz_move_and_gradient.png'.format(file_name))
    plt.show()

def view_3D(df, make_dir_path, file_name):
    # XYZ座標の3次元可視化(初期位置)
    fig = plt.figure(figsize=(25, 15))
    ax = fig.add_subplot(111, projection='3d')

    x = np.array(df['X'], dtype=np.float32)
    y = np.array(df['Y'], dtype=np.float32)
    z = np.array(df['Z'], dtype=np.float32)

    ax.scatter(x[0], y[0], z[0], label='start', marker='o', color='r', s=150)
    ax.scatter(x[-1], y[-1], z[-1], label='goal', marker='o', color='b', s=150)
    plt.plot(x,y,z, label='root',linewidth=3,color='green')
    for num in range(1, len(x[:-1])):
        ax.scatter(x[num], y[num], z[num], color='black', marker="${}$".format(num), s=250)

    ax.legend(fontsize=15)
    # elev: 縦方向の回転, azim: 横方向の回転
    # ax.view_init(elev=10, azim=50)
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_zlabel('Z', fontsize=20)
    plt.title("{} Coordinate by sensor with 3D view(default angle)".format(file_name), fontsize=30)
    plt.savefig('./'+str(make_dir_path)+'/{}_3Dview_default_angle.png'.format(file_name))
    plt.show()

    # XYZ座標の3次元可視化(真横)
    fig = plt.figure(figsize=(25, 15))
    ax = fig.add_subplot(111, projection='3d')

    x = np.array(df['X'], dtype=np.float32)
    y = np.array(df['Y'], dtype=np.float32)
    z = np.array(df['Z'], dtype=np.float32)

    ax.scatter(x[0], y[0], z[0], label='start', marker='o', color='r', s=150)
    ax.scatter(x[-1], y[-1], z[-1], label='goal', marker='o', color='b', s=150)
    plt.plot(x,y,z, label='root',linewidth=3,color='green')
    for num in range(1, len(x[:-1])):
        ax.scatter(x[num], y[num], z[num], color='black', marker="${}$".format(num), s=250)

    ax.legend(fontsize=15)
    # elev: 縦方向の回転, azim: 横方向の回転
    ax.view_init(elev=0, azim=0)
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_zlabel('Z', fontsize=20)
    plt.title("{} Coordinate by sensor with 3D view(height angle)".format(file_name), fontsize=30)
    plt.savefig('./'+str(make_dir_path)+'/{}_3Dview_height_angle.png'.format(file_name))
    plt.show()

def view_2d(df, make_dir_path, file_name):
    # XY座標の2次元可視化
    fig = plt.figure(figsize=(25, 15))
    ax = fig.add_subplot(111)

    x = np.array(df['X'], dtype=np.float32)
    y = np.array(df['Y'], dtype=np.float32)

    ax.scatter(x[0], y[0], label='start', marker='o', color='r', s=150)
    ax.scatter(x[-1], y[-1], label='goal', marker='o', color='b', s=150)
    plt.plot(x,y, label='root',linewidth=3,color='green')
    for num in range(1, len(x[:-1])):
        ax.scatter(x[num], y[num], color='black', marker="${}$".format(num), s=250)

    ax.legend(fontsize=15)
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    plt.title("{} Coordinate by sensor with 2D view".format(file_name), fontsize=30)
    plt.savefig('./'+str(make_dir_path)+'/{}_2Dview.png'.format(file_name))
    plt.show()