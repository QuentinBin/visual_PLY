import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def heapMapPlot(data,title,savepath='depth_0.png',sexi='bwr'):
    '''
    基于相关性系数计算结果来绘制热力图
    '''
    data=np.array(data)
    sns.heatmap(data,fmt='d',cmap=sexi)  #"YlGnBu"
    plt.title(title)
    plt.savefig(savepath)

if __name__ == "__main__":
    data = np.load("depth_0.npy")
    heapMapPlot(data,"aaa")