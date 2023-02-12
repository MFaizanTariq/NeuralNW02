from venv.train import *
from venv.data import *
import math
import sqlite3
import numpy as np

def main_prog():
    con = sqlite3.connect('neu_nw.db')
    cur = con.cursor()

    cur.execute("CREATE TABLE IF NOT EXISTS weights (w_id INTEGER PRIMARY KEY, w INTEGER)")

    con.commit()

    print("select option:")
    print("1. Train:")
    print("2. Detect:")
    num = int(input())
    
    test_rc1 = np.zeros(600)
    test_rc2 = np.ones(600)
    region = np.append(test_rc1, test_rc2, axis=0)

    t_data = train_data()

    test_data = train_data2()

    if num == 1:
        for x in range(50):
            print("Running: EPOCH # ", x)
            for y in range(1199):
                train(t_data[y,0], t_data[y,1], region[y])

    correct = 0

    if num == 2:
        for y in range(499):
            out = detect(test_data[y, 0], test_data[y, 1])
            
            if y < 250:
                if out < 0.25:
                    correct += 1
            elif y > 249:
                if out > 0.75:
                    correct += 1

    print(correct)

def detect(x1, x2):
    con = sqlite3.connect('neu_nw.db')
    cur = con.cursor()
    cur.execute("SELECT w FROM weights")
    w1 = cur.fetchall()
    wt = [w1[0][0], w1[1][0], w1[2][0], w1[3][0], w1[4][0], w1[5][0], w1[6][0], w1[7][0], w1[8][0], w1[9][0], w1[10][0],
          w1[11][0], w1[12][0], w1[13][0], w1[14][0], w1[15][0], w1[16][0], w1[17][0], w1[18][0], w1[19][0], w1[20][0],
          w1[21][0], w1[22][0], w1[23][0], w1[24][0], w1[25][0], w1[26][0], w1[27][0], w1[28][0], w1[29][0], w1[30][0],
          w1[21][0], w1[22][0], w1[23][0], w1[24][0], w1[25][0], w1[26][0], w1[27][0], w1[28][0], w1[29][0], w1[30][0],
          w1[31][0], w1[32][0], w1[33][0], w1[34][0], w1[35][0], w1[36][0], w1[37][0], w1[38][0], w1[39][0], w1[40][0],
          w1[41][0], w1[42][0], w1[43][0], w1[44][0], w1[45][0]]

    out1 = neuron1(x1, x2, wt[0], wt[1])
    out2 = neuron1(x1, x2, wt[2], wt[3])
    out3 = neuron1(x1, x2, wt[4], wt[5])
    out4 = neuron1(x1, x2, wt[6], wt[7])
    out5 = neuron1(x1, x2, wt[8], wt[9])
    out6 = neuron1(x1, x2, wt[10], wt[11])
    out7 = neuron1(x1, x2, wt[12], wt[13])
    out8 = neuron1(x1, x2, wt[14], wt[15])
    out9 = neuron1(x1, x2, wt[16], wt[17])
    out10 = neuron1(x1, x2, wt[18], wt[19])
    out11 = neuron1(x1, x2, wt[20], wt[21])
    out12 = neuron1(x1, x2, wt[22], wt[23])
    out13 = neuron1(x1, x2, wt[24], wt[25])
    out14 = neuron1(x1, x2, wt[26], wt[27])
    out15 = neuron1(x1, x2, wt[28], wt[29])

    out = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15,
                wt[30], wt[31], wt[32], wt[33], wt[34], wt[35], wt[36], wt[37], wt[38], wt[39], wt[40], wt[41], wt[42], 
                wt[43], wt[44])

    return out

def neuron1(x1, x2, w1, w2):
    net = x1*w1 + x2*w2
    out = 1 / (1 + math.exp(-net))
    return out

def neuron2(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, 
            w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15):
    net =   x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5 + x6*w6 + x7*w7 + x8*w8 + x9*w9 + x10*w10 + x11*w11 + x12*w12+ x13*w13+ x14*w14 + x15*w15
    out = 1 / (1 + math.exp(-net))
    return out