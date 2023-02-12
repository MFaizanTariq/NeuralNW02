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
    print("3. Initialize weights")
    num = int(input())
    
    test_rc1 = np.zeros(600)
    test_rc2 = np.ones(600)
    region = np.append(test_rc1, test_rc2, axis=0)

    t_data = train_data()

    test_data = train_data2()

    if num == 1:
        for x in range(1):
            print("Running: EPOCH # ", x)
            for y in range(1199):
                train(t_data[y,0], t_data[y,1], region[y])

    correct = 0

    if num == 2:
        for y in range(499):
            out = detect(test_data[y, 0], test_data[y, 1])
            
            if y < 250:
                if out < 0.5:
                    correct += 1
            elif y > 249:
                if out > 0.5:
                    correct += 1

    if num == 3:
        init_weight()
        print("Weight initialized")

    print(correct)

def detect(x1, x2):
    con = sqlite3.connect('neu_nw.db')
    cur = con.cursor()
    cur.execute("SELECT w FROM weights")
    w1 = cur.fetchall()
    wt = [w1[0][0], w1[1][0], w1[2][0], w1[3][0], w1[4][0], w1[5][0], w1[6][0], w1[7][0], w1[8][0], w1[9][0], w1[10][0],
          w1[11][0], w1[12][0], w1[13][0], w1[14][0], w1[15][0], w1[16][0], w1[17][0], w1[18][0], w1[19][0], w1[20][0],
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

def detect_2(x1, x2):
    con = sqlite3.connect('neu_nw.db')
    cur = con.cursor()
    cur.execute("SELECT w FROM weights")
    w1 = cur.fetchall()

    wt = [w1[0][0], w1[1][0], w1[2][0], w1[3][0], w1[4][0], w1[5][0], w1[6][0], w1[7][0], w1[8][0], w1[9][0], w1[10][0],
          w1[11][0], w1[12][0], w1[13][0], w1[14][0], w1[15][0], w1[16][0], w1[17][0], w1[18][0], w1[19][0], w1[20][0],
          w1[21][0], w1[22][0], w1[23][0], w1[24][0], w1[25][0], w1[26][0], w1[27][0], w1[28][0], w1[29][0], w1[30][0],
          w1[21][0], w1[22][0], w1[23][0], w1[24][0], w1[25][0], w1[26][0], w1[27][0], w1[28][0], w1[29][0], w1[30][0],
          w1[31][0], w1[32][0], w1[33][0], w1[34][0], w1[35][0], w1[36][0], w1[37][0], w1[38][0], w1[39][0], w1[40][0],
          w1[41][0], w1[42][0], w1[43][0], w1[44][0], w1[45][0], w1[46][0], w1[47][0], w1[48][0], w1[49][0], w1[50][0],
          w1[51][0], w1[52][0], w1[53][0], w1[54][0], w1[55][0], w1[56][0], w1[57][0], w1[58][0], w1[59][0], w1[60][0],
          w1[61][0], w1[62][0], w1[63][0], w1[64][0], w1[65][0], w1[66][0], w1[67][0], w1[68][0], w1[69][0], w1[70][0],
          w1[71][0], w1[72][0], w1[73][0], w1[74][0], w1[75][0], w1[76][0], w1[77][0], w1[78][0], w1[79][0], w1[80][0],
          w1[81][0], w1[82][0], w1[83][0], w1[84][0], w1[85][0], w1[86][0], w1[87][0], w1[88][0], w1[89][0], w1[90][0],
          w1[91][0], w1[92][0], w1[93][0], w1[94][0], w1[95][0], w1[96][0], w1[97][0], w1[98][0], w1[99][0], w1[100][0],
          w1[101][0], w1[102][0], w1[103][0], w1[104][0], w1[105][0], w1[106][0], w1[107][0], w1[108][0], w1[109][0], w1[110][0],
          w1[111][0], w1[112][0], w1[113][0], w1[114][0], w1[115][0], w1[116][0], w1[117][0], w1[118][0], w1[119][0], w1[120][0],
          w1[121][0], w1[122][0], w1[123][0], w1[124][0], w1[125][0], w1[126][0], w1[127][0], w1[128][0], w1[129][0], w1[130][0],
          w1[131][0], w1[132][0], w1[133][0], w1[134][0], w1[135][0], w1[136][0], w1[137][0], w1[138][0], w1[139][0], w1[140][0],
          w1[141][0], w1[142][0], w1[143][0], w1[144][0], w1[145][0], w1[146][0], w1[147][0], w1[148][0], w1[149][0], w1[150][0],
          w1[151][0], w1[152][0], w1[153][0], w1[154][0], w1[155][0], w1[156][0], w1[157][0], w1[158][0], w1[159][0], w1[160][0],
          w1[161][0], w1[162][0], w1[163][0], w1[164][0], w1[165][0], w1[166][0], w1[167][0], w1[168][0], w1[169][0], w1[170][0],
          w1[171][0], w1[172][0], w1[173][0], w1[174][0], w1[175][0], w1[176][0], w1[177][0], w1[178][0], w1[179][0], w1[180][0],
          w1[181][0], w1[182][0], w1[183][0], w1[184][0], w1[185][0], w1[186][0], w1[187][0], w1[188][0], w1[189][0]
          ]

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

    out16 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15,
    wt[30], wt[31], wt[32], wt[33], wt[34], wt[35], wt[36], wt[37], wt[38], wt[39], wt[40], wt[41], wt[42], wt[43], wt[44])
    out17 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15,
    wt[45], wt[46], wt[47], wt[48], wt[49], wt[50], wt[51], wt[52], wt[53], wt[54], wt[55], wt[56], wt[57], wt[58], wt[59])
    out18 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15,
    wt[60], wt[61], wt[62], wt[63], wt[64], wt[65], wt[66], wt[67], wt[68], wt[69], wt[70], wt[71], wt[72], wt[73], wt[74])
    out19 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15,
    wt[75], wt[76], wt[77], wt[78], wt[79], wt[80], wt[81], wt[82], wt[83], wt[84], wt[85], wt[86], wt[87], wt[88], wt[89])
    out20 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15,
    wt[90], wt[91], wt[92], wt[93], wt[94], wt[95], wt[96], wt[97], wt[98], wt[99], wt[100], wt[101], wt[102], wt[103], wt[104])
    out21 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15,
    wt[105], wt[106], wt[107], wt[108], wt[109], wt[110], wt[111], wt[112], wt[113], wt[114], wt[115], wt[116], wt[117], wt[118], wt[119])
    out22 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15,
    wt[120], wt[121], wt[122], wt[123], wt[124], wt[125], wt[126], wt[127], wt[128], wt[129], wt[130], wt[131], wt[132], wt[133], wt[134])
    out23 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15,
    wt[135], wt[136], wt[137], wt[138], wt[139], wt[140], wt[141], wt[142], wt[143], wt[144], wt[145], wt[146], wt[147], wt[148], wt[149])
    out24 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15,
    wt[150], wt[151], wt[152], wt[153], wt[154], wt[155], wt[156], wt[157], wt[158], wt[159], wt[160], wt[161], wt[162], wt[163], wt[164])
    out25 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15,
    wt[165], wt[166], wt[167], wt[168], wt[169], wt[170], wt[171], wt[172], wt[173], wt[174], wt[175], wt[176], wt[177], wt[178], wt[179])

    out = neuron3(out16, out17, out18, out19, out20, out21, out22, out23, out24, out25,
                  wt[180], wt[181], wt[182], wt[183], wt[184], wt[185], wt[186], wt[187], wt[188], wt[189])

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

def neuron3(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
            w1, w2, w3, w4, w5, w6, w7, w8, w9, w10):
    net =   x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5 + x6*w6 + x7*w7 + x8*w8 + x9*w9 + x10*w10
    out = 1 / (1 + math.exp(-net))
    return out