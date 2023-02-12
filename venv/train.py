from venv.bpn import *
import sqlite3


def train(x1, x2, trgt):
    from venv.bpn import neuron1, neuron2
    con = sqlite3.connect('neu_nw.db')
    cur = con.cursor()
    cur.execute("SELECT w FROM weights")
    w1 = cur.fetchall()
    wt = [w1[0][0], w1[1][0], w1[2][0], w1[3][0], w1[4][0], w1[5][0], w1[6][0], w1[7][0], w1[8][0], w1[9][0], w1[10][0], 
          w1[11][0], w1[12][0], w1[13][0], w1[14][0], w1[15][0], w1[16][0], w1[17][0], w1[18][0], w1[19][0], w1[20][0],
          w1[21][0], w1[22][0], w1[23][0], w1[24][0], w1[25][0], w1[26][0], w1[27][0], w1[28][0], w1[29][0], w1[30][0],
          w1[21][0], w1[22][0], w1[23][0], w1[24][0], w1[25][0], w1[26][0], w1[27][0], w1[28][0], w1[29][0], w1[30][0],
          w1[31][0], w1[32][0], w1[33][0], w1[34][0], w1[35][0], w1[36][0], w1[37][0], w1[38][0], w1[39][0], w1[40][0],
          w1[41][0], w1[42][0], w1[43][0], w1[44][0]]

    neu = 0.1

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
    
    delta = out * (1 - out) * (trgt - out)

    wt[30] += neu * out1 * delta
    wt[31] += neu * out2 * delta
    wt[32] += neu * out3 * delta
    wt[33] += neu * out4 * delta
    wt[34] += neu * out5 * delta
    wt[35] += neu * out6 * delta
    wt[36] += neu * out7 * delta
    wt[37] += neu * out8 * delta
    wt[38] += neu * out9 * delta
    wt[39] += neu * out10 * delta
    wt[40] += neu * out11 * delta
    wt[41] += neu * out12 * delta
    wt[42] += neu * out13 * delta
    wt[43] += neu * out14 * delta
    wt[44] += neu * out15 * delta

    wt[0] += neu * out1 * (1 - out1) * (delta * wt[30]) * x1
    wt[1] += neu * out1 * (1 - out1) * (delta * wt[30]) * x2
    wt[2] += neu * out2 * (1 - out2) * (delta * wt[31]) * x1
    wt[3] += neu * out2 * (1 - out2) * (delta * wt[31]) * x2
    wt[4] += neu * out3 * (1 - out3) * (delta * wt[32]) * x1
    wt[5] += neu * out3 * (1 - out3) * (delta * wt[32]) * x2
    wt[6] += neu * out4 * (1 - out4) * (delta * wt[33]) * x1
    wt[7] += neu * out4 * (1 - out4) * (delta * wt[33]) * x2
    wt[8] += neu * out5 * (1 - out5) * (delta * wt[34]) * x1
    wt[9] += neu * out5 * (1 - out5) * (delta * wt[34]) * x2
    wt[10] += neu * out6 * (1 - out6) * (delta * wt[35]) * x1
    wt[11] += neu * out6 * (1 - out6) * (delta * wt[35]) * x2
    wt[12] += neu * out7 * (1 - out7) * (delta * wt[36]) * x1
    wt[13] += neu * out7 * (1 - out7) * (delta * wt[36]) * x2
    wt[14] += neu * out8 * (1 - out8) * (delta * wt[37]) * x1
    wt[15] += neu * out8 * (1 - out8) * (delta * wt[37]) * x2
    wt[16] += neu * out9 * (1 - out9) * (delta * wt[38]) * x1
    wt[17] += neu * out9 * (1 - out9) * (delta * wt[38]) * x2
    wt[18] += neu * out10 * (1 - out10) * (delta * wt[39]) * x1
    wt[19] += neu * out10 * (1 - out10) * (delta * wt[39]) * x2
    wt[20] += neu * out11 * (1 - out11) * (delta * wt[40]) * x1
    wt[21] += neu * out11 * (1 - out11) * (delta * wt[40]) * x2
    wt[22] += neu * out12 * (1 - out12) * (delta * wt[41]) * x1
    wt[23] += neu * out12 * (1 - out12) * (delta * wt[41]) * x2
    wt[24] += neu * out13 * (1 - out13) * (delta * wt[42]) * x1
    wt[25] += neu * out13 * (1 - out13) * (delta * wt[42]) * x2
    wt[26] += neu * out14 * (1 - out14) * (delta * wt[43]) * x1
    wt[27] += neu * out14 * (1 - out14) * (delta * wt[43]) * x2
    wt[28] += neu * out15 * (1 - out15) * (delta * wt[44]) * x1
    wt[29] += neu * out15 * (1 - out15) * (delta * wt[44]) * x2


    # wt[0] = round(wt[0], 6)
    # wt[1] = round(wt[1], 6)
    # wt[2] = round(wt[2], 6)
    # wt[3] = round(wt[3], 6)
    # wt[4] = round(wt[4], 6)
    # wt[5] = round(wt[5], 6)
    # wt[6] = round(wt[6], 6)
    # wt[7] = round(wt[7], 6)
    # wt[8] = round(wt[8], 6)
    # wt[9] = round(wt[9], 6)
    # wt[10] = round(wt[10], 6)
    # wt[11] = round(wt[11], 6)
    # wt[12] = round(wt[12], 6)
    # wt[13] = round(wt[13], 6)
    # wt[14] = round(wt[14], 6)
    # wt[15] = round(wt[15], 6)
    # wt[16] = round(wt[16], 6)
    # wt[17] = round(wt[17], 6)
    # wt[18] = round(wt[18], 6)
    # wt[19] = round(wt[19], 6)

    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[0],1))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[1],2))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[2],3))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[3],4))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[4],5))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[5],6))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[6],7))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[7],8))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[8],9))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[9],10))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[10],11))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[11],12))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[12],13))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[13],14))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[14],15))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[15],16))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[16],17))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[17],18))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[18],19))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[19],20))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[20],21))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[21],22))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[22],23))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[23],24))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[24],25))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[25],26))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[26],27))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[27],28))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[28],29))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[29],30))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[30],31))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[31],32))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[32],33))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[33],34))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[34],35))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[35],36))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[36],37))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[37],38))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[38],39))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[39],40))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[40],41))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[41],42))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[42],43))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[43],44))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[44],45))

    con.commit()

def train2(x1, x2, x3, trgt1, trgt2):
    from bpn import neuron
    con = sqlite3.connect('neu_nw.db')
    cur = con.cursor()
    cur.execute("SELECT w FROM weights")
    w1 = cur.fetchall()
    wt = [w1[0][0], w1[1][0], w1[2][0], w1[3][0], w1[4][0], w1[5][0], w1[6][0], w1[7][0], w1[8][0], w1[9][0],
          w1[10][0], w1[11][0], w1[12][0], w1[13][0], w1[14][0], w1[15][0], w1[16][0], w1[17][0], w1[18][0],
          w1[19][0], w1[20][0], w1[21][0], w1[22][0], w1[23][0], w1[24][0]]

    print(wt)
    neu = 1

    res1 = neuron(x1, x2, x3, wt[0], wt[1], wt[2])
    res2 = neuron(x1, x2, x3, wt[3], wt[4], wt[5])
    res3 = neuron(x1, x2, x3, wt[6], wt[7], wt[8])
    out1 = res1[0]
    out2 = res2[0]
    out3 = res3[0]
    res4 = neuron(out1, out2, out3, wt[9], wt[10], wt[11])
    res5 = neuron(out1, out2, out3, wt[12], wt[13], wt[14])
    out4 = res4[0]
    out5 = res5[0]

    delta4 = out4 * (1 - out4) * (trgt1 - out4)
    delta5 = out5 * (1 - out4) * (trgt2 - out5)

    wt[9] += neu * out1 * delta4
    wt[10] += neu * out2 * delta4
    wt[11] += neu * out3 * delta4
    wt[12] += neu * out1 * delta5
    wt[13] += neu * out2 * delta5
    wt[14] += neu * out3 * delta5

    wt[0] += neu * out1 * (1 - out1) * (delta4 * wt[9] + delta5 * wt[12])
    wt[1] += neu * out1 * (1 - out1) * (delta4 * wt[10] + delta5 * wt[13])
    wt[2] += neu * out1 * (1 - out1) * (delta4 * wt[11] + delta5 * wt[14])
    wt[3] += neu * out2 * (1 - out2) * (delta4 * wt[9] + delta5 * wt[12])
    wt[4] += neu * out2 * (1 - out2) * (delta4 * wt[10] + delta5 * wt[13])
    wt[5] += neu * out2 * (1 - out2) * (delta4 * wt[11] + delta5 * wt[14])
    wt[6] += neu * out3 * (1 - out3) * (delta4 * wt[9] + delta5 * wt[12])
    wt[7] += neu * out3 * (1 - out3) * (delta4 * wt[10] + delta5 * wt[13])
    wt[8] += neu * out3 * (1 - out3) * (delta4 * wt[11] + delta5 * wt[14])

    wt[0] = round(wt[0], 6)
    wt[1] = round(wt[1], 6)
    wt[2] = round(wt[2], 6)
    wt[3] = round(wt[3], 6)
    wt[4] = round(wt[4], 6)
    wt[5] = round(wt[5], 6)
    wt[6] = round(wt[6], 6)
    wt[7] = round(wt[7], 6)
    wt[8] = round(wt[8], 6)
    wt[9] = round(wt[9], 6)
    wt[10] = round(wt[10], 6)
    wt[11] = round(wt[11], 6)
    wt[12] = round(wt[12], 6)
    wt[13] = round(wt[13], 6)
    wt[14] = round(wt[14], 6)

    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[0],1))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[1],2))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[2],3))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[3],4))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[4],5))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[5],6))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[6],7))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[7],8))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[8],9))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[9],10))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[10],11))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[11],12))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[12],13))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[13],14))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[14],15))

    con.commit()

    print(wt)
    print(res1)
    print(res2)
    print(res3)
    print(res4)
    print(res5)