from venv.bpn import *
import sqlite3
import numpy as np

def train(x1, x2, trgt):
    from venv.bpn import neuron1, neuron2
    con = sqlite3.connect('neu_nw.db')
    cur = con.cursor()
    cur.execute("SELECT w FROM weights")
    w1 = cur.fetchall()
    wt = [w1[0][0], w1[1][0], w1[2][0], w1[3][0], w1[4][0], w1[5][0], w1[6][0], w1[7][0], w1[8][0], w1[9][0], w1[10][0], 
          w1[11][0], w1[12][0], w1[13][0], w1[14][0], w1[15][0], w1[16][0], w1[17][0], w1[18][0], w1[19][0], w1[20][0],
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


    for i in range (45):
        cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[i], i+1))

    con.commit()


def train_2(x1, x2, trgt):
    from venv.bpn import neuron1, neuron2, neuron3
    con = sqlite3.connect('neu_nw.db')
    cur = con.cursor()
    cur.execute("SELECT w FROM weights")
    w1 = cur.fetchall()
    wt = [w1[0][0], w1[1][0], w1[2][0], w1[3][0], w1[4][0], w1[5][0], w1[6][0], w1[7][0], w1[8][0], w1[9][0], w1[10][0],
          w1[11][0], w1[12][0], w1[13][0], w1[14][0], w1[15][0], w1[16][0], w1[17][0], w1[18][0], w1[19][0], w1[20][0],
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

    out16 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, wt[30],
          wt[31], wt[32], wt[33], wt[34], wt[35], wt[36], wt[37], wt[38], wt[39], wt[40], wt[41], wt[42], wt[43], wt[44])
    out17 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, wt[45],
          wt[46], wt[47], wt[48], wt[49], wt[50], wt[51], wt[52], wt[53], wt[54], wt[55], wt[56], wt[57], wt[58], wt[59])
    out18 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, wt[60],
          wt[61], wt[62], wt[63], wt[64], wt[65], wt[66], wt[67], wt[68], wt[69], wt[70], wt[71], wt[72], wt[73], wt[74])
    out19 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, wt[75],
          wt[76], wt[77], wt[78], wt[79], wt[80], wt[81], wt[82], wt[83], wt[84], wt[85], wt[86], wt[87], wt[88], wt[89])
    out20 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, wt[90],
          wt[91], wt[92], wt[93], wt[94], wt[95], wt[96], wt[97], wt[98], wt[99], wt[100], wt[101], wt[102], wt[103], wt[104])
    out21 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, wt[105],
    wt[106], wt[107], wt[108], wt[109], wt[110], wt[111], wt[112], wt[113], wt[114], wt[115], wt[116], wt[117], wt[118], wt[119])
    out22 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, wt[120],
    wt[121], wt[122], wt[123], wt[124], wt[125], wt[126], wt[127], wt[128], wt[129], wt[130], wt[131], wt[132], wt[133], wt[134])
    out23 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, wt[135],
    wt[136], wt[137], wt[138], wt[139], wt[140], wt[141], wt[142], wt[143], wt[144], wt[145], wt[146], wt[147], wt[148], wt[149])
    out24 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, wt[150],
    wt[151], wt[152], wt[153], wt[154], wt[155], wt[156], wt[157], wt[158], wt[159], wt[160], wt[161], wt[162], wt[163], wt[164])
    out25 = neuron2(out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, wt[165],
    wt[166], wt[167], wt[168], wt[169], wt[170], wt[171], wt[172], wt[173], wt[174], wt[175], wt[176], wt[177], wt[178], wt[179])

    out = neuron3(out16, out17, out18, out19, out20, out21, out22, out23, out24, out25,
                wt[180], wt[181], wt[182], wt[183], wt[184], wt[185], wt[186], wt[187], wt[188], wt[189])

    delta = out * (1 - out) * (trgt - out)

    wt[180] += neu * out16 * delta
    wt[181] += neu * out17 * delta
    wt[182] += neu * out18 * delta
    wt[183] += neu * out19 * delta
    wt[184] += neu * out20 * delta
    wt[185] += neu * out21 * delta
    wt[186] += neu * out22 * delta
    wt[187] += neu * out23 * delta
    wt[188] += neu * out24 * delta
    wt[189] += neu * out25 * delta

    delta_hid = delta * (wt[180] + wt[181] + wt[182] + wt[183] + wt[184] + wt[185] + wt[186] + wt[187] + wt[188] + wt[189])

    delta_16 = out16 * (1 - out16) * delta_hid
    wt[30] += neu * out1 * delta_16
    wt[31] += neu * out2 * delta_16
    wt[32] += neu * out3 * delta_16
    wt[33] += neu * out4 * delta_16
    wt[34] += neu * out5 * delta_16
    wt[35] += neu * out6 * delta_16
    wt[36] += neu * out7 * delta_16
    wt[37] += neu * out8 * delta_16
    wt[38] += neu * out9 * delta_16
    wt[39] += neu * out10 * delta_16
    wt[40] += neu * out11 * delta_16
    wt[41] += neu * out12 * delta_16
    wt[42] += neu * out13 * delta_16
    wt[43] += neu * out14 * delta_16
    wt[44] += neu * out15 * delta_16

    delta_17 = out17 * (1 - out17) * delta_hid
    wt[45] += neu * out1 * delta_17
    wt[46] += neu * out2 * delta_17
    wt[47] += neu * out3 * delta_17
    wt[48] += neu * out4 * delta_17
    wt[49] += neu * out5 * delta_17
    wt[50] += neu * out6 * delta_17
    wt[51] += neu * out7 * delta_17
    wt[52] += neu * out8 * delta_17
    wt[53] += neu * out9 * delta_17
    wt[54] += neu * out10 * delta_17
    wt[55] += neu * out11 * delta_17
    wt[56] += neu * out12 * delta_17
    wt[57] += neu * out13 * delta_17
    wt[58] += neu * out14 * delta_17
    wt[59] += neu * out15 * delta_17


    delta_18 = out18 * (1 - out18) * delta_hid
    wt[60] += neu * out1 * delta_18
    wt[61] += neu * out2 * delta_18
    wt[62] += neu * out3 * delta_18
    wt[63] += neu * out4 * delta_18
    wt[64] += neu * out5 * delta_18
    wt[65] += neu * out6 * delta_18
    wt[66] += neu * out7 * delta_18
    wt[67] += neu * out8 * delta_18
    wt[68] += neu * out9 * delta_18
    wt[69] += neu * out10 * delta_18
    wt[70] += neu * out11 * delta_18
    wt[71] += neu * out12 * delta_18
    wt[72] += neu * out13 * delta_18
    wt[73] += neu * out14 * delta_18
    wt[74] += neu * out15 * delta_18


    delta_19 = out19 * (1 - out19) * delta_hid
    wt[75] += neu * out1 * delta_19
    wt[76] += neu * out2 * delta_19
    wt[77] += neu * out3 * delta_19
    wt[78] += neu * out4 * delta_19
    wt[79] += neu * out5 * delta_19
    wt[80] += neu * out6 * delta_19
    wt[81] += neu * out7 * delta_19
    wt[82] += neu * out8 * delta_19
    wt[83] += neu * out9 * delta_19
    wt[84] += neu * out10 * delta_19
    wt[85] += neu * out11 * delta_19
    wt[86] += neu * out12 * delta_19
    wt[87] += neu * out13 * delta_19
    wt[88] += neu * out14 * delta_19
    wt[89] += neu * out15 * delta_19

    delta_20 = out20 * (1 - out20) * delta_hid
    wt[90] += neu * out1 * delta_20
    wt[91] += neu * out2 * delta_20
    wt[92] += neu * out3 * delta_20
    wt[93] += neu * out4 * delta_20
    wt[94] += neu * out5 * delta_20
    wt[95] += neu * out6 * delta_20
    wt[96] += neu * out7 * delta_20
    wt[97] += neu * out8 * delta_20
    wt[98] += neu * out9 * delta_20
    wt[99] += neu * out10 * delta_20
    wt[100] += neu * out11 * delta_20
    wt[101] += neu * out12 * delta_20
    wt[102] += neu * out13 * delta_20
    wt[103] += neu * out14 * delta_20
    wt[104] += neu * out15 * delta_20

    delta_21 = out21 * (1 - out21) * delta_hid
    wt[105] += neu * out1 * delta_21
    wt[106] += neu * out2 * delta_21
    wt[107] += neu * out3 * delta_21
    wt[108] += neu * out4 * delta_21
    wt[109] += neu * out5 * delta_21
    wt[110] += neu * out6 * delta_21
    wt[111] += neu * out7 * delta_21
    wt[112] += neu * out8 * delta_21
    wt[113] += neu * out9 * delta_21
    wt[114] += neu * out10 * delta_21
    wt[115] += neu * out11 * delta_21
    wt[116] += neu * out12 * delta_21
    wt[117] += neu * out13 * delta_21
    wt[118] += neu * out14 * delta_21
    wt[119] += neu * out15 * delta_21

    delta_22 = out22 * (1 - out22) * delta_hid
    wt[120] += neu * out1 * delta_22
    wt[121] += neu * out2 * delta_22
    wt[122] += neu * out3 * delta_22
    wt[123] += neu * out4 * delta_22
    wt[124] += neu * out5 * delta_22
    wt[125] += neu * out6 * delta_22
    wt[126] += neu * out7 * delta_22
    wt[127] += neu * out8 * delta_22
    wt[128] += neu * out9 * delta_22
    wt[129] += neu * out10 * delta_22
    wt[130] += neu * out11 * delta_22
    wt[131] += neu * out12 * delta_22
    wt[132] += neu * out13 * delta_22
    wt[133] += neu * out14 * delta_22
    wt[134] += neu * out15 * delta_22

    delta_23 = out23 * (1 - out23) * delta_hid
    wt[135] += neu * out1 * delta_23
    wt[136] += neu * out2 * delta_23
    wt[137] += neu * out3 * delta_23
    wt[138] += neu * out4 * delta_23
    wt[139] += neu * out5 * delta_23
    wt[140] += neu * out6 * delta_23
    wt[141] += neu * out7 * delta_23
    wt[142] += neu * out8 * delta_23
    wt[143] += neu * out9 * delta_23
    wt[144] += neu * out10 * delta_23
    wt[145] += neu * out11 * delta_23
    wt[146] += neu * out12 * delta_23
    wt[147] += neu * out13 * delta_23
    wt[148] += neu * out14 * delta_23
    wt[149] += neu * out15 * delta_23

    delta_24 = out24 * (1 - out24) * delta_hid
    wt[150] += neu * out1 * delta_24
    wt[151] += neu * out2 * delta_24
    wt[152] += neu * out3 * delta_24
    wt[153] += neu * out4 * delta_24
    wt[154] += neu * out5 * delta_24
    wt[155] += neu * out6 * delta_24
    wt[156] += neu * out7 * delta_24
    wt[157] += neu * out8 * delta_24
    wt[158] += neu * out9 * delta_24
    wt[159] += neu * out10 * delta_24
    wt[160] += neu * out11 * delta_24
    wt[161] += neu * out12 * delta_24
    wt[162] += neu * out13 * delta_24
    wt[163] += neu * out14 * delta_24
    wt[164] += neu * out15 * delta_24

    delta_25 = out25 * (1 - out25) * delta_hid
    wt[165] += neu * out1 * delta_25
    wt[166] += neu * out2 * delta_25
    wt[167] += neu * out3 * delta_25
    wt[168] += neu * out4 * delta_25
    wt[169] += neu * out5 * delta_25
    wt[170] += neu * out6 * delta_25
    wt[171] += neu * out7 * delta_25
    wt[172] += neu * out8 * delta_25
    wt[173] += neu * out9 * delta_25
    wt[174] += neu * out10 * delta_25
    wt[175] += neu * out11 * delta_25
    wt[176] += neu * out12 * delta_25
    wt[177] += neu * out13 * delta_25
    wt[178] += neu * out14 * delta_25
    wt[179] += neu * out15 * delta_25

    delta_inp = delta_hid * (wt[30] + wt[31] + wt[32] + wt[33] + wt[34] + wt[35] + wt[36] + wt[37] + wt[38] + wt[39]
                          +  wt[40] + wt[41] + wt[42] + wt[43] + wt[44] + wt[45] + wt[46] + wt[47] + wt[48] + wt[49]
                          +  wt[50] + wt[51] + wt[52] + wt[53] + wt[54] + wt[55] + wt[56] + wt[57] + wt[58] + wt[59]
                          +  wt[60] + wt[61] + wt[62] + wt[63] + wt[64] + wt[65] + wt[66] + wt[67] + wt[68] + wt[69]
                          +  wt[70] + wt[71] + wt[72] + wt[73] + wt[74] + wt[75] + wt[76] + wt[77] + wt[78] + wt[79]
                          +  wt[80] + wt[81] + wt[82] + wt[83] + wt[84] + wt[85] + wt[86] + wt[87] + wt[88] + wt[89]
                          +  wt[90] + wt[91] + wt[92] + wt[93] + wt[94] + wt[95] + wt[96] + wt[97] + wt[98] + wt[99]
                 + wt[100] + wt[101] + wt[102] + wt[103] + wt[104] + wt[105] + wt[106] + wt[107] + wt[108] + wt[109]
                 + wt[110] + wt[111] + wt[112] + wt[113] + wt[114] + wt[115] + wt[116] + wt[117] + wt[118] + wt[119]
                 + wt[120] + wt[121] + wt[122] + wt[123] + wt[124] + wt[125] + wt[126] + wt[127] + wt[128] + wt[129]
                 + wt[130] + wt[131] + wt[132] + wt[133] + wt[134] + wt[135] + wt[136] + wt[137] + wt[138] + wt[139]
                 + wt[140] + wt[141] + wt[142] + wt[143] + wt[144] + wt[145] + wt[146] + wt[147] + wt[148] + wt[149]
                 + wt[150] + wt[151] + wt[152] + wt[153] + wt[154] + wt[155] + wt[156] + wt[157] + wt[158] + wt[159]
                 + wt[160] + wt[161] + wt[162] + wt[163] + wt[164] + wt[165] + wt[166] + wt[167] + wt[168] + wt[169]
                 + wt[170] + wt[171] + wt[172] + wt[173] + wt[174] + wt[175] + wt[176] + wt[177] + wt[178] + wt[179]
                             )

    delta1 = delta_inp * out1 * (1 - out1)
    wt[0] += neu * x1 * delta1
    wt[1] += neu * x2 * delta1

    delta2 = delta_inp * out2 * (1 - out2)
    wt[2] += neu * x1 * delta2
    wt[3] += neu * x2 * delta2

    delta3 = delta_inp * out3 * (1 - out3)
    wt[4] += neu * x1 * delta3
    wt[5] += neu * x2 * delta3

    delta4 = delta_inp * out4 * (1 - out4)
    wt[6] += neu * x1 * delta4
    wt[7] += neu * x2 * delta4

    delta5 = delta_inp * out5 * (1 - out5)
    wt[8] += neu * x1 * delta5
    wt[9] += neu * x2 * delta5

    delta6 = delta_inp * out6 * (1 - out6)
    wt[10] += neu * x1 * delta6
    wt[11] += neu * x2 * delta6

    delta7 = delta_inp * out7 * (1 - out7)
    wt[12] += neu * x1 * delta7
    wt[13] += neu * x2 * delta7

    delta8 = delta_inp * out8 * (1 - out8)
    wt[14] += neu * x1 * delta8
    wt[15] += neu * x2 * delta8

    delta9 = delta_inp * out9 * (1 - out9)
    wt[16] += neu * x1 * delta9
    wt[17] += neu * x2 * delta9

    delta10 = delta_inp * out10 * (1 - out10)
    wt[18] += neu * x1 * delta10
    wt[19] += neu * x2 * delta10

    delta11 = delta_inp * out11 * (1 - out11)
    wt[20] += neu * x1 * delta11
    wt[21] += neu * x2 * delta11

    delta12 = delta_inp * out12 * (1 - out12)
    wt[22] += neu * x1 * delta12
    wt[23] += neu * x2 * delta12

    delta13 = delta_inp * out13 * (1 - out13)
    wt[24] += neu * x1 * delta13
    wt[25] += neu * x2 * delta13

    delta14 = delta_inp * out14 * (1 - out14)
    wt[26] += neu * x1 * delta14
    wt[27] += neu * x2 * delta14

    delta15 = delta_inp * out15 * (1 - out15)
    wt[28] += neu * x1 * delta15
    wt[29] += neu * x2 * delta15

    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[0], 1))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[1], 2))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[2], 3))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[3], 4))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[4], 5))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[5], 6))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[6], 7))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[7], 8))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[8], 9))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[9], 10))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[10], 11))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[11], 12))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[12], 13))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[13], 14))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[14], 15))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[15], 16))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[16], 17))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[17], 18))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[18], 19))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[19], 20))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[20], 21))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[21], 22))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[22], 23))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[23], 24))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[24], 25))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[25], 26))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[26], 27))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[27], 28))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[28], 29))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[29], 30))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[30], 31))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[31], 32))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[32], 33))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[33], 34))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[34], 35))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[35], 36))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[36], 37))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[37], 38))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[38], 39))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[39], 40))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[40], 41))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[41], 42))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[42], 43))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[43], 44))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[44], 45))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[45], 46))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[46], 47))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[47], 48))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[48], 49))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[49], 50))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[50], 51))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[51], 52))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[52], 53))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[53], 54))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[54], 55))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[55], 56))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[56], 57))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[57], 58))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[58], 59))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[59], 60))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[60], 61))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[61], 62))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[62], 63))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[63], 64))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[64], 65))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[65], 66))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[66], 67))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[67], 68))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[68], 69))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[69], 70))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[70], 71))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[71], 72))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[72], 73))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[73], 74))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[74], 75))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[75], 76))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[76], 77))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[77], 78))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[78], 79))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[79], 80))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[80], 81))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[81], 82))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[82], 83))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[83], 84))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[84], 85))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[85], 86))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[86], 87))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[87], 88))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[88], 89))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[89], 90))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[90], 91))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[91], 92))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[92], 93))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[93], 94))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[94], 95))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[95], 96))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[96], 97))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[97], 98))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[98], 99))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[99], 100))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[100], 101))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[101], 102))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[102], 103))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[103], 104))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[104], 105))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[105], 106))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[106], 107))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[107], 108))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[108], 109))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[109], 110))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[110], 111))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[111], 112))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[112], 113))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[113], 114))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[114], 115))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[115], 116))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[116], 117))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[117], 118))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[118], 119))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[119], 120))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[120], 121))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[121], 122))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[122], 123))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[123], 124))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[124], 125))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[125], 126))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[126], 127))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[127], 128))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[128], 129))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[129], 130))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[130], 131))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[131], 132))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[132], 133))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[133], 134))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[134], 135))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[135], 136))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[136], 137))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[137], 138))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[138], 139))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[139], 140))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[140], 141))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[141], 142))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[142], 143))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[143], 144))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[144], 145))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[145], 146))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[146], 147))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[147], 148))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[148], 149))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[149], 150))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[150], 151))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[151], 152))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[152], 153))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[153], 154))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[154], 155))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[155], 156))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[156], 157))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[157], 158))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[158], 159))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[159], 160))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[160], 161))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[161], 162))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[162], 163))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[163], 164))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[164], 165))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[165], 166))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[166], 167))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[167], 168))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[168], 169))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[169], 170))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[170], 171))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[171], 172))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[172], 173))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[173], 174))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[174], 175))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[175], 176))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[176], 177))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[177], 178))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[178], 179))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[179], 180))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[180], 181))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[181], 182))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[182], 183))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[183], 184))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[184], 185))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[185], 186))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[186], 187))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[187], 188))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[188], 189))
    cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[189], 190))
    con.commit()

def init_weight():
    con = sqlite3.connect('neu_nw.db')
    cur = con.cursor()
    wt = np.random.rand(190)

    for i in range (190):
        cur.execute("UPDATE weights SET w=? WHERE w_id=?", (wt[i], i+1))
    con.commit()
