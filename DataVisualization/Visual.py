from graphviz import Digraph
import numpy as np
import os
from joblib import Parallel, delayed

np.set_printoptions(precision=4)

def plot_petri(petri_gra,loc):
    dot = Digraph()
    data = petri_gra
    data = np.array(data,dtype=int)
    # print(data)

    deli_inde = int((data.shape[1]-1)/2)

    for i in range(len(data)):
        # dot.node("P"+str(i+1),"P"+str(i+1))
        # dot.node("P" + str(i + 1), "●",labelfloat=True)
        if data[i][-1] >= 1 :
            no_str = "P"+str(i + 1)+"\n"
            for j in range(data[i][-1]):
                no_str += "● "

            dot.node("P" + str(i + 1), no_str)
        else:
            dot.node("P" + str(i + 1), "P" + str(i + 1) + "\n\n")


    for i in range(deli_inde):
        # dot.node("P"+str(i+1),"P"+str(i+1))
        # dot.node("P" + str(i + 1), "●",labelfloat=True)
        dot.node("t" + str(i + 1), "t"+str(i + 1)+"",shape="box")
    for i in range(len(data)):
        for j in range(deli_inde):
            if data[i][j] == 1:
                dot.edge(str("P" + str(i+1)),str("t" + str(j+1)))
    # print(deli_inde)

    metrix_right = data[:,deli_inde:-1]
    # print(metrix_right)

    for i in range(len(metrix_right)):
        for j in range(metrix_right.shape[1]):
            if metrix_right[i][j] == 1:
                dot.edge(str("t" + str(j + 1)),str("P" + str(i + 1)))


    dot.format = 'png'
    try:
        dot.render(loc)
    except Exception:
        return

def plot_arri_gra(v_list, edage_list, arctrans_list, loc):
    dot = Digraph()
    for i in range(len(v_list)):
        dot.node("M" + str(i + 1), "M" + str(i) + "\n" + str(v_list[i]),shape="box")

    for edage,arctrans in zip(edage_list, arctrans_list):
        dot.edge(str("M" + str(edage[0] + 1)),str("M" + str(edage[1] + 1)),label= ("t" + str(arctrans+1)))
    dot.attr(fontsize='20')
    dot.format = 'png'
    try:
        dot.render(loc)
    except Exception:
        return

def plot_spn(v_list, edage_list, arctrans_list, labda, sv, midubiaoji, mu_biaoji, loc="test-output/test.gv"):
    dot = Digraph()
    # print(pangbiao_list)
    for i in range(len(v_list)):
        dot.node("M" + str(i + 1), "M" + str(i) + "\n" + str(v_list[i]),shape="box")

    for edage,arctrans in zip(edage_list, arctrans_list):
        at_idx = int(arctrans)
        # at_idx = int(re.findall(r"\d+\.?\d*", str(arctrans))[0]) - 1
        dot.edge(str("M" + str(edage[0] + 1)),str("M" + str(edage[1] + 1)),label= (str("t%s" % (arctrans + 1)) + " [" + str(labda[at_idx]) + "]"))


    dot.attr(label=r'\n Steady State Probability: \n' + str(np.array(sv)) +'\n Token Probability Density Function:\n' +
                   str(np.array(midubiaoji)) +'\n The Average Number of Tokens in the Place :\n' + str(np.array(mu_biaoji)) + "\n Sum of the Average Numbers of Tokens:\n" +
                   str(np.array([np.sum(mu_biaoji)])))
    dot.attr(fontsize='20')
    dot.format = 'png'

    try:
        dot.render(loc)
    except Exception:
        return


def save_i_pic(data,w_pic_loc,counter):
    plot_petri(data['petri_net'], os.path.join(w_pic_loc, "data(petri)%s" % str(counter)))
    plot_spn(data['arr_vlist'], data['arr_edge'], data['arr_tranidx'],
             data['spn_labda'], data['spn_steadypro'], data['spn_markdens'],
             data['spn_allmus'], os.path.join(w_pic_loc, "data(arr)%s" % str(counter))
             )
    os.remove(os.path.join(w_pic_loc, "data(arr)%s" % str(counter)))
    os.remove(os.path.join(w_pic_loc, "data(petri)%s" % str(counter)))

def visual_data(all_data,w_pic_loc,pall_job):
    Parallel(n_jobs=pall_job)(delayed(save_i_pic)(all_data["data%s"%str(i+1)], w_pic_loc, i + 1) for i in range(len(all_data)))
    # counter = 1
    # for data in all_data.values():
    #     plot_petri(data['petri_net'], os.path.join(w_pic_loc, "data(petri)%s" % str(counter)))
    #     plot_spn(data['arr_vlist'], data['arr_edge'], data['arr_tranidx'],
    #                     data['spn_labda'], data['spn_steadypro'], data['spn_markdens'],
    #                     data['spn_allmus'], os.path.join(w_pic_loc, "data(arr)%s" % str(counter))
    #                     )
    #     os.remove(os.path.join(w_pic_loc, "data(arr)%s" % str(counter)))
    #     os.remove(os.path.join(w_pic_loc, "data(petri)%s" % str(counter)))
    #     counter += 1
    print("save  pic successful!!")