


# shapes 裡只能有兩個label
# is glaucoma
def IsGlaucoma(shapes):

    if len(shapes) != 2:
        print("\nLabelBox != 2, LabelBox:{0}".format(len(shapes)))
        return

    w1 = 0
    h1 = 0
    
    w2 = 0
    h2 = 0

    for data in shapes:
        print(data)
        # small (1)
        if data['label'] == "cup":
            bx1 = data['points'][0][0]
            bx2 = data['points'][1][0]
            by1 = data['points'][0][1]
            by2 = data['points'][1][1]
            w1 = abs(bx2 - bx1)
            h1 = abs(by2 - by1)

        # big   (2)
        if data['label'] == "disk":
            ax1 = data['points'][0][0]
            ax2 = data['points'][1][0]
            ay1 = data['points'][0][1]
            ay2 = data['points'][1][1]
            w2 = abs(ax2 - ax1)
            h2 = abs(ay2 - ay1)

    print("\nw1: {0}, w2: {1}, h1:{2}, h2:{3}".format(w1, w2, h1, h2))

    if w1 == 0 or w2 == 0:
        print("No find class label.")

    if w1 < w2 and h1 < h2:
        ratio_w = w1 / w2
        ratio_h = h1 / h2
        print("")
        # print("w1: {0}, h1: {1}, w2: {2}, h2: {3}".format(w1, h1, w2, h2))
        # print("")
        # print("ratio_w: {0}, ratio_h: {1}".format(ratio_w, ratio_h))
        # print("")
        if ratio_w > 0.7 and ratio_h > 0.7:
            S = by1 - ay1
            I = ay2 - by2
            print("\nS: {0}, I: {1}".format(S, I))
            if I < 0 or S < 0:
                # 框預測超過大的框會跑來這
                print("I or S is minus.")
            elif I > S:
                print("I > S, No glaucoma.")
            else:
                print("Yes glaucoma.")
        else:
            # 比率小於0.7
            print("ratio < 0.7, No glaucoma")

    else:
        # 可能長或寬超過比較大的框才跑來這
        print("Width or height Error!")