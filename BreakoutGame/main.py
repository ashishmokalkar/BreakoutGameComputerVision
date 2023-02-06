from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)
detector = HandDetector(detectionCon=0.8, maxHands=1)
game_background = cv2.imread("Resources/background4.png")
bat = cv2.imread("Resources/bat1.png")
ball = cv2.imread("Resources/ball.png")
game_over = cv2.imread("Resources/gameover3.png")
score = cv2.imread("Resources/score.png")
startgame = cv2.imread("Resources/startgame1.jpg")
youwon = cv2.imread("Resources/youwon.jpg")

ball = cv2.resize(ball, (30, 30), interpolation=cv2.INTER_LINEAR)

ball_init_pos_h = 537
ball_end_pos_h = 567
ball_init_pos_w = 360
ball_end_pos_w = 390

ball_x_speed = -15
ball_y_speed = -13

bat_init_pos_h = 572
bat_end_pos_h = 600
bat_init_pos_w = 288
bat_end_pos_w = 462

background_init_pos_h = 0
background_end_pos_h = 700
background_init_pos_w = 0
background_end_pos_w = 750

points_bottom = np.array([[0, 613], [750, 613], [750, 700], [0, 700]])
points_left = np.array([[0, 0], [13, 0], [13, 700], [0, 700]])
points_right = np.array([[735, 0], [750, 0], [750, 700], [735, 700]])
points_top = np.array([[0, 0], [750, 0], [750, 15], [0, 15]])

blocks_1 = np.array([[25, 25], [150, 25], [150, 75], [25, 75]])
blocks_2 = np.array([[175, 25], [300, 25], [300, 75], [175, 75]])
blocks_3 = np.array([[325, 25], [450, 25], [450, 75], [325, 75]])
blocks_4 = np.array([[475, 25], [600, 25], [600, 75], [475, 75]])
blocks_5 = np.array([[625, 25], [725, 25], [725, 75], [625, 75]])

color_list = [(242, 169, 32), (134, 46, 234), (209, 200, 55), (39, 178, 89), (223, 69, 160)]

line1_blocks = [blocks_1, blocks_2, blocks_3, blocks_4, blocks_5]

blocks_2_3 = np.array([[25, 125], [150, 125], [150, 175], [25, 175]])
blocks_2_5 = np.array([[175, 125], [300, 125], [300, 175], [175, 175]])
blocks_2_1 = np.array([[325, 125], [450, 125], [450, 175], [325, 175]])
blocks_2_2 = np.array([[475, 125], [600, 125], [600, 175], [475, 175]])
blocks_2_4 = np.array([[625, 125], [725, 125], [725, 175], [625, 175]])
line2_blocks = [blocks_2_3, blocks_2_5, blocks_2_1, blocks_2_2, blocks_2_4]
blocks_coord_l2 = [25, 150, 175, 300, 325, 450, 475, 600, 625, 725]
blocks_coord_l1 = [25, 150, 175, 300, 325, 450, 475, 600, 625, 725]
game_over_flag = False
start_flag = False
you_won_flag = False

score_value = 0
startgame = cv2.resize(startgame, (200, 136),
                             interpolation=cv2.INTER_LINEAR)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    copiedImage = img.copy()
    anothercopied = img.copy()
    img = cv2.resize(img, (750, 700),
                             interpolation=cv2.INTER_LINEAR)
    copiedImage = cv2.resize(copiedImage, (750,700),
                              interpolation=cv2.INTER_LINEAR)
    anothercopied = cv2.resize(copiedImage, (750, 700),
                             interpolation=cv2.INTER_LINEAR)
    hands, img = detector.findHands(img)  # with draw

    if not start_flag:
        img[172:308, 450:650] = startgame
        cv2.imshow("Image", img)
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        if not start_flag:
            fingers1 = detector.fingersUp(hand1)
            print(fingers1)
            if fingers1 == [1, 1, 1, 0, 0]:
                if lmList1[12][0] > 450 and lmList1[12][0] < 650 and lmList1[12][1] > 172 and lmList1[12][1] < 308 :
                    start_flag = True
        else:
            bat_init_pos_w = lmList1[8][0] - 87
            bat_end_pos_w = lmList1[8][0] + 87
            if bat_init_pos_w < 0:
                bat_init_pos_w = 0
                print("init: ", bat_init_pos_w)
                bat_end_pos_w = 174
                print("end: ", bat_end_pos_w)
            elif bat_end_pos_w > 750:
                bat_init_pos_w = 576
                bat_end_pos_w = 750

    if start_flag:
        copiedImage[background_init_pos_h:background_end_pos_h, background_init_pos_w:background_end_pos_w] = game_background  # First parameter is height, second parameter is width
        copiedImage = cv2.addWeighted(img, 0.8, copiedImage, 0.2, 0)
        cv2.fillPoly(copiedImage, pts=[points_bottom], color=(148, 75, 9))
        cv2.fillPoly(copiedImage, pts=[points_left], color=(148, 75, 9))
        cv2.fillPoly(copiedImage, pts=[points_right], color=(148, 75, 9))
        cv2.fillPoly(copiedImage, pts=[points_top], color=(148, 75, 9))
        a = 0
        for i in line2_blocks:
            cv2.fillPoly(copiedImage, pts=[i], color=color_list[a])
            a = a + 1
        b = 0
        for j in line1_blocks:
            cv2.fillPoly(copiedImage, pts=[j], color=color_list[b])
            b = b + 1
        copiedImage[bat_init_pos_h:bat_end_pos_h, bat_init_pos_w:bat_end_pos_w] = bat
        k = 0
        m = 0
        while ((k + 1) < len(blocks_coord_l2)):
            if ((ball_end_pos_w) > blocks_coord_l2[k] and ball_init_pos_w < blocks_coord_l2[k+1] and ball_init_pos_h > 125 and ball_init_pos_h < 175) :
                ball_y_speed = ball_y_speed * -1
                line2_blocks.pop(m)
                blocks_coord_l2.pop(k+1)
                blocks_coord_l2.pop(k)
                print(blocks_coord_l2)
                print("Removed")
                score_value = int(score_value) + 1
            k = k + 2
            m = m + 1
        n = 0
        l = 0
        while ((n + 1) < len(blocks_coord_l1)):
            if ((ball_end_pos_w) > blocks_coord_l1[n] and ball_init_pos_w < blocks_coord_l1[
                n + 1] and ball_init_pos_h > 25 and ball_init_pos_h < 75):
                ball_y_speed = ball_y_speed * -1
                line1_blocks.pop(l)
                blocks_coord_l1.pop(n + 1)
                blocks_coord_l1.pop(n)
                print(blocks_coord_l1)
                print("Removed")
                score_value = int(score_value) + 1
            n = n + 2
            l = l + 1
        if len(line1_blocks) == 0 and len(line2_blocks) == 0:
            you_won_flag = True

        if ball_end_pos_h >= 572 and ball_end_pos_w >= bat_init_pos_w and ball_end_pos_w <= bat_end_pos_w:
            ball_y_speed = ball_y_speed * -1
        if (ball_init_pos_h <= 15):
            ball_y_speed = ball_y_speed * -1
        if (ball_init_pos_w <= 13):
            ball_x_speed = -ball_x_speed
        if (ball_end_pos_w >= 737):
            ball_x_speed = -ball_x_speed
        if (ball_end_pos_h >= 610):
            game_over_flag = True
        ball_init_pos_w = ball_init_pos_w + ball_x_speed
        ball_end_pos_w = ball_end_pos_w + ball_x_speed
        ball_init_pos_h = ball_init_pos_h + ball_y_speed
        ball_end_pos_h = ball_end_pos_h + ball_y_speed

        # copiedImage[650:690, 308:442] = score
        image = cv2.putText(copiedImage, 'SCORE: ', (300,664), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (159, 208, 27), 3, cv2.LINE_AA)

        score_value = str(score_value) + ""

        image = cv2.putText(copiedImage, score_value, (445, 664), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (159, 208, 27), 3, cv2.LINE_AA)

        if you_won_flag:
            youwon = cv2.resize(youwon, (300, 227), interpolation=cv2.INTER_LINEAR)
            copiedImage[237:464, 225:525] = youwon
            cv2.imshow("Image", copiedImage)
        if game_over_flag:
            game_over = cv2.resize(game_over, (300, 227), interpolation=cv2.INTER_LINEAR)
            copiedImage[237:464, 225:525] = game_over
            cv2.imshow("Image", copiedImage)
        else:
            copiedImage[ball_init_pos_h:ball_end_pos_h, ball_init_pos_w:ball_end_pos_w] = ball
            cv2.imshow("Image", copiedImage)

    cv2.waitKey(1)
cap.release()