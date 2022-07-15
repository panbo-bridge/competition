#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
import pyautogui
# while 1:
#   time.sleep(30);
# time.sleep(1)
def get_trace(x_trace):
    move_x = []
    sum_value = 0
    for i in range(len(x_trace)):
        if i == 0:
            move_value = x_trace[i] -0
        else:
            move_value = x_trace[i] - x_trace[i-1]

        if sum_value + move_value < 258:
            sum_value  = sum_value + move_value
            move_x.append(move_value)
        else:
            # for i in range(50):
            #     move_x.append(0)
            move_x.append(258-sum_value)
            # for i in range(50):
            #     move_x.append(0)
    return move_x
trace_ = "0 1 2 2 2 4 5 7 9 11 19 22 27 32 39 45 51 60 65 71 78 83 90 97 102 109 119 128 134 142 149 154 161 168 171 176 183 192 198 204 207 214 220 229 232 238 241 244 245 248 250 252 254 257 259 260 262 264 265 267 267 269 270 270 273 274 275 278 280 283 287 289 292 295 298 298 301 303 306 310 313 314 317 319 319 321 322 322 323 324 325 327 327 329 330 331 334 335 338 340 340 340"
trace_l = []
for i in trace_.split(" "):
    trace_l.append(int(i))
trace = get_trace(trace_l)
# while True:
#     current_x = 1100
#     pyautogui.moveTo(1100, 600)
#     for i in trace:
#         current_x = current_x + i
#         pyautogui.dragTo(current_x,600, button='left')
import autopy,PIL
#
# autopy.mouse.smooth_move(880, 480) # 移动鼠标
#
# while True:
#     autopy.mouse.move(880, 480) # 平滑移动鼠标（上面那个是瞬间的）
#     # autopy.mouse.click() # 单击
#     autopy.mouse.toggle(None,True) # 按下左键
#     # autopy.mouse.smooth_move(1138, 480)
#     current = 880
#     for item in trace:
#         time.sleep(0.01)
#         current = current + item
#         autopy.mouse.move(current, 480) # 平滑移动鼠标（上面那个是瞬间的）
#     time.sleep(1)
#     autopy.mouse.toggle(None,False) # 松开左键

import pynput

mouse = pynput.mouse.Controller()

# 移动鼠标到绝对坐标与相对坐标

# mouse.move(dx=5, dy=-5)

# 鼠标按下与释放
trace_ = [(1082, 594), (1083, 595), (1084, 595), (1086, 595), (1089, 595), (1091, 595), (1094, 595), (1097, 596), (1098, 596), (1101, 596), (1103, 596), (1103, 596), (1106, 596), (1106, 596), (1108, 596), (1111, 596), (1112, 596), (1115, 596), (1117, 596), (1118, 596), (1120, 596), (1122, 596), (1123, 596), (1125, 596), (1126, 596), (1127, 596), (1128, 596), (1130, 596), (1131, 596), (1134, 596), (1138, 596), (1142, 596), (1145, 596), (1148, 596), (1151, 596), (1152, 596), (1155, 596), (1156, 596), (1158, 596), (1161, 597), (1162, 597), (1165, 598), (1168, 598), (1171, 599), (1172, 599), (1176, 599), (1179, 600), (1182, 600), (1185, 600), (1188, 600), (1190, 601), (1197, 601), (1204, 601), (1206, 601), (1210, 602), (1212, 604), (1213, 604), (1215, 604), (1216, 604), (1216, 604), (1218, 604), (1220, 604), (1222, 604), (1225, 604), (1228, 604), (1229, 604), (1233, 604), (1236, 604), (1237, 604), (1240, 605), (1243, 605), (1246, 605), (1248, 605), (1251, 606), (1253, 606), (1257, 607), (1259, 608), (1261, 608), (1264, 609), (1267, 610), (1271, 610), (1277, 611), (1281, 611), (1283, 612), (1286, 613), (1288, 613), (1291, 614), (1294, 615), (1295, 615), (1298, 617), (1301, 618), (1305, 618), (1307, 618), (1310, 620), (1314, 620), (1315, 620), (1318, 620), (1320, 621), (1321, 621), (1322, 622), (1323, 622), (1324, 622), (1324, 622), (1326, 623), (1327, 623), (1328, 623), (1329, 623), (1330, 623), (1330, 623), (1331, 623), (1332, 623), (1333, 624), (1335, 624), (1336, 624), (1338, 625), (1340, 625), (1341, 625), (1342, 625), (1343, 625), (1343, 626), (1345, 626), (1346, 626), (1348, 626), (1349, 626), (1350, 626), (1352, 626), (1356, 626), (1356, 626), (1359, 625), (1360, 625), (1361, 625), (1364, 624), (1365, 624), (1367, 624), (1369, 623), (1369, 623), (1370, 622), (1371, 622), (1371, 621), (1372, 621), (1374, 621), (1375, 620), (1376, 620), (1379, 620), (1379, 620), (1380, 619), (1381, 619), (1382, 618), (1384, 617), (1385, 617), (1385, 617), (1385, 617), (1386, 616), (1388, 616), (1391, 615), (1393, 614), (1395, 613), (1398, 613), (1399, 613), (1401, 612), (1402, 612), (1404, 611), (1405, 611), (1406, 611), (1407, 611), (1409, 611), (1410, 610), (1412, 609), (1414, 609), (1417, 608), (1417, 607), (1418, 607), (1419, 607), (1419, 607), (1420, 607), (1421, 607)]
print(len(trace_))
import random
while True:
    mouse.position = trace_[0]
    mouse.press(pynput.mouse.Button.left)
    current = 1100
    for i in trace_:
        a = random.randint(0,3)/100.0
        time.sleep(a)
        mouse.position = i
    time.sleep(0.5)
    mouse.release(pynput.mouse.Button.left)






#
# # 点击鼠标次数
# mouse.click(pynput.mouse.Button.left, count=2)
#
# # 滚轮
# mouse.scroll(dx=0, dy=2)

# def on_move(x, y):
#     print("pointer moved to {}".format((x, y)))
#
# def on_click(x, y, button, pressed):
#     print("{} at {}".format("pressed" if pressed else "released", (x, y)))
#
# def on_scroll(x, y, dx, dy):
#     print("scroll {} at {}".format("down" if dy < 0 else "up", (x, y)))
#
# # 鼠标添加监听器
# with pynput.mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
#     listener.join()
