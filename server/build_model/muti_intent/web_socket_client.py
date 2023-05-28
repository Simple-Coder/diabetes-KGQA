"""
Created by xiedong
@Date: 2023/5/27 21:43
"""
import asyncio
import websockets

import websocket


# 接收到服务器消息时的回调函数
def on_message(ws, message):
    print("Received message:", message)


# WebSocket客户端连接成功时的回调函数
def on_open(ws):
    print("Connected to server")
    ws.send("Hello, server!")


# WebSocket客户端连接关闭时的回调函数
def on_close(ws):
    print("Connection closed")


# 创建WebSocket客户端实例，并设置回调函数
ws = websocket.WebSocketApp("ws://localhost:9001/",
                            on_message=on_message,
                            on_open=on_open,
                            on_close=on_close)
# 运行客户端
ws.run_forever()

# async def hello(uri):
#     async with websockets.connect(uri) as websocket:
#         await websocket.send("请问二型糖尿病的临床表现是什么,需要吃什么药啊")
#         # print("< Hello world!")
#         while True:
#             recv_text = await websocket.recv()
#             print("收到： {}".format(recv_text))
#
#
# asyncio.get_event_loop().run_until_complete(
#     hello('ws://localhost:9001'))
