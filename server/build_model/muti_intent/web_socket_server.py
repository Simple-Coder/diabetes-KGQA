"""
Created by xiedong
@Date: 2023/5/27 21:36
"""
import asyncio
import websockets

from websocket_server import WebsocketServer


# 客户端连接时的回调函数
def new_client(client, server):
    print("New client connected and was given id %d" % client['id'])
    server.send_message_to_all("A new client has joined!")


# 接收到客户端消息时的回调函数
def message_received(client, server, message):
    print("Client(%d) said: %s" % (client['id'], message))
    server.send_message_to_all("Client(%d) said: %s" % (client['id'], message))


# 创建WebSocket服务器实例，监听在指定端口
server = WebsocketServer('0.0.0.0',9001)
# 设置回调函数
server.set_fn_new_client(new_client)
server.set_fn_message_received(message_received)
# 启动服务器
server.run_forever()

#
# # 处理 WebSocket 连接
# async def handle_websocket(websocket, path):
#     async for message in websocket:
#         print(f"Received message: {message}")
#
#         # 发送消息给客户端
#         response = f"Server received: {message}"
#         await websocket.send(response)
#
#
# # 启动 WebSocket 服务器
# start_server = websockets.serve(handle_websocket, 'localhost', 8765)
#
#
# # 运行事件循环
# async def main():
#     async with start_server:
#         print("WebSocket server started")
#         await asyncio.Future()  # 持续运行，直到手动中断
#
#
# asyncio.run(main())


import asyncio
import websockets

# import asyncio
# import websockets
# import time
#
#
# async def echo(websocket, path):
#     async for message in websocket:
#         message = "I got your message: {}".format(message)
#         print("服务端收到:", message)
#         await websocket.send(message)
#
#         while True:
#             t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#             if str(t).endswith("0"):
#                 await websocket.send(t)
#                 break
#
#
# asyncio.get_event_loop().run_until_complete(
#     websockets.serve(echo, '127.0.0.1', 9001))
# asyncio.get_event_loop().run_forever()
