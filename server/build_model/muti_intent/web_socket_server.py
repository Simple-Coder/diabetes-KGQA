"""
Created by xiedong
@Date: 2023/5/27 21:36
"""
import asyncio
import websockets

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


import asyncio
import websockets
import time


async def echo(websocket, path):
    async for message in websocket:
        message = "I got your message: {}".format(message)
        await websocket.send(message)

        while True:
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if str(t).endswith("0"):
                await websocket.send(t)
                break


asyncio.get_event_loop().run_until_complete(
    websockets.serve(echo, 'localhost', 8765))
asyncio.get_event_loop().run_forever()

