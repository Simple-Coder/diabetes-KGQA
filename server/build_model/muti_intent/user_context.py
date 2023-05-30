"""
Created by xiedong
@Date: 2023/5/30 21:20
"""


class UserContext:
    def __init__(self, clientId, username, query):
        self.clientId = clientId
        self.username = username
        self.query = query
        self.all_slots = []
        self.slot_info = None

    def getusername(self):
        return self.username

    def getContextSlot(self):
        return self.slot_info

    def setContextSlot(self, slot_info):
        self.slot_info = slot_info

    def getquery(self):
        return self.query

    def setusername(self, username):
        self.username = username

    def setAllSlots(self, all_slots):
        self.all_slots = all_slots

    def getAllSlots(self):
        return self.all_slots
