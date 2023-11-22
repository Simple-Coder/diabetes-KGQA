"""
Created by xiedong
@Date: 2023/6/6 21:16
"""
from muti_server.nlg.nlg_config import IntentEnum, intent_threshold_config, gossip_corpus


def build_intent_enum(intent, conf):
    if not intent:
        return IntentEnum.Others
    elif intent == "others":
        return IntentEnum.Others
    elif intent in gossip_corpus.keys():
        return IntentEnum.Gossip
    elif intent == "accept":
        return IntentEnum.Accept

    # 到这里一定是诊断意图了，根据意图强度来确认回复策略
    elif conf >= intent_threshold_config["accept"]:
        # slot_info["intent_strategy"] = "accept"
        # return IntentEnum.Accept
        return IntentEnum.Medical
    elif conf >= intent_threshold_config["deny"]:
        # slot_info["intent_strategy"] = "clarify"
        return IntentEnum.Clarify
    else:
        # slot_info["intent_strategy"] = "deny"
        return IntentEnum.DENY


def recognize_medical(intent_enum):
    if not intent_enum:
        return False

    if intent_enum == IntentEnum.Medical \
            or intent_enum == IntentEnum.Accept \
            or intent_enum == IntentEnum.Clarify \
            or intent_enum == IntentEnum.DENY:
        return True
    return False


# def build_intent_strategy(intent_enum, conf):
#     if intent_enum == IntentEnum.Others:
#         return IntentEnum.DENY
#     elif intent_enum == IntentEnum.Gossip:
#         return IntentEnum.DENY
#     # 根据意图强度来确认回复策略
#     elif conf >= intent_threshold_config["accept"]:
#         # slot_info["intent_strategy"] = "accept"
#         # return IntentEnum.Accept
#         return IntentEnum.Medical
#     elif conf >= intent_threshold_config["deny"]:
#         # slot_info["intent_strategy"] = "clarify"
#         return IntentEnum.Clarify
#     else:
#         # slot_info["intent_strategy"] = "deny"
#         return IntentEnum.DENY


def get_white_multi_hop_nlu(query, all_intents, all_slots):
    white_querys_multi_hop = ["糖尿病忌吃食物的替代品有哪些?"]

    if query not in white_querys_multi_hop:
        return all_intents, all_slots
    # TODO:处理多跳，数据结构长什么样

    return ('disease_food_replace', 0.9, 2), ('糖尿病')
