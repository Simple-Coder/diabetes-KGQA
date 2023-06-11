"""
Created by xiedong
@Date: 2023/6/6 21:16
"""
from muti_server.nlg.nlg_config import IntentEnum, intent_threshold_config, gossip_corpus


def build_intent_strategy(intent, conf):
    if intent == "others":
        return IntentEnum.Others
    if intent in gossip_corpus.keys():
        return IntentEnum.Gossip
    if intent == "accept":
        return IntentEnum.Accept
    # 根据意图强度来确认回复策略
    if conf >= intent_threshold_config["accept"]:
        # slot_info["intent_strategy"] = "accept"
        return IntentEnum.Medical
    elif conf >= intent_threshold_config["deny"]:
        # slot_info["intent_strategy"] = "clarify"
        return IntentEnum.Clarify
    else:
        # slot_info["intent_strategy"] = "deny"
        return IntentEnum.DENY
