"""
Created by xiedong
@Date: 2023/6/6 21:17
"""
from muti_server.nlg.nlg_config import gossip_corpus, semantic_slot
from muti_server.utils.logger_conf import my_log
from muti_server.utils.json_utils import json_str

log = my_log.logger


def can_search_neo4j(intent):
    log.info("can_search_neo4j intent:{}".format(intent))
    if intent == "others":
        return False
    if intent in gossip_corpus.keys():
        return False
    return True


def get_slot_info(intent):
    log.info("get_slot_info-start,intent:{}".format(intent))
    slot_info = semantic_slot.get(intent)
    return slot_info


def fill_slot_info(intent, recognize_slot_infos, dialog_context):
    try:
        if not can_search_neo4j(intent):
            log.warn("[dst] fill_slot_info-ext,current intent:{},not search".format(intent))
            return None
        log.info("[dst] fill_slot_info-start,intent:{},slots:{}".format(intent, json_str(recognize_slot_infos)))
        slot_info = get_slot_info(intent)
        # 填槽
        slots = slot_info.get("slot_list")
        slot_values = {}
        for slot in slots:
            slot_values[slot] = None
            for ent_info in recognize_slot_infos:
                slot_key = ent_info[0]
                slot_val = ent_info[1]
                slot_values[slot_key] = slot_val
                # print()
                # for e in ent_info["entities"]:
                #     if slot.lower() == e['type']:
                #         slot_values[slot] = entity_link(e['word'], e['type'])

        # last_slot_values = load_user_dialogue_context(username)["slot_values"]
        # for k in slot_values.keys():
        #     if slot_values[k] is None:
        #         slot_values[k] = last_slot_values.get(k, None)

        slot_info["slot_values"] = slot_values

        log.info("[dst] fill_slot_info-intent:{} end...,slot_info:{}".format(intent, json_str(slot_info)))
        return slot_info
    except Exception as e:
        log.error(
            "[dst] fill_slot_info-error,intent:{},slots:{} error:{}".format(intent, json_str(recognize_slot_infos), e))
        return None
