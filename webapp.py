import streamlit as st
import pandas as pd
import numpy as np
from utils import refine_question, binary_class, determine_scene, determine_linkage, determine_person, determine_deduction

if 'first_stage_response' not in st.session_state:
    st.session_state.first_stage_response = "情況的總結"
    st.session_state.classification_result = "內外之分"
    st.session_state.inner_num = 0
    st.session_state.outer_num = 0
    st.session_state.change_num = 0
    st.session_state.inner_scene = "內卦未定"
    st.session_state.outer_scene = "外卦未定"
    st.session_state.gossip = "尚未定"
    st.session_state.inner_description = "誰是誰"
    st.session_state.outer_description = "誰是誰"

st.write("施主最近有什麼煩惱?")
prompt = st.text_area('', key='prompt', value = "秦始皇不喜歡吃三文治")
ask_button = st.button('確定')

if ask_button:
    with st.spinner("正在理解"):
        st.session_state.first_stage_response = refine_question(prompt)

st.write(st.session_state.first_stage_response)

confirm_button = st.button('確認')

if confirm_button:
    with st.spinner("正在分析"):
        st.session_state.classification_result = binary_class(st.session_state.first_stage_response)

st.write(st.session_state.classification_result)

st.write("請輸入三個數字:")
col1, col2, col3 = st.columns(3)

# Place a number input box in each column
with col1:
    num1 = st.number_input("Number 1", value=0, step=1)
with col2:
    num2 = st.number_input("Number 2", value=0, step=1)
with col3:
    num3 = st.number_input("Number 3", value=0, step=1)

generate_gossip = st.button("起卦")

if generate_gossip:
    st.session_state.inner_num = num1 % 8
    st.session_state.outer_num = num2 % 8
    st.session_state.change_num = num3 % 6
    with st.spinner("正在決定卦象"):
        st.session_state.inner_person, st.session_state.outer_person = determine_person(st.session_state.classification_result)
        st.session_state.inner_scene, st.session_state.outer_scene, st.session_state.gossip = determine_scene(st.session_state.inner_num, st.session_state.outer_num)

st.write(f"內卦: {st.session_state.inner_num}, 外卦: {st.session_state.outer_num}, 變爻: {st.session_state.change_num}")
st.write("內卦: ", st.session_state.inner_scene, " 外卦: ", st.session_state.outer_scene, " ", st.session_state.gossip)

inner_describe = st.button("內卦象徵")

if inner_describe:
    with st.spinner("正在顯示內卦象徵"):
        st.session_state.inner_description = determine_linkage(
            person=st.session_state.inner_person,
            first_stage_response=st.session_state.first_stage_response,
            scene_num=st.session_state.inner_num
        )

st.write(st.session_state.inner_description)


outer_describe = st.button("外卦象徵")

if outer_describe:
    with st.spinner("正在顯示外卦象徵"):
        st.session_state.outer_description = determine_linkage(
            person=st.session_state.outer_person,
            first_stage_response=st.session_state.first_stage_response,
            scene_num=st.session_state.outer_num
        )

st.write(st.session_state.outer_description)

st.write("施主的問題是什麼?")
question_prompt = st.text_area('', key='question_prompt', value = "秦始皇喜歡吃意粉嗎?")
confirm_question = st.button('遞交')

if confirm_question:
    with st.spinner("正在理解問題"):
        question_response = determine_deduction(first_stage_response=st.session_state.first_stage_response,
                                                inner_person_description=st.session_state.inner_description,
                                                outer_person_description=st.session_state.outer_description,
                                                question=question_prompt)
        st.write(question_response)