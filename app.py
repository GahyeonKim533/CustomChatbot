import os
import csv
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from streamlit_folium import st_folium
from datetime import datetime
import openai
from dotenv import load_dotenv
from streamlit_geolocation import streamlit_geolocation

# 환경 변수를 로드합니다.
load_dotenv()

# API 키를 환경 변수에서 가져옵니다.
API_KEY = os.getenv("API_KEY")
openai.api_key = API_KEY

# Streamlit 세션 상태를 초기화합니다. 이는 대화 내역을 저장하는 데 사용됩니다.
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 사용자 질문에 대한 응답을 처리하는 함수입니다.
def ask_gpt3(question):
    # 이전 대화 내역을 기반으로 GPT-4에게 요청할 쿼리를 생성합니다.
    conversation = [{"role": "user", "content": message} for message in st.session_state.chat_history if "질문" in message]
    conversation.append({"role": "user", "content": f"우린 역할극을 할거야. 이 질문에 니가 마치 의사인 것 처럼 \
                         가정해서 병명을 진단하고 답변해줘 역할극 티는 내지 말고: {question}"})

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo-preview",  # 적절한 모델을 지정합니다.
        messages=conversation,
        max_tokens=800,
        temperature=0.7,
    )
    # 로그 파일에 질문과 답변을 기록합니다.
    with open('log.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([question, response.choices[0].message.content, str(datetime.now())[:19]])
        
    return response.choices[0].message.content


# 지도 생성
def create_map(df_hospitals, my_locations):
    # 서울의 중심에 지도 생성
    latitude, longitude = my_locations
    m = folium.Map(location=[latitude, longitude], zoom_start=12)
    # 데이터 프레임의 각 행에 대하여
    for index, row in df_hospitals.iterrows():
        # 마커 추가
        folium.Marker(
            [row['경도'], row['위도']],
            tooltip=row['기관명'],  # 팝업에 표시될 내용row['기관명']  # 마우스 오버시 표시될 내용
        ).add_to(m)
        
    folium.Marker(
            [latitude, longitude],
            tooltip='내 위치',
            icon = folium.map.Icon('red')
        ).add_to(m)

    return m


# 메인 함수입니다.
if __name__ == "__main__":
    df_hospitals = pd.read_csv('의료기관.csv',encoding='cp949')
    #df_hospitals = df_hospitals.loc[(abs(df_hospitals['경도'] - 37.4871305)< 0.03) & (abs(df_hospitals['위도'] - 126.9011842)< 0.03)]
    df_hospitals = df_hospitals.reset_index()
    
    # Streamlit 페이지 설정
    st.set_page_config(layout="wide", page_title="LLM기반의 의료 상담 앱")
    # HTML 컴포넌트를 사용하여 위치 정보 표시
    # 페이지 제목
    st.title('LLM기반의 의료 상담 앱')

    # 질문 입력을 위한 텍스트 박스
    question = st.text_input("증상을 입력하세요", "")

    # 질문에 대한 답변을 생성하는 버튼
    if st.button('AI 분석 답변 생성 시작'):
        if question:
            answer = ask_gpt3(question)  # GPT-3 모델을 호출하여 답변을 받습니다.
            st.session_state.chat_history.append(f"질문: {question}")
            st.session_state.chat_history.append(f"답변: {answer}")
            # 모든 대화 내역을 화면에 표시합니다.
            for message in st.session_state.chat_history:
                st.text(message)
        else:
            st.error("질문을 제공해주세요.")  # 필수 입력이 없을 경우 사용자에게 알림


    st.write("#### 현재 위치를 기준으로 주변 병원을 추천드리겠습니다.")
    location = streamlit_geolocation()
    locations = location['latitude'], location['longitude']
    if location['latitude']:
        try: 
            map = create_map(df_hospitals, locations)
            folium_static(map)
        except:
            st.write("위치 정보를 가져올 수 없습니다. 위치 서비스가 활성화되어 있는지 확인하세요.")