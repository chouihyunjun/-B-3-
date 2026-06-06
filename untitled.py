import os
from google.colab import userdata
from huggingface_hub import InferenceClient

# 토큰 로드
try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    os.environ["HF_TOKEN"] = HF_TOKEN
except Exception as e:
    print("토큰 로드 실패: 코랩 Secrets를 확인하세요.")

# 모델
MODEL = "Qwen/Qwen2.5-7B-Instruct"
client = InferenceClient(token=HF_TOKEN)


MUSIC_DATABASE = {
    "기쁨": [
        {"title": "틀자마자 저절로 리듬타게 되는 뉴욕 플리🗽기분 확 살아나는 팝송 모음🎶 신나는 노동요🕺🏻", "url": "https://youtu.be/EIBTbf_AMx4?si=qGi5c1kH5_ZzktQ3"},
        {"title": "🚘 운전할 때 잠이 확 깨는 드라이브 팝송│💥 심장이 요동치는 플레이리스트🌿 기분 좋게 시작하는 하루", "url": "https://youtu.be/46iib26GrMM?si=Vb_bgdz0N8seQAq"}
    ],
    "슬픔": [
        {"title": "지치고 힘들 때, 번아웃을 견디게 해주던 플레이리스트 PLAYLIST", "url": "https://youtu.be/fFswLRG4edc?si=-JhNLj_jUz7mOoLY"},
        {"title": "[Playlist] 지친 당신을 위로해 줄 새벽 감성 유스케 플레이리스트🌙", "url": "https://youtu.be/HbBjHIwtjlI?si=iuEVMf5in1HD01R4"}
    ],
    "분노": [
        {"title": "𝐏𝐥𝐚𝐲𝐥𝐢𝐬𝐭🔥전투력 59000%상승💥개쎄지는메탈플리🎵광고없는 플레이리스트", "url": "https://youtu.be/m9f5XgDLlWM?si=WyIvAU7zX24nnX9D"},
        {"title": "[playlist] 개빡칠 때 듣는 노래🤬 | 화났을 때 듣기 좋은 팝송 모음", "url": "https://youtu.be/71hZutqP_cM?si=wekoJjIIQz6aL5d6"}
    ],
    "스트레스": [
        {"title": "스트레스 끝🔥기분 확 올라가는, 신나는 노래들🎧 | EDM | Playlist🎧 | 클럽음악 | 노동요 | 플레이리스트", "url": "https://youtu.be/4gWG7keb51U?si=k2HkrctP1faqlhQT"},
        {"title": "[퇴근 후 지쳤을 때 듣는 재즈] 잘 견뎠어 ☕이제부터 자유다 #스트레스해소 #힐링뮤직", "url": "https://youtu.be/P3y-M4Evmno?si=uEY3oMsvQ7vH9fvR"}
    ],
    "잔잔함": [
        {"title": "[Playlist] 가만히 틀어놓기 좋은 잔잔한 띵곡 모음 | 국내 인디노래모음 플레이리스트", "url": "https://youtu.be/7UCih6xc9kE?si=9v3hWFd-dIkA0roW"},
        {"title": "𝗽𝗹𝗮y𝗹𝗶𝘀𝘁🎧 틀어놓기만 해도 분위기를 채워주는 잔잔한 플레이리스트", "url": "https://youtu.be/3lZYvVRCWO0?si=XkunI9qaUZLLgaOW"}
    ],
    "불안": [
        {"title": "불안한 마음을 위한 힐링음악☁스트레스해소음악,명상음악,이완음악,스파음악,수면음악", "url": "https://youtu.be/cRbYxsLHyJU?si=aMfWdHax9nZwi4HH"},
        {"title": "[playlist] 마음이 복잡할때 듣는 플리", "url": "https://youtu.be/JD1Q3BtscqA?si=_Y9daRRkhqfqFTwm"}
    ],
    "우울": [
        {"title": "[playlist] 위로받고 싶은 너에게 | 우울할 때 듣는 팝송 모음", "url": "https://youtu.be/YwO8hrtWD7s?si=HzJLyMVD9MWi5yp-"},
        {"title": "[Playlist] 요즘 힘들었던 당신을 ‘위로’해줄 노래✨", "url": "https://youtu.be/4e1YnHauXCU?si=G07xw_RAtNuhgzeu"}
    ],
    "설렘": [
        {"title": "𝐏𝐥𝐚𝐲𝐥𝐢𝐬𝐭 | 듣자마자 설레는 달달한 사랑 노래 플레이 리스트❤️", "url": "https://youtu.be/cePhzednLig?si=2GjWHec4DNyfU2X_"},
        {"title": "[𝐩𝐥𝐚y𝐥𝐢𝐬𝐭] 도입부만 들어도 설레는 사랑노래❤️", "url": "https://youtu.be/jC_IOd-ZjnE?si=akuOeH7OiirZWbIW"}
    ]
}

def analyze_emotion(user_input: str):
    messages = [
        {
            "role": "system",
            "content": (
                "당신은 사용자의 문장을 분석하는 감정 분류기입니다.\n"
                "사용자의 입력 문장을 보고 오직 [기쁨, 슬픔, 분노, 스트레스, 잔잔함, 불안, 우울, 설렘] 중 하나의 단어로만 답변하세요.\n"
                "수식어나 다른 설명은 절대 하지 말고 딱 한 단어만 출력하세요. 예: 기쁨"
            )
        },
        {
            "role": "user",
            "content": f"이 문장의 감정을 한 단어로 분류해줘: '{user_input}'"
        }
    ]

    response = client.chat_completion(messages=messages, model=MODEL, max_tokens=10)
    emotion = response.choices[0].message.content.strip()
    return emotion

# 실행 루프

print("\n사용자 감정에 맞는 플레이리스트를 매칭해주는 스마트 무드플레이리스트 시작!")
print("종료하고 싶다면 '/quit'을 입력하세요.")

while True:
    유저_입력 = input("\n오늘 당신의 기분은 어떤가요? > ").strip()

    if 유저_입력 == "/quit":
        print("\n프로그램을 종료합니다.")
        break
    if not 유저_입력:
        continue

    print("AI가 감정을 분석하는 중입니다...")

    감정결과 = analyze_emotion(유저_입력)
    print(f"AI 분석 결과 오늘 당신의 감정 상태는: [{감정결과}] 입니다.")

    print("\n[당신을 위한 플레이리스트 추천 리스트] ──────────────────")
    if 감정결과 in MUSIC_DATABASE:
        추천곡들 = MUSIC_DATABASE[감정결과]
        for 곡 in 추천곡들:
            print(f"▶ 제목: {곡['title']}")
            print(f" 링크: {곡['url']}\n")
    else:
        print("감정에 맞는 플레이리스트를 찾지 못했습니다. 대신 검색 링크를 드릴게요")
        print(f"링크: https://www.youtube.com/results?search_query={감정결과}+플레이리스트")
    print("───────────────────────────────────────────")
