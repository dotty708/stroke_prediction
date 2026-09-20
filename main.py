"""뇌졸중 예측 실습실 - 데이터 소개 화면"""

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    """데이터를 읽어 온다. bmi 열의 빈 값은 그대로 둔다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


def value_kind(series: pd.Series) -> str:
    """값의 종류를 우리말로 알려 준다."""
    if pd.api.types.is_bool_dtype(series):
        return "참/거짓"
    if pd.api.types.is_integer_dtype(series):
        return "숫자 (정수)"
    if pd.api.types.is_float_dtype(series):
        return "숫자 (소수)"
    return "글자 (범주)"


# ─────────────────────────────  제목  ─────────────────────────────
st.title(f"{APP_ICON} {APP_TITLE}")
st.subheader("1단계 · 이 데이터는 어떤 데이터일까?")

try:
    df = load_data()
except Exception as error:  # 인터넷 연결이 없을 때를 위한 안내
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인한 뒤 새로고침해 주세요.")
    st.caption(f"오류 내용: {error}")
    st.stop()

st.write(
    "뇌졸중(stroke)이 일어난 사람과 일어나지 않은 사람의 건강·생활 정보가 담긴 데이터입니다. "
    "먼저 데이터의 크기와 열의 의미를 살펴봅시다."
)

# ─────────────────────────  큰 숫자 카드 4개  ─────────────────────────
total_people = len(df)
total_columns = df.shape[1]
stroke_people = int((df["stroke"] == 1).sum()) if "stroke" in df.columns else 0
stroke_ratio = stroke_people / total_people * 100 if total_people else 0.0

card1, card2, card3, card4 = st.columns(4)
card1.metric("전체 사람 수", f"{total_people:,}명")
card2.metric("열 개수", f"{total_columns}개")
card3.metric("뇌졸중이 있는 사람 수", f"{stroke_people:,}명")
card4.metric("뇌졸중이 있는 사람 비율", f"{stroke_ratio:.2f}%")

st.divider()

# ───────────────────────────  열 설명 표  ───────────────────────────
st.subheader("📋 열 이름과 뜻 정리하기")
st.write("**우리말 뜻** 칸은 비어 있습니다. 교재를 보면서 직접 채워 넣어 보세요.")

column_table = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": ["" for _ in df.columns],
        "값의 종류": [value_kind(df[col]) for col in df.columns],
        "빈 값 개수": [int(df[col].isna().sum()) for col in df.columns],
    }
)

edited_table = st.data_editor(
    column_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", disabled=True),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 보고 직접 적어 보세요.",
            placeholder="여기에 뜻을 적어 보세요",
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류", disabled=True),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", disabled=True),
    },
    key="column_meaning_table",
)

filled = int((edited_table["우리말 뜻"].astype(str).str.strip() != "").sum())
st.caption(f"채운 칸: {filled} / {len(edited_table)}")

st.divider()

# ─────────────────────────  데이터 앞부분 5줄  ─────────────────────────
st.subheader("🔍 데이터 앞부분 5줄 살펴보기")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# ───────────────────────────  데이터 출처  ───────────────────────────
st.subheader("✍️ 데이터 출처 적기")
st.write("교재에 나온 데이터 출처를 아래 칸에 그대로 적어 보세요.")
source_text = st.text_area(
    "데이터 출처",
    value="",
    height=100,
    placeholder="여기에 데이터 출처를 적어 보세요.",
    key="data_source",
    label_visibility="collapsed",
)

if source_text.strip():
    st.success("적은 출처")
    st.write(source_text)
