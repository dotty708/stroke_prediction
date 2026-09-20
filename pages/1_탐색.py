"""뇌졸중 예측 실습실 - 2단계 탐색 화면"""

import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

STROKE_COLORS = {"뇌졸중 없음": "#4C78A8", "뇌졸중 있음": "#E45756"}

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    """첫 화면과 같은 데이터를 같은 주소에서 읽어 온다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


def rate_table(frame: pd.DataFrame, column: str, labels: dict, group_name: str) -> pd.DataFrame:
    """어떤 열의 값마다 사람 수와 뇌졸중 비율을 구한다."""
    rows = []
    for value, label in labels.items():
        part = frame[frame[column] == value]
        people = len(part)
        stroke_people = int((part["stroke"] == 1).sum())
        rows.append(
            {
                group_name: label,
                "사람 수": people,
                "뇌졸중 있는 사람 수": stroke_people,
                "뇌졸중 비율(%)": round(stroke_people / people * 100, 2) if people else 0.0,
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────  제목  ─────────────────────────────
st.title(f"{APP_ICON} {APP_TITLE}")
st.subheader("2단계 · 데이터 속을 들여다보기")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인한 뒤 새로고침해 주세요.")
    st.caption(f"오류 내용: {error}")
    st.stop()

df["뇌졸중 여부"] = df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

st.divider()

# ──────────────────────  1. 나이와 평균 혈당의 분포  ──────────────────────
st.subheader("① 나이와 평균 혈당은 어떻게 퍼져 있을까?")

left, right = st.columns(2)

with left:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이(세)", "count": "사람 수"},
    )
    fig_age.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig_age, use_container_width=True)

with right:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"},
        color_discrete_sequence=["#72B7B2"],
    )
    fig_glucose.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig_glucose, use_container_width=True)

st.divider()

# ─────────────────  2. 두 그룹으로 나누어 상자그림으로 비교  ─────────────────
st.subheader("② 뇌졸중을 겪은 사람과 겪지 않은 사람 비교하기")

box_left, box_right = st.columns(2)

with box_left:
    fig_box_age = px.box(
        df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        title="나이 비교",
        labels={"age": "나이(세)"},
        color_discrete_map=STROKE_COLORS,
    )
    fig_box_age.update_layout(showlegend=False)
    st.plotly_chart(fig_box_age, use_container_width=True)

with box_right:
    fig_box_glucose = px.box(
        df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="평균 혈당 비교",
        labels={"avg_glucose_level": "평균 혈당"},
        color_discrete_map=STROKE_COLORS,
    )
    fig_box_glucose.update_layout(showlegend=False)
    st.plotly_chart(fig_box_glucose, use_container_width=True)

mean_table = (
    df.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .round(2)
    .reset_index()
    .rename(columns={"age": "나이 평균(세)", "avg_glucose_level": "평균 혈당의 평균"})
)
mean_table["사람 수"] = (
    df.groupby("뇌졸중 여부")["stroke"].size().reindex(mean_table["뇌졸중 여부"]).values
)
st.write("**두 그룹의 평균값**")
st.dataframe(mean_table, hide_index=True, use_container_width=True)

st.divider()

# ──────────────  3. 고혈압·심장병에 따른 뇌졸중 비율 막대그래프  ──────────────
st.subheader("③ 고혈압과 심장병은 뇌졸중과 관계가 있을까?")

bar_left, bar_right = st.columns(2)

with bar_left:
    hyper_table = rate_table(df, "hypertension", {0: "고혈압 없음", 1: "고혈압 있음"}, "고혈압")
    fig_hyper = px.bar(
        hyper_table,
        x="고혈압",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="고혈압에 따른 뇌졸중 비율",
        color="고혈압",
        color_discrete_sequence=["#9ECAE1", "#E45756"],
    )
    fig_hyper.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_hyper.update_layout(showlegend=False, yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(fig_hyper, use_container_width=True)
    st.dataframe(hyper_table, hide_index=True, use_container_width=True)

with bar_right:
    heart_table = rate_table(df, "heart_disease", {0: "심장병 없음", 1: "심장병 있음"}, "심장병")
    fig_heart = px.bar(
        heart_table,
        x="심장병",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="심장병에 따른 뇌졸중 비율",
        color="심장병",
        color_discrete_sequence=["#9ECAE1", "#E45756"],
    )
    fig_heart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_heart.update_layout(showlegend=False, yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(fig_heart, use_container_width=True)
    st.dataframe(heart_table, hide_index=True, use_container_width=True)

st.divider()

# ─────────────  4. bmi가 비어 있는 사람들의 뇌졸중 비율 비교  ─────────────
st.subheader("④ 체질량지수(bmi)가 비어 있는 사람들은 어떨까?")

missing_bmi = df[df["bmi"].isna()]
missing_people = len(missing_bmi)
missing_stroke = int((missing_bmi["stroke"] == 1).sum())
all_people = len(df)
all_stroke = int((df["stroke"] == 1).sum())

bmi_table = pd.DataFrame(
    [
        {
            "구분": "bmi가 비어 있는 사람",
            "사람 수": missing_people,
            "뇌졸중 있는 사람 수": missing_stroke,
            "뇌졸중 비율(%)": round(missing_stroke / missing_people * 100, 2) if missing_people else 0.0,
        },
        {
            "구분": "전체 사람",
            "사람 수": all_people,
            "뇌졸중 있는 사람 수": all_stroke,
            "뇌졸중 비율(%)": round(all_stroke / all_people * 100, 2) if all_people else 0.0,
        },
    ]
)
st.dataframe(bmi_table, hide_index=True, use_container_width=True)
st.caption("두 비율을 견주어 보고, 빈 값이 우연히 생긴 것인지 생각해 봅시다.")

st.divider()

# ────────────────────  5. 흡연 상태별 사람 수 세기  ────────────────────
st.subheader("⑤ 흡연 상태에 따라 사람 수는 얼마나 될까?")

smoking_table = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
smoking_table["전체 중 비율(%)"] = (smoking_table["사람 수"] / all_people * 100).round(2)
st.dataframe(smoking_table, hide_index=True, use_container_width=True)
