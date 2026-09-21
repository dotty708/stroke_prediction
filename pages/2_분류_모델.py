"""뇌졸중 예측 실습실 - 3단계 분류 모델 화면"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

SEED = 42  # 난수를 고정하는 값

ALL_FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]  # bmi 제외

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

STROKE_LABELS = {0: "뇌졸중 없음", 1: "뇌졸중 있음"}
STROKE_COLORS = {"뇌졸중 없음": "#4C78A8", "뇌졸중 있음": "#E45756"}

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    """첫 화면과 같은 데이터를 같은 주소에서 읽어 온다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


def split_train_test(frame: pd.DataFrame):
    """번호 순으로 정렬한 뒤, 열 명씩 묶어 앞 세 명은 테스트용, 나머지 일곱 명은 훈련용으로 나눈다."""
    sorted_frame = frame.sort_values("id").reset_index(drop=True)
    position_in_group = sorted_frame.index % 10
    is_test = position_in_group < 3
    test_frame = sorted_frame[is_test].reset_index(drop=True)
    train_frame = sorted_frame[~is_test].reset_index(drop=True)
    return train_frame, test_frame


def balance_by_undersampling(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """훈련용 데이터의 두 그룹(뇌졸중/아님) 크기를 더 적은 쪽에 맞춘다."""
    counts = frame["stroke"].value_counts()
    minority_count = int(counts.min())
    parts = []
    for stroke_value, count in counts.items():
        part = frame[frame["stroke"] == stroke_value]
        if count > minority_count:
            part = part.sample(n=minority_count, random_state=seed)
        parts.append(part)
    balanced = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced


def accuracy(model, features: pd.DataFrame, answers: pd.Series) -> float:
    return accuracy_score(answers, model.predict(features))


def build_tree_dot(tree_model: DecisionTreeClassifier, feature_names: list[str]):
    """의사결정트리를 st.graphviz_chart로 그릴 수 있는 DOT 문자열로 바꾼다."""
    tree_ = tree_model.tree_
    classes = list(tree_model.classes_)
    stroke_index = classes.index(1) if 1 in classes else None

    lines = [
        "digraph Tree {",
        'node [shape=box, style="rounded,filled", fontsize=11];',
        "edge [fontsize=11];",
    ]
    leaf_count = 0
    leaf_no_count = 0
    used_features: set[str] = set()

    def node_stats(node_id: int):
        samples = int(tree_.n_node_samples[node_id])
        stroke_count = int(tree_.value[node_id][0][stroke_index]) if stroke_index is not None else 0
        ratio = stroke_count / samples * 100 if samples else 0.0
        return samples, stroke_count, ratio

    def recurse(node_id: int):
        nonlocal leaf_count, leaf_no_count
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        samples, stroke_count, ratio = node_stats(node_id)
        is_leaf = left == -1 and right == -1

        if is_leaf:
            leaf_count += 1
            predicted = classes[int(np.argmax(tree_.value[node_id][0]))]
            predicted_label = "뇌졸중" if predicted == 1 else "아님"
            if predicted == 0:
                leaf_no_count += 1
            color = "#F4CCCC" if predicted == 1 else "#CFE2F3"
            label = (
                f"답: {predicted_label}\\n"
                f"사람 수 {samples}명\\n"
                f"뇌졸중 {stroke_count}명 ({ratio:.1f}%)"
            )
            lines.append(f'{node_id} [label="{label}", fillcolor="{color}"];')
        else:
            feature_name = feature_names[tree_.feature[node_id]]
            used_features.add(feature_name)
            threshold = tree_.threshold[node_id]
            question = f"{FEATURE_LABELS[feature_name]} <= {threshold:.1f} ?"
            label = (
                f"{question}\\n"
                f"사람 수 {samples}명\\n"
                f"뇌졸중 {stroke_count}명 ({ratio:.1f}%)"
            )
            lines.append(f'{node_id} [label="{label}", fillcolor="#FFFFFF"];')
            recurse(left)
            recurse(right)
            lines.append(f'{node_id} -> {left} [label="예"];')
            lines.append(f'{node_id} -> {right} [label="아니요"];')

    recurse(0)
    lines.append("}")
    return "\n".join(lines), leaf_count, leaf_no_count, used_features


# ─────────────────────────────  제목  ─────────────────────────────
st.title(f"{APP_ICON} {APP_TITLE}")
st.subheader("3단계 · 뇌졸중을 예측하는 모델 만들기")
st.caption("stroke = 1이면 뇌졸중(양성), stroke = 0이면 뇌졸중이 아님(음성)으로 두었습니다.")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인한 뒤 새로고침해 주세요.")
    st.caption(f"오류 내용: {error}")
    st.stop()

st.divider()

# ───────────────────────────  속성 고르기  ───────────────────────────
st.subheader("① 모델에 넣을 속성 고르기")
selected_features = st.multiselect(
    "입력으로 쓸 속성을 골라 보세요.",
    options=ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: FEATURE_LABELS[c],
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 골라야 모델을 만들 수 있어요. 위에서 두 개 이상 골라 주세요.")
    st.stop()

# ─────────────────────  훈련용·테스트용 나누기  ─────────────────────
train_df, test_df = split_train_test(df)

if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)
    st.caption(f"체질량지수(bmi)의 빈 값은 훈련용의 중앙값인 {bmi_median:.1f}로 채웠습니다.")

st.caption(
    f"훈련용 사람 수: {len(train_df):,}명 · 테스트용 사람 수: {len(test_df):,}명"
)

train_balanced = balance_by_undersampling(train_df, SEED)

X_train = train_balanced[selected_features]
y_train = train_balanced["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

st.caption(
    f"크기를 맞춘 뒤 훈련에 쓰는 사람 수: {len(train_balanced):,}명 "
    f"(뇌졸중 {int((y_train == 1).sum())}명 · 아님 {int((y_train == 0).sum())}명)"
)

st.divider()

# ───────────────────────────  모델 만들기  ───────────────────────────
logreg = LogisticRegression(max_iter=1000, random_state=SEED)
logreg.fit(X_train, y_train)

tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=SEED)
tree_clf.fit(X_train, y_train)

baseline = DummyClassifier(strategy="most_frequent", random_state=SEED)
baseline.fit(train_df[selected_features], train_df["stroke"])

log_train_acc = accuracy(logreg, X_train, y_train)
log_test_acc = accuracy(logreg, X_test, y_test)
tree_train_acc = accuracy(tree_clf, X_train, y_train)
tree_test_acc = accuracy(tree_clf, X_test, y_test)
base_train_acc = accuracy(baseline, train_df[selected_features], train_df["stroke"])
base_test_acc = accuracy(baseline, X_test, y_test)

st.subheader("② 세 모델의 정확도 비교하기")

card1, card2, card3 = st.columns(3)
with card1:
    st.metric("로지스틱 회귀(확률로 답하는 모델)", f"{log_test_acc * 100:.2f}%")
    st.caption(f"훈련 정확도 {log_train_acc * 100:.2f}% · 테스트 정확도 {log_test_acc * 100:.2f}%")
with card2:
    st.metric("의사결정트리(질문으로 답하는 모델)", f"{tree_test_acc * 100:.2f}%")
    st.caption(f"훈련 정확도 {tree_train_acc * 100:.2f}% · 테스트 정확도 {tree_test_acc * 100:.2f}%")
with card3:
    st.metric("가장 흔한 답만 하는 모델", f"{base_test_acc * 100:.2f}%")
    st.caption(f"훈련 정확도 {base_train_acc * 100:.2f}% · 테스트 정확도 {base_test_acc * 100:.2f}%")
st.caption(
    "'가장 흔한 답만 하는 모델'은 입력을 보지 않고, 크기를 맞추기 전 훈련용에서 "
    "더 많았던 답(뇌졸중 아님)만 계속 말하는 모델입니다."
)

st.divider()

# ─────────────────────  산점도 + 경계선 + 나무 영역  ─────────────────────
st.subheader("③ 두 속성으로 그려 보는 경계선과 나무의 칸")

axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_col = st.selectbox(
        "가로축으로 쓸 속성",
        options=selected_features,
        format_func=lambda c: FEATURE_LABELS[c],
        index=0,
        key="x_axis_select",
    )
with axis_col2:
    default_y_index = 1 if len(selected_features) > 1 else 0
    y_col = st.selectbox(
        "세로축으로 쓸 속성",
        options=selected_features,
        format_func=lambda c: FEATURE_LABELS[c],
        index=default_y_index,
        key="y_axis_select",
    )

if x_col == y_col:
    st.warning("가로축과 세로축은 서로 다른 속성으로 골라 주세요.")
    st.stop()

other_cols = [c for c in selected_features if c not in (x_col, y_col)]
fixed_values = {c: float(test_df[c].median()) for c in other_cols}

if fixed_values:
    fixed_text = " · ".join(f"{FEATURE_LABELS[c]} = {v:.2f}" for c, v in fixed_values.items())
    st.caption(f"그림에 없는 속성은 테스트 데이터의 중앙값으로 고정했습니다: {fixed_text}")
else:
    st.caption("고른 속성이 두 개뿐이라 따로 고정할 속성이 없습니다.")

# 그림의 가로·세로 범위 정하기
x_min, x_max = float(test_df[x_col].min()), float(test_df[x_col].max())
y_min, y_max = float(test_df[y_col].min()), float(test_df[y_col].max())
x_pad = (x_max - x_min) * 0.05 or 1.0
y_pad = (y_max - y_min) * 0.05 or 1.0
x_range = (x_min - x_pad, x_max + x_pad)
y_range = (y_min - y_pad, y_max + y_pad)

# 의사결정트리가 나눈 칸을 옅은 색으로 칠하기 위한 격자
grid_n = 120
xx = np.linspace(x_range[0], x_range[1], grid_n)
yy = np.linspace(y_range[0], y_range[1], grid_n)
XX, YY = np.meshgrid(xx, yy)

grid_df = pd.DataFrame({x_col: XX.ravel(), y_col: YY.ravel()})
for c, v in fixed_values.items():
    grid_df[c] = v
grid_df = grid_df[selected_features]

tree_region = tree_clf.predict(grid_df).astype(float).reshape(XX.shape)

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x=xx,
        y=yy,
        z=tree_region,
        zmin=0,
        zmax=1,
        colorscale=[[0, "rgba(76,120,168,0.20)"], [1, "rgba(228,87,86,0.20)"]],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 영역",
    )
)

for stroke_value, label in STROKE_LABELS.items():
    subset = test_df[test_df["stroke"] == stroke_value]
    fig.add_trace(
        go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            mode="markers",
            name=label,
            marker=dict(color=STROKE_COLORS[label], size=6, opacity=0.75),
        )
    )

# 로지스틱 회귀의 0.5 경계선 계산하기
weights = dict(zip(selected_features, logreg.coef_[0]))
intercept = float(logreg.intercept_[0])
fixed_contribution = intercept + sum(weights[c] * v for c, v in fixed_values.items())

line_note = None
if abs(weights[y_col]) > 1e-12:
    line_x = np.array(x_range)
    line_y = -(weights[x_col] * line_x + fixed_contribution) / weights[y_col]
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="로지스틱 회귀 경계(0.5)",
            line=dict(color="black", dash="dash"),
        )
    )
    if (line_y.min() > y_range[1]) or (line_y.max() < y_range[0]):
        line_note = "로지스틱 회귀의 경계선이 이 그림의 세로 범위 밖에 있어 보이지 않습니다."
elif abs(weights[x_col]) > 1e-12:
    line_x_value = -fixed_contribution / weights[x_col]
    fig.add_trace(
        go.Scatter(
            x=[line_x_value, line_x_value],
            y=list(y_range),
            mode="lines",
            name="로지스틱 회귀 경계(0.5)",
            line=dict(color="black", dash="dash"),
        )
    )
    if line_x_value < x_range[0] or line_x_value > x_range[1]:
        line_note = "로지스틱 회귀의 경계선이 이 그림의 가로 범위 밖에 있어 보이지 않습니다."
else:
    line_note = "이 두 속성만으로는 로지스틱 회귀의 경계선을 그릴 수 없습니다."

fig.update_layout(
    title="테스트 데이터와 두 모델의 경계",
    xaxis_title=FEATURE_LABELS[x_col],
    yaxis_title=FEATURE_LABELS[y_col],
    xaxis=dict(range=list(x_range)),
    yaxis=dict(range=list(y_range)),
    legend_title="실제 뇌졸중 여부",
)

st.plotly_chart(fig, use_container_width=True)

if line_note:
    st.caption(line_note)

st.divider()

# ───────────────────────────  의사결정트리 그림  ───────────────────────────
st.subheader("④ 의사결정트리가 던진 질문 살펴보기")

dot_string, leaf_count, leaf_no_count, used_features = build_tree_dot(tree_clf, selected_features)
st.graphviz_chart(dot_string)

st.write(f"- 답을 내는 마디는 모두 **{leaf_count}칸**이고, 그중 **{leaf_no_count}칸**이 '아님'이라고 답합니다.")
for feature in selected_features:
    asked = "물었습니다" if feature in used_features else "묻지 않았습니다"
    st.write(f"- {FEATURE_LABELS[feature]}: 이 나무가 실제로 {asked}.")
