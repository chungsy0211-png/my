import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균 기온 데이터를 이용해 기온 변화 추세를 분석하고 "
    "회귀선을 이용해 미래 기온을 예측합니다."
)

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

try:
    df = pd.read_csv(URL, encoding="utf-8-sig")
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# --------------------------------------------------
# 데이터 전처리
# --------------------------------------------------
df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["연도"] = df["날짜"].dt.year
df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

# 2025년까지의 데이터만 사용
df = df[df["연도"] <= 2025].copy()

# 연도별 평균기온과 관측일수 계산
yearly = (
    df.dropna(subset=["평균기온"])
    .groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일수가 300일 미만인 해 제외
yearly = yearly[yearly["관측일수"] >= 300].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)

if len(yearly) < 2:
    st.error("회귀분석을 할 수 있는 데이터가 충분하지 않습니다.")
    st.stop()

# --------------------------------------------------
# 전체 기간 회귀분석
# 독립변수 = 1908년부터 지난 연수
# --------------------------------------------------
yearly["지난연수"] = yearly["연도"] - 1908

X_all = yearly[["지난연수"]]
y_all = yearly["연평균기온"]

model_all = LinearRegression()
model_all.fit(X_all, y_all)

slope_all = model_all.coef_[0]
intercept_all = model_all.intercept_

# 1년에 몇 도 변화하는가
# → 100년에 몇 도 변화하는가
slope_all_100 = slope_all * 100

# 상관계수
correlation = yearly["지난연수"].corr(yearly["연평균기온"])

# --------------------------------------------------
# 최근 20년 회귀분석
# --------------------------------------------------
recent_end_year = int(yearly["연도"].max())
recent_start_year = recent_end_year - 19

recent = yearly[
    (yearly["연도"] >= recent_start_year) &
    (yearly["연도"] <= recent_end_year)
].copy()

if len(recent) >= 2:

    recent["지난연수"] = recent["연도"] - 1908

    X_recent = recent[["지난연수"]]
    y_recent = recent["연평균기온"]

    model_recent = LinearRegression()
    model_recent.fit(X_recent, y_recent)

    slope_recent = model_recent.coef_[0]
    intercept_recent = model_recent.intercept_

    slope_recent_100 = slope_recent * 100

else:
    model_recent = None
    slope_recent = np.nan
    slope_recent_100 = np.nan
    intercept_recent = np.nan

# --------------------------------------------------
# 기간 정보
# --------------------------------------------------
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)

# --------------------------------------------------
# 회귀 기울기 크게 표시
# --------------------------------------------------
st.subheader("🌡️ 100년에 몇 도 오르는가?")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 기간",
        f"{slope_all_100:+.2f} °C / 100년"
    )
    st.caption(
        f"{start_year}~{end_year}년 데이터를 이용한 회귀선"
    )

with col2:
    if not np.isnan(slope_recent_100):
        st.metric(
            "최근 20년",
            f"{slope_recent_100:+.2f} °C / 100년"
        )
        st.caption(
            f"{recent_start_year}~{recent_end_year}년 데이터를 이용한 회귀선"
        )
    else:
        st.metric("최근 20년", "계산 불가")

st.info(
    "기울기는 연평균기온의 변화량을 100년 기준으로 환산한 값입니다. "
    "예를 들어 +1.50 °C / 100년은 회귀 추세상 100년에 약 1.50°C 상승한다는 뜻입니다."
)

# --------------------------------------------------
# 회귀 분석 정보
# --------------------------------------------------
st.subheader("📊 분석에 사용된 데이터")

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric("회귀에 사용한 연도 수", f"{data_count}년")

with info2:
    st.metric("시작 연도", f"{start_year}년")

with info3:
    st.metric("끝 연도", f"{end_year}년")

with info4:
    st.metric("상관계수", f"{correlation:.3f}")

st.write(
    f"전체 기간 회귀선은 **{start_year}년부터 {end_year}년까지** "
    f"관측일수가 300일 이상인 연도의 연평균기온을 사용했습니다."
)

# --------------------------------------------------
# 연도 슬라이더
# --------------------------------------------------
st.subheader("🔮 연도별 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

selected_x = np.array([[selected_year - 1908]])
predicted_temp = model_all.predict(selected_x)[0]

st.metric(
    f"{selected_year}년 예상 연평균기온",
    f"{predicted_temp:.2f} °C"
)

# --------------------------------------------------
# Plotly 산점도 + 전체 회귀선
# --------------------------------------------------
st.subheader("📈 서울 연평균기온과 전체 기간 회귀선")

# 1900~2100년 회귀선
prediction_years = np.arange(1900, 2101)

prediction_x = (
    prediction_years - 1908
).reshape(-1, 1)

prediction_temps = model_all.predict(prediction_x)

fig = go.Figure()

# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        text=[
            f"{year}년<br>"
            f"평균기온: {temp:.2f}°C<br>"
            f"관측일수: {days}일"
            for year, temp, days in zip(
                yearly["연도"],
                yearly["연평균기온"],
                yearly["관측일수"]
            )
        ],
        hovertemplate="%{text}<extra></extra>"
    )
)

# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temps,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(width=3)
    )
)

# 선택한 연도의 예측값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=12),
        hovertemplate=(
            f"{selected_year}년<br>"
            f"예상 연평균기온: {predicted_temp:.2f}°C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(
        range=[1900, 2100],
        dtick=10
    ),
    hovermode="closest",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 회귀식
# --------------------------------------------------
st.subheader("📐 전체 기간 회귀식")

st.write(
    f"연평균기온 = {intercept_all:.4f} + "
    f"({slope_all:.4f} × 지난 연수)"
)

st.write(
    f"지난 연수 = 연도 − 1908"
)

# --------------------------------------------------
# 전체 기간 vs 최근 20년 비교
# --------------------------------------------------
st.subheader("📊 전체 기간과 최근 20년의 기울기 비교")

comparison = pd.DataFrame({
    "기간": [
        f"{start_year}~{end_year}년",
        f"{recent_start_year}~{recent_end_year}년"
    ],
    "연도 수": [
        len(yearly),
        len(recent)
    ],
    "100년당 기온 변화": [
        slope_all_100,
        slope_recent_100
    ]
})

comparison["100년당 기온 변화"] = (
    comparison["100년당 기온 변화"].round(2)
)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 원본 데이터 보기
# --------------------------------------------------
with st.expander("📋 회귀에 사용된 연도별 데이터 보기"):

    display_df = yearly[
        [
            "연도",
            "연평균기온",
            "관측일수",
            "지난연수"
        ]
    ].copy()

    display_df["연평균기온"] = (
        display_df["연평균기온"].round(2)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
