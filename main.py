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
st.write("서울의 연평균 기온 데이터를 이용해 연도별 기온 변화를 분석하고 미래 기온을 예측합니다.")

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

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

# 2025년까지만 사용
df = df[df["연도"] <= 2025].copy()

# 평균기온이 실제로 존재하는 관측일만 계산
yearly = (
    df.dropna(subset=["평균기온"])
      .groupby("연도")
      .agg(
          연평균기온=("평균기온", "mean"),
          관측일수=("평균기온", "count")
      )
      .reset_index()
)

# 관측일이 300일 이상인 해만 사용
yearly = yearly[yearly["관측일수"] >= 300].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)

# --------------------------------------------------
# 회귀 분석
# 독립변수 = 1908년부터 지난 연수
# --------------------------------------------------
yearly["지난연수"] = yearly["연도"] - 1908

X = yearly[["지난연수"]]
y = yearly["연평균기온"]

model = LinearRegression()
model.fit(X, y)

# 회귀선의 기울기와 절편
slope = model.coef_[0]
intercept = model.intercept_

# 예측값
yearly["회귀예측기온"] = model.predict(X)

# 상관계수
correlation = yearly["지난연수"].corr(yearly["연평균기온"])

# --------------------------------------------------
# 기본 정보
# --------------------------------------------------
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)

st.subheader("📊 회귀 분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("회귀에 사용한 연도 수", f"{data_count}년")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

with col4:
    st.metric("상관계수", f"{correlation:.3f}")

st.write(
    f"회귀선은 **{start_year}년부터 {end_year}년까지** "
    f"관측일이 300일 이상인 연도의 연평균기온을 사용해 계산했습니다."
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
predicted_temp = model.predict(selected_x)[0]

st.metric(
    f"{selected_year}년 예상 연평균기온",
    f"{predicted_temp:.2f} °C"
)

# --------------------------------------------------
# Plotly 산점도 + 회귀선
# --------------------------------------------------
st.subheader("📈 서울 연평균기온과 회귀선")

# 회귀선은 1900~2100년 전체 범위에 표시
prediction_years = np.arange(1900, 2101)
prediction_x = (prediction_years - 1908).reshape(-1, 1)
prediction_temps = model.predict(prediction_x)

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        text=[
            f"{year}년<br>평균기온: {temp:.2f}°C<br>관측일수: {days}일"
            for year, temp, days in zip(
                yearly["연도"],
                yearly["연평균기온"],
                yearly["관측일수"]
            )
        ],
        hovertemplate="%{text}<extra></extra>"
    )
)

# 회귀선
fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temps,
        mode="lines",
        name="회귀선",
        line=dict(width=3)
    )
)

# 선택한 연도의 예측점
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
        dtick=10
    ),
    hovermode="closest",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 회귀식
# --------------------------------------------------
st.subheader("📐 회귀식")

st.write(
    f"**연평균기온 = {intercept:.4f} + "
    f"({slope:.4f} × 지난 연수)**"
)

st.write(
    f"여기서 **지난 연수 = 연도 − 1908** 입니다."
)

# --------------------------------------------------
# 데이터 표
# --------------------------------------------------
with st.expander("📋 회귀에 사용된 연도별 데이터 보기"):
    display_df = yearly[
        ["연도", "연평균기온", "관측일수", "지난연수", "회귀예측기온"]
    ].copy()

    display_df["연평균기온"] = display_df["연평균기온"].round(2)
    display_df["회귀예측기온"] = display_df["회귀예측기온"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
