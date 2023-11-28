import os

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from neuralforecast.core import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MAE


# Patch TST를 사용한 학습 및 예측입니다.
# 관련 내용은 'https://arxiv.org/abs/2211.14730'에서 확인할 수 있습니다.

DATA_PATH = "../data" # 학습에 사용할 csv 파일이 저장된 폴더입니다.
TRAIN_FILE = "train_aug_std-25_window-9_2023_08_25_version_1_interpolate_use_all.csv" # 학습 및 예측에 사용할 파일입니다.


def df_to_timeseries(df):
    """
    데이터 프레임을 neuralforecast에서 예상하는 입력 형식으로 변환합니다.
    """
    date_cols = [col for col in df.columns if col.startswith("2")]
    input_df = pd.melt(df[["ID"] + date_cols], id_vars="ID", value_vars=date_cols) # 데이터 프레임을 ID와 날짜로 long format으로 변환합니다.
    input_df = input_df.rename(
        columns={"ID": "unique_id", "variable": "ds", "value": "y"} # neuralforecast에 맞게 열 이름을 변경합니다.
    )
    input_df["ds"] = pd.to_datetime(input_df["ds"]) # 날짜 데이터를 'object' 타입에서 'datetime' 형식으로 변경합니다.

    static_df = df.iloc[:, :6].copy() # 시간에 상관없는 카테고리 형태의 데이터를 라벨 인코딩을 위해 따로 저장합니다.
    static_df = static_df.drop("제품", axis=1)

    # 각 분류에 대해 라벨 인코딩을 정의하고 변환합니다.
    # 하지만 현재 사용하는 patchTST는 각 상품마다 따로 학습을 진행하므로 이 값을 사용하진 않습니다.
    main_encoder = LabelEncoder() 
    mid_encoder = LabelEncoder()
    sub_encoder = LabelEncoder()
    brand_encoder = LabelEncoder()

    static_df["대분류"] = main_encoder.fit_transform(static_df["대분류"])
    static_df["중분류"] = mid_encoder.fit_transform(static_df["중분류"])
    static_df["소분류"] = sub_encoder.fit_transform(static_df["소분류"])
    static_df["브랜드"] = brand_encoder.fit_transform(static_df["브랜드"])

    static_df = static_df.rename(columns={"ID": "unique_id"})

    # 변환한 데이터 프레임들을 반환합니다. 이중 날짜별로 변하는 값인 input_df만 사용합니다.
    return input_df, static_df


def train_model(df):
    """정의한 모델과 데이터 프레임으로 학습을 진행합니다."""
    nf = NeuralForecast(models=models, freq="D") # 모델을 사용해 NeuralForecast 클래스를 생성합니다.
    preds_df = nf.cross_validation(df=df, val_size=horizon) # 교차검증으로 모델을 학습합니다.
    return nf, preds_df


def long_to_wide(df, col):
    # 데이터 프레임을 long format에서 col에 있는 값들을 column으로 가지는 wide format으로 변환합니다.
    if "unique_id" in df.columns:
        temp = df[["unique_id", "ds"] + [col]]
    else:
        temp = df[["ds"] + [col]]
        temp = temp.reset_index()

    temp = temp.rename(columns={"unique_id": "ID"})
    temp["ds"] = temp["ds"].dt.date
    return pd.pivot(temp, columns="ds", index="ID", values=col)


if __name__ == "__main__":
    TRAIN_PATH = os.path.join(DATA_PATH, TRAIN_FILE)

    # 보간된 데이터 프레임인 경우 원본 데이터 길이만큼만 사용합니다. (2021년 데이터는 사용하지 않습니다)
    df = pd.read_csv(TRAIN_PATH)
    if df.shape[1] > 500:
        df = pd.concat([df.iloc[:, :6], df.iloc[:, 6 + 365 :]], axis=1)
    date_cols = [col for col in df.columns if col.startswith("2")]

    # 각 상품마다 최대 최솟값을 구해 minmax 정규화를 해줍ㄴ다.
    df_mins = df[date_cols].min(axis=1)
    df_maxs = df[date_cols].max(axis=1)
    df_denom = df_maxs - df_mins
    df_denom = df_denom.map(lambda x: x if x != 0 else 1)

    train_df = df.copy()
    train_df[date_cols] = df[date_cols].apply(lambda x: (x - df_mins) / df_denom)

    # 대분류마다 데이터 프레임을 나눠줍니다.
    train_mid_subs = {}
    for main_cat in train_df["대분류"].unique():
        train_mid_subs[main_cat] = train_df.query("대분류==@main_cat")

    # 대분류마다 나눠진 데이터프레임 각각 neuralforecast 입력에 맞게 변환합니다.
    train_mid_subs_ts = {}
    for mid_cat, subs in train_mid_subs.items():
        sub_df, _ = df_to_timeseries(subs)
        train_mid_subs_ts[mid_cat] = sub_df

    # 모델 파라미터를 정의합니다.
    horizon = 21
    params = dict(
        h=horizon,
        input_size=96,
        max_steps=5000,
        loss=MAE(),
        activation="relu",
        batch_size=4096,
        learning_rate=5e-7,
        early_stop_patience_steps=5,
    )

    # 모델을 정의합니다. PatchTST를 사용합니다.
    models = [
        PatchTST(**params),
    ]

    # 대분류마다 각각 학습을 진행하고 결과를 저장합니다.
    nfs = []
    preds = []
    for mid_cat, sub_ts in train_mid_subs_ts.items():
        nf, pred = train_model(sub_ts)
        nfs.append(nf)
        preds.append(pred)

    # 대분류마다 따로 예측을 진행하고 최종 결과를 합쳐줍니다.
    pred_fin = []
    for nf, (_, d) in zip(nfs, train_mid_subs_ts.items()):
        pred_fin.append(nf.predict(d))

    preds_fin = pd.concat(pred_fin, axis=0)

    # 합친 결과를 제출 형식에 맞게 변환하고, 정규화된 값들을 다시 복원해줍니다.
    pred_fin_wide = (
        long_to_wide(preds_fin, "PatchTST")
        .apply(lambda x: x * df_denom + df_mins)
        .clip(0)
        .round()
        .applymap(lambda x: int(x))
        .reset_index()
    )

    # 최종 결과를 저장합니다.
    result_name = (
        f"../submissions/PatchTST_main_indep_{TRAIN_FILE.split('.')[0]}.csv"
    )
    pred_fin_wide.to_csv(result_name, index=False)
    print(f"File saved at ../submissions/PatchTST_main_indep_{TRAIN_FILE.split('.')[0]}")

    del nfs
