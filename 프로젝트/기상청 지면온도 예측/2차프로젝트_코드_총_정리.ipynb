{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "라이브러리"
      ],
      "metadata": {
        "id": "YFRJk5OOGlqY"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EtZesFSE_UqC"
      },
      "outputs": [],
      "source": [
        "!pip install xgboost\n",
        "!pip install catboost\n",
        "!pip install optuna\n",
        "!pip install pycaret"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "import xgboost as xgb\n",
        "from catboost import CatBoostRegressor\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.ensemble import ExtraTreesRegressor\n",
        "from lightgbm import LGBMRegressor\n",
        "from sklearn.metrics import mean_absolute_error\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.model_selection import GridSearchCV, KFold\n",
        "from sklearn.preprocessing import PolynomialFeatures\n",
        "from sklearn.ensemble import VotingRegressor\n",
        "import optuna\n",
        "from optuna import Trial, visualization\n",
        "import time\n",
        "from pycaret.regression import *"
      ],
      "metadata": {
        "id": "UgkSCBbW_YKW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "전처리(사계절분리)"
      ],
      "metadata": {
        "id": "xMH-phBWGcdR"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 원본데이터\n",
        "train = pd.read_csv('/content/drive/MyDrive/surface_tp_train.csv')\n",
        "test=pd.read_csv('/content/drive/MyDrive/surface_tp_test.csv')\n",
        "train = train.iloc[:,1:]\n",
        "test = test.iloc[:,1:]\n",
        "\n",
        "# 컬럼명 정리\n",
        "col = ['stn','year','mmddhh','ta','td','hm','ws','rn','re','ww','ts','si','ss','sn']\n",
        "col2 = ['stn','year','mmddhh','ta','td','hm','ws','rn','re','ww','si','ss','sn']\n",
        "train.columns = col\n",
        "test.columns = col2"
      ],
      "metadata": {
        "id": "pEHQKGvWAOEU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 날짜에 따른 사계절 분리\n",
        "\n",
        "spring_train = pd.concat(\n",
        "    [\n",
        "        train[train[\"surface_tp_train.mmddhh\"].astype(str).str.startswith(\"2\")],\n",
        "        train[train[\"surface_tp_train.mmddhh\"].astype(str).str.startswith(\"3\")],\n",
        "        train[train[\"surface_tp_train.mmddhh\"].astype(str).str.startswith(\"4\")],\n",
        "    ]\n",
        ")\n",
        "summer_train = pd.concat(\n",
        "    [\n",
        "        train[train[\"surface_tp_train.mmddhh\"].astype(str).str.startswith(\"5\")],\n",
        "        train[train[\"surface_tp_train.mmddhh\"].astype(str).str.startswith(\"6\")],\n",
        "        train[train[\"surface_tp_train.mmddhh\"].astype(str).str.startswith(\"7\")],\n",
        "    ]\n",
        ")\n",
        "fall_train = pd.concat(\n",
        "    [\n",
        "        train[train[\"surface_tp_train.mmddhh\"].astype(str).str.startswith(\"8\")],\n",
        "        train[train[\"surface_tp_train.mmddhh\"].astype(str).str.startswith(\"9\")],\n",
        "        train[\n",
        "            train[\"surface_tp_train.mmddhh\"].astype(str).str.startswith(\"10\")\n",
        "            & (train[\"surface_tp_train.mmddhh\"].astype(str).str.len() == 6)\n",
        "        ],\n",
        "    ]\n",
        ")\n",
        "winter_train = pd.concat(\n",
        "    [\n",
        "        train[\n",
        "            train[\"surface_tp_train.mmddhh\"].astype(str).str.startswith(\"11\")\n",
        "            & (train[\"surface_tp_train.mmddhh\"].astype(str).str.len() == 6)\n",
        "        ],\n",
        "        train[\n",
        "            train[\"surface_tp_train.mmddhh\"].astype(str).str.startswith(\"12\")\n",
        "            & (train[\"surface_tp_train.mmddhh\"].astype(str).str.len() == 6)\n",
        "        ],\n",
        "        train[\n",
        "            train[\"surface_tp_train.mmddhh\"].astype(str).str.startswith(\"1\")\n",
        "            & (train[\"surface_tp_train.mmddhh\"].astype(str).str.len() == 5)\n",
        "        ],\n",
        "    ]\n",
        ")\n",
        "\n",
        "spring_test = pd.concat(\n",
        "    [\n",
        "        test[test[\"surface_tp_test.mmddhh\"].astype(str).str.startswith(\"2\")],\n",
        "        test[test[\"surface_tp_test.mmddhh\"].astype(str).str.startswith(\"3\")],\n",
        "        test[test[\"surface_tp_test.mmddhh\"].astype(str).str.startswith(\"4\")],\n",
        "    ]\n",
        ")\n",
        "summer_test = pd.concat(\n",
        "    [\n",
        "        test[test[\"surface_tp_test.mmddhh\"].astype(str).str.startswith(\"5\")],\n",
        "        test[test[\"surface_tp_test.mmddhh\"].astype(str).str.startswith(\"6\")],\n",
        "        test[test[\"surface_tp_test.mmddhh\"].astype(str).str.startswith(\"7\")],\n",
        "    ]\n",
        ")\n",
        "fall_test = pd.concat(\n",
        "    [\n",
        "        test[test[\"surface_tp_test.mmddhh\"].astype(str).str.startswith(\"8\")],\n",
        "        test[test[\"surface_tp_test.mmddhh\"].astype(str).str.startswith(\"9\")],\n",
        "        test[\n",
        "            test[\"surface_tp_test.mmddhh\"].astype(str).str.startswith(\"10\")\n",
        "            & (test[\"surface_tp_test.mmddhh\"].astype(str).str.len() == 6)\n",
        "        ],\n",
        "    ]\n",
        ")\n",
        "winter_test = pd.concat(\n",
        "    [\n",
        "        test[\n",
        "            test[\"surface_tp_test.mmddhh\"].astype(str).str.startswith(\"11\")\n",
        "            & (test[\"surface_tp_test.mmddhh\"].astype(str).str.len() == 6)\n",
        "        ],\n",
        "        test[\n",
        "            test[\"surface_tp_test.mmddhh\"].astype(str).str.startswith(\"12\")\n",
        "            & (test[\"surface_tp_test.mmddhh\"].astype(str).str.len() == 6)\n",
        "        ],\n",
        "        test[\n",
        "            test[\"surface_tp_test.mmddhh\"].astype(str).str.startswith(\"1\")\n",
        "            & (test[\"surface_tp_test.mmddhh\"].astype(str).str.len() == 5)\n",
        "        ],\n",
        "    ]\n",
        ")"
      ],
      "metadata": {
        "id": "nKKePXC6A6M0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "전처리(결측치 대체)"
      ],
      "metadata": {
        "id": "aUy7nnDgGqie"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 결측치 유무 확인 코드\n",
        "\n",
        "missing_values_2 = [-99, -99.9, -999, -99.90]\n",
        "\n",
        "for column in train.columns:\n",
        "    # 각 컬럼에서 결측치가 있는지 확인\n",
        "    missing_values = train[column].isin(missing_values_2).sum()\n",
        "\n",
        "    if missing_values > 0:\n",
        "        print(f\"{column}: {missing_values} missing values\")\n",
        "    else:\n",
        "        print(f\"{column}: No missing values\")"
      ],
      "metadata": {
        "id": "4YCaiTnCA1V7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 평균,중위수,보간법으로 결측치를 대체하는 함수 V1(Q),V2(Q),V3(Q)"
      ],
      "metadata": {
        "id": "BeuUsHowB3Ko"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def V1(Q):\n",
        "  df2 = df.replace([-99, -99.9, -999], np.nan)\n",
        "  df2[Q][df2[df2[Q].isna()].index] = np.nanmean(df2[Q]) # 컬럼Q nan값 평균값 대체\n",
        "  df2 = df2.fillna(0)   # 나머지 결측치 0으로 변경\n",
        "\n",
        "  # 필요없는 컬럼들 drop\n",
        "  to_drops = [\n",
        "      \"year\"]\n",
        "  # X, y로 분리\n",
        "  X = df2.drop([\"ts\"], axis='columns')\n",
        "  y = df2[\"ts\"]\n",
        "\n",
        "  # object => LabelEncoder, !object => StandardScaler\n",
        "  object_cols = []\n",
        "  non_object_cols = []\n",
        "  for col in X.columns:\n",
        "      if X[col].dtype == 'object':\n",
        "          object_cols.append(col)\n",
        "      else:\n",
        "          non_object_cols.append(col)\n",
        "\n",
        "  # LabelEncoder 적용\n",
        "  le = LabelEncoder()\n",
        "  for col in object_cols:\n",
        "      X[col] = le.fit_transform(X[col])\n",
        "\n",
        "  # StandardScaler 적용\n",
        "  ss = StandardScaler()\n",
        "  for col in non_object_cols:\n",
        "      X[col] = ss.fit_transform(X[[col]])\n",
        "\n",
        "  # lightGBM + 교차검증 + MAE(평가지표)\n",
        "  # LightGBM 모델 생성\n",
        "  model = LGBMRegressor(random_state=42)\n",
        "\n",
        "  # 교차 검증 설정\n",
        "  kf = KFold(n_splits=5, random_state=42, shuffle=True)\n",
        "  scoring = make_scorer(mean_absolute_error, greater_is_better=True)\n",
        "  cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)\n",
        "\n",
        "  # 전체 교차 검증 결과 출력\n",
        "  print(f\"평균대체{Q}\")\n",
        "  print(f\"전체 MAE: {cv_scores}\")\n",
        "  print(f\"평균 MAE: {np.mean(cv_scores)}\")\n",
        "  print(f\"MAE 표준편차: {np.std(cv_scores)}\")"
      ],
      "metadata": {
        "id": "mEcOaLzqA5Xn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def V2(Q):\n",
        "  df2 = df.replace([-99, -99.9, -999], np.nan)\n",
        "  df2[Q][df2[df2[Q].isna()].index] = np.nanmedian(df2[Q])    # ta컬럼 nan값 중위수로 대체\n",
        "  df2 = df2.fillna(0)   # 나머지 결측치 0으로 변경\n",
        "\n",
        "  # @@@데이터 로드@@@ \"eval(input(...))\"을 입력 변수로 수정해주세요\n",
        "  dfm = df2\n",
        "\n",
        "  # 필요없는 컬럼들 drop\n",
        "  to_drops = [\n",
        "      \"year\",\n",
        "  ]\n",
        "  for to_drop in to_drops:\n",
        "      if to_drop in dfm.columns:\n",
        "          dfm = dfm.drop(to_drop, axis='columns')\n",
        "\n",
        "  # X, y로 분리\n",
        "  X = dfm.drop([\"ts\"], axis='columns')\n",
        "  y = dfm[\"ts\"]\n",
        "\n",
        "  # object => LabelEncoder, !object => StandardScaler\n",
        "  object_cols = []\n",
        "  non_object_cols = []\n",
        "  for col in X.columns:\n",
        "      if X[col].dtype == 'object':\n",
        "          object_cols.append(col)\n",
        "      else:\n",
        "          non_object_cols.append(col)\n",
        "\n",
        "  # LabelEncoder 적용\n",
        "  le = LabelEncoder()\n",
        "  for col in object_cols:\n",
        "      X[col] = le.fit_transform(X[col])\n",
        "\n",
        "  # StandardScaler 적용\n",
        "  ss = StandardScaler()\n",
        "  for col in non_object_cols:\n",
        "      X[col] = ss.fit_transform(X[[col]])\n",
        "\n",
        "  # lightGBM + 교차검증 + MAE(평가지표)\n",
        "  # LightGBM 모델 생성\n",
        "  model = LGBMRegressor(random_state=42)\n",
        "\n",
        "  # 교차 검증 설정\n",
        "  kf = KFold(n_splits=5, random_state=42, shuffle=True)\n",
        "  scoring = make_scorer(mean_absolute_error, greater_is_better=True)\n",
        "  cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)\n",
        "\n",
        "  # 전체 교차 검증 결과 출력\n",
        "  print(f\"중위수대체{Q}\")\n",
        "  print(f\"전체 MAE: {cv_scores}\")\n",
        "  print(f\"평균 MAE: {np.mean(cv_scores)}\")\n",
        "  print(f\"MAE 표준편차: {np.std(cv_scores)}\")"
      ],
      "metadata": {
        "id": "sLE2dnveBpFq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def V3(Q):\n",
        "  df2 = df.replace([-99, -99.9, -999], np.nan)\n",
        "  df2[Q] = df2[Q].interpolate(method='linear', limit_direction='forward', x=df2[df2.corr()[Q].drop(Q).idxmax()])  # ta컬럼 nan값 보간법으로 대체.  df2.corr()['ta'].drop('ta').idxmax(): ta컬럼과 상관관계가 가장 높은 컬럼\n",
        "  df2=df2.fillna(0)   # 나머지 결측치 0으로 변경\n",
        "\n",
        "  # @@@데이터 로드@@@ \"eval(input(...))\"을 입력 변수로 수정해주세요\n",
        "  dfm = df2\n",
        "\n",
        "  # 필요없는 컬럼들 drop\n",
        "  to_drops = [\n",
        "      \"year\",\n",
        "  ]\n",
        "  for to_drop in to_drops:\n",
        "      if to_drop in dfm.columns:\n",
        "          dfm = dfm.drop(to_drop, axis='columns')\n",
        "\n",
        "  # X, y로 분리\n",
        "  X = dfm.drop([\"ts\"], axis='columns')\n",
        "  y = dfm[\"ts\"]\n",
        "\n",
        "  # object => LabelEncoder, !object => StandardScaler\n",
        "  object_cols = []\n",
        "  non_object_cols = []\n",
        "  for col in X.columns:\n",
        "      if X[col].dtype == 'object':\n",
        "          object_cols.append(col)\n",
        "      else:\n",
        "          non_object_cols.append(col)\n",
        "\n",
        "  # LabelEncoder 적용\n",
        "  le = LabelEncoder()\n",
        "  for col in object_cols:\n",
        "      X[col] = le.fit_transform(X[col])\n",
        "\n",
        "  # StandardScaler 적용\n",
        "  ss = StandardScaler()\n",
        "  for col in non_object_cols:\n",
        "      X[col] = ss.fit_transform(X[[col]])\n",
        "\n",
        "  # lightGBM + 교차검증 + MAE(평가지표)\n",
        "  # LightGBM 모델 생성\n",
        "  model = LGBMRegressor(random_state=42)\n",
        "\n",
        "  # 교차 검증 설정\n",
        "  kf = KFold(n_splits=5, random_state=42, shuffle=True)\n",
        "  scoring = make_scorer(mean_absolute_error, greater_is_better=True)\n",
        "  cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)\n",
        "\n",
        "  # 전체 교차 검증 결과 출력\n",
        "  print(f\"보간법{Q}\")\n",
        "  print(f\"전체 MAE: {cv_scores}\")\n",
        "  print(f\"평균 MAE: {np.mean(cv_scores)}\")\n",
        "  print(f\"MAE 표준편차: {np.std(cv_scores)}\")"
      ],
      "metadata": {
        "id": "31FEqxzXB6pT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 각 컬럼별 결측치를 대체하는 최상의 방법이 무엇인지 확인하는 for문\n",
        "\n",
        "col_list = ['ta','td','hm','ws','rn','re','ts','si','ss','sn']\n",
        "for i in col_list:\n",
        "  V1(i)\n",
        "  V2(i)\n",
        "  V3(i)\n",
        "  print('-'*50)"
      ],
      "metadata": {
        "id": "A1p-6J94B8Hx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 피처 엔지니어링\n",
        "df2['mm'] = df2['mmddhh'] // 10000           # 'mmddhh' 컬럼에서 'mm', 'dd', 'hh' 분리\n",
        "df2['dd'] = (df2['mmddhh'] // 100) % 100\n",
        "df2['hh'] = df2['mmddhh'] % 100"
      ],
      "metadata": {
        "id": "Gm_XNl0sCDUw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# spring_train 만들기(다른 계절도 동일)\n",
        "\n",
        "# spring_train\n",
        "df2 = df.replace([-99, -99.9, -999], np.nan)\n",
        "df2['ta'][df2[df2['ta'].isna()].index] = np.nanmedian(df2['ta'])\n",
        "df2['td'][df2[df2['td'].isna()].index] = np.nanmean(df2['td'])\n",
        "df2['hm'] = df2['hm'].interpolate(method='linear', limit_direction='forward', x=df2[df2.corr()['hm'].drop('hm').idxmax()])\n",
        "df2['ws'][df2[df2['ws'].isna()].index] = np.nanmedian(df2['ws'])\n",
        "df2['rn'] = df2['rn'].interpolate(method='linear', limit_direction='forward', x=df2[df2.corr()['rn'].drop('rn').idxmax()])\n",
        "df2['re'][df2[df2['re'].isna()].index] = np.nanmean(df2['re'])\n",
        "df2['ts'][df2[df2['ts'].isna()].index] = np.nanmean(df2['ts'])\n",
        "df2['si'] = df2['si'].interpolate(method='linear', limit_direction='forward', x=df2[df2.corr()['si'].drop('si').idxmax()])\n",
        "df2['ss'] = df2['ss'].interpolate(method='linear', limit_direction='forward', x=df2[df2.corr()['ss'].drop('ss').idxmax()])\n",
        "df2['sn'] = df2['sn'].interpolate(method='linear', limit_direction='forward', x=df2[df2.corr()['sn'].drop('sn').idxmax()])\n",
        "df2=df2.fillna(0)\n",
        "\n",
        "df2['mm'] = df2['mmddhh'] // 10000           # 'mmddhh' 컬럼에서 'mm', 'dd', 'hh' 분리\n",
        "df2['dd'] = (df2['mmddhh'] // 100) % 100\n",
        "df2['hh'] = df2['mmddhh'] % 100"
      ],
      "metadata": {
        "id": "liX_soyXCGeW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df2.to_csv('Feature Engineering(spring_train).csv',index=False)"
      ],
      "metadata": {
        "id": "mMv9Tgy4CKXP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# spring_test 만들기 (다른 계절도 동일)"
      ],
      "metadata": {
        "id": "nBJ8lga6CM8l"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# spring_train의 각 컬럼별 선택한 방법을 spring_test의 각 컬럼에도 동일하게 적용시켜주는 함수\n",
        "\n",
        "def test(mean,median,bogan): # 변수는 컬럼 리스트로 넣을 것\n",
        "  '''train에서 평균법이 최상의방법이었던 컬럼명은 첫번째 리스트로, 중위수법이 최상이었던 컬럼명은 두번째 리스트로, 보간법이 최상이었던 컬럼명은 세번째 리스트로\n",
        "  예시::test(['ta','td','sn'],[],['hm','ws','rn','re','si','ss']  '''\n",
        "  df = df_test.copy()\n",
        "  dft = df_train.copy()\n",
        "  df2 = df.replace([-99, -99.9, -999], np.nan)  # 결측치값들 nan으로 변환\n",
        "  dft2 = dft.replace([-99, -99.9, -999], np.nan)\n",
        "\n",
        "  for i in mean: # 평균법\n",
        "    df2[i][df2[df2[i].isna()].index] = np.nanmean(dft2[i])\n",
        "\n",
        "  for i in median: # 중위수\n",
        "    df2[i][df2[df2[i].isna()].index] = np.nanmedian(dft2[i])\n",
        "\n",
        "  list_bo1 = bogan # 보간법\n",
        "  list_bo2 = [i+'_test' for i in list_bo1]\n",
        "  for i, j in zip(list_bo2,list_bo1):\n",
        "    dft2[i] = df2[j]\n",
        "    df2[j] = dft2[i].interpolate(method='linear', limit_direction='forward', x=dft2[dft2.corr()[i].drop(i).idxmax()])\n",
        "    dft2 = dft2.drop(i,axis=1)\n",
        "\n",
        "  df2 = df2.fillna(0) # 나머지 결측치 0으로 변경\n",
        "\n",
        "  df2['mm'] = df2['mmddhh'] // 10000           # 'mmddhh' 컬럼에서 'mm', 'dd', 'hh' 분리\n",
        "  df2['dd'] = (df2['mmddhh'] // 100) % 100\n",
        "  df2['hh'] = df2['mmddhh'] % 100\n",
        "  return df2"
      ],
      "metadata": {
        "id": "AD6Qds_HCa8g"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df2=test(['td','re'],['ta','ws'],['hm','rn','si','ss','sn'])#봄\n",
        "df2.to_csv('Feature Engineering(spring_test).csv',index=False)"
      ],
      "metadata": {
        "id": "JR9Kc0BQCtT3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "머신러닝 모델탐색"
      ],
      "metadata": {
        "id": "vgbHxQ_lG2Bd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 모델탐색"
      ],
      "metadata": {
        "id": "7HC0M9lwCyMY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# AutoML로 적당한 모델 탐색\n",
        "setup(data=X, target=y)\n",
        "best_model = compare_models(sort='MAE')"
      ],
      "metadata": {
        "id": "yZJ8HbZADAdq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 상위 5개 모델 직접 확인"
      ],
      "metadata": {
        "id": "ozsbHJG-DMui"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_spring_train = pd.read_csv('/content/Feature Engineering(spring_train).csv')\n",
        "df_spring_test = pd.read_csv('/content/Feature Engineering(spring_test).csv')"
      ],
      "metadata": {
        "id": "FCOL31EHDVOT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "## pre 데이터 생성\n",
        "pref = df_spring_train.copy()\n",
        "preg = df_spring_test.copy()\n",
        "\n",
        "pref['stn'] = pref['stn'].astype(str)\n",
        "\n",
        "# 1. 필요없는 컬럼들 drop\n",
        "to_drops = [\n",
        "    \"year\",\n",
        "]\n",
        "for to_drop in to_drops:\n",
        "    if to_drop in pref.columns:\n",
        "        pref = pref.drop(to_drop, axis=\"columns\")\n",
        "    if to_drop in preg.columns:\n",
        "        preg = preg.drop(to_drop, axis=\"columns\")\n",
        "\n",
        "y = pref['ts']\n",
        "pref = pref.drop('ts',axis=1)\n",
        "\n",
        "# 2. 라벨 인코더 미리 적용, 정규화도 미리 적용하고, 변화된 부분만 다시 적용\n",
        "object_cols = []\n",
        "non_object_cols = []\n",
        "for col in pref.columns:\n",
        "    if pref[col].dtype == \"object\":\n",
        "        object_cols.append(col)\n",
        "    else:\n",
        "        non_object_cols.append(col)\n",
        "\n",
        "# LabelEncoder 적용\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "for col in object_cols:\n",
        "    le = LabelEncoder()\n",
        "    le = le.fit(pref[col])\n",
        "    pref[col] = le.transform(pref[col])\n",
        "\n",
        "    for label in np.unique(preg[col]):\n",
        "        if label not in le.classes_:\n",
        "            le.classes_ = np.append(le.classes_, label)\n",
        "    preg[col] = le.transform(preg[col])\n",
        "\n",
        "# StandardScaler 적용\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "for col in non_object_cols:\n",
        "    ss = StandardScaler()\n",
        "    pref[col] = ss.fit_transform(pref[[col]])\n",
        "\n",
        "    if col in preg.columns:\n",
        "        preg[col] = ss.transform(preg[[col]])\n",
        "X = pref"
      ],
      "metadata": {
        "id": "Q7bO0Cf4DhjW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train,x_test,y_train,y_test =train_test_split(X,y,random_state=42)"
      ],
      "metadata": {
        "id": "my-hTZ3pDorf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#ExtraTreesRegressor\n",
        "model = ExtraTreesRegressor(n_estimators=100, max_features=\"auto\", random_state=42)\n",
        "model.fit(x_train,y_train)\n",
        "print(model.score(x_train,y_train),model.score(x_test,y_test))\n",
        "mean_absolute_error(y_test,model.predict(x_test))"
      ],
      "metadata": {
        "id": "pscrhPm1DrhG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# CatBoostRegressor\n",
        "model = CatBoostRegressor(random_state=42)\n",
        "model.fit(x_train,y_train)\n",
        "print(model.score(x_train,y_train),model.score(x_test,y_test))\n",
        "mean_absolute_error(y_test,model.predict(x_test))"
      ],
      "metadata": {
        "id": "cnJTjKf7EG4_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# xgboost\n",
        "model = xgb.XGBRegressor(random_state=42)\n",
        "model.fit(x_train,y_train)\n",
        "print(model.score(x_train,y_train),model.score(x_test,y_test))\n",
        "mean_absolute_error(y_test,model.predict(x_test))"
      ],
      "metadata": {
        "id": "Lmz06QiHEJah"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# LGBMRegressor\n",
        "model = LGBMRegressor(random_state=42)\n",
        "model.fit(x_train,y_train)\n",
        "print(model.score(x_train,y_train),model.score(x_test,y_test))\n",
        "mean_absolute_error(y_test,model.predict(x_test))"
      ],
      "metadata": {
        "id": "1jeqMJgYEMXD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#RandomForestRegressor\n",
        "model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)\n",
        "model.fit(x_train,y_train)\n",
        "print(model.score(x_train,y_train),model.score(x_test,y_test))\n",
        "mean_absolute_error(y_test,model.predict(x_test))"
      ],
      "metadata": {
        "id": "3tVXtR3YEOYp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# ExtraTreesRegressor,CatBoostRegressor,XGBRegressor로 앙상블 결정.\n",
        "# ExtraTreesRegressor,CatBoostRegressor(2개) vs ExtraTreesRegressor,CatBoostRegressor,XGBRegressor(3개)"
      ],
      "metadata": {
        "id": "1gIg_Tx0EUj6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 2개 모델 앙상블\n",
        "\n",
        "# CatBoostRegressor 모델 초기화\n",
        "catboost_model = CatBoostRegressor(random_state=42)\n",
        "\n",
        "# ExtraTreesRegressor 모델 초기화\n",
        "extra_trees_model = ExtraTreesRegressor(random_state=42)\n",
        "\n",
        "# VotingRegressor를 사용하여 앙상블 모델 초기화\n",
        "ensemble_model = VotingRegressor([('catboost', catboost_model), ('extra_trees', extra_trees_model)])\n",
        "\n",
        "ensemble_model.fit(x_train, y_train)\n",
        "print(ensemble_model.score(x_train,y_train),ensemble_model.score(x_test,y_test))\n",
        "mean_absolute_error(y_test,ensemble_model.predict(x_test))"
      ],
      "metadata": {
        "id": "SVUqfsuzEh7p"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 3개 모델 앙상블\n",
        "\n",
        "# CatBoostRegressor 모델 초기화\n",
        "catboost_model = CatBoostRegressor(random_state=42)\n",
        "\n",
        "# XGBoost 모델 초기화\n",
        "xgboost_model = xgb.XGBRegressor(random_state=42)\n",
        "\n",
        "# ExtraTreesRegressor 모델 초기화\n",
        "extra_trees_model = ExtraTreesRegressor(random_state=42)\n",
        "\n",
        "# VotingRegressor를 사용하여 앙상블 모델 초기화\n",
        "ensemble_model = VotingRegressor([('catboost', catboost_model), ('xgboost', xgboost_model), ('extra_trees', extra_trees_model)])\n",
        "\n",
        "ensemble_model.fit(x_train, y_train)\n",
        "print(ensemble_model.score(x_train,y_train),ensemble_model.score(x_test,y_test))\n",
        "mean_absolute_error(y_test,ensemble_model.predict(x_test))"
      ],
      "metadata": {
        "id": "EdRY3pJkEnUL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 2개 모델 앙상블 하는 것으로 결정"
      ],
      "metadata": {
        "id": "u7k4XwIsE8py"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# PolynomialFeatures로 특성을 늘리면 사계절 모두 좋은 결과를 얻는다는 것 확인"
      ],
      "metadata": {
        "id": "gNGorq9EFHRP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pf = PolynomialFeatures(degree=2)\n",
        "X_poly = pf.fit_transform(X)\n",
        "x_train,x_test,y_train,y_test=train_test_split(X_poly,y,random_state=42)\n",
        "# CatBoostRegressor 모델 초기화\n",
        "catboost_model = CatBoostRegressor(random_state=42)\n",
        "\n",
        "# ExtraTreesRegressor 모델 초기화\n",
        "extra_trees_model = ExtraTreesRegressor(random_state=42)\n",
        "\n",
        "# VotingRegressor를 사용하여 앙상블 모델 초기화\n",
        "ensemble_model = VotingRegressor([('catboost', catboost_model), ('extra_trees', extra_trees_model)])\n",
        "\n",
        "ensemble_model.fit(x_train, y_train)\n",
        "print(ensemble_model.score(x_train,y_train),ensemble_model.score(x_test,y_test))\n",
        "mean_absolute_error(y_test,ensemble_model.predict(x_test))"
      ],
      "metadata": {
        "id": "AezPVQw9E-t3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# optuna 파라메터튜닝 + LGBM 모델"
      ],
      "metadata": {
        "id": "TmjklkI4FOL4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pf = PolynomialFeatures(degree=2)\n",
        "X_poly = pf.fit_transform(X)\n",
        "x_train, x_test, y_train, y_test = train_test_split(X_poly, y, random_state=42)"
      ],
      "metadata": {
        "id": "XkfOtwGmFdlv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Objective 함수 정의\n",
        "def objective(trial):\n",
        "    # 하이퍼파라미터 탐색 범위 설정\n",
        "    params = {\n",
        "        'num_leaves': trial.suggest_int('num_leaves', 10, 200),\n",
        "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),\n",
        "        'max_depth': trial.suggest_int('max_depth', 3, 10),\n",
        "        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),\n",
        "    }\n",
        "\n",
        "    # LGBMRegressor 모델 초기화\n",
        "    model = LGBMRegressor(random_state=42, **params)\n",
        "\n",
        "    # 모델 학습\n",
        "    model.fit(x_train, y_train)\n",
        "\n",
        "    # 평가 지표 (MAE) 계산\n",
        "    mae = mean_absolute_error(y_test, model.predict(x_test))\n",
        "\n",
        "    return mae\n",
        "\n",
        "# 데이터 분할\n",
        "# x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)\n",
        "\n",
        "# Optuna 최적화 설정\n",
        "study = optuna.create_study(direction='minimize')\n",
        "\n",
        "# 최적화 진행 상황 출력\n",
        "def print_study_statistics(study, trial):\n",
        "    print(f'Trial {trial.number}/{len(study.trials)} - Loss: {trial.value:.4f} - Best Loss: {study.best_value:.4f} - Params: {trial.params}')\n",
        "\n",
        "study.optimize(objective, n_trials=100, callbacks=[print_study_statistics])\n",
        "\n",
        "# 최적 하이퍼파라미터 출력\n",
        "print('Best Parameters:', study.best_params)\n",
        "print('Best MAE:', study.best_value)\n",
        "\n",
        "# 최적 하이퍼파라미터로 모델 재학습\n",
        "best_params = study.best_params\n",
        "model = LGBMRegressor(random_state=42, **best_params)\n",
        "model.fit(x_train, y_train)\n",
        "\n",
        "# 모델 성능 평가\n",
        "print('Train Score:', model.score(x_train, y_train))\n",
        "print('Test Score:', model.score(x_test, y_test))\n",
        "print('MAE:', mean_absolute_error(y_test, model.predict(x_test)))"
      ],
      "metadata": {
        "id": "qCCHi0JlF3uc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# optuna 파라메터튜닝 + LGBM 모델의 검증점수가 기존의 앙상블 모델보다 훨씬 더 좋은 성능을 냄."
      ],
      "metadata": {
        "id": "GAWa2nrZF7Qn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 따라서 optuna 파라메터튜닝 + LGBM 모델로 학습\n",
        "pf = PolynomialFeatures(degree=2)\n",
        "X_poly = pf.fit_transform(X)\n",
        "model = LGBMRegressor(random_state=42,num_leaves=197,learning_rate=0.2237698754656206,max_depth=10,min_child_samples=5)\n",
        "model.fit(X_poly,y)"
      ],
      "metadata": {
        "id": "FKD_pQW7GEIg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#최종 답 추출\n",
        "preg_poly = pf.transform(preg)\n",
        "y_pred = model.predict(preg_poly)\n",
        "print(y_pred)\n",
        "print(len(preg),len(df_spring_test))"
      ],
      "metadata": {
        "id": "vpbwiBNsGQgR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 답 csv파일로 저장\n",
        "pd.concat([df_spring_test, pd.DataFrame(y_pred)], axis=\"columns\").to_csv(\"output_spring.csv\")"
      ],
      "metadata": {
        "id": "rkXgu3SbGT-g"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}