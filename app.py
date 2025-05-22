import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# 必要なライブラリのインポート
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ページ設定
st.set_page_config(
    page_title="時系列予測アプリ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 日本語フォント設定（環境に応じて変更が必要な場合があります）
plt.rcParams['font.family'] = 'Meiryo', 'MS Gothic', 'Arial', 'sans-serif'

# アプリのタイトル
st.title('📈 時系列予測アプリ')
st.markdown("""
このアプリは、アップロードされた時系列データに対して複数の予測モデルを適用し、結果を比較・可視化します。
対応モデル: ARIMA, SARIMAX, VAR, LSTM
""")

# セッション状態の初期化
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}

# サイドバー設定
st.sidebar.header('設定')

# ファイルアップロード
uploaded_file = st.sidebar.file_uploader(
    "Excelファイルをアップロードしてください", 
    type=["xlsx", "xls"]
)

def load_data(file):
    """Excelファイルを読み込む"""
    if file is not None:
        try:
            df = pd.read_excel(file)
            return df
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
            return None
    return None

# データ読み込み
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.session_state.df = df

# データが読み込まれたら設定を表示
if st.session_state.df is not None:
    df = st.session_state.df
    
    # カラム選択
    st.sidebar.subheader('データ設定')
    date_col = st.sidebar.selectbox('日付列を選択', df.columns, index=0)
    target_col = st.sidebar.selectbox('目的変数列を選択', df.columns, index=min(1, len(df.columns)-1))
    
    # 日付列をインデックスに設定
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df.sort_index(inplace=True)
    except Exception as e:
        st.error(f"日付列の処理中にエラーが発生しました: {e}")
    
    # 時系列の頻度設定
    freq_options = ['自動検出', '日次 (D)', '週次 (W)', '月次 (M)', '四半期 (Q)', '年次 (Y)']
    freq_setting = st.sidebar.selectbox('時系列の頻度', freq_options, index=0)
    
    # 外部変数の選択（複数選択可）
    exog_cols = st.sidebar.multiselect(
        '外部変数を選択（任意）',
        [col for col in df.columns if col != target_col]
    )
    
    # トレーニング期間の設定
    st.sidebar.subheader('分析期間設定')
    train_start = st.sidebar.date_input(
        '学習開始日',
        value=df.index.min().to_pydatetime().date(),
        min_value=df.index.min().to_pydatetime().date(),
        max_value=df.index.max().to_pydatetime().date()
    )
    
    train_end = st.sidebar.date_input(
        '学習終了日',
        value=df.index.max().to_pydatetime().date() - timedelta(days=30),
        min_value=df.index.min().to_pydatetime().date(),
        max_value=df.index.max().to_pydatetime().date()
    )
    
    # 予測期間の設定
    forecast_steps = st.sidebar.number_input(
        '予測ステップ数',
        min_value=1,
        max_value=365,
        value=30,
        step=1
    )
    
    # モデル選択
    st.sidebar.subheader('モデル設定')
    use_arima = st.sidebar.checkbox('ARIMA', value=True)
    use_sarimax = st.sidebar.checkbox('SARIMAX', value=len(exog_cols) > 0)
    use_var = st.sidebar.checkbox('VAR', value=len(exog_cols) > 0)
    use_lstm = st.sidebar.checkbox('LSTM', value=True)
    
    # LSTM固有の設定
    if use_lstm:
        st.sidebar.subheader('LSTM設定')
        lstm_timesteps = st.sidebar.number_input(
            'タイムステップ数（履歴の長さ）',
            min_value=1,
            max_value=100,
            value=10,
            step=1
        )
        
        lstm_epochs = st.sidebar.number_input(
            'エポック数',
            min_value=1,
            max_value=1000,
            value=100,
            step=10
        )
        
        lstm_units = st.sidebar.number_input(
            'LSTMユニット数',
            min_value=1,
            max_value=256,
            value=50,
            step=10
        )
        
        lstm_dropout = st.sidebar.slider(
            'Dropout率',
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.05
        )
        
        lstm_scaler = st.sidebar.selectbox(
            'スケーリング方法',
            ['MinMaxScaler', 'StandardScaler'],
            index=0
        )
    
    # データの前処理
    train_data = df[train_start:train_end]
    
    # モデル実行ボタン
    if st.sidebar.button('モデルを実行'):
        with st.spinner('モデルを実行中...'):
            st.session_state.models = {}
            st.session_state.results = {}
            
            # 予測用の日付インデックスを作成
            last_date = train_data.index[-1]
            if freq_setting == '自動検出':
                freq = pd.infer_freq(df.index)
                if freq is None:
                    freq = 'D'  # デフォルトで日次
            else:
                freq = freq_setting.split(' ')[1][1:-1]  # '日次 (D)' から 'D' を抽出
            
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_steps,
                freq=freq
            )
            
            # 予測結果を格納するDataFrame
            forecast_df = pd.DataFrame(index=future_dates)
            
            # ARIMAモデル
            if use_arima:
                try:
                    with st.spinner('ARIMAモデルを実行中...'):
                        # 自動ARIMAで最適なパラメータを探索
                        model_arima = auto_arima(
                            train_data[target_col],
                            seasonal=False,
                            trace=False,
                            error_action='ignore',
                            suppress_warnings=True
                        )
                        
                        # モデルのトレーニング
                        model_fit = model_arima.fit(train_data[target_col])
                        
                        # 予測
                        forecast = model_fit.predict(n_periods=forecast_steps)
                        forecast_df['ARIMA'] = forecast.values
                        
                        # モデルと結果を保存
                        st.session_state.models['ARIMA'] = model_fit
                        st.session_state.results['ARIMA'] = {
                            'forecast': forecast_df[['ARIMA']],
                            'metrics': {},
                            'model_info': str(model_fit.summary())
                        }
                        
                except Exception as e:
                    st.error(f'ARIMAモデルの実行中にエラーが発生しました: {e}')
            
            # SARIMAXモデル（外部変数あり）
            if use_sarimax and exog_cols:
                try:
                    with st.spinner('SARIMAXモデルを実行中...'):
                        # 自動SARIMAXで最適なパラメータを探索
                        model_sarimax = auto_arima(
                            train_data[target_col],
                            X=train_data[exog_cols],
                            seasonal=True,
                            trace=False,
                            error_action='ignore',
                            suppress_warnings=True
                        )
                        
                        # モデルのトレーニング
                        model_fit = SARIMAX(
                            train_data[target_col],
                            exog=train_data[exog_cols],
                            order=model_sarimax.order,
                            seasonal_order=model_sarimax.seasonal_order
                        ).fit(disp=False)
                        
                        # 予測用の外部変数を準備
                        # 単純に最後の値をコピーして使用（実際のアプリではより高度な処理が必要）
                        exog_forecast = pd.DataFrame(
                            np.tile(train_data[exog_cols].values[-1], (forecast_steps, 1)),
                            columns=exog_cols,
                            index=future_dates
                        )
                        
                        # 予測
                        forecast = model_fit.get_forecast(steps=forecast_steps, exog=exog_forecast)
                        forecast_df['SARIMAX'] = forecast.predicted_mean
                        
                        # モデルと結果を保存
                        st.session_state.models['SARIMAX'] = model_fit
                        st.session_state.results['SARIMAX'] = {
                            'forecast': forecast_df[['SARIMAX']],
                            'metrics': {},
                            'model_info': str(model_fit.summary())
                        }
                        
                except Exception as e:
                    st.error(f'SARIMAXモデルの実行中にエラーが発生しました: {e}')
            
            # VARモデル（多変量時系列）
            if use_var and len(exog_cols) > 0:
                try:
                    with st.spinner('VARモデルを実行中...'):
                        # データの準備
                        var_data = train_data[[target_col] + exog_cols].dropna()
                        
                        # 最適なラグを選択
                        # 実際のアプリではより高度な方法で選択する必要があります
                        model_var = VAR(var_data)
                        results = model_var.fit()
                        
                        # 予測
                        lag_order = results.k_ar
                        forecast_input = var_data.values[-lag_order:]
                        forecast = results.forecast(forecast_input, steps=forecast_steps)
                        
                        # 予測結果をDataFrameに変換
                        forecast_df['VAR'] = forecast[:, 0]  # 目的変数の予測のみを取得
                        
                        # グレンジャー因果性検定
                        gc_results = {}
                        for col in exog_cols:
                            try:
                                gc_test = grangercausalitytests(var_data[[target_col, col]], maxlag=min(5, len(var_data)//3), verbose=False)
                                p_values = [round(gc_test[i+1][0]['ssr_ftest'][1], 4) for i in range(len(gc_test))]
                                gc_results[col] = {
                                    'p_values': p_values,
                                    'min_p_value': min(p_values)
                                }
                            except:
                                pass
                        
                        # インパルス応答関数
                        irf = results.irf(10)
                        
                        # モデルと結果を保存
                        st.session_state.models['VAR'] = results
                        st.session_state.results['VAR'] = {
                            'forecast': forecast_df[['VAR']],
                            'metrics': {},
                            'model_info': str(results.summary()),
                            'granger_causality': gc_results,
                            'irf': irf
                        }
                        
                except Exception as e:
                    st.error(f'VARモデルの実行中にエラーが発生しました: {e}')
            
            # LSTMモデル
            if use_lstm:
                try:
                    with st.spinner('LSTMモデルを実行中...'):
                        # データの準備
                        features = [target_col] + exog_cols if exog_cols else [target_col]
                        lstm_data = train_data[features].copy()
                        
                        # スケーリング
                        if lstm_scaler == 'MinMaxScaler':
                            scaler = MinMaxScaler()
                        else:
                            scaler = StandardScaler()
                            
                        scaled_data = scaler.fit_transform(lstm_data)
                        
                        # 時系列データの作成
                        X, y = [], []
                        for i in range(len(scaled_data) - lstm_timesteps):
                            X.append(scaled_data[i:(i + lstm_timesteps)])
                            y.append(scaled_data[i + lstm_timesteps, 0])  # 目的変数のみ
                            
                        X = np.array(X)
                        y = np.array(y)
                        
                        # モデルの構築
                        model = Sequential([
                            LSTM(units=lstm_units, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                            Dropout(lstm_dropout),
                            LSTM(units=lstm_units, return_sequences=False),
                            Dropout(lstm_dropout),
                            Dense(1)
                        ])
                        
                        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                        
                        # モデルのトレーニング
                        history = model.fit(
                            X, y,
                            epochs=lstm_epochs,
                            batch_size=32,
                            validation_split=0.1,
                            verbose=0
                        )
                        
                        # 予測のための入力データを準備
                        last_sequence = scaled_data[-lstm_timesteps:]
                        future_predictions = []
                        
                        for _ in range(forecast_steps):
                            # 予測
                            next_step = model.predict(last_sequence.reshape(1, lstm_timesteps, len(features)), verbose=0)
                            future_predictions.append(next_step[0, 0])
                            
                            # 予測値を入力に追加（自己回帰的に予測）
                            new_row = np.zeros(len(features))
                            new_row[0] = next_step[0, 0]  # 目的変数の予測値
                            
                            # 外部変数がある場合は、最後の値をコピー（実際のアプリではより高度な処理が必要）
                            if len(features) > 1:
                                new_row[1:] = last_sequence[-1, 1:]
                                
                            last_sequence = np.vstack([last_sequence[1:], new_row])
                        
                        # 予測結果を元のスケールに戻す
                        dummy_array = np.zeros((len(future_predictions), len(features)))
                        dummy_array[:, 0] = future_predictions
                        forecast_values = scaler.inverse_transform(dummy_array)[:, 0]
                        
                        # 予測結果をDataFrameに格納
                        forecast_df['LSTM'] = forecast_values
                        
                        # モデルと結果を保存
                        st.session_state.models['LSTM'] = model
                        st.session_state.results['LSTM'] = {
                            'forecast': forecast_df[['LSTM']],
                            'metrics': {},
                            'training_history': history.history,
                            'model_summary': []
                        }
                        
                        # モデルのサマリーを取得
                        stringlist = []
                        model.summary(print_fn=lambda x: stringlist.append(x))
                        st.session_state.results['LSTM']['model_summary'] = '\n'.join(stringlist)
                        
                except Exception as e:
                    st.error(f'LSTMモデルの実行中にエラーが発生しました: {e}')
            
            # 予測結果を保存
            st.session_state.forecast_df = forecast_df
            st.session_state.train_data = train_data
            
            # 評価指標の計算
            if len(forecast_df.columns) > 0:
                st.success('モデルの実行が完了しました！')
            else:
                st.warning('実行可能なモデルが選択されていません。')
    
    # 結果の表示
    if 'forecast_df' in st.session_state and not st.session_state.forecast_df.empty:
        st.subheader('予測結果')
        
        # 予測結果のプロット
        fig = go.Figure()
        
        # 実績データのプロット
        fig.add_trace(go.Scatter(
            x=train_data.index,
            y=train_data[target_col],
            mode='lines',
            name='実績',
            line=dict(color='blue')
        ))
        
        # 各モデルの予測結果をプロット
        for model_name in st.session_state.forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=st.session_state.forecast_df.index,
                y=st.session_state.forecast_df[model_name],
                mode='lines',
                name=f'{model_name} 予測',
                line=dict(dash='dash')
            ))
        
        # レイアウトの設定
        fig.update_layout(
            title=f'{target_col} の時系列予測',
            xaxis_title='日付',
            yaxis_title=target_col,
            legend_title='凡例',
            template='plotly_white',
            hovermode='x unified',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # タブで各モデルの詳細を表示
        tab_titles = list(st.session_state.results.keys())
        tabs = st.tabs(tab_titles)
        
        for i, (model_name, tab) in enumerate(zip(tab_titles, tabs)):
            with tab:
                result = st.session_state.results[model_name]
                
                # モデル情報の表示
                st.subheader(f'{model_name} モデル情報')
                
                if model_name == 'LSTM':
                    # LSTMのモデルサマリーを表示
                    st.text('モデルアーキテクチャ:')
                    st.code(result['model_summary'], language='text')
                    
                    # トレーニング履歴のプロット
                    st.subheader('トレーニング履歴')
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        y=result['training_history']['loss'],
                        mode='lines',
                        name='Training Loss'
                    ))
                    if 'val_loss' in result['training_history']:
                        fig_loss.add_trace(go.Scatter(
                            y=result['training_history']['val_loss'],
                            mode='lines',
                            name='Validation Loss'
                        ))
                    fig_loss.update_layout(
                        title='モデルの損失',
                        xaxis_title='Epoch',
                        yaxis_title='Loss',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                    
                elif model_name == 'VAR':
                    # VARモデルのグレンジャー因果性検定結果を表示
                    st.subheader('グレンジャー因果性検定')
                    gc_data = []
                    for col, gc_result in result['granger_causality'].items():
                        gc_data.append({
                            '変数': col,
                            '最小p値': f"{gc_result['min_p_value']:.4f}",
                            '因果関係の有無': 'あり' if gc_result['min_p_value'] < 0.05 else 'なし'
                        })
                    
                    if gc_data:
                        st.table(pd.DataFrame(gc_data))
                    
                    # インパルス応答関数のプロット
                    st.subheader('インパルス応答関数')
                    try:
                        irf_fig = result['irf'].plot(plot_stderr=False, figsize=(12, 8))
                        st.pyplot(irf_fig[0].figure)
                    except Exception as e:
                        st.error(f'インパルス応答関数のプロットに失敗しました: {e}')
                    
                else:
                    # その他のモデルのサマリーを表示
                    st.text('モデルサマリー:')
                    st.text(result.get('model_info', '利用不可'))
                
                # 予測結果のテーブルを表示
                st.subheader('予測結果')
                st.dataframe(result['forecast'].style.format('{:.2f}'))
                
                # 予測結果のダウンロードボタン
                csv = result['forecast'].to_csv().encode('utf-8')
                st.download_button(
                    label=f'{model_name} 予測結果をダウンロード',
                    data=csv,
                    file_name=f'{model_name}_forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
    
    # データの表示
    st.subheader('データプレビュー')
    st.dataframe(df.head())
    
    # 時系列プロット
    st.subheader('時系列プロット')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[target_col],
        mode='lines',
        name=target_col
    ))
    
    if exog_cols:
        for col in exog_cols:
            # スケールを合わせるため正規化
            normalized_col = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            normalized_col = normalized_col * (df[target_col].max() - df[target_col].min()) + df[target_col].min()
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized_col,
                mode='lines',
                name=f'{col} (正規化)',
                opacity=0.5
            ))
    
    fig.update_layout(
        title=f'{target_col} の時系列',
        xaxis_title='日付',
        yaxis_title=target_col,
        legend_title='凡例',
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # データの統計情報
    st.subheader('統計情報')
    st.dataframe(df[[target_col] + exog_cols].describe() if exog_cols else df[[target_col]].describe())
    
    # 相関行列
    if exog_cols:
        st.subheader('相関行列')
        corr = df[[target_col] + exog_cols].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=corr.round(2).values,
            texttemplate='%{text}',
            textfont={"size":10}
        ))
        fig_corr.update_layout(
            title='相関行列',
            width=800,
            height=800
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# アプリの実行方法を表示
else:
    st.info('👈 左のサイドバーからExcelファイルをアップロードしてください。')
    
    st.markdown("""
    ### アプリの使い方
    1. 左のサイドバーからExcelファイルをアップロード
    2. 日付列と目的変数列を選択
    3. 必要に応じて外部変数を選択
    4. 分析期間と予測期間を設定
    5. 使用するモデルを選択（ARIMA, SARIMAX, VAR, LSTM）
    6. 「モデルを実行」ボタンをクリック
    
    ### 対応ファイル形式
    - Excelファイル (.xlsx, .xls)
    
    ### 注意事項
    - データの欠損値は自動的に補間されますが、適切な前処理を推奨します
    - 大規模なデータセットの場合、処理に時間がかかることがあります
    - LSTMモデルは特に計算リソースを消費します
    """)
