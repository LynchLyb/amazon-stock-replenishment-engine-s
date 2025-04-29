from flask import Flask, request, jsonify
from prophet import Prophet
import pandas as pd

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # 输入数据转换为 DataFrame
    df = pd.DataFrame(data)
    df['ds'] = pd.to_datetime(df['ds'])  # 确保日期类型正确

    # 拟合 Prophet 模型
    model = Prophet()
    model.fit(df)

    # 预测未来 14 天
    future = model.make_future_dataframe(periods=14)
    forecast = model.predict(future)

    # 只保留预测值
    result = forecast[['ds', 'yhat']].tail(14)
    result_json = result.to_dict(orient='records')

    return jsonify(result_json)

if __name__ == '__main__':
    app.run(port=5000)