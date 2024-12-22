from flask import Flask, render_template, request, jsonify
from linkkit import linkkit
import datetime
import logging
import random
import time
import json
import threading
import pandas as pd
#*
'''lk = linkkit.LinkKit(
    host_name="cn-shanghai",
    product_key="k255wPG4Ykb",
    device_name="KWZRtest001",
    device_secret="34971c1e3bb9548e8bfcfb65e581b5f0")
lk.enable_logger(logging.DEBUG)

mqtt_topic_post = f'/sys/k255wPG4Ykb/KWZRtest001/thing/event/property/post'  # 用于发布消息的主题
# mqtt_topic_post_reply = f'/sys/k255wPG4Ykb/KWZRtest001/thing/event/property/post_reply'
'''

# 上传数据的线程控制变量
stop_upload_flag = False
upload_thread = None
upload_status = 0

# 加载 CSV 数据并按时间排序
data = pd.read_csv("D:/github/IOT-project-based-on-MQTT/data/Data.csv")  # 确保文件路径正确
data_sorted = data.sort_values(by=data.columns[0])  # 假设第一列为时间列

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('publish.html') 

@app.route('/connect', methods=['POST'])
def connect():
    # 尝试连接到 MQTT 服务器
    try:
        # 调用 connect_async() 启动连接
        result = lk.connect_async()
        time.sleep(2)  # 等待连接完成

        # 主动订阅 post_reply 主题
        if result == 0:  # 0 表示连接启动成功
            return jsonify({
                'timestamp': str(datetime.datetime.now()),
                'status': 'connected',
                'message': 'Connected successfully!'
            })
        else:  # 非 0 表示连接失败
            return jsonify({
                'timestamp': str(datetime.datetime.now()),
                'status': 'connection_failed',
                'message': f'Failed to initiate connection. Return code: {result}'
            })
    except Exception as e:
        return jsonify({
            'timestamp': str(datetime.datetime.now()),
            'status': 'error',
            'message': f'Error occurred: {str(e)}'
        })
    

@app.route('/disconnect', methods=['POST'])
def disconnect():
    try:
        # 调用 disconnect() 方法断开连接
        lk.disconnect()
        return jsonify({
            'timestamp': str(datetime.datetime.now()),
            'status': 'disconnected',
            'message': 'Disconnected successfully!'
        })
    except Exception as e:
        return jsonify({
            'timestamp': str(datetime.datetime.now()),
            'status': 'error',
            'message': f'Error occurred while disconnecting: {str(e)}'
        })
    
@app.route('/publishRandom', methods=['POST'])
def publishRandom():
    try:
        # 调用随机数据发布函数
        result = post_random_data(mqtt_topic_post)

        if result:
            return jsonify({'timestamp': str(datetime.datetime.now()), 'status': 'success', 'message': 'Random data published.'})
        else:
            return jsonify({'timestamp': str(datetime.datetime.now()), 'status': 'failed', 'message': 'Failed to publish random data.'})
    except Exception as e:
        return jsonify({'timestamp': str(datetime.datetime.now()), 'status': 'error', 'message': f'Error occurred: {str(e)}'})

@app.route('/startPublish', methods=['POST'])
def start_upload():
    global stop_upload_flag, upload_thread

    # 如果已有上传线程在运行，则返回错误
    if upload_thread and upload_thread.is_alive():
        return jsonify({
            'timestamp': str(datetime.datetime.now()),
            'status': 'error',
            'message': 'Upload is already in progress.'
        })

    # 启动上传线程
    stop_upload_flag = False
    upload_thread = threading.Thread(target=upload_data)
    upload_thread.start()

    return jsonify({
        'timestamp': str(datetime.datetime.now()),
        'status': 'started',
        'message': 'Data upload started.'
    })


@app.route('/stopPublish', methods=['POST'])
def stop_upload():
    global stop_upload_flag

    # 设置停止标志
    stop_upload_flag = True

    return jsonify({
        'timestamp': str(datetime.datetime.now()),
        'status': 'stopped',
        'message': 'Data upload stopped.'
    })

@app.route('/publishStatus', methods=['GET'])
def publish_status():
    global upload_status, stop_upload_flag, upload_thread

    # 检查线程是否存在和是否正在运行
    thread_running = upload_thread.is_alive() if upload_thread else False

    # 返回上传状态信息
    return jsonify({
        'count': upload_status,  # 已发布数据的条目数
        'complete': stop_upload_flag,  # 是否停止上传
        'error': None  # 可以用于扩展添加错误信息
    })


# 随机数据上传函数
def post_random_data(topic):
    try:
        # 随机生成属性数据
        prop_data = {
            "CurrentTemperature": round(random.uniform(-10.0, 40.0), 2),  # 随机温度
            "CurrentHumidity": round(random.uniform(0.0, 100.0), 2),  # 随机湿度
            "CurrentPressure": random.randint(900, 1100),  # 随机气压
            "DetectTime": str(int(round(time.time() * 1000)))  # 时间戳（毫秒）
        }

        # 构造消息负载
        payload = {
            "id": "123",
            "version": "1.0",
            "params": prop_data,
            "method": "thing.event.property.post"
        }

        # 发布消息到 MQTT
        rc, request_id = lk.publish_topic(topic, json.dumps(payload))
        
        if rc == 0:
            print(f"Random data published successfully: request_id={request_id}")
            return True
        else:
            print(f"Failed to publish random data, rc={rc}")
            return False
    except Exception as e:
        print(f"Error in post_random_data: {str(e)}")
        return False
    

def upload_data():
    global stop_upload_flag, upload_status

    upload_status = 0

    try:
        
        # 遍历排序后的数据
        for _, row in data_sorted.iterrows():
            if stop_upload_flag:
                print("Upload stopped by user.")
                break
        
            # 构造消息负载
            payload = {
                "id": '123',  # 使用时间戳作为唯一 ID
                "version": "1.0",
                "params": {
                    "CurrentTemperature": float(row[1]),  # 确保为浮点数
                    "CurrentHumidity": float(row[2]),  # 确保为浮点数
                    "CurrentPressure": int(float(row[3])),  # 确保为整数
                    "DetectTime": str(int(time.mktime(datetime.datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S").timetuple()) * 1000)),  # 转换为字符串的毫秒级时间戳
                },
                "method": "thing.event.property.post"
            }

            # 发布消息到 MQTT
            rc, request_id = lk.publish_topic(mqtt_topic_post, json.dumps(payload))

            if rc == 0:
                upload_status += 1  # 成功发布后增加 upload_status
                print(f"Data published successfully: request_id={request_id}")
            else:
                print(f"Failed to publish data: rc={rc}")

            # 控制发送频率（假设每秒发送一次）
            time.sleep(1)

    except Exception as e:
        print(f"Error during data upload: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)