import base64
import csv
import hashlib
import hmac
import json
import ssl
import threading
import time
import paho.mqtt.client as mqtt
import schedule
from datetime import datetime
import src.core.global_var as gv
'''
# MQTT 连接配置
YourHost = "k24ta6M9Wm2.iot-as-mqtt.cn-shanghai.aliyuncs.com"
YourClientId = "k24ta6M9Wm2.device1"
YourIotInstanceId = "iot-06z00iczh7cuxzc"
YourConsumerGroupId = "DEFAULT_GROUP"
ALIBABA_CLOUD_ACCESS_KEY_ID = "LTAI5tNRm2i6koqHEX6KP6Cq"
ALIBABA_CLOUD_ACCESS_KEY_SECRET = "HQiZbOXkXgo9d23lyCLh5MAxZH3v1W"
conn = None

csv_file = "data/forecast_data/receive_data.csv"
'''

def disconnect_mqtt():
    global conn
    try:
        if conn and conn.is_connected():
            conn.disconnect()
            gv.global_var.user_initiated_disconnect = True  # 设置断开标志
            print("MQTT connection is disconnected.")
    except Exception as e:
        print('Error while trying to disconnect:', e)


# 回调函数：连接成功时调用
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("/topic/#")


# 回调函数：接收到消息时调用
def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload}")
    transform_data(msg.payload)


# 连接和订阅
def connect_and_subscribe(YourClientId, accessKey_id, accessKey_secret):
    global conn
    accessKey = accessKey_id
    accessSecret = accessKey_secret
    consumerGroupId = YourConsumerGroupId
    iotInstanceId = YourIotInstanceId
    clientId = YourClientId

    signMethod = "hmacsha1"
    timestamp = current_time_millis()
    username = clientId + "|authMode=aksign" + ",signMethod=" + signMethod + ",timestamp=" + timestamp \
               + ",authId=" + accessKey + ",iotInstanceId=" + iotInstanceId \
               + ",consumerGroupId=" + consumerGroupId + "|"
    signContent = "authId=" + accessKey + "&timestamp=" + timestamp
    password = do_sign(accessSecret.encode("utf-8"), signContent.encode("utf-8"))

    # 创建 MQTT 客户端
    conn = mqtt.Client(clientId)
    conn.username_pw_set(username, password)
    conn.tls_set(certfile=None, keyfile=None, cert_reqs=ssl.CERT_NONE, tls_version=ssl.PROTOCOL_TLS)

    # 设置回调函数
    conn.on_connect = on_connect
    conn.on_message = on_message

    try:
        # 连接到 MQTT 代理服务器
        conn.connect(YourHost, 1883, 60)
        # 启动一个新线程以保持连接
        thread = threading.Thread(target=mqtt_loop)
        thread.start()
    except Exception as e:
        print('Connecting failed:', e)
        raise e


# 维持 MQTT 连接的循环
def mqtt_loop():
    try:
        conn.loop_forever()
    except Exception as e:
        print('Error while in MQTT loop:', e)


def current_time_millis():
    return str(int(round(time.time() * 1000)))


def do_sign(secret, sign_content):
    m = hmac.new(secret, sign_content, digestmod=hashlib.sha1)
    return base64.b64encode(m.digest()).decode("utf-8")


# 检查连接，如果未连接则重新建连
def do_check(conn):
    global clientId, accessKey, accessSecret
    if clientId is None or accessKey is None or accessSecret is None:
        print('Please input clientId, accessKey, and accessSecret.')
        return
    print('Checking connection, is_connected:', conn.is_connected())
    if not conn.is_connected():
        try:
            if not gv.global_var.user_initiated_disconnect:
                connect_and_subscribe(clientId, accessKey, accessSecret)
        except Exception as e:
            print('Error while trying to reconnect:', e)


# 定时任务方法，检查连接状态
def connection_check_timer():
    while 1:
        schedule.run_pending()
        time.sleep(10)


# 把时间戳转换成字符串
def timestamp_to_time(timestamp):
    if (type(timestamp) == float):
        dt_obj = datetime.fromtimestamp(timestamp / 1000.0)
    else:
        dt_obj = datetime.fromtimestamp(timestamp / 1000)
    return dt_obj.strftime('%Y-%m-%d T %H:%M:%S')


# 将时间字符串转换为时间戳
def time_to_timestamp(time_str):
    dt_obj = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
    return int(dt_obj.timestamp()) * 1000


def transform_data(payload):
    try:
        str_result = payload.decode('utf-8')
        print(type(payload))  # str类型
        print(str_result)  # 打印原始数据
        result = json.loads(str_result)
        prop_data = {}
        prop_data["time"] = int(result['items']['DetectTime']['value'])
        prop_data["temperature"] = float(result['items']['CurrentTemperature'].get("value", None))
        prop_data["humidity"] = float(result['items']['CurrentHumidity'].get("value", None))
        prop_data["pressure"] = int(result['items']['CurrentPressure'].get("value", None))
        ans_data = {}
        print(str(len(gv.global_var.topic_list)))
        for topic in gv.global_var.topic_list:
            if prop_data[topic] is not None:
                ans_data[topic] = prop_data[topic]
        if ans_data != {}:
            ans_data['time'] = prop_data['time']
            ans_data["printed"] = False
            gv.global_var.receive_data.append(ans_data)
            print(format_topicData(ans_data))
    except Exception as e:
        print("Error while processing the data:", e)


def format_topicData(prop_data):
    formatted_ans = "time:" + timestamp_to_time(prop_data["time"]) + " "
    formatted_ans += (" ".join([f"{k}:{v}" for k, v in prop_data.items() if k != "printed" and k != "time"]) + "\n")
    return formatted_ans


def format_time(month, day, hour):
    return "2024-" + month + "-" + day + " " + hour + ":00"


def format_topiclist(topic_list):
    formatted_ans = ""
    for topic in topic_list:
        formatted_ans += topic + " "
    return formatted_ans


# 读取数据，整理
def read_data():
    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            gv.global_var.receive_data = []
            for row in reader:
                prop_data = {}
                for i in range(0, 3):
                    if row[i] != "":
                        if i == 0:
                            prop_data["time"] = int(row[i])
                        elif i == 1:
                            prop_data["temperature"] = float(row[i])
                        elif i == 2:
                            prop_data["humidity"] = float(row[i])
                        elif i == 3:
                            prop_data["pressure"] = int(row[i])
                if prop_data:
                    gv.global_var.receive_data.append(prop_data)
            print("Data reading complete. Total entries fetched:", len(gv.global_var.receive_data))
            return gv.global_var.receive_data
    except Exception as e:
        print('Error while trying to read data:', e)
        return None
