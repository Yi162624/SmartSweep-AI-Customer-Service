"""
    为AI agent提供函数工具
"""
import json
import os
import random
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from langchain_core.tools import tool
from RAG.rag_service import RagSummarizerService
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
import re


# 实现rag
rag = RagSummarizerService()

user_ids = ["1001","1002","1003","1004","1005","1006","1007","1008","1009","1010"]

month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]

external_data = {}

# 全局位置提供者
# _location_provider = None

# 正则表达式
_IPV4_RE = re.compile(
    r"^(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\."
    r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\."
    r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\."
    r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$"
)

def _is_valid_ipv4(ip: str) -> bool:
    # 判段传入的 ip 的格式是否正确
    return bool(_IPV4_RE.match(ip or ""))

def _get_public_ip() -> str:
    """
    获取当前设备所在的公网 IPv4 地址: 不能挂梯子
    """
    # 若配置中有public_ip_sources 则返回这个配置，若没有 则以https://ipv4.icanhazip.com为源列表
    ip_sources = agent_conf.get("public_ip_sources", [
        "https://ipv4.icanhazip.com",
    ])
    print(f"[_get_public_ip] ip_sources: {ip_sources}")
    timeout = float(agent_conf.get("public_ip_timeout",3))
    for source in ip_sources:
        try:
            # 发送一个HTTP请求
            with urlopen(source,timeout=timeout) as resp:
                print(f"[_get_public_ip] resp: {resp.url}")
                # 从 HTTP 响应中读取原始字节数据，解码成字符串，再去除首尾空白字符，最终得到纯 IP 地址
                ip = resp.read().decode("utf-8").strip()
                if _is_valid_ipv4(ip):
                    print(f"[_get_public_ip] ip: {ip}")
                    return ip
        except Exception:
            continue
    return ""

GAODE_BASE_URL = agent_conf.get("gaode_base_url")
GAODE_TIMEOUT = float(agent_conf.get("gaode_timeout"))

def _gaode_get(path: str,params: dict) -> dict:
    """
    高德地图 API 返回的 JSON 数据，被解析成的 Python 字典。
    """
    gaode_key = agent_conf.get("gaode_key", "")
    print(f"gaode_key: {gaode_key}")
    if not gaode_key:
        raise ValueError("agent.yml中未配置gaodekey")

    query = dict(params)         # 复制一份
    query["key"] = gaode_key     # 再复制的字典中添加一个键
    url = f"{GAODE_BASE_URL}{path}?{urlencode((query))}"

    try:
        # 向url发送一个HTTP请求
        with urlopen(url,timeout=GAODE_TIMEOUT) as resp:
            print(f"resp: {resp.url}")
            data = resp.read().decode("utf-8").strip()
            print(f"data: {data}")
            # 把json对象转换为python的字典或者列表，然后返回
            return json.loads(data)
    except HTTPError as e:
        raise RuntimeError(f"高德HTTP错误: {e.code}") from e
    except URLError as e:
        raise RuntimeError(f"高德网络错误: {e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"高德请求异常: {str(e)}") from e

def _resolve_city_to_adcode(city: str) -> tuple[str, str]:
    """
    把城市名称（如“北京”）解析成高德地图的行政区划代码（adcode）和规范化的城市名
    """
    geo = _gaode_get("/v3/geocode/geo",{"address": city})
    """  geo结构
    geo = {
    "status": "1",
    "info": "OK",
    "infocode": "10000",    # 状态码
    "geocodes": [
        {"adcode": "110000", "city": "北京市", ...}
    ]} """
    print(f"程序读到的gaodekey是: '{agent_conf.get('gaodekey')}'")
    print(geo)

    if geo.get("status") != "1" or not geo.get("geocodes"):
        raise RuntimeError(f"城市解析失败: {geo.get('info', 'unknown')}")

    first = geo["geocodes"][0]      # 返回列表中的第一个字典
    adcode = first["adcode"]        # 从第一个地理编码信息中提取行政区划代码（adcode）
    if not adcode:
        raise RuntimeError("城市解析成功但未返回adcode")

    # 获取当前所在城市
    resolved_city = first.get("city") or first.get("district") or city
    if isinstance(resolved_city, list):     # 判断resolved_city 是不是列表
        resolved_city = "".join(resolved_city)           # 把一个列表中的所有元素拼接成一个字符串

    return str(resolved_city),str(adcode)



@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    # return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨率极低"
    if not city or not city.strip():
        return "未提供城市信息，无法查询天气"

    try:
        # 获取 规范化的城市名 和 行政区划代码
        resolved_city,adcode = _resolve_city_to_adcode(city.strip())
        print(f"[get_weather] 城市名，行政区化代码：{resolved_city , adcode}")
        # 调用天气查询API，得到高德的天气数据
        weather = _gaode_get(
            "/v3/weather/weatherInfo",        # 天气查询API
            {"city": adcode, "extensions": "base"},     # 请求参数
        )

        if weather.get("status") != "1" or not weather.get("lives"):
            return f"城市{resolved_city}天气查询失败：{weather.get('info', 'unknown')}"

        # 从高德API中获取需要的信息
        live = weather["lives"][0]
        condition = live.get("weather", "未知")
        temperature = live.get("temperature", "未知")
        humidity = live.get("humidity", "未知")
        wind_direction = live.get("winddirection", "未知")
        wind_power = live.get("windpower", "未知")
        report_time = live.get("reporttime", "未知")

        return (
            f"城市{resolved_city}天气为{condition}，气温{temperature}摄氏度，"
            f"空气湿度{humidity}%，{wind_direction}风{wind_power}级，"
            f"数据发布时间{report_time}。"
        )

    except Exception as e:
        logger.error(f"[get_weather]天气查询失败 city={city} err={str(e)}")
        return f"城市{city}天气查询失败，请稍后重试"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    """
    通过浏览器API获取真实的用户所在城市
    """
    try:
        public_ip = _get_public_ip()       # 获取公网的地址
        print(f"[get_user_location] public_ip: {public_ip}")
        params = {"ip": public_ip} if public_ip else {}
        # 获取高德 API 返回的数据（字典）
        ip_info = _gaode_get("/v3/ip",params)
        print(f"[get_user_location] ip_info: {ip_info}")

        if ip_info.get("status") != "1":
            logger.warning(        # warning为日志级别
                f"[get_user_location]高德返回失败 info={ip_info.get('info')}"
                f"infocode={ip_info.get('infocode')} ip={public_ip or 'none'}"
            )
            return "未知城市"

        # 获取城市和省 不存在则返回空字符串
        city = ip_info.get("city","")
        province = ip_info.get("province","")

        # 如果返回的是列表 则拼接在一起（多余的代码）
        if isinstance(city,list):
            city = "".join(city)
        if isinstance(province,list):
            province = "".join(province)

        city = str(city).strip()
        province = str(province).strip()
        print(f"[ get_user_location ] city , province: {city,province}")

        if city:
            return city
        if province:
            return province

        logger.warning(
            f"[get_user_location]未得到城市信息 info={ip_info.get('info')}"
            f"infocode={ip_info.get('infocode')} ip={public_ip or 'none'} raw={ip_info}"
        )

        return "未知城市"

    except Exception as e:
        logger.warning(f"[get_user_location]获取信息失败 err={str(e)}")
        return "未知城市"


@tool(description="从向量存储中检索参考资料")     # RAG
def rag_summarsize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取月份，以字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data() -> str:
    """
    {
        "user_id": {
            "month": {"特征": xxx, "效率": xxx, ...},
            "month": {"特征": xxx, "效率": xxx, ...},
            "month": {"特征": xxx, "效率": xxx, ...},
            ...
        }
        "user_id": {
            "month": {"特征": xxx, "效率": xxx, ...},
            "month": {"特征": xxx, "效率": xxx, ...},
            "month": {"特征": xxx, "效率": xxx, ...},
            ...
        }
        "user_id": {
            "month": {"特征": xxx, "效率": xxx, ...},
            "month": {"特征": xxx, "效率": xxx, ...},
            "month": {"特征": xxx, "效率": xxx, ...},
            ...
        }
    }
    """
    if not external_data:
        # 获取绝对路径
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):     # 检查文件是否存在
            raise FileNotFoundError(f"外部数据文件不存在{external_data_path}不存在")     # 主动抛出异常

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:      # 从第二行开始读取所有的行
                # .split(",")：按照 , 去分割列表
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")      # 替换字符
                feature: str = arr[1].replace('"',"")
                efficiency: str = arr[2].replace('"',"")
                consumables: str = arr[3].replace('"',"")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    # 如果用户不存在，则创建一个新的字典
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取用户的使用记录，以字符串形式返回")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()       # 从外部系统中获取用户的使用记录

    try:
        return external_data[user_id][month]        # 返回用户的使用记录
    except KeyError:
        logger.warning(f"[generate_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="无参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"


if __name__ == "__main__":
    # print(fetch_external_data(user_id="1021", month="2025-06"))
    city = get_user_location()
    print(city)
    ip = _get_public_ip()
    print(ip)
    # resolved_city,abcode = _resolve_city_to_adcode("上海")
    # print(resolved_city)
    # print(abcode)


