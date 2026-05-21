"""
长期记忆服务模块 - MySQL数据库持久化存储
基于MySQL实现用户信息和历史对话的持久化存储，实现增删改查
"""
import json
import re
from datetime import datetime
from typing import Dict, Optional
from utils.database import get_db_cursor, db_config, DatabaseInitializer
from utils.logger_handler import logger


class MemoryService:
    """用户记忆服务类"""

    def __init__(self):
        self._table_name = db_config.get_table_name("memory")  # 获取数据库表名
        self._ensure_table_exists()  # 确保数据库表存在

    def _ensure_table_exists(self):
        """确保数据库表存在，如不存在则创建"""
        try:
            # 初始化数据库表（如果不存在）
            DatabaseInitializer.init_database()
        except Exception as e:
            logger.error(f"[memory_service] 初始化数据库表失败: {e}")

    def _get_or_create_user_memory(self, username: str) -> int:
        """
        确保这条记忆记录存在，并返回它的 ID

        Args:
            username: 用户名

        Returns:
            用户记忆ID (memory表的主键id)
        """
        with get_db_cursor() as cursor:      # 获取数据库游标
            # 先查 memory 表，看记忆记录是否已存在
            cursor.execute(
                f"SELECT `id` FROM `{self._table_name}` WHERE `username` = %s",
                (username,)       # 传给 %s 的实际参数
            )
            result = cursor.fetchone()      # 获取查询结果

            if result:
                return result['id']    # 返回已存在的记忆ID

            # 记忆不存在，需要先从 users 表查到 user_id
            users_table = db_config.get_table_name("users")    # 把"users"这个表名，根据配置加上前缀，返回最终要用的表名。
            cursor.execute(
                f"SELECT `id` FROM `{users_table}` WHERE `username` = %s",
                (username,)
            )
            user_record = cursor.fetchone()     # 获取查询结果
            if not user_record:
                raise ValueError(f"用户 '{username}' 尚未注册，无法创建记忆记录")

            user_id = user_record['id']    # 从 users 表获取 user_id

            # 插入记忆记录，关联到 users 表的 user_id
            cursor.execute(
                f"INSERT INTO `{self._table_name}` (`user_id`, `username`, `history`) VALUES (%s, %s, %s)",
                (user_id, username, json.dumps([]))
            )
            logger.info(f"[memory_service] 新用户记忆记录已创建: username={username}, user_id={user_id}")
            return cursor.lastrowid    # 取回刚刚数据库自动生成的那个ID

    def save_user_info(self, username: str, info: Dict):
        """
        把用户的长期记忆信息保存到数据库中

        Args:
            username: 用户名
            info: 用户信息字典，包含设备型号、历史问题等
        """
        # 确保记忆记录存在，返回它的 ID
        user_id = self._get_or_create_user_memory(username)

        with get_db_cursor() as cursor:    # 获取数据库游标
            if "device_model" in info:     # 如果 info 字典中包含 device_model 键
                cursor.execute(            # 修改或更新已有的数据
                    f"UPDATE `{self._table_name}` SET `device_model` = %s WHERE `id` = %s",
                    (info["device_model"], user_id)     # 传给 %s 的实际参数
                )
                logger.info(f"[memory_service] 设备型号已保存: username={username}, model={info['device_model']}")

            if "history" in info:     # 如果 info 字典中包含 history 键
                history_json = json.dumps(info["history"], ensure_ascii=False)    # 把历史记录列表转换为 JSON 字符串
                cursor.execute(
                    f"UPDATE `{self._table_name}` SET `history` = %s WHERE `id` = %s",
                    (history_json, user_id)
                )

    def load_user_info(self, username: str) -> Dict:
        """
        加载用户长期记忆信息

        Args:
            username: 用户名
        Returns:
            用户信息字典，包含设备型号、历史问题等
        """
        with get_db_cursor() as cursor:    # 获取数据库游标
            cursor.execute(            # 把括号里的SQL语句发送给 MySQL 数据库去执行。
                f"SELECT `device_model`, `history` FROM `{self._table_name}` WHERE `username` = %s",
                (username,)
            )
            result = cursor.fetchone()      # 获取查询结果

            if result:
                history = result['history']
                if isinstance(history, str):    # 如果 history 字段是字符串类型
                    try:
                        # 尝试把 JSON 字符串转换为 Python 列表
                        history = json.loads(history)
                    except json.JSONDecodeError:
                        history = []
                elif history is None:
                    history = []

                return {
                    "device_model": result['device_model'],
                    "history": history
                }

        return {}

    def append_history(self, username: str, question: str, answer: str):
        """
        追加用户历史问题和回答到长期记忆记录

        Args:
            username: 用户名
            question: 用户问题
            answer: 助手回答
        """
        user_info = self.load_user_info(username)   # 加载用户长期记忆信息
        history = user_info.get("history", [])      # 获取历史记录列表，默认空列表

        history.append({              # 追加用户历史问题和回答
            "question": question,     # 用户问题
            "answer": answer,         # 助手回答
            "timestamp": datetime.now().isoformat()    # 记录时间戳
        })

        if len(history) > 20:
            history = history[-20:]    # 保留最近 20 条记录

        if not user_info.get("device_model"):    # 如果用户没有设备型号
            extracted_model = self._extract_device_model(question)    # 从用户问题中提取设备型号
            if extracted_model:    # 如果提取到设备型号
                user_info["device_model"] = extracted_model    # 保存设备型号
                logger.info(f"[memory_service] 检测到设备型号: {extracted_model}")

        self.save_user_info(username, user_info)    # 把长期记忆保存到数据库
        logger.info(f"[memory_service] 历史记录已追加: username={username}")

    @staticmethod
    def _extract_device_model(text: str) -> Optional[str]:
        """
        从文本中提取设备型号

        Args:
            text: 用户问题文本

        Returns:
            提取到的设备型号，如果没有则返回None
        """
        patterns = [
            r"(?:型号是?|型号为?|用的?[是为]?)(\w+[\w\s]*(?:扫地|扫拖)?(?:机器人|机))",
            r"(?:我的|我买了|买了)(\w+[\w\s]*(?:扫地|扫拖)?(?:机器人|机))",
            r"(?:是|用)(\w+[\w\s]*(?:pro|plus|ultra|max)?(?:扫地|扫拖)?(?:机器人|机))",
        ]

        for pattern in patterns:     # 遍历所有正则表达式
            match = re.search(pattern, text)    # 搜索文本中是否包含设备型号
            if match:
                logger.info(f"[memory_service] 提取到设备型号: {match.group(1).strip()}")
                return match.group(1).strip()     # 返回提取到的型号，去掉首尾空格

        return None  


memory_service = MemoryService()


def save_user_info(username: str, info: Dict):
    """保存用户信息（兼容旧接口）"""
    memory_service.save_user_info(username, info)


def load_user_info(username: str) -> Dict:
    """加载用户信息（兼容旧接口）"""
    return memory_service.load_user_info(username)


def append_history(username: str, question: str, answer: str):
    """追加历史记录（兼容旧接口）"""
    memory_service.append_history(username, question, answer)



