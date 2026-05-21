"""
MySQL 数据库连接和操作模块
提供数据库连接池管理、基础CRUD操作及初始化功能
"""
import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager
from typing import Optional, Dict, Any
import queue
from queue import Queue
from threading import Lock
from utils.logger_handler import logger
from utils.config_handler import load_database_config


class DatabaseConfig:
    """数据库配置类"""

    _instance: Optional['DatabaseConfig'] = None    # 单例模式实例
    _connection_params: Dict[str, Any] = {}         # 数据库连接参数

    def __new__(cls):
        # cls 是类本身，self 是实例本身
        """单例模式实现"""
        # 检查是否已存在实例
        if cls._instance is None:
            # 创建新实例
            cls._instance = super().__new__(cls)    # 调用父类的__new__方法，创建一个新的对象
            cls._instance._load_config()            # 加载数据库配置
        return cls._instance

    def _load_config(self):
        """从配置文件加载数据库配置"""
        try:
            # 加载数据库配置
            config = load_database_config()
            self._connection_params = {
                "host": config.get("host", "localhost"),
                "port": config.get("port", 3306),
                "user": config.get("user", "root"),
                "password": config.get("password", ""),
                "database": config.get("database", "sweeper_robot"),
                "charset": config.get("charset", "utf8mb4"),
                "connect_timeout": config.get("connect_timeout", 10),
                "read_timeout": config.get("read_timeout", 30),
                "write_timeout": config.get("write_timeout", 30)
            }
            self._table_prefix = config.get("table_prefix", "")
            logger.info("[database] 数据库配置加载成功")
        except Exception as e:
            logger.error(f"[database] 配置加载失败，使用默认配置: {e}")
            # 配置加载失败时使用默认配置
            self._connection_params = {
                "host": "localhost",
                "port": 3306,
                "user": "root",
                "password": "",
                "database": "sweeper_robot",
                "charset": "utf8mb4",
                "connect_timeout": 10,
                "read_timeout": 30,
                "write_timeout": 30
            }
            self._table_prefix = ""

    @property
    def connection_params(self) -> Dict[str, Any]:
        """获取数据库连接参数"""
        return self._connection_params.copy()

    def get_table_name(self, table: str) -> str:
        """获取带前缀的表名"""
        return f"{self._table_prefix}{table}" if self._table_prefix else table


class ConnectionPool:
    """
    数据库连接池
    - 用 queue.Queue 存储空闲连接：有空闲直接拿，没空闲就阻塞等待
    - 用 threading.Lock 保护总连接数：防止多线程同时创建连接导致超限
    """

    def __init__(self, config: DatabaseConfig, max_connections: int = 10):
        self.config = config              # 数据库配置实例
        self._max_size = max_connections    # 最大连接数，默认 10 个

        # 存放空闲连接的队列（线程安全的"书架"）
        self._idle = Queue(maxsize=max_connections)

        self._total = 0                      # 当前总共创建了多少个连接
        self._lock = Lock()                  # 保护总连接数的锁

    def _create_connection(self) -> pymysql.Connection:
        """创建一个新的数据库连接"""
        conn = pymysql.connect(**self.config.connection_params)
        logger.debug("[database] 创建新数据库连接")
        return conn

    def get_connection(self) -> pymysql.Connection:
        """
        从池中获取一个数据库连接
        三步走：
          1. 先看空闲队列里有没有 → 有就直接拿
          2. 队列空了，看能不能新建 → 没超上限就新建
          3. 超上限了，阻塞等待别人归还
        """
        # 第一步：从空闲队列拿
        try:
            conn = self._idle.get_nowait()    # 从空闲队列拿连接
            return conn
        except queue.Empty:
            pass

        # 第二步：队列空了，要不要新建？
        with self._lock:         # 用锁保护总连接数
            if self._total < self._max_size:      # 没超上限
                self._total += 1                  # 新增连接数+1
                return self._create_connection()    # 创建新连接

        # 第三步：达到上限了，阻塞等待别人归还
        try:
            conn = self._idle.get(timeout=30)    # get(timeout=30) 等待 30 秒，有空闲连接就返回
            return conn
        except queue.Empty:
            raise pymysql.Error(
                f"数据库连接池已满（最大 {self._max_size} 个），等待 30 秒后仍然没有空闲连接"
            )

    def return_connection(self, conn: pymysql.Connection):
        """
        将连接归还到池中
        池没满 → 放回队列，唤醒正在等待 get() 的线程
        池满了 → 关闭多余连接，防止连接泄漏
        """
        try:
            self._idle.put_nowait(conn)          # 放回空闲队列
        except queue.Full:                  # 队列已满，关闭连接
            with self._lock:
                self._total -= 1
            conn.close()
            logger.debug("[database] 连接池已满，关闭多余连接")

    @contextmanager
    def get_cursor(self, commit: bool = True):
        """
        上下文管理器：从池中借连接 → 使用 → 归还到池中
        用 with 语句调用，自动处理提交、回滚、归还
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()        # 从池中获取连接
            cursor = conn.cursor(DictCursor)    # 创建游标
            yield cursor                        # 返回游标给调用方使用（函数暂停，调用方用完后再回来继续执行）
            if commit:
                # 提交事务（将所有操作写入数据库）
                conn.commit()
        except pymysql.Error as e:
            if conn:
                # 回滚事务（撤销之前未提交的操作）
                conn.rollback()
            logger.error(f"[database] 数据库操作失败: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                # 关键区别：不再 close()，而是归还到池中
                self.return_connection(conn)


db_config = DatabaseConfig()            # 数据库配置单例
connection_pool = ConnectionPool(db_config)  # 数据库连接池单例


def get_db_cursor(commit: bool = True):
    """获取数据库游标的便捷函数"""
    return connection_pool.get_cursor(commit=commit)


class DatabaseInitializer:
    """数据库初始化器"""

    @staticmethod
    def init_database():
        """初始化数据库表结构"""
        init_sql = """
        -- 用户表
        CREATE TABLE IF NOT EXISTS `{table_users}` (
            `id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '用户ID',
            `username` VARCHAR(50) NOT NULL UNIQUE COMMENT '用户名',
            `password_hash` VARCHAR(255) NOT NULL COMMENT '密码哈希',
            `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
            INDEX `idx_username` (`username`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表';

        -- 用户记忆表
        CREATE TABLE IF NOT EXISTS `{table_memory}` (
            `id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '记忆ID',
            `user_id` INT NOT NULL COMMENT '用户ID（关联users表）',
            `username` VARCHAR(50) NOT NULL COMMENT '用户名（冗余存储，便于查询）',
            `device_model` VARCHAR(100) DEFAULT NULL COMMENT '设备型号',
            `history` JSON DEFAULT NULL COMMENT '历史对话记录（JSON格式）',
            `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
            UNIQUE KEY `uk_user_id` (`user_id`),
            INDEX `idx_username` (`username`),
            FOREIGN KEY (`user_id`) REFERENCES `{table_users}`(`id`) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户记忆表';
        """

        try:
            with get_db_cursor() as cursor:                      # 获取数据库游标
                for statement in init_sql.strip().split(';'):    # 按分号分隔每个SQL语句
                    statement = statement.strip()                # 移除首尾空格
                    if statement:
                        sql = statement.format(                  # 替换占位符为实际表名
                            table_users=db_config.get_table_name("users"),      # 替换用户表名
                            table_memory=db_config.get_table_name("memory")     # 替换用户记忆表名
                        )
                        cursor.execute(sql)                     # 执行SQL语句
            logger.info("[database] 数据库表初始化完成")
        except pymysql.Error as e:
            logger.error(f"[database] 数据库表初始化失败: {e}")
            raise

    @staticmethod      # 静态方法，无需实例化即可调用
    def check_connection() -> bool:
        """检查数据库连接是否正常"""
        try:
            with get_db_cursor() as cursor:         # 获取数据库游标
                cursor.execute("SELECT 1")          # 检查数据库连接是否正常
            logger.info("[database] 数据库连接正常")
            return True
        except Exception as e:
            logger.error(f"[database] 数据库连接检查失败: {e}")
            return False
