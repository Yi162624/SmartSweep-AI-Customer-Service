"""
用户认证模块 - 提供用户注册、登录、密码验证等功能
支持SHA256密码哈希加密
"""
import hashlib
import secrets
from typing import Optional, Tuple
from utils.database import get_db_cursor, db_config
from utils.logger_handler import logger


class AuthService:
    """用户认证服务类"""

    def __init__(self):
        """初始化用户认证服务"""
        self._table_name = db_config.get_table_name("users")

    def _hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        对密码进行SHA256哈希加密

        Args:
            password: 明文密码
            salt: 盐值，如不提供则自动生成

        Returns:
            (哈希后的密码, 盐值)
        """
        if salt is None:
            # 生成随机盐值
            salt = secrets.token_hex(16)

        # 把密码和盐值拼接起来 进行哈希加密
        hash_obj = hashlib.sha256((password + salt).encode('utf-8'))
        # 返回哈希后的密码和盐值
        return hash_obj.hexdigest(), salt
        
    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """
        验证密码是否正确

        Args:
            password: 明文密码
            password_hash: 数据库中存储的密码哈希
            salt: 盐值

        Returns:
            密码是否匹配
        """
        # 对明文密码进行哈希加密
        computed_hash, _ = self._hash_password(password, salt)
        # 对比哈希结果是否匹配
        return computed_hash == password_hash

    def register(self, username: str, password: str) -> Tuple[bool, str]:
        """
        用户注册

        Args:
            username: 用户名
            password: 明文密码

        Returns:
            (是否成功, 消息)
        """
        if not username or len(username) < 2:
            return False, "用户名至少需要2个字符"

        if not password or len(password) < 6:
            return False, "密码至少需要6个字符"

        try:
            with get_db_cursor() as cursor:   # 获取数据库游标
                cursor.execute(
                    f"SELECT `id` FROM `{self._table_name}` WHERE `username` = %s",
                    (username,)
                )
                if cursor.fetchone():      # 如果查询结果不为空，说明用户名已存在
                    return False, "用户名已存在"

                password_hash, salt = self._hash_password(password)     # 对密码进行哈希加密

                cursor.execute(
                    # 插入用户信息到数据库表
                    f"INSERT INTO `{self._table_name}` (`username`, `password_hash`) VALUES (%s, %s)",
                    (username, f"{password_hash}:{salt}")
                )

            logger.info(f"[auth] 用户注册成功: {username}")
            return True, "注册成功"

        except Exception as e:
            logger.error(f"[auth] 用户注册失败: {e}")
            return False, f"注册失败: {str(e)}"

    def login(self, username: str, password: str) -> Tuple[bool, str]:
        """
        用户登录验证

        Args:
            username: 用户名
            password: 明文密码

        Returns:
            (是否成功, 消息)
        """
        try:
            with get_db_cursor() as cursor:
                cursor.execute(
                    f"SELECT `id`, `password_hash` FROM `{self._table_name}` WHERE `username` = %s",
                    (username,)
                )
                result = cursor.fetchone()     # 把 SQL 查询结果取出来

                if not result:
                    return False, "用户名或密码错误"

                password_hash, salt = result['password_hash'].split(':')     # 从数据库中获取的密码哈希和盐值

                if self._verify_password(password, password_hash, salt):     # 验证密码是否正确
                    logger.info(f"[auth] 用户登录成功: {username}")
                    return True, "登录成功"
                else:
                    logger.warning(f"[auth] 用户登录失败（密码错误）: {username}")
                    return False, "用户名或密码错误"

        except Exception as e:
            logger.error(f"[auth] 用户登录异常: {e}")
            return False, f"登录异常: {str(e)}"

    def change_password(self, username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        """
        修改密码

        Args:
            username: 用户名
            old_password: 旧密码
            new_password: 新密码

        Returns:
            (是否成功, 消息)
        """
        if not new_password or len(new_password) < 6:
            return False, "新密码至少需要6个字符"

        success, msg = self.login(username, old_password)      # 验证旧密码是否正确
        if not success:
            return False, "旧密码验证失败"

        try:
            with get_db_cursor() as cursor:
                password_hash, salt = self._hash_password(new_password)     # 对新密码进行哈希加密
                cursor.execute(
                    # 更新用户密码
                    f"UPDATE `{self._table_name}` SET `password_hash` = %s WHERE `username` = %s",
                    (f"{password_hash}:{salt}", username)     
                )

            logger.info(f"[auth] 用户修改密码成功: {username}")
            return True, "密码修改成功"

        except Exception as e:
            logger.error(f"[auth] 修改密码失败: {e}")
            return False, f"修改密码失败: {str(e)}"

    def user_exists(self, username: str) -> bool:
        """
        检查用户是否存在

        Args:
            username: 用户名

        Returns:
            用户是否存在
        """
        with get_db_cursor() as cursor:
            cursor.execute(
                f"SELECT `id` FROM `{self._table_name}` WHERE `username` = %s",
                (username,)
            )
            return cursor.fetchone() is not None      # 如果查询结果不为空，返回 True，否则 False


auth_service = AuthService()     # 创建 AuthService 实例，用于处理用户认证相关操作


def register(username: str, password: str) -> Tuple[bool, str]:
    """用户注册（兼容接口）"""
    return auth_service.register(username, password)


def login(username: str, password: str) -> Tuple[bool, str]:
    """用户登录（兼容接口）"""
    return auth_service.login(username, password)


def change_password(username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
    """修改密码（兼容接口）"""
    return auth_service.change_password(username, old_password, new_password)


def user_exists(username: str) -> bool:
    """检查用户是否存在（兼容接口）"""
    return auth_service.user_exists(username)
