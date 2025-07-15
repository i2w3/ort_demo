import time
import logging
from pathlib import Path
from typing import Optional, Union

# 添加颜色支持的格式化器类
class ColoredFormatter(logging.Formatter):
    """自定义的彩色日志格式化器"""
    
    # 定义ANSI颜色代码
    COLORS = {
        'RESET': '\033[0m',
        'TIME': '\033[36m',  # 青色用于时间
        'DEBUG': '\033[94m',  # 蓝色
        'INFO': '\033[92m',   # 绿色
        'WARNING': '\033[93m', # 黄色
        'ERROR': '\033[91m',   # 红色
        'CRITICAL': '\033[41m\033[97m', # 白色文字红色背景
    }
    
    def format(self, record):
        # 保存原始的格式化字符串
        orig_format = self._style._fmt
        
        # 根据日志级别设置颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            # 为时间添加特定颜色，为级别添加对应颜色
            colored_fmt = (
                f"{self.COLORS['TIME']}%(asctime)s{self.COLORS['RESET']} - "
                f"%(name)s - {self.COLORS[levelname]}%(levelname)s{self.COLORS['RESET']} - "
                f"%(message)s"
            )
            self._style._fmt = colored_fmt
            
        # 调用原始的format方法
        result = super().format(record)
        
        # 恢复原始格式
        self._style._fmt = orig_format
        
        return result

class CustomLogger:
    # 类变量，所有实例共享
    LOCAL_TIME:str = time.strftime("%Y-%m-%dT%H%M%S", time.localtime())
    DEFAULT_LOGGING_PATH:Path = Path(f"./logs/{LOCAL_TIME}/{LOCAL_TIME}.log")
    
    def __init__(self, logger_level: Union[str, int] = "debug", 
                       logging_path: Optional[Path] = None, 
                       file_logging: bool = True,
                       logger_name: Optional[str] = None
                       ) -> None:
        """
        初始化日志记录器。
        :param logger_level: 日志级别，默认为 "debug"。
        :param logging_path: 日志文件保存路径，默认为 None。
        :param file_logging: 是否将日志写入文件，默认为 True。
        :param logger_name: 日志器名称，默认为 None。
        """
        
        self.logger = logging.getLogger(logger_name or __name__)
        self.logger.setLevel(self._get_log_level(logger_level))
        self.logger.propagate = False  # 避免日志重复输出
        
        # 清除已有的处理器，防止重复添加
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # 创建标准格式化器（用于文件日志）
        standard_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # 创建彩色格式化器（用于控制台输出）
        colored_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(colored_formatter)  # 使用彩色格式化器
        self.logger.addHandler(ch)

        # 文件处理器
        if file_logging:
            # 使用传入的路径或类变量
            self.logging_path = logging_path if logging_path is not None else CustomLogger.DEFAULT_LOGGING_PATH
            Path.mkdir(self.logging_path.parent, parents=True, exist_ok=True)

            fh = logging.FileHandler(self.logging_path, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(standard_formatter)  # 文件使用标准格式化器
            self.logger.addHandler(fh)
        else:
            self.logging_path = None
    
    def _get_log_level(self, level: Union[str, int]) -> int:
        """将字符串日志级别转换为logging模块的整数级别"""
        if isinstance(level, int):
            return level
            
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }
        return level_map.get(level.lower(), logging.DEBUG)
    
    # 快速调用 self.logger
    def debug(self, msg, *args, **kwargs):
        """记录DEBUG级别日志"""
        self.logger.debug(msg, *args, **kwargs)
        
    def info(self, msg, *args, **kwargs):
        """记录INFO级别日志"""
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        """记录WARNING级别日志"""
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        """记录ERROR级别日志"""
        self.logger.error(msg, *args, **kwargs)
        
    def critical(self, msg, *args, **kwargs):
        """记录CRITICAL级别日志"""
        self.logger.critical(msg, *args, **kwargs)
    
    def set_level(self, level: Union[str, int]):
        """动态设置日志级别"""
        self.logger.setLevel(self._get_log_level(level))
        
    def get_logging_path(self) -> str:
        """获取 logging_path """
        if self.logging_path:
            return str(self.logging_path)
        else:
            raise ValueError("Logging path is not set.")