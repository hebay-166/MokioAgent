#!/usr/bin/env python3
"""
一个简单的 Python 脚本 —— 欢迎程序
"""

def greet(name: str) -> str:
    """返回问候语"""
    return f"你好，{name}！欢迎使用 Python 代码生成工具。"

def main():
    """主函数"""
    user = "用户"
    message = greet(user)
    print(message)
    print("当前工作目录中的文件：")
    
    import os
    for f in os.listdir("."):
        print(f"  - {f}")

if __name__ == "__main__":
    main()