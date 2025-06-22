#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名: chat_with_memory.py
功能描述: 
    这是一个基于 OpenAI API 的智能对话程序，具有记忆功能（保存对话历史）。
    程序可以与 AI 模型进行多轮对话，并在对话过程中保持上下文的连贯性。

主要功能:
    1. 支持与 AI 模型进行自然语言对话
    2. 维护完整的对话历史，使 AI 能够理解上下文
    3. 使用流式响应，实时显示 AI 回复
    4. 支持自定义 API 设置（通过环境变量配置）

使用方法:
    1. 设置环境变量：
       - LLM_API_KEY: OpenAI API 密钥
       - LLM_BASE_URL: API 基础 URL
    2. 导入并使用 chat_with_memory 函数进行对话

依赖项:
    - openai: OpenAI Python 客户端
    - os: 用于读取环境变量

作者: [您的名字]
创建日期: [创建日期]
最后修改: [最后修改日期]
版本: 1.0.0
"""

from openai import OpenAI
import os

# 初始化一个空列表用于存储对话历史记录
# 每条记录都是一个字典，包含 role(角色)和 content(内容)
history = []

# 从环境变量中获取 OpenAI API 的密钥和基础 URL
LLM_API_KEY = os.getenv("LLM_API_KEY")  # 获取 API 密钥
LLM_BASE_URL = os.getenv("LLM_BASE_URL") # 获取 API 基础 URL

# 初始化 OpenAI 客户端
# 使用环境变量中的 API 密钥和基础 URL 创建客户端实例
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def chat_with_memory(input_content):
    """
    处理用户输入并与 AI 模型进行对话，同时保持对话历史记录
    
    参数:
        input_content (str): 用户输入的内容
        
    返回:
        str: AI 助手的回复内容
    """

    # 将用户的新输入添加到对话历史中
    # role='user' 表示这是用户的输入
    history.append({'role': 'user', 'content': input_content})

    # 使用完整的对话历史构建消息列表
    # 这样 AI 模型可以理解整个对话上下文
    messages = history

    # 调用 OpenAI API 创建对话补全
    # stream=True 启用流式响应，可以逐步获取 AI 的回复
    response = client.chat.completions.create(
        model='THUDM/glm-4-9b-chat',  # 注释掉的模型选项
        # model='Pro/THUDM/glm-4-9b-chat', # 使用 GLM-4 模型(收费)
        messages=messages,
        stream=True                       # 启用流式输出
    )

    # 打印助手回复的开头提示
    print(f'助手：', end='')
    
    # 用于存储完整的回复内容
    reply = ""
    
    # 逐块接收和处理 AI 的回复
    for chunk in response:
        # 获取当前块的文本内容
        chunk_text = chunk.choices[0].delta.content
        # 将当前块添加到完整回复中
        reply += chunk_text
        # 实时打印当前块内容，不换行
        print(chunk_text, end='')

    # 将 AI 的完整回复添加到对话历史中
    # role='assistant' 表示这是 AI 助手的回复
    history.append({'role': 'assistant', 'content': reply})

    return reply


def main():
    """
    主函数：启动循环对话模式
    用户可以持续与 AI 进行对话，输入 'quit' 退出程序
    """
    print("欢迎使用智能对话助手！")
    print("您可以与我进行对话，输入 'quit' 退出程序。")
    print("=" * 50)
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入您的问题: ")
            
            # 检查是否要退出
            if user_input.lower() == 'quit':
                print("\n感谢使用！再见！")
                break
            
            # 检查输入是否为空
            if not user_input.strip():
                print("请输入有效的内容。")
                continue
            
            # 调用对话函数
            chat_with_memory(user_input)
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，正在退出...")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            print("请重试或输入 'quit' 退出程序。")


def show_history():
    """
    显示对话历史记录
    """
    if not history:
        print("\n暂无对话历史记录。")
        return
    
    print("\n=== 对话历史记录 ===")
    for i, message in enumerate(history, 1):
        role = "我" if message['role'] == 'user' else "助手"
        content = message['content'][:100] + "..." if len(message['content']) > 100 else message['content']
        print(f"{i}. {role}: {content}")
    print("=" * 30)


def clear_history():
    """
    清空对话历史记录
    """
    global history
    history = []
    print("\n对话历史记录已清空。")


if __name__ == "__main__":
    main()