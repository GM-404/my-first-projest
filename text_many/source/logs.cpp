#include <iostream>
#include "logs.h"

// 打印提示函数
void log(const char *msg) 
{
    std::cout << msg << std::endl;    
}

// 打印 int 类型的变量
void logs_int_mes(int var)  
{
    std::cout << var << std::endl;
}

// 先前变量信息
void logs_var_before()
{
    extern int val1;
    extern int val2;
    extern float val3;
    extern double val4;
    extern bool val5;

    std::cout << "Before swap:" << std::endl;
    std::cout << "val1 = " << val1 << std::endl;
    std::cout << "val2 = " << val2 << std::endl;
    std::cout << "val3 = " << val3 << std::endl;
    std::cout << "val4 = " << val4 << std::endl;
    std::cout << "val5 = " << val5 << std::endl;
    std::cout << "val1 val2 size: " << sizeof(val1) << std::endl;
    std::cout << "val3 size: " << sizeof(val3) << std::endl;
    std::cout << "val4 size: " << sizeof(val4) << std::endl;
    std::cout << "val5 size: " << sizeof(val5) << std::endl;
}

// 设置日志级别
void LOG::SetLevel(int level) {
    m_LogLevel = level;
}

// 输出错误信息
void LOG::Error(const char* message) {
    if (m_LogLevel >= LogLevelError)
        std::cout << "[ERROR]: " << message << std::endl;
}

// 输出警告信息
void LOG::Warn(const char* message) {
    if (m_LogLevel >= LogLevelWarning)
        std::cout << "[WARNING]: " << message << std::endl;
}

// 输出一般信息
void LOG::Info(const char* message) {
    if (m_LogLevel >= LogLevelInfo)
        std::cout << "[INFO]: " << message << std::endl;
}