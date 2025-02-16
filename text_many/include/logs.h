#ifndef LOGS_H
#define LOGS_H

// 打印提示函数
void log(const char *msg); 
// 打印 int 类型的变量
void logs_int_mes(int var);  
// 先前变量信息
void logs_var_before();    
// 后来变量信息
void logs_var_aftre();     

// 定义日志级别常量
enum   InfoLevel : unsigned int
{
    InfoLevelError = 0,
    InfoLevelWarning = 1,
    InfoLevelInfo = 2
};
// const int LogLevelError = 0;
// const int LogLevelWarning = 1;
// const int LogLevelInfo = 2;

// 定义了一个类 LOG 用来输出日志信息
class LOG {
private:
    int m_LogLevel;  // 声明成员变量

public:
    void SetLevel(int level);  // 设置日志级别
    void Error(const char* message);  // 输出错误信息
    void Warn(const char* message);  // 输出警告信息
    void Info(const char* message);  // 输出一般信息
};

#endif