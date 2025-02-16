#include <iostream>        //包含头文件iostream
#include "swap.h"
#include "max.h"
#include "logs.h"

using namespace std;       //使用标准命名空间

//定义全局变量
int val1 = 1;
int val2 = 2;
float  val3 = 10.3f;
double val4 = 10.3;
bool val5 = true;
int the_max_val;
int the_min_val;

int main (int argc, char **argv)
{
    cout << "Hello, World!" << endl;   //输出Hello, World!
    
    //logs_var_before();     //打印出来先前变量的相关信息

    // swap(val1, val2);   //交换两个变量的值  

    // logs_var_aftre();      //打印出来后来变量的相关信息

    //cin.get();       //输入函数

    // LOG log;
    // log.SetLevel(LogLevelInfo);
    // log.Error("Hello");
    // log.Warn("Hello");  
    // log.Info("Hello");  
    return 0;
}