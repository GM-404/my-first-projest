#include <iostream>        //包含头文件iostream
#include <array>           //包含头文件array
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
    son_two son("GM");   //声明一个子类对象 son，传入一个字符串参数 "Cherno"。
    father_two* entity = &son;   //将子类对象 son 的地址赋值给基类指针 entity。
    std::cout << entity->GetName() << std::endl;   //调用基类指针 entity 的 GetName 函数，输出 "GM"。
    

    std :: array<int, 5> arr = {1, 2, 3, 4, 5};   //定义一个长度为5的数组
    for (size_t i = 0; i < arr.size(); i++)  //边界检查
    {
        std::cout << arr[i] << std::endl;   //输出数组中的元素
    }
    
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