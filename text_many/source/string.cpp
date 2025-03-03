#include <iostream>
#include <array>
#include <string>
#include "string.h"

//默认作者
const char* name = "GM";   //定义一个字符串类型的常量     修改字符串的话就需要添加const

void make_array()     //数组生成函数
{
    std :: array<int, 5> arr = {1, 2, 3, 4, 5};   //定义一个长度为5的数组
    for (size_t i = 0; i < arr.size(); i++)  //边界检查
    {
        std::cout << arr[i] << std::endl;   //输出数组中的元素
    }
};
void string_write(const char* msg)
{
    name = msg;
    std::cout << name << std::endl;
}
void PrintString(std::string& str)    //表示传入的是一个字符串类型的引用，而不是复制
{
    std::cout << str << std::endl;
}

