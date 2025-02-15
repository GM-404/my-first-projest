#include <iostream>        //包含头文件iostream
#include "swap.h"
#include "max.h"
#include "logs.h"

using namespace std;

int main (int argc, char **argv)
{
    Player player;
    player.x = 1;
    player.y = 1;
    player.speed = 1;

    logs_var_before();     //打印出来先前变量的相关信息

    swap(val1, val2);   //交换两个变量的值  

    logs_var_aftre();      //打印出来后来变量的相关信息

    //cin.get();       //输入函数
    return 0;
}