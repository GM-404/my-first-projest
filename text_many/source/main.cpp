#include <iostream>
#include "swap.h"
#include "max.h"
using namespace std;

/*
//不包头文件直接声明函数也是可以的
int max(int &num1, int &num2);    //max功能
*/
int min(int &num1, int &num2)    //min功能
{
    int min;
    if (num1 < num2) {
        min = num1;
    } 
    else{
        min = num2;
    }
    log("the_min_val",5);
    return min;     //返回最小值
}
 //打印函数
void log(const char *msg , int &val) 
{
    cout << msg << endl;     cout << val << endl;
}
int main (int argc, char **argv)
{
    int val1 = 10;
    int val2 = 20;
    int the_max_val;
    int the_min_val;

    cout << "Before swap:" << endl;
    cout << "val1 = "<<val1 << endl;
    cout << "val2 = "<<val2 << endl;

    swap(val1, val2);

    cout << "After swap:" << endl;
    cout << "val1 = "<<val1 << endl;
    cout << "val2 = "<<val2 << endl;

    the_max_val = max(val1,val2);
    cout << "the_max_val =  "<<the_max_val << endl;
    the_min_val = min(val1,val2);

    //  cin.get();       //输入函数
    return 0;
}