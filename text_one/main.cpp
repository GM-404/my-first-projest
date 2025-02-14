#include <iostream>

using namespace std;

void swap (int &a,int &b)       //交换功能
{
    int temp;
    temp = a;
    a = b;
    b = temp;
}

int max(int &num1, int &num2)    //选择最大值功能
{
    int max;
    if (num1 > num2) 
    {
        max = num1;
    } 
    else
    {
        max = num2;
    }
    return max;     //返回最大值
}

//常量区域
 int max_one;

int main (int argc, char **argv)
{
    int val1 = 10;
    int val2 = 20;
    cout << "Before swap:" << endl;
    cout << "val1 = "<<val1 << endl;
    cout << "val2 = "<<val2 << endl;
    // 输入函数  cin.get();

    swap(val1, val2);
    max_one  = max(val1, val2);

    cout << "After swap:" << endl;
    cout << "val1 = "<<val1 << endl;
    cout << "val2 = "<<val2 << endl;
    cout << "max =  " << max_one << endl;
    return 0;
}