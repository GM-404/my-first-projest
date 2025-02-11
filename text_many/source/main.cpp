#include <iostream>        //包含头文件iostream
#include "swap.h"
#include "max.h"

using namespace std;       //使用标准命名空间

int main (int argc, char **argv)
{
    int val1 = 10;
    int val2 = 20;
    int the_max_val;

    cout << "Before swap:" << endl;
    cout << "val1 = "<<val1 << endl;
    cout << "val2 = "<<val2 << endl;

    swap(val1, val2);

    cout << "After swap:" << endl;
    cout << "val1 = "<<val1 << endl;
    cout << "val2 = "<<val2 << endl;

    the_max_val = max(val1,val2);
    cout << "the_max_val =  "<<the_max_val << endl;
    
    return 0;
}