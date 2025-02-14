#include <iostream>
#include "swap.h"
#include "max.h"
#include "logs.h"

using namespace std;


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
    log("hello world");
    return 0;
}