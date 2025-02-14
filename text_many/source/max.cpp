#include "max.h"

int max(int &num1, int &num2)    //max功能
{
    int max;
    bool flag_val1;
    if (num1 > num2) {
        max = num1;
        flag_val1 = true;
    } 
    else{
        max = num2;
        flag_val1 = false;
    }
    return max,flag_val1;     //返回最大值
}