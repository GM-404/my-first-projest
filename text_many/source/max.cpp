#include "max.h"

int max(int &num1, int &num2)    //max功能
{
    int max;
    if (num1 > num2) {
        max = num1;
    } 
    else{
        max = num2;
    }
    return max;     //返回最大值
}