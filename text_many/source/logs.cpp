 #include <iostream>
 #include "logs.h"

 using namespace std;
 
 //定义全局变量
extern int val1;
extern int val2;
extern float  val3;
extern double val4;
extern bool val5;
extern int the_max_val;
extern int the_min_val;

 //打印提示函数
 void log(const char *msg) 
 {
     cout << msg << endl;    
 }
 //先前变量信息
 void logs_var_before()
 {
    cout << "Before swap:" << endl;
    cout << "val1 = "<<val1 << endl;
    cout << "val2 = "<<val2 << endl;
    cout << "val3 = "<<val3 << endl;
    cout << "val4 = "<<val4 << endl;
    cout << "val5 = "<<val5 << endl;
    cout << "val1 val2 size:" << sizeof(val1) << endl;
    cout << "val3  size:" << sizeof(val3) << endl;
    cout << "val4  size:" << sizeof(val4) << endl;
    cout << "val5  size:" << sizeof(val5) << endl;
 }
 //后来变量信息
void logs_var_aftre()
{
    int flag_val;
    cout << "After swap:" << endl;
    cout << "val1 = "<<val1 << endl;
    cout << "val2 = "<<val2 << endl;

    the_max_val,flag_val = max(val1,val2);

    if (flag_val)    //判断最大值是哪个
    {
        cout << "the_max_val is val1  "<<the_max_val << endl;
    }
    else
    {
        cout << "the_max_val is val2  "<<the_max_val << endl;
    }
}