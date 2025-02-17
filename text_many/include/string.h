#ifndef STRING_H
#define STRING_H

void make_array();     //数组生成函数
void string_write(const char* msg);     //字符串写入函数
void PrintString(std::string& str);    //表示传入的是一个字符串类型的引用，而不是复制

#endif

// string names = string("GM") + " is a";   //定义一个字符串类型的变量
// const char* name = "GM";   //定义一个字符串类型的常量     修改字符串的话就需要添加const
// char name1[] = "GM";   //定义一个字符串类型的数组
// name1[0] = 'b';   //修改字符串中的某个字符
// cout << name1 << endl;
// cout << names << endl;  

// names += " good boy";   //字符串的拼接
// cout << names << endl;   

// int contains = names.find("boy");//查找字符串中是否包含某个字符串
// bool contains1 = names.find("boy") != string::npos;  //查找字符串中是否包含某个字符串  npos是一个常量，表示查找不到的情况

// cout << contains << endl;   //输出查找到的字符串的位置
// cout << contains1 << endl;    //输出字符串 
//PrintString(names);   //输出字符串