#include <iostream>        //包含头文件iostream
#include <array>           //包含头文件array
#include <string>          //包含头文件string
#include <stdlib.h>        //包含头文件stdlib
#include <memory>          //包含头文件memory   //智能指针
#include "swap.h"
#include "max.h"
#include "logs.h"
#include "string.h"
#include <cctype>           //包含头文件cctype

// 全局变量
int val1;
int val2;
float val3;
double val4;
bool val5;

class  Entity_ptr
{
public:
    Entity_ptr()
    {
        std::cout << "Created Entity!" << std::endl;
    }
    ~Entity_ptr()
    {
        std::cout << "Destroyed Entity!" << std::endl;
    }
    void Print()
    {
        std::cout << "Hello!" << std::endl;
    }
};
using namespace std;       //使用标准命名空间

int main (int argc, char **argv)
{

    {
        std::unique_ptr<Entity_ptr> entity = std::make_unique<Entity_ptr>();    //创建一个智能指针
        
        entity->Print();
    }

    std::shared_ptr<Entity_ptr> sharedEntity2;    //创建一个共享指针复制上内容
    {
        std::shared_ptr<Entity_ptr> sharedEntity = std::make_shared<Entity_ptr>();    //创建一个共享指针
        sharedEntity2 = sharedEntity;    //复制上内容

    }
    std::cin.get();
    return 0;
}