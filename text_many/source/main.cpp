#include <iostream>        //����ͷ�ļ�iostream
#include <array>           //����ͷ�ļ�array
#include <string>          //����ͷ�ļ�string
#include <stdlib.h>        //����ͷ�ļ�stdlib
#include <memory>          //����ͷ�ļ�memory   //����ָ��
#include "swap.h"
#include "max.h"
#include "logs.h"
#include "string.h"
#include <cctype>           //����ͷ�ļ�cctype

// ȫ�ֱ���
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
using namespace std;       //ʹ�ñ�׼�����ռ�

int main (int argc, char **argv)
{

    {
        std::unique_ptr<Entity_ptr> entity = std::make_unique<Entity_ptr>();    //����һ������ָ��
        
        entity->Print();
    }

    std::shared_ptr<Entity_ptr> sharedEntity2;    //����һ������ָ�븴��������
    {
        std::shared_ptr<Entity_ptr> sharedEntity = std::make_shared<Entity_ptr>();    //����һ������ָ��
        sharedEntity2 = sharedEntity;    //����������

    }
    std::cin.get();
    return 0;
}