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

//s_speed =  s_Level > 5 ? 10 : 5;   //三目运算符
//std:: string rank = s_Level > 5 ? "GM" : "Cherno: ";   //三目运算符
//s_speed = s_level > 5 ? (s_leve>10 ? 10 : 5 ): 0;   //嵌套三目运算符

/*********************************在堆上创建一个对象****************************************** */
// class entity_one1
// {
// private:
//        int m_Score;
//        std::string m_Name;
// public:
//        entity_one1()
//               : m_Name("GM"), m_Score(0) {}    //构造函数初始化成员变量
//        entity_one1(const std::string& name)
//               : m_Name(name), m_Score(0) {}    //构造函数初始化变量
//        void Get_name() const
//        {
//             std::cout << m_Name << std::endl;
//        }
// };

// entity_one1* e1;    //在堆上创建一个对象
//     {
//         entity_one1* entity = new entity_one1("Cherno");
//         e1 = entity;
//         e1->Get_name();
//         delete entity;   //一定要删除堆上的对象
//     }

/********************************************new关键字*********************************************** */
// int a = 2 ;
// int* b  = new int[30];     //在堆上创建一个对象  new关键字的作用是在堆上创建一个对象，返回的是对象的地址
// cout << a << endl;
// cout << *b << endl;  
// entity_one* entity = new(b) entity_one("Cherno");   //在堆上创建一个对象 ,分配到b的地址上
// entity_one* entity1 = (entity_one*)malloc(sizeof(entity_one));   //与上一句等价，但是不会调用构造函数
// delete entity;   //一定要删除堆上的对象
// delete[] b;   //删除堆上的数组对象
/*****************************************41运算符及其重载************************************************ */
// struct Vector2
// {
//     float x, y;
//     Vector2(float x, float y)
//         : x(x), y(y) {}
//     Vector2 Add(const Vector2& other) const
//     {
//         return Vector2(x + other.x, y + other.y);
//     }
//     Vector2 multiply(const Vector2& other) const
//     {
//         return Vector2(x * other.x, y * other.y);
//     }
//     Vector2 operator + (const Vector2& other) const    //接线员操作
//     {
//         return Add(other);
//     }
//     Vector2 operator * (const Vector2& other) const
//     {
//         return multiply(other);
//     }
//     bool operator == (const Vector2& other) const
//     {
//         return x == other.x && y == other.y;
//     }
//     bool operator != (const Vector2& other) const
//     {
//         return !(*this == other);          //return !(operator == (other));
//     }
// };
//主函数
// Vector2 pos = {1.0f, 1.0f};
// Vector2 speed = {1.0f, 1.0f};
// Vector2 powerup = {1.2f, 1.2f};

// Vector2 result = pos.Add(speed);
// cout << result.x << ", " << result.y << endl;
// Vector2 result1 = pos + speed;                       //下面两句与上面等效
// cout << result1.x << ", " << result1.y << endl;
// Vector2 result2 = pos + speed * powerup;
// Vector2 results = pos.Add(speed.multiply(powerup));  //与上一句意思相同

/**************************************************************************************************** */

/**************************************************************************************************** */

/**************************************************************************************************** */

/**************************************************************************************************** */

/**************************************************************************************************** */

/**************************************************************************************************** */

/**************************************************************************************************** */

/**************************************************************************************************** */

/**************************************************************************************************** */

/**************************************************************************************************** */