#ifndef LOGS_H
#define LOGS_H

//private:   //私有成员  只能自己或者友元函数访问
//protected:   //保护成员  只能自己或者子类访问
//public:   //公有成员  谁都可以访问
 
//以*为分界，const在左边就是内容不能变，在右边是地址不能变  （靠近谁锁谁）
//const int* a;   //指针的内容不能变    也就是不用*a = 2;这样的操作    指针，锁定！
//int* const a;   //指针的地址不能变    也就是不用a = &b;这样的操作    锁定地址，
// 打印提示函数
void log(const char *msg); 
// 打印 int 类型的变量
void logs_int_mes(int var);  
// 先前变量信息
void logs_var_before();    
// 后来变量信息
void logs_var_aftre();     

// 定义日志级别常量
enum   InfoLevel : unsigned int
{
    LogLevelError = 0,
    LogLevelWarning = 1,
    LogLevelInfo = 2
};
// const int LogLevelError = 0;
// const int LogLevelWarning = 1;
// const int LogLevelInfo = 2;

// 定义了一个类 LOG 用来输出日志信息
class LOG {
private:
    int m_LogLevel;  // 声明成员变量

public:
    void SetLevel(int level);  // 设置日志级别
    void Error(const char* message);  // 输出错误信息
    void Warn(const char* message);  // 输出警告信息
    void Info(const char* message);  // 输出一般信息
};
//定义了一个类来声明构造函数和析构函数
class Entity{   
public:
    int x, y;
    Entity() =delete;   //删除默认构造函数   
    Entity(int val1,int val2);   //构造函数 类似于python中的__init__方法
    ~Entity();   //析构函数
};
//定义一个类来说明可以构造两个构造函数
class entity_one
{
private:
       int m_Score;
       std::string m_Name;
public:
       entity_one()
              : m_Name("GM"), m_Score(0) {}    //构造函数初始化成员变量
       entity_one(const std::string& name)
              : m_Name(name), m_Score(0) {}    //构造函数初始化变量
       void Get_name() const
       {
            std::cout << m_Name << std::endl;
       }
};
//主函数如下
// entity_one e1;
// entity_one e2("Cherno");
// 输出两个人名
// e1.Get_name();
// e2.Get_name();

//定义两个类来声明父类和子类
class father
{
public:
    int x, y;
    void print();
    void print_father_val();
};
class son : public father    //声明子类继承父类  认爹
{
public:
    const char* name;        //儿子特有的指针类型
    void print();
    void print_son_name(const char* name);
};
//主函数书写如下
/*子类调用子类函数，或者父类不同名函数时直接调用，子类调用父类同名函数时压要加入.father::<函数名>*/
// son  player;   //声明一个子类对象
// player.print();   //调用子类的函数
// player.father::print();   //调用父类的函数
// player.print_father_val();

//定义两个类来声明父类和子类中的虚函数(即重写函数)   
//纯虚函数是在基类中声明的虚函数，但是在基类中没有定义，而是强制留给派生类去实现。
class father_two
{
public:
    //虚函数的作用是允许在派生类中对其进行重写，
    virtual std::string GetName();   //虚拟的，返回字符串类型的Getname函数,这个函数返回father_two
    //virtual std::string GetName() const = 0;   //纯虚函数
};
class son_two : public father_two
{
private:
    std::string m_name;
public:
    son_two(const std::string& name)   //构造函数  接受一个 const std::string& 类型的参数 name   其中&表示引用
    : m_name(name){};                 //把传入的 name 参数值赋给 m_name以完成初始化，
    
    std::string GetName();   //重写了基类 father_two 中的 GetName 函数。
    //使用 override 关键字明确表示这是对基类虚函数的重写，这样可以让编译器进行检查，如果没有正确重写基类的虚函数，编译器会报错。

};
//主函数书写如下
// son_two son("GM");   //声明一个子类对象 son，传入一个字符串参数 "Cherno"。
// father_two* entity = &son;   //将子类对象 son 的地址赋值给基类指针 entity。
// std::cout << entity->GetName() << std::endl;   //调用基类指针 entity 的 GetName 函数，输出 "GM"。

//const的类里面的锁定使用   这个时候只有定义成 const entity e;这个才能使用 e.GetName(); 可以使用mutable关键字来改变这一情况
class entity 
{
private:
    std :: string m_Name;
    mutable int a  = 0 ;  //可改变的
public:
    const std::string& GetName() const 
    {
        return  m_Name;
    }
};
#endif