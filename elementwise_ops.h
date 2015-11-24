#ifndef SP_MAP_OPS_H
#define SP_MAP_OPS_H
#include<cmath>

/************************************
 * Elementary elementwise operations
 ************************************/


namespace sparray{

template<typename T>
T linear_unary_op(T x, T a, T b){
    return a*x + b;
}


template<typename T>
T power_unary_op(T x, T a, T b){
    return a * pow(x, b);
}


template<typename T>
T linear_binary_op(T x, T y, T a, T b){
    return a*x + y*b;
}


template<typename T>
T mul_binary_op(T x, T y, T a, T b){
    return a*x*y + b;
}


} // namespace sparray

#endif
