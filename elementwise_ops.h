#ifndef SP_MAP_OPS_H
#define SP_MAP_OPS_H
#include<cmath>

/************************************
 * Elementary elementwise operations
 ************************************/


namespace sparray{

typedef unsigned char npy_bool_t;   // XXX

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


//////// COMPARISONS /////////////

template<typename S>
npy_bool_t equal(S x, S y){
    return x == y ? 1 : 0;
}

template<typename S>
npy_bool_t less_equal(S x, S y){
    return x <= y ? 1 : 0;
}

template<typename S>
npy_bool_t greater_equal(S x, S y){
    return x >= y ? 1 : 0;
}


template<typename S>
npy_bool_t not_equal(S x, S y){
    return x != y ? 1 : 0;
}

template<typename S>
npy_bool_t less(S x, S y){
    return x < y ? 1 : 0;
}

template<typename S>
npy_bool_t greater(S x, S y){
    return x > y ? 1 : 0;
}


} // namespace sparray

#endif
