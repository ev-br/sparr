#ifndef SP_MAP_OPS_H
#define SP_MAP_OPS_H
#include<cmath>

/************************************
 * Elementary elementwise operations
 ************************************/


namespace sparray{

typedef unsigned char npy_bool_t;   // XXX

template<typename T>
T add(T x, T y){
    return x + y;
}

template<typename T>
T mul(T x, T y){
    return x*y;
}

template<typename T>
T sub(T x, T y){
    return x - y;
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
