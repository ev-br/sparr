#include<iostream>
#include<string.h>
#include"fixed_cap.h"

int main(){

    using namespace sparray;

    fixed_capacity<> v;
    FC_ELEM(v, 0) = -1;
    FC_ELEM(v, 1) = 1;
    std::cout << v << "\n";

    FC_ELEM(v, 0+1) = -101;
    std::cout << v << "\n";


    /*********************/

    // operator<() comparison
    fixed_capacity<> vv;
    FC_ELEM(vv, 1) = -101;

    std::cout << "\nas compared to " << v << "\n";

    for(int val=-2; val < 3; ++val){
        FC_ELEM(vv, 0) = val;
        std::cout << vv << " : " << bool_outp(fixed_capacity_cmp<>()(v, vv)) << "\n";
    }

    /*********************/

    // copy ctor & copy assignment op

    fixed_capacity<int, 2> vvv(v);
    std::cout<< "copy ctor(v): "<< vvv << "\n";

    fixed_capacity<int, 2> v4 = v;
    std::cout<< "op=(v): "<< vvv << "\n";


    /*********************/

    // C array ctor: intentionally left out. XXX: restore?

    int a[2] = {3, 4};
    fixed_capacity<int, 2> v5(a);
    std::cout<< "c arr ctor: "<< v5 << "\n";

    fixed_capacity<int, 2> v6;
    memcpy(FC_ELEMS(v6), a, sizeof(a));

    std::cout << "w/ memcpy:" <<  v6 << "\n";
    std::cout << "sizeof v6: " << sizeof(v6) << "\n";
}
