#include<iostream>
#include"sp_map.h"
#include"fixed_cap.h"
#include"elementwise_ops.h"

#define NDIM 2


int main(){

    using namespace sparray;

    /********************/

    {
    // ctors
        map_array_t<double> ma;
        std::cout << ma.count_nonzero() << "  "<< ma.fill_value();
        std::cout << "  "<< ma.shape()<< "\n";

        map_array_t<double> ma1(ma);
        std::cout << ma1.count_nonzero() << "  "<< ma1.fill_value();
        std::cout << "  " << ma1.shape()<< "\n";

        std::cout << "ndim: " << ma1.ndim() << "\n";
    }

    /*******************/

    // get/set_one
    {
        std::cout<< "\n*** get/set_one:\n";

        single_index_type arr[NDIM] = {2, 3};
        fixed_capacity<> vect(arr);
        map_array_t<double> ma;

        ma.set_one(vect, 8);
        std::cout << vect << " " << ma.get_one(vect) << "  ";
        std::cout << "shape = " << ma.shape()<< "\n" ;

        ma.set_one(vect, 111);
        std::cout << vect << " " << ma.get_one(vect) << "  ";
        std::cout << "shape = " << ma.shape()<< "\n" ;

        // test overloads on the C array
        ma.set_one(arr, -101);
        std::cout << "w/array: " << ma.get_one(arr) << "  ";
        std::cout << " shape = " << ma.shape() << "\n";

        // check that shape gets updated
        single_index_type a2[NDIM] = {5, 6};
        ma.set_one(a2, 222);
        std::cout << ma;
    }

    /*******************/

    // in-place elementwise operations

    {
        std::cout <<"\n\n******\n";
        map_array_t<double> ma;
        single_index_type a[NDIM] = {3, 4};
        ma.set_one(a, 101.);

        a[0] += 1;
        ma.set_one(a, -2);

        std::cout << ma << "\n";

        ma.inplace_unary_op(linear_unary_op<double>, 1., 1.);
        std::cout << ma << "\n";

        // check that there's no aliasing
        map_array_t<double> ma2(ma);
        ma2.inplace_unary_op(power_unary_op<double>, 1., 2.);

        std::cout << " res = " << ma2 << "\n";
        std::cout << " orig= " << ma << "\n";
    }

    /*******************/

    // todense
    {
        map_array_t<double> ma;
        single_index_type a[NDIM];
        size_t n=3, m=5;

        double value = 0.;
        for (size_t i=0; i<n; ++i){
            a[0] = i;
            for (size_t j=0; j<m; ++j){
                a[1] = j;
                ma.set_one(a, value);
                value += 1.;
            }
        }
        std::cout << "\n*****************\n";
        std::cout << ma << "\n";
    }

}
