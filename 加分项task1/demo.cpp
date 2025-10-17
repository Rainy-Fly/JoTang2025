#include"autodiff/reverse/var.hpp"
#include"autodiff/forward/utils/gradient.hpp"
#include "autodiff/reverse/var.hpp"
#include "autodiff/reverse/var/eigen.hpp"
#include"Eigen/Dense"
#include<iostream>
using namespace Eigen;
using namespace std;
using autodiff::var;
using autodiff::MatrixXvar;
using autodiff::VectorXvar;
using autodiff::detail::jacobian;
using autodiff::detail::wrt;
using autodiff::detail::at;
var func(MatrixXvar &X,MatrixXvar &W,MatrixXvar M,VectorXvar &b){
    VectorXvar out= (X*W)*M+b;  
    return out.sum();
}

int main(){
    MatrixXvar X(10,3);
    MatrixXvar W(3,2);
    MatrixXvar M(2,1);
    VectorXvar b(10);

    W.setRandom();
    b.setRandom();
    X.setRandom();
    M.setRandom();
    MatrixXd dW;
    dW=jacobian(func,wrt(W),at(X,W,M,b));
    cout<<dW<<endl;
    return 0;
}

