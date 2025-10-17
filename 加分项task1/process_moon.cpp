#include "autodiff/reverse/var.hpp"
#include "autodiff/reverse/var/eigen.hpp"
#include<Eigen/Dense>
#include<vector>
#include<sstream>
#include<iostream>
#include<fstream>
#include<string>
#include<random>
#include <unordered_map>
#include<unordered_set>
#include <tuple>
using namespace std;
using autodiff::var;
using autodiff::MatrixXvar;
using autodiff::VectorXvar;
using autodiff::gradient;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using autodiff::wrt;
using autodiff::at;

//超参数
const int batch=40;
const double lr=0.01;
const int neu1=16;
const int neu2=8;

//标准化
template<typename T>
void Normalize(T &param){
    auto mean=param.mean();
    auto std=param.std();
    param=(param-mean)/std;
}

//单个样本
struct Sample{
    var x1;
    var x2;
    var label;
    Sample(var a,var b,var c):x1(a),x2(b),label(c){};
};

//每个batch打乱选择
unordered_set<int> Used; 
unordered_map<int,Sample> SamplesMap;
random_device rd;
mt19937 gen(rd());
uniform_int_distribution<int> dis;
tuple<MatrixXvar,VectorXvar> Shuffle(uniform_int_distribution<int> dis){
    vector<var> x;
    vector<var> y;
    MatrixXvar X;
    VectorXvar Y;
    x.reserve(batch*2);
    y.reserve(batch);
    while(true){
        int choice=dis(gen);
        if(Used.find(choice)==Used.end()) continue;
        if(y.size()>=batch) break;
        x.emplace_back(SamplesMap[choice].x1);
        x.emplace_back(SamplesMap[choice].x2)
        y.emplace_back(SamplesMap[choice].label);
    }
    X=Eigen::Map<MatrixXvar>(x.data(),batch,2);
    Y=Eigen::Map<VectorXvar>(y.data(),batch);
    return make_tuple<X,Y>;
}


//激活函数
MatrixXvar Sigmoid(MatrixXvar& x){
    MatrixXvar out= 1/(1+(-x).array().exp());
    return out;
}
MatrixXvar Tanh(MatrixXvar& x){
    x=x.array().tanh();
    MatrixXvar out=x;
    return out;
}

//acc计算并输出
class Acc{
    private:
        int sampleNum;
        int correct;
        double acc;
    public:
        Acc():sampleNum(0),acc(0){};
        void Match(VectorXvar& predict,VectorXvar& label){
            this.sampleNum+=predict.size();
            for(int i=0;i<predict.size();i++){
                if((predict(i)>=0.5 && label(i)==1)||(predict(i)<0.5 && label(i)==0)){
                    correct++;
                }
            }
        }
        void PrintAcc(){
            double acc=double(correct)/sampleNum;
            cout<<"当前准确率："<<acc<<endl;
            }
        };

//forward
var Forward(MatrixXvar& X,VectorXvar& label,MatrixXvar& W1,MatrixXvar& W2,MatrixXvar& W3,VectorXvar& b1,VectorXvar& b2,VectorXvar& b3,bool isVal=false,VectorXvar& handle_predict=VectorXvar()){
    Normalize(X);
    MatrixXvar h1=(X*W1)+b1;
    h1=Tanh(h1);
    VectorXvar h2=(h1*W2)+b2;
    h2=Tanh(h2);
    VectorXvar predict=(h2*W3)+b3;
    predict=Sigmoid(predict);
    if(isVal) {
        handle_predict=predict;
    }
    var loss=(predict-label).array().square().sum()/(2*batch);
    return loss;
}

//计算梯度
tuple<MatrixXd,MatrixXd,MatrixXd,MatrixXd,VectorXd,VectorXd,VectorXd> Diff(MatrixXvar& W1,MatrixXvar& W2,MatrixXvar& W3,VectorXvar& b1,VectorXvar& b2,VectorXvar& b3,bool isVal=false)
{   var loss=gradient(Forward,wrt(W1,W2,W3,b1,b2,b3),at(W1,W2,W3,b1,b2,b3));
    MatrixXd W1_=W1.cast<double>()-lr*grad(W1);
    MatrixXd W2_=W2.cast<double>()-lr*grad(W2);
    MatrixXd W3_=W3.cast<double>()-lr*grad(W3);
    VectorXd b1_=b1.cast<double>()-lr*grad(b1);
    VectorXd b2_=b2.cast<double>()-lr*grad(b2);
    VectorXd b3_=b3.cast<double>()-lr*grad(b3);
    return make_tuple<W1_,W2_,W3_,b1_,b2_,b3_>;
}

int main(){
    //文件读取
    ifstream fin("../moons.csv");
    if(!fin){
        cerr<<"无法打开文件"<<endl;
        return -1;
    }
    string line;
    int index=0;
    while(getline(fin,line)){
        string a,b,c;
        stringstream ss(line);
        getline(ss,a,',');
        getline(ss,b,',');
        getline(ss,c,',');
        double x1,x2;
        int label;
        x1=stod(a);
        x2=stod(b);
        label=stoi(c);
        cout<<"读取一个"<<endl;
        SamplesMap[index]=Sample(x1,x2,label);
    }
        
    fin.close();
    cout<<"读取完毕"<<endl;

    //权重偏置初始化
    MatrixXvar W1(2,neu1);
    MatrixXvar W2(neu1,neu2);
    MatrixXvar W3(neu2,1);
    VectorXvar b1=VectorXvar::Zero(neu1);
    VectorXvar b2=VectorXvar::Zero(neu2);
    VectorXvar b3=VectorXvar::Zero(1);
    W1.setRandom();
    W2.setRandom();
    W3.setRandom();
    Normalize(W1);
    Normalize(W2);
    Normalize(W3);      

    //分布器赋值
    int totalNum=SamplesMap.size();
    uniform_int_distribution<int> dis(0,totalNum-1);

    //训练
    int count=0;
    while(count<=int(totalNum*0.8)){
        Acc trainacc=Acc();
        auto[X,label]=Shuffle(dis);
        MatrixXd W1_,W2_,W3_;
        VectorXd b1_,b2_,b3_;
        auto [W1_,W2_,W3_,b1_,b2_,b3_]=Diff(W1,W2,W3,b1,b2,b3);
        W1=W1_.cast<var>();
        W2=W2_.cast<var>();
        W3=W3_.cast<var>();
        b1=b1_.cast<var>();
        b2=b2_.cast<var>();
        b3=b3_.cast<var>();
        count++;
    }

    //验证
    Acc valacc=Acc();
    while(count<=totalNum*0.2){
        auto [X,label]=Shuffle(dis);
        VectorXvar predict;
        var loss=Forward(X,label,W1,W2,W3,b1,b2,b3,predict);
        valacc.Match(predict,label);
    }
    valacc.PrintAcc();
    return 0;
}

