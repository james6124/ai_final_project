#include <iostream>

using namespace std;

int mul(int n){
    int ans=1;
    for(int i=1;i<=n;i++){
        ans=ans*i;
    }
    return ans;
}

int main(){
    int sum=0;
    for(int i=2;i<=169;i++){
        int up=mul(169)/mul(169-i);
        

    }
}