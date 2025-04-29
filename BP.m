function [TestingTime, TestingAccuracy] = BP(train_data, test_data)
classes=18;
neurons=200;
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;
[P,minI,maxI] = premnmx(P);
TV.P = tramnmx ( TV.P , minI, maxI ) ;
s=length(T');
output=zeros(s,18);
for i=1:s
    output(i,T(i)')=1;
end
net=newff(minmax(P),[neurons classes],{'tansig' 'tansig'},'traingdx');
net.trainparam.show = 50 ;
net.trainparam.epochs = 1000 ;
net.trainparam.goal = 0.001 ;
net.trainParam.lr = 0.05 ;
net=train(net,P,output');
start_time_test=cputime;
Y=sim(net,TV.P);
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;
[s1,s2]=size(Y);
T_sim=[];
hitNum=0;
for i=1:s2
    [m,Index(i)]=max(Y(:,i));
    T_sim(i)=Index(i);
    if(Index(i)==TV.T(i))
        hitNum=hitNum+1;
    end
end
TestingAccuracy = 100 * hitNum / s2;