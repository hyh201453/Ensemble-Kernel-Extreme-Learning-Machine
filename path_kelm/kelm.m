function [R2,mae2,mbe,error2,mape,TestOutT,T_test]=kelm(Pn_train,Tn_train,Pn_test,T_test,outputps,N)
%% ��ȡ��������ϵ�� C �ͺ˺������� S
Regularization_coefficient = rand*20;
Kernel_para = rand*20;
Kernel_type = 'rbf';
%% ѵ��
[TrainOutT,OutputWeight] = kelmTrain(Pn_train,Tn_train,Regularization_coefficient,Kernel_type,Kernel_para);

%% ѵ����Ԥ��
InputWeight = OutputWeight;
[TestOutT] = kelmPredict(Pn_train,InputWeight,Kernel_type,Kernel_para,Pn_test);
%% ���Լ���ȷ��
TestOutT = mapminmax('reverse',TestOutT,outputps);
errorTest = TestOutT - T_test;
MSEErrorTest = mse(errorTest);

N1=length(T_test);
R2 = (N1*sum(TestOutT.*T_test)-sum(TestOutT)*sum(T_test))^2/((N1*sum((TestOutT).^2)-(sum(TestOutT))^2)*(N1*sum((T_test).^2)-(sum(T_test))^2)); 


%%  ���ָ�����
%  R2
R2 = 1 - norm(T_test -  TestOutT)^2 / norm(T_test -  mean(T_test ))^2;
disp(['���Լ����ݵ�R2Ϊ��', num2str(R2)])

%  MAE
mae2 = sum(abs(TestOutT - T_test )) ./ N ;
disp(['���Լ����ݵ�MAEΪ��', num2str(mae2)])
%  MBE
mbe = sum(TestOutT - T_test ) ./ N ;
disp(['���Լ����ݵ�MBEΪ��', num2str(mbe)])
%rmse
error2 = sqrt(sum((TestOutT - T_test ).^2) ./ N);
disp(['���Լ����ݵ�RMSEΪ��', num2str(error2)])
%mape
mape=mean(abs(TestOutT - T_test )./T_test);
disp(['���Լ����ݵ�MAPEΪ��', num2str(mape)])
