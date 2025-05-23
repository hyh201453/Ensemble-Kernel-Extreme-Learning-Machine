
function [TrainOutT,OutputWeight] = kelmTrain(P_train,T_train,Regularization_coefficient,Kernel_type,Kernel_para)
%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = P_train;
n = length(T_train);
C = Regularization_coefficient;
[maxClass] = max(T_train);
T = T_train;
Omega_train = kernel_matrix(P',Kernel_type, Kernel_para);
OutputWeight=((Omega_train+speye(n)/C)\(T'));
Y=(Omega_train * OutputWeight)';
TrainOutT = Y;
end