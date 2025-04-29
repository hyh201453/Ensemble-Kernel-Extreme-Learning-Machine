function [OutputWeight] = train_kELM(Regularization_coefficient,Kernel_para,train_data,valid_data,Kernel_type)
n = size(train_data,1);
C = Regularization_coefficient;
P = train_data(:,2:size(train_data,2))';
T = train_data(:,1)';
clear train_data;
TV.T=valid_data(:,1)';                 
TV.P=valid_data(:,2:size(valid_data,2))';
clear valid_data;
NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
sorted_target=sort(cat(2,T,TV.T),2);
label=zeros(1,1);                   
label(1,1)=sorted_target(1,1);
j=1;
for i = 2:(NumberofTrainingData+NumberofTestingData)
    if sorted_target(1,i) ~= label(1,j)
        j=j+1;
        label(1,j) = sorted_target(1,i);
    end
end
number_class=j;                    
NumberofOutputNeurons=number_class;
temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
for i = 1:NumberofTrainingData
    for j = 1:number_class
        if label(1,j) == T(1,i)
            break;
        end
    end
    temp_T(j,i)=1;               
end
T=temp_T*2-1;                   
temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
for i = 1:NumberofTestingData
    for j = 1:number_class
        if label(1,j) == TV.T(1,i)
            break;
        end
    end
    temp_TV_T(j,i)=1;            
end
TV.T=temp_TV_T*2-1;             
Omega_train = kernel_matrix(P',Kernel_type, Kernel_para);
OutputWeight=((Omega_train+speye(n)/C)\(T'));
Y=(Omega_train * OutputWeight)';
MissClassificationRate_Training=0;
for i = 1 : size(T, 2)
    [x, label_index_expected]=max(T(:,i));
    [x, label_index_actual]=max(Y(:,i));
    if label_index_actual~=label_index_expected
        MissClassificationRate_Training=MissClassificationRate_Training+1;
    end
end
TrainAccuracy=1-MissClassificationRate_Training/size(T,2);
Omega_test = kernel_matrix(P',Kernel_type, Kernel_para,TV.P');
TY=(Omega_test' * OutputWeight)';                         
MissClassificationRate_Testing=0;
for i = 1 : size(TV.T, 2)
    [x, label_index_expected]=max(TV.T(:,i));
    [x, label_index_actual]=max(TY(:,i));
    if label_index_actual~=label_index_expected
        MissClassificationRate_Testing=MissClassificationRate_Testing+1;
    end
end
ValidAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);
end