function [TestTime,label_index_actual,TestAccuracy] = test_kELM(train_data,test_data,InputWeight,Kernel_type,Kernel_para)
P = train_data(:,2:size(train_data,2))';
T = train_data(:,1)';
TV.P = test_data(:,2:size(test_data,2))';
TV.T = test_data(:,1)';
start_time_test=cputime;
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
Omega_test = kernel_matrix(P',Kernel_type, Kernel_para,TV.P');
TY=(Omega_test' * InputWeight)';                            
end_time_test=cputime;
TestTime=end_time_test-start_time_test;
MissClassificationRate_Testing=0;
for i = 1 : size(TV.P, 2)
    [x, label_index_expected(i)]=max(TV.T(:,i));
    [x, label_index_actual(i)]=max(TY(:,i));
    if label_index_actual(i)~=label_index_expected(i)
        MissClassificationRate_Testing=MissClassificationRate_Testing+1;
    end
end
TestAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);
TestAccuracy=TestAccuracy*100;
end