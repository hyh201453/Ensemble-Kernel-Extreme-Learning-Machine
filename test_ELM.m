function [TestTime, TestAccuracy] = test_ELM(InputData, ELM_Model, Elm_Type, ActivationFunction)
REGRESSION=0;
CLASSIFIER=1;
test_data=InputData;
TV.T=test_data(:,1)';                  
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;
start_time_test=cputime;
NumberofTestingData=size(TV.P,2);
if Elm_Type~=REGRESSION
    sorted_target=sort(TV.T,2);
    label=zeros(1,1);                   
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;                    
    NumberofOutputNeurons=number_class;
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
end
InputWeight=ELM_Model.InputWeight;
BiasofHiddenNeurons=ELM_Model.BiasofHiddenNeurons;
OutputWeight=ELM_Model.OutputWeight;
tempH_test=InputWeight*TV.P;
clear TV.P;  
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'} 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}   
        H_test = sin(tempH_test);        
    case {'hardlim'}      
        H_test = hardlim(tempH_test);        
    case {'tribas'}      
         H_test = tribas(tempH_test);        
    case {'radbas'}       
         H_test = radbas(tempH_test);                    
end
TY=(H_test' * OutputWeight)';
end_time_test=cputime;
TestTime=end_time_test-start_time_test;
if Elm_Type == REGRESSION
    TestAccuracy=sqrt(mse(TV.T - TY));
end
if Elm_Type == CLASSIFIER 
    MissClassificationRate_Testing=0;
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);
    TestAccuracy=TestAccuracy*100;
end